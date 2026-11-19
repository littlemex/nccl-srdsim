// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// The simulator's failure paths, driven from a hostile peer.
//
// The happy paths are covered by `test_srdsim_transport.cpp`. This file only sends things a correct
// sender would never send, because that is where a simulator earns its keep: a fabric emulator that
// crashes on a malformed frame cannot be used to investigate a crash.
//
// Every case here maps to a numbered item in `layer2/CONTRACT.md`, and every case was chosen from the
// contract before the fix was written, so that the test says what the contract says rather than what
// the code happens to do.
//
//   A1  contract I1. A frame whose length is enormous is refused without allocating it, the link is
//       closed, and the process survives. On 2026-09-04 a desynchronised stream produced exactly this
//       and survived only because the garbage length happened to be small.
//   A2  contract I2. `offset + length` overflowing 64 bits does not pass the bounds check.
//   A3  contract I3. The same, on the signal path.
//   A4  contract I4. `serve` after `stop` is refused rather than starting a thread nobody joins.
//   A5  contract I6. Rebinding a peer closes the descriptor it replaces.
//   A6  contract I7. An operation id the scheduler never issued is not reported complete.
//   A7  contract I8. A zero chunk size is refused with a reason instead of dividing by zero.
//   A8  contract I9. A frame the peer got wrong and a write that failed locally move different
//       counters, because they have different owners.
//   A9  contract C-WIRE-2. After each refusal the stream is still framed: a well-formed frame that
//       follows still applies. Checked for every refusal that is survivable.
//   A10 contract C-WIRE-3. A refusal that cannot leave the stream framed closes the link instead.
// Not here, and worth saying why: the correctness of a device signal update. Two host-side designs were
// tried and both were wrong -- a read-modify-write loses increments between agents, and a host-side
// absolute store overwrites the reset the GPU performs. The mechanism is now a device-side atomic add,
// which cannot be checked without a GPU. `pseudo/test_plugin_conformance.cpp` covers everything about
// the plugin that is host-side; this one property is verified on hardware and nowhere else.
//   A11 contract I5. `shutdown_all` racing a sender never leaves a half-written frame on the wire.
//       The receiver is the witness: a Shutdown header interleaved into the middle of a payload would
//       desynchronise it, and a desynchronised receiver reports refusals.
//
// Build: c++ -std=c++17 -O2 -I layer2 tests/test_srdsim_adversarial.cpp -o test_srdsim_adversarial

#include <atomic>
#include <chrono>
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include "srdsim_schedule.h"
#include "srdsim_transport.h"

namespace {

int g_checks = 0;
std::vector<std::string> g_failures;

void expect(bool cond, const std::string& what) {
    ++ g_checks;
    if (not cond) g_failures.push_back(what);
}

std::string ctx(const char* fmt, ...) {
    char buf[400];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return std::string(buf);
}

constexpr size_t kWindowBytes = 1024;

// A receiver in its own process, so that a defect which kills the process is observed as a dead child
// rather than as a dead test run. That distinction is the point of A1: the previous code could
// terminate on a wire field, and a same-process test would simply have vanished.
struct Report {
    uint64_t applied_placements = 0;
    uint64_t applied_signals = 0;
    uint64_t refused_frames = 0;
    uint64_t failed_writes = 0;
    uint64_t errors = 0;
    unsigned char window[kWindowBytes] = {0};
    uint64_t connections = 0;
    // Set by the child once it has finished serving, so a truncated report is distinguishable from a
    // report that says zero.
    uint64_t sentinel = 0;
};

constexpr uint64_t kSentinel = 0x5645'5249'4649'4544ull;

// Runs a receiver on one connection, applies whatever the parent sends, and reports.
// `signal_window_is_device` forces the signal window through the device hooks, which is how a local
// write failure is provoked without a GPU: the hook simply refuses.
struct ChildResult {
    bool exited_cleanly = false;
    int exit_status = -1;
    Report report{};
};

bool refuse_device_copy(void*, const void*, size_t, void*) { return false; }
bool refuse_device_add(void*, uint64_t, void*) { return false; }

[[noreturn]] void run_receiver(int listen_fd, int report_fd, bool device_windows,
                               size_t data_window_bytes) {
    std::vector<char> data(data_window_bytes, 0);
    std::vector<uint64_t> signals(kWindowBytes / sizeof(uint64_t), 0);

    srdsim::WindowTable windows;
    windows.add({data.data(), data.size(), device_windows});
    windows.add({signals.data(), signals.size() * sizeof(uint64_t), device_windows});

    srdsim::ReceiveAgent::Hooks hooks{};
    hooks.log = [](const char*) {};
    if (device_windows) {
        hooks.device_copy = refuse_device_copy;
        hooks.device_add = refuse_device_add;
    }
    srdsim::ReceiveAgent agent(&windows, hooks);

    const int conn = ::accept(listen_fd, nullptr, nullptr);
    ::close(listen_fd);
    if (conn < 0) _exit(2);
    srdsim::set_nodelay(conn);
    if (not agent.serve(conn)) _exit(3);
    agent.join();

    Report r{};
    r.applied_placements = agent.applied_placements();
    r.applied_signals = agent.applied_signals();
    r.refused_frames = agent.refused_frames();
    r.failed_writes = agent.failed_writes();
    r.errors = agent.errors();
    r.connections = agent.connections();
    std::memcpy(r.window, data.data(), std::min(kWindowBytes, data.size()));
    r.sentinel = kSentinel;
    const bool ok = srdsim::write_all(report_fd, &r, sizeof(r));
    ::close(report_fd);
    _exit(ok ? 0 : 4);
}

// The parent half: forks a receiver, hands the caller a connected socket to be hostile on, and
// collects the report.
struct Session {
    int fd = -1;
    pid_t pid = -1;
    int report_fd = -1;
};

// `data_window_bytes` sizes window 0. It is a parameter because one case needs a window larger than
// the socket's send buffer: a payload that cannot be written in one `send` is what puts the sender
// inside a frame long enough for a teardown to race it, and a 1 KiB payload never does.
bool open_session(Session* s, bool device_windows, size_t data_window_bytes = kWindowBytes) {
    uint16_t port = 0;
    const int listen_fd = srdsim::listen_any(&port);
    if (listen_fd < 0) return false;
    int sv[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) { ::close(listen_fd); return false; }

    const pid_t pid = ::fork();
    if (pid < 0) { ::close(listen_fd); return false; }
    if (pid == 0) {
        ::close(sv[0]);
        run_receiver(listen_fd, sv[1], device_windows, data_window_bytes);
        _exit(9);   // not reached
    }
    ::close(sv[1]);
    ::close(listen_fd);
    const int fd = srdsim::dial("127.0.0.1", port);
    if (fd < 0) { ::waitpid(pid, nullptr, 0); ::close(sv[0]); return false; }
    s->fd = fd;
    s->pid = pid;
    s->report_fd = sv[0];
    return true;
}

ChildResult close_session(Session* s) {
    ChildResult out;
    ::close(s->fd);
    Report r{};
    const bool got = srdsim::read_all(s->report_fd, &r, sizeof(r));
    ::close(s->report_fd);
    int status = 0;
    ::waitpid(s->pid, &status, 0);
    out.exited_cleanly = WIFEXITED(status);
    out.exit_status = out.exited_cleanly ? WEXITSTATUS(status) : -1;
    if (got and r.sentinel == kSentinel) out.report = r;
    return out;
}

srdsim::WireHeader placement_header(uint32_t window, uint64_t offset, uint64_t length) {
    srdsim::WireHeader h{};
    h.kind = static_cast<uint32_t>(srdsim::WireKind::Placement);
    h.window_id = window;
    h.offset = offset;
    h.length = length;
    return h;
}

srdsim::WireHeader signal_header(uint32_t window, uint64_t offset, uint64_t value) {
    srdsim::WireHeader h{};
    h.kind = static_cast<uint32_t>(srdsim::WireKind::Signal);
    h.window_id = window;
    h.offset = offset;
    h.length = 0;
    h.value = value;
    return h;
}

// Sends a well-formed placement of `n` bytes of 'A' at offset 0 of window 0. Used after every
// survivable refusal to show the stream is still framed.
bool send_good_placement(int fd, size_t n) {
    const auto h = placement_header(0, 0, n);
    std::vector<char> payload(n, 'A');
    return srdsim::write_all(fd, &h, sizeof(h)) and srdsim::write_all(fd, payload.data(), n);
}

bool send_shutdown(int fd) {
    srdsim::WireHeader bye{};
    bye.kind = static_cast<uint32_t>(srdsim::WireKind::Shutdown);
    return srdsim::write_all(fd, &bye, sizeof(bye));
}

// ---------------------------------------------------------------------------------------------
// A1, A9, A10: an over-long length
// ---------------------------------------------------------------------------------------------

void test_absurd_length() {
    Session s;
    if (not open_session(&s, /*device_windows=*/false)) {
        expect(false, "A1 the session opened");
        return;
    }
    // Two thirds of the address space. A receiver that allocates this either dies or swaps the machine
    // to death; a receiver that trusts it to drain the payload waits forever.
    const auto h = placement_header(0, 0, uint64_t(1) << 63);
    expect(srdsim::write_all(s.fd, &h, sizeof(h)), "A1 the over-long header was written");
    // The contract says the link is closed, so this write may or may not succeed depending on timing.
    // Either is acceptable; what is not acceptable is the child dying.
    (void)send_good_placement(s.fd, 8);
    (void)send_shutdown(s.fd);

    const auto res = close_session(&s);
    expect(res.exited_cleanly and res.exit_status == 0,
           ctx("A1 the receiver survived an over-long length and reported (exited=%d status=%d)",
               int(res.exited_cleanly), res.exit_status));
    expect(res.report.refused_frames >= 1,
           ctx("A1 the over-long frame was refused (refused=%llu)",
               (unsigned long long)res.report.refused_frames));
    expect(res.report.applied_placements == 0,
           ctx("A10 nothing was applied after the link was closed (applied=%llu)",
               (unsigned long long)res.report.applied_placements));
    expect(res.report.failed_writes == 0,
           ctx("A8 an over-long frame is a peer fault, not a local write failure (failed=%llu)",
               (unsigned long long)res.report.failed_writes));
}

// ---------------------------------------------------------------------------------------------
// A2: offset + length overflowing
// ---------------------------------------------------------------------------------------------

void test_offset_length_overflow() {
    Session s;
    if (not open_session(&s, false)) { expect(false, "A2 the session opened"); return; }

    // offset + length wraps to 8, which is inside a 1024 byte window. A check written as
    // `offset + length > bytes` accepts this and writes at offset 2^64 - 8.
    const uint64_t offset = ~uint64_t(0) - 7;   // 2^64 - 8
    const auto h = placement_header(0, offset, 16);
    expect(srdsim::write_all(s.fd, &h, sizeof(h)), "A2 the overflowing header was written");
    (void)send_good_placement(s.fd, 8);
    (void)send_shutdown(s.fd);

    const auto res = close_session(&s);
    expect(res.exited_cleanly and res.exit_status == 0,
           ctx("A2 the receiver survived an overflowing range (exited=%d status=%d)",
               int(res.exited_cleanly), res.exit_status));
    expect(res.report.refused_frames >= 1,
           ctx("A2 the overflowing frame was refused (refused=%llu)",
               (unsigned long long)res.report.refused_frames));
    // Nothing may have been written into the window by that frame. A byte written at a wrapped
    // offset would land outside the window, so the window itself cannot witness it; what can is that
    // the frame was refused and the process is alive, which the two checks above cover.
}

// ---------------------------------------------------------------------------------------------
// A3: the same on the signal path
// ---------------------------------------------------------------------------------------------

void test_signal_offset_overflow() {
    Session s;
    if (not open_session(&s, false)) { expect(false, "A3 the session opened"); return; }

    // offset + 8 wraps to 0, which is inside any window. The signal path's check has the same shape
    // as the placement path's and the same overflow.
    const auto h = signal_header(1, ~uint64_t(0) - 7, 1);
    expect(srdsim::write_all(s.fd, &h, sizeof(h)), "A3 the overflowing signal header was written");
    expect(send_good_placement(s.fd, 4), "A9 a good frame follows a refused signal");
    (void)send_shutdown(s.fd);

    const auto res = close_session(&s);
    expect(res.exited_cleanly and res.exit_status == 0,
           ctx("A3 the receiver survived an overflowing signal offset (status=%d)", res.exit_status));
    expect(res.report.refused_frames >= 1,
           ctx("A3 the overflowing signal was refused (refused=%llu)",
               (unsigned long long)res.report.refused_frames));
    expect(res.report.applied_signals == 0,
           ctx("A3 no signal was applied (applied=%llu)",
               (unsigned long long)res.report.applied_signals));
    // A signal carries no payload, so the stream stays framed and the following placement must land.
    expect(res.report.applied_placements == 1,
           ctx("A9 the frame after a refused signal still applied (applied=%llu)",
               (unsigned long long)res.report.applied_placements));
    expect(res.report.window[0] == 'A' and res.report.window[3] == 'A' and res.report.window[4] == 0,
           "A9 and it applied exactly its own four bytes");
}

// ---------------------------------------------------------------------------------------------
// A9: an unknown window, which is survivable because the payload can be drained
// ---------------------------------------------------------------------------------------------

void test_unknown_window_keeps_stream_framed() {
    Session s;
    if (not open_session(&s, false)) { expect(false, "A9 the session opened"); return; }

    const auto h = placement_header(/*window=*/99, 0, 64);
    std::vector<char> payload(64, 'z');
    expect(srdsim::write_all(s.fd, &h, sizeof(h)) and
           srdsim::write_all(s.fd, payload.data(), payload.size()),
           "A9 the unknown-window frame was written");
    expect(send_good_placement(s.fd, 16), "A9 a good frame follows an unknown window");
    (void)send_shutdown(s.fd);

    const auto res = close_session(&s);
    expect(res.exited_cleanly and res.exit_status == 0,
           ctx("A9 the receiver survived an unknown window (status=%d)", res.exit_status));
    expect(res.report.refused_frames == 1,
           ctx("A9 exactly one frame was refused (refused=%llu)",
               (unsigned long long)res.report.refused_frames));
    expect(res.report.applied_placements == 1,
           ctx("A9 the following frame still applied (applied=%llu)",
               (unsigned long long)res.report.applied_placements));
    expect(res.report.window[0] == 'A' and res.report.window[15] == 'A' and res.report.window[16] == 0,
           "A9 the following frame applied its own bytes and nothing else");
}

// ---------------------------------------------------------------------------------------------
// A8: a peer fault and a local write failure are different numbers
// ---------------------------------------------------------------------------------------------

void test_peer_fault_and_local_failure_are_separate() {
    Session s;
    // Device windows with hooks that refuse: every write fails locally while the peer is blameless.
    if (not open_session(&s, /*device_windows=*/true)) {
        expect(false, "A8 the session opened");
        return;
    }
    expect(send_good_placement(s.fd, 32), "A8 a well-formed placement was sent");
    const auto sig = signal_header(1, 0, 1);
    expect(srdsim::write_all(s.fd, &sig, sizeof(sig)), "A8 a well-formed signal was sent");
    // And one frame the peer did get wrong, so both counters have something to hold.
    const auto bad = placement_header(99, 0, 0);
    expect(srdsim::write_all(s.fd, &bad, sizeof(bad)), "A8 an unknown-window frame was sent");
    (void)send_shutdown(s.fd);

    const auto res = close_session(&s);
    expect(res.exited_cleanly and res.exit_status == 0,
           ctx("A8 the receiver survived refused writes (status=%d)", res.exit_status));
    expect(res.report.failed_writes == 2,
           ctx("A8 both well-formed frames counted as failed writes (failed=%llu, want 2)",
               (unsigned long long)res.report.failed_writes));
    expect(res.report.refused_frames == 1,
           ctx("A8 the peer's mistake counted separately (refused=%llu, want 1)",
               (unsigned long long)res.report.refused_frames));
    expect(res.report.applied_placements == 0 and res.report.applied_signals == 0,
           ctx("A8 a failed write is not an application (placements=%llu signals=%llu)",
               (unsigned long long)res.report.applied_placements,
               (unsigned long long)res.report.applied_signals));
    expect(res.report.errors == res.report.refused_frames + res.report.failed_writes,
           "A8 errors() still equals the sum, so existing callers keep their meaning");
}

// ---------------------------------------------------------------------------------------------
// A4: serve after stop
// ---------------------------------------------------------------------------------------------

void test_serve_after_stop_is_refused() {
    std::vector<char> data(kWindowBytes, 0);
    srdsim::WindowTable windows;
    windows.add({data.data(), data.size(), false});

    srdsim::ReceiveAgent::Hooks hooks{};
    hooks.log = [](const char*) {};
    srdsim::ReceiveAgent agent(&windows, hooks);

    expect(agent.connections() == 0, "A4 a fresh agent holds no connections");
    agent.stop();

    int sv[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        expect(false, "A4 a socket pair was created");
        return;
    }
    const bool accepted = agent.serve(sv[0]);
    expect(not accepted, "A4 serve after stop is refused");
    expect(agent.connections() == 0, "A4 and no connection was recorded");
    // The agent took no ownership, so closing here is the caller's job and must not double-close.
    ::close(sv[0]);
    ::close(sv[1]);
}

// ---------------------------------------------------------------------------------------------
// A5: rebinding a peer closes the descriptor it replaces
// ---------------------------------------------------------------------------------------------

void test_rebind_closes_the_old_descriptor() {
    int a[2], b[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, a) != 0) { expect(false, "A5 first pair"); return; }
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, b) != 0) { expect(false, "A5 second pair"); return; }

    srdsim::SendLinks links;
    links.set(0, a[0]);
    links.set(0, b[0]);   // rebinding peer 0

    // The old descriptor must be closed. `fcntl` with F_GETFD is the cheap way to ask.
    const bool old_closed = ::fcntl(a[0], F_GETFD) == -1;
    expect(old_closed, "A5 the replaced descriptor was closed");
    expect(::fcntl(b[0], F_GETFD) != -1, "A5 the new descriptor is still open");
    expect(links.fd_for(0) == b[0], "A5 the peer is bound to the new descriptor");

    links.shutdown_all();
    ::close(a[1]);
    ::close(b[1]);
}

// ---------------------------------------------------------------------------------------------
// A6, A7: the scheduler's refusals
// ---------------------------------------------------------------------------------------------

void test_scheduler_refusals() {
    {
        srdsim::Scheduler::Options o;
        o.chunk_bytes = 0;
        srdsim::Scheduler sched(o);
        srdsim::Op op;
        op.id = 1;
        op.bytes = 128;
        std::string why;
        expect(not sched.submit(op, &why), "A7 a zero chunk size is refused");
        expect(why.find("chunk") != std::string::npos,
               ctx("A7 and the reason names the option (%s)", why.c_str()));
    }
    {
        srdsim::Scheduler::Options o;
        srdsim::Scheduler sched(o);
        expect(not sched.complete(7),
               "A6 an id the scheduler never issued is not reported complete");
        expect(sched.issued_ids() == 0, "A6 no id has been issued yet");

        srdsim::Op op;
        op.id = 7;
        op.bytes = 0;            // no placements, so the operation is trivial
        op.has_signal = false;
        std::string why;
        expect(sched.submit(op, &why), ctx("A6 the operation was accepted (%s)", why.c_str()));
        expect(sched.issued_ids() == 7, "A6 the largest issued id is reported");
        // Nothing to release, so it is complete once reaped; before that it is pending.
        std::vector<srdsim::Unit> batch;
        sched.progress(4, &batch);
        expect(sched.complete(7), "A6 an id that was issued and finished is complete");
        expect(not sched.complete(8), "A6 an id above the largest issued is still refused");
    }
}

// ---------------------------------------------------------------------------------------------
// A11: shutdown_all racing a sender
// ---------------------------------------------------------------------------------------------

void test_shutdown_does_not_interleave_with_a_sender() {
    // Four mebibytes, which no default socket send buffer holds, so `write_all` for the payload needs
    // many `send` calls and blocks in the middle of the frame. That is the state a teardown has to be
    // safe against, and a small payload almost never reaches it.
    constexpr size_t kBigWindow = 4u << 20;
    Session s;
    if (not open_session(&s, /*device_windows=*/false, kBigWindow)) {
        expect(false, "A11 the session opened");
        return;
    }

    // A small send buffer is what makes this deterministic rather than lucky. With the default buffer
    // the kernel swallows a whole frame in one call and the sender is almost never inside one, so the
    // race the test exists for is reached only by chance -- the first version of this test passed
    // against the very defect it was written for. Four kilobytes against a four mebibyte payload means
    // the sender is blocked in `send` for essentially the entire run.
    int small = 4096;
    ::setsockopt(s.fd, SOL_SOCKET, SO_SNDBUF, &small, sizeof(small));

    srdsim::SendLinks links;
    links.set(/*peer=*/0, s.fd);

    // One thread sends well-formed frames as fast as it can; the main thread shuts the link down
    // underneath it. If the Shutdown frame or the close can slip between a header and its payload, the
    // receiver reads a header from inside a payload and starts refusing frames. That is the observable
    // this checks, because "a descriptor was closed under a writer" has no direct witness.
    constexpr size_t kPayload = kBigWindow;
    std::vector<char> payload(kPayload, 'B');
    std::atomic<uint64_t> sent{0};
    std::atomic<bool> stop_sending{false};

    std::thread sender([&] {
        while (not stop_sending.load(std::memory_order_acquire)) {
            if (not links.send_placement(0, 0, 0, payload.data(), kPayload, 1))
                break;                       // the link went away, which is the expected ending
            ++ sent;
            // Queueing is not sending. With a non-blocking transport the frame leaves only when
            // somebody pushes it, and in production that somebody is the progress loop.
            if (not links.flush(0)) break;
        }
    });

    // Waiting on `sent` is no longer enough. A send now returns once the frame is queued, not once it
    // has left, so `sent` climbs instantly and says nothing about what reached the peer. What does say
    // it is the outbox draining to empty: those bytes are on the wire. Tearing down before that would
    // close the link with everything still queued, and the receiver would have applied nothing -- which
    // is exactly how this test started failing when the transport became non-blocking.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (std::chrono::steady_clock::now() < deadline) {
        links.flush(0);
        if (sent.load() >= 1 and links.pending_bytes(0) == 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    expect(sent.load() >= 1, "A11 at least one frame was queued");
    // Now the sender is inside the next frame, with a four mebibyte payload against a four kilobyte
    // socket buffer, so a teardown lands mid-frame.
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    // The order matters and getting it wrong deadlocks. `shutdown_all` waits for the peer's send lock,
    // which is exactly what the fix under test makes it do, and a sender in a loop re-acquires that
    // lock immediately after every frame. Telling the sender to stop first bounds the wait to the one
    // frame already in flight, which is still mid-frame and still the race this test exists for. The
    // first version called shutdown first and starved on a machine whose receiver drained more slowly
    // than the laptop it was written on.
    stop_sending.store(true, std::memory_order_release);
    links.shutdown_all();
    sender.join();

    // `close_session` closes `s.fd`, which `shutdown_all` already closed. Clear it so the descriptor
    // is not closed twice; a double close can hit a descriptor the runtime has since reused.
    s.fd = -1;
    Report r{};
    const bool got = srdsim::read_all(s.report_fd, &r, sizeof(r));
    ::close(s.report_fd);
    int status = 0;
    ::waitpid(s.pid, &status, 0);

    expect(got and r.sentinel == kSentinel, "A11 the receiver reported");
    expect(WIFEXITED(status) and WEXITSTATUS(status) == 0,
           ctx("A11 the receiver exited cleanly (exited=%d status=%d)",
               int(WIFEXITED(status)), WIFEXITED(status) ? WEXITSTATUS(status) : -1));
    expect(r.refused_frames == 0,
           ctx("A11 the receiver never read a header from inside a payload (refused=%llu, after %llu "
               "frames were queued)",
               (unsigned long long)r.refused_frames, (unsigned long long)sent.load()));
    expect(r.failed_writes == 0,
           ctx("A11 and nothing failed locally (failed=%llu)",
               (unsigned long long)r.failed_writes));
    // Deliberately not asserted: that a frame landed. It cannot be made to hold reliably here and
    // pretending otherwise cost three redesigns of this test. A four mebibyte frame is torn down while
    // in flight on purpose, so whether the receiver finishes reading it before the link ends is a
    // race by construction -- which is the point of the case, and the reason the outcome is "no
    // desynchronisation and no crash" rather than "n frames arrived". That a well-formed frame lands
    // is what T1 in the transport test establishes, without a teardown racing it.
    //
    // The exhaustive framing model covers the property this case is really about, over every frame
    // sequence rather than one: no reachable state leaves the receiver reading a header from inside a
    // payload. This case exists to show the same thing survives a real socket and a real teardown.
}

}   // namespace

int main() {
    test_absurd_length();
    test_offset_length_overflow();
    test_signal_offset_overflow();
    test_unknown_window_keeps_stream_framed();
    test_peer_fault_and_local_failure_are_separate();
    test_serve_after_stop_is_refused();
    test_rebind_closes_the_old_descriptor();
    test_scheduler_refusals();
    test_shutdown_does_not_interleave_with_a_sender();

    if (not g_failures.empty()) {
        std::printf("FAIL: %d checks, %zu failures\n", g_checks, g_failures.size());
        for (const auto& f : g_failures) std::printf("  [FAIL] %s\n", f.c_str());
        return 1;
    }
    std::printf("PASS: %d checks, 0 failures\n", g_checks);
    return 0;
}
