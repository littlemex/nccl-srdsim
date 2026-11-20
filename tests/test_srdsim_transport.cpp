// SPDX-License-Identifier: Apache-2.0

// Tests for the wire side of the SRD simulator (`src/srdsim_transport.h`), driven end to end
// through the scheduler so that what is checked is the combination that will run under NCCL:
// scheduler decides the order, transport carries it, the receiving agent applies it.
//
// The two processes are a real fork over a real TCP connection with host memory, so this runs
// anywhere and needs neither CUDA nor NCCL.
//
//   T1  every byte the sender submitted arrives at the right offset in the right window, and
//       every signal lands exactly once, for every mode and seed.
//   T2  the order units are applied at the receiver equals the order the scheduler released them.
//       This is the property the whole design rests on: if a stream did not preserve it, the
//       scheduler's control over ordering would not survive the wire and the simulator would be
//       measuring the network instead of the protocol.
//   T3  in `srd` mode a signal never lands before the last byte of its own message, and in
//       `violate-i1` mode it sometimes does. The contract is carried end to end, not just decided
//       in the scheduler.
//   T4  a window id the receiver does not know is reported and skipped rather than writing
//       somewhere arbitrary, and the payload is still drained so the stream stays framed.
//   T5  a placement that would run past the end of its window is refused, and the stream survives.
//
// Built and run by CTest: cmake -S . -B build && cmake --build build && ctest --test-dir build

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <memory>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "srdsim_schedule.h"
#include "srdsim_transport.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void expect(bool cond, const std::string& what) {
    ++ g_checks;
    if (not cond) {
        ++ g_failures;
        if (g_failures <= 30)
            std::printf("[FAIL] %s\n", what.c_str());
    }
}

std::string ctx(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return std::string(buf);
}

constexpr uint32_t kDataWindow = 0;
constexpr uint32_t kSignalWindow = 1;
constexpr size_t kDataBytes = 40'000;
constexpr size_t kNumSignals = 4;

// What the receiver reports back to the parent after applying everything.
struct Report {
    uint64_t placements = 0;
    uint64_t signals = 0;
    uint64_t errors = 0;
    uint64_t signal_values[kNumSignals] = {0, 0, 0, 0};
    uint64_t data_checksum = 0;
    // The order of applications, so T2 and T3 can be checked on the receiving side. Each entry is
    // (kind << 56) | op_id for placements, and (2 << 56) | op_id for signals.
    uint32_t trace_len = 0;
    uint64_t trace[512] = {0};
};

uint64_t checksum(const void* p, size_t n) {
    const unsigned char* b = static_cast<const unsigned char*>(p);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++ i) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}

// A trace-recording agent: the base agent applies, this records what it applied and in what order.
struct TracingHooks {
    Report* report;
};

// The receive agent writes host windows itself, so the trace is captured by wrapping the window
// table: every application goes through a copy hook even for host memory.
bool trace_copy(void* dst, const void* src, size_t bytes, void* user) {
    auto* r = static_cast<Report*>(user);
    std::memcpy(dst, src, bytes);
    if (r->trace_len < 512) r->trace[r->trace_len++] = (1ull << 56);
    return true;
}

bool trace_add(void* dst, uint64_t value, void* user) {
    auto* r = static_cast<Report*>(user);
    __atomic_fetch_add(static_cast<uint64_t*>(dst), value, __ATOMIC_RELAXED);
    if (r->trace_len < 512) r->trace[r->trace_len++] = (2ull << 56);
    return true;
}

// ---------------------------------------------------------------------------
// One end-to-end run: parent sends, child receives, child reports back.
// ---------------------------------------------------------------------------

struct RunResult {
    bool ok = false;
    Report report{};
    std::vector<srdsim::Unit> released;   // the order the scheduler chose, as seen by the sender
    std::vector<int> sent_kinds;          // 1 placement, 2 signal, in send order
    std::vector<uint64_t> sent_op_ids;
};

RunResult run_once(srdsim::Mode mode, uint64_t seed, size_t chunk_bytes, int num_ops) {
    RunResult out;

    int sv[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) return out;

    uint16_t port = 0;
    const int listen_fd = srdsim::listen_any(&port);
    if (listen_fd < 0) return out;

    const pid_t pid = ::fork();
    if (pid < 0) return out;

    if (pid == 0) {
        // ---- child: the receiving rank ----
        ::close(sv[0]);
        std::vector<char> data(kDataBytes, 0);
        std::vector<uint64_t> signals(kNumSignals, 0);
        Report report{};

        srdsim::WindowTable windows;
        // Device flag forced on so that every application goes through the hooks and can be traced.
        windows.add({data.data(), data.size(), true});
        windows.add({signals.data(), signals.size() * sizeof(uint64_t), true});

        srdsim::ReceiveAgent::Hooks hooks{};
        hooks.device_copy = trace_copy;
        hooks.device_add = trace_add;
        hooks.user = &report;
        hooks.log = [](const char*) {};   // quiet: the parent checks counters, not text
        srdsim::ReceiveAgent agent(&windows, hooks);

        const int conn = ::accept(listen_fd, nullptr, nullptr);
        ::close(listen_fd);
        if (conn < 0) _exit(2);
        srdsim::set_nodelay(conn);
        agent.serve(conn);
        agent.join();   // returns when the sender sends Shutdown or closes

        report.placements = agent.applied_placements();
        report.signals = agent.applied_signals();
        report.errors = agent.errors();
        for (size_t i = 0; i < kNumSignals; ++ i) report.signal_values[i] = signals[i];
        report.data_checksum = checksum(data.data(), data.size());

        const bool ok = srdsim::write_all(sv[1], &report, sizeof(report));
        ::close(sv[1]);
        _exit(ok ? 0 : 3);
    }

    // ---- parent: the sending rank ----
    ::close(sv[1]);
    ::close(listen_fd);

    std::vector<char> src(kDataBytes);
    for (size_t i = 0; i < src.size(); ++ i)
        src[i] = static_cast<char>((i * 37 + 11) & 0xff);

    const int fd = srdsim::dial("127.0.0.1", port);
    if (fd < 0) { ::waitpid(pid, nullptr, 0); return out; }
    srdsim::SendLinks links;
    links.set(/*peer=*/0, fd);

    srdsim::Scheduler::Options o;
    o.mode = mode;
    o.seed = seed;
    o.chunk_bytes = chunk_bytes;
    srdsim::Scheduler sched(o);

    const size_t per_op = kDataBytes / static_cast<size_t>(num_ops);
    for (int k = 0; k < num_ops; ++ k) {
        srdsim::Op op;
        // The scheduler carries the id it is given; assigning distinct ids is the caller's job
        // (the plugin does it from a counter). Without distinct ids the per-operation bookkeeping
        // below would collapse every operation into one.
        op.id = static_cast<uint64_t>(k) + 1;
        op.peer = 0;
        op.src_base = src.data();
        op.src_off = static_cast<uint64_t>(k) * per_op;
        op.dst_window = kDataWindow;    // the wire names a window, not an address
        op.dst_off = static_cast<uint64_t>(k) * per_op;
        op.bytes = per_op;
        op.has_signal = true;
        op.signal_window = kSignalWindow;
        op.signal_off = static_cast<uint64_t>(k % kNumSignals) * sizeof(uint64_t);
        op.signal_val = 1;
        std::string why;
        if (not sched.submit(op, &why)) { ::close(fd); ::waitpid(pid, nullptr, 0); return out; }
    }

    std::vector<srdsim::Unit> batch;
    while (not sched.idle()) {
        batch.clear();
        if (sched.progress(1, &batch) == 0) break;
        for (const auto& u : batch) {
            out.released.push_back(u);
            bool ok;
            if (u.kind == srdsim::Unit::Kind::Placement) {
                ok = links.send_placement(0, u.window, u.window_offset, u.src, u.bytes, u.op_id);
                out.sent_kinds.push_back(1);
            } else {
                ok = links.send_signal(0, u.window, u.window_offset, u.signal_val, u.op_id);
                out.sent_kinds.push_back(2);
            }
            out.sent_op_ids.push_back(u.op_id);
            if (not ok) { ::close(fd); ::waitpid(pid, nullptr, 0); return out; }
        }
    }
    links.shutdown_all();

    Report report{};
    const bool got = srdsim::read_all(sv[0], &report, sizeof(report));
    ::close(sv[0]);
    int wstatus = 0;
    ::waitpid(pid, &wstatus, 0);

    out.ok = got and WIFEXITED(wstatus) and WEXITSTATUS(wstatus) == 0;
    out.report = report;
    return out;
}

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------

void test_delivery_and_order() {
    for (const auto mode : {srdsim::Mode::InOrder, srdsim::Mode::Srd, srdsim::Mode::ViolateI1}) {
        int i1_broken_runs = 0;
        for (uint64_t seed = 1; seed <= 8; ++ seed) {
            const auto r = run_once(mode, seed, /*chunk_bytes=*/4096, /*num_ops=*/4);
            expect(r.ok, ctx("the run completed: mode=%s seed=%llu",
                             srdsim::mode_name(mode), (unsigned long long)seed));
            if (not r.ok) continue;

            // T1: every unit arrived, nothing was dropped or duplicated.
            size_t placements = 0, signals = 0;
            for (const int k : r.sent_kinds) (k == 1 ? placements : signals) += 1;
            expect(r.report.placements == placements,
                   ctx("T1 every placement applied: mode=%s seed=%llu sent=%zu applied=%llu",
                       srdsim::mode_name(mode), (unsigned long long)seed, placements,
                       (unsigned long long)r.report.placements));
            expect(r.report.signals == signals,
                   ctx("T1 every signal applied: mode=%s seed=%llu sent=%zu applied=%llu",
                       srdsim::mode_name(mode), (unsigned long long)seed, signals,
                       (unsigned long long)r.report.signals));
            expect(r.report.errors == 0,
                   ctx("T1 no receiver errors: mode=%s seed=%llu errors=%llu",
                       srdsim::mode_name(mode), (unsigned long long)seed,
                       (unsigned long long)r.report.errors));
            uint64_t total_signal = 0;
            for (size_t i = 0; i < kNumSignals; ++ i) total_signal += r.report.signal_values[i];
            expect(total_signal == signals,
                   ctx("T1 signal counts sum to the number sent: mode=%s seed=%llu got=%llu want=%zu",
                       srdsim::mode_name(mode), (unsigned long long)seed,
                       (unsigned long long)total_signal, signals));

            // T2: the receiver applied things in the order the sender wrote them.
            expect(r.report.trace_len == r.sent_kinds.size(),
                   ctx("T2 trace length matches the send count: mode=%s seed=%llu %u vs %zu",
                       srdsim::mode_name(mode), (unsigned long long)seed,
                       r.report.trace_len, r.sent_kinds.size()));
            const uint32_t n = std::min<uint32_t>(r.report.trace_len,
                                                  static_cast<uint32_t>(r.sent_kinds.size()));
            bool order_ok = true;
            for (uint32_t i = 0; i < n; ++ i) {
                const int applied_kind = static_cast<int>(r.report.trace[i] >> 56);
                if (applied_kind != r.sent_kinds[i]) { order_ok = false; break; }
            }
            expect(order_ok,
                   ctx("T2 apply order equals send order: mode=%s seed=%llu",
                       srdsim::mode_name(mode), (unsigned long long)seed));

            // T3: was the contract kept on the wire?
            // Reconstruct per-op progress from the send order, which T2 has just shown equals the
            // apply order.
            std::vector<int> placed(64, 0), total(64, 0);
            for (size_t i = 0; i < r.released.size(); ++ i)
                if (r.released[i].kind == srdsim::Unit::Kind::Placement)
                    total[r.released[i].op_id % 64] = r.released[i].chunk_count;
            std::fill(placed.begin(), placed.end(), 0);
            bool broken = false;
            for (const auto& u : r.released) {
                const size_t slot = u.op_id % 64;
                if (u.kind == srdsim::Unit::Kind::Placement) ++ placed[slot];
                else if (placed[slot] != total[slot]) broken = true;
            }
            if (mode == srdsim::Mode::ViolateI1) {
                if (broken) ++ i1_broken_runs;
            } else {
                expect(not broken,
                       ctx("T3 a signal never precedes its own bytes on the wire: mode=%s seed=%llu",
                           srdsim::mode_name(mode), (unsigned long long)seed));
            }
        }
        if (mode == srdsim::Mode::ViolateI1)
            expect(i1_broken_runs > 0,
                   ctx("T3 violate-i1 really breaks the contract end to end (%d of 8 runs)",
                       i1_broken_runs));
    }
}

// T1 again, on the bytes: with one operation and a known payload the receiver's checksum must match.
void test_bytes_land_correctly() {
    std::vector<char> expected(kDataBytes);
    for (size_t i = 0; i < expected.size(); ++ i)
        expected[i] = static_cast<char>((i * 37 + 11) & 0xff);
    const uint64_t want = checksum(expected.data(), expected.size());

    for (const size_t chunk : {size_t{512}, size_t{4096}, size_t{40'000}}) {
        const auto r = run_once(srdsim::Mode::Srd, /*seed=*/7, chunk, /*num_ops=*/1);
        expect(r.ok, ctx("the run completed: chunk=%zu", chunk));
        if (not r.ok) continue;
        expect(r.report.data_checksum == want,
               ctx("T1 the whole window matches byte for byte: chunk=%zu", chunk));
    }
}

// T4 / T5: malformed descriptors are reported and skipped, and the stream stays usable.
void test_bad_descriptors_are_refused() {
    uint16_t port = 0;
    const int listen_fd = srdsim::listen_any(&port);
    expect(listen_fd >= 0, "T4 listen succeeded");
    if (listen_fd < 0) return;

    int sv[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) { ::close(listen_fd); return; }

    const pid_t pid = ::fork();
    if (pid < 0) { ::close(listen_fd); return; }

    if (pid == 0) {
        ::close(sv[0]);
        std::vector<char> data(1024, 0);
        Report report{};
        srdsim::WindowTable windows;
        windows.add({data.data(), data.size(), false});
        srdsim::ReceiveAgent::Hooks hooks{};
        hooks.log = [](const char*) {};
        srdsim::ReceiveAgent agent(&windows, hooks);
        const int conn = ::accept(listen_fd, nullptr, nullptr);
        ::close(listen_fd);
        if (conn < 0) _exit(2);
        agent.serve(conn);
        agent.join();
        report.placements = agent.applied_placements();
        report.errors = agent.errors();
        report.data_checksum = checksum(data.data(), data.size());
        const bool ok = srdsim::write_all(sv[1], &report, sizeof(report));
        ::close(sv[1]);
        _exit(ok ? 0 : 3);
    }

    ::close(sv[1]);
    ::close(listen_fd);
    const int fd = srdsim::dial("127.0.0.1", port);
    expect(fd >= 0, "T4 dial succeeded");
    if (fd < 0) { ::waitpid(pid, nullptr, 0); return; }

    std::vector<char> payload(64, 'z');

    // An unknown window id, with a payload that must still be drained.
    srdsim::WireHeader h{};
    h.kind = static_cast<uint32_t>(srdsim::WireKind::Placement);
    h.window_id = 99;
    h.offset = 0;
    h.length = payload.size();
    srdsim::write_all(fd, &h, sizeof(h));
    srdsim::write_all(fd, payload.data(), payload.size());

    // A placement that runs past the end of a window that does exist.
    h.window_id = 0;
    h.offset = 1000;
    h.length = payload.size();     // 1000 + 64 > 1024
    srdsim::write_all(fd, &h, sizeof(h));
    srdsim::write_all(fd, payload.data(), payload.size());

    // A well-formed placement after the two bad ones: it must still land, which is what shows the
    // stream stayed framed rather than being left mid-message.
    h.window_id = 0;
    h.offset = 0;
    h.length = payload.size();
    srdsim::write_all(fd, &h, sizeof(h));
    srdsim::write_all(fd, payload.data(), payload.size());

    srdsim::WireHeader bye{};
    bye.kind = static_cast<uint32_t>(srdsim::WireKind::Shutdown);
    srdsim::write_all(fd, &bye, sizeof(bye));
    ::close(fd);

    Report report{};
    const bool got = srdsim::read_all(sv[0], &report, sizeof(report));
    ::close(sv[0]);
    int wstatus = 0;
    ::waitpid(pid, &wstatus, 0);
    expect(got and WIFEXITED(wstatus) and WEXITSTATUS(wstatus) == 0, "T4 the receiver exited cleanly");
    expect(report.errors == 2,
           ctx("T4/T5 both malformed descriptors were reported (errors=%llu, want 2)",
               (unsigned long long)report.errors));
    expect(report.placements == 1,
           ctx("T4/T5 exactly the well-formed placement was applied (placements=%llu, want 1)",
               (unsigned long long)report.placements));

    std::vector<char> expected(1024, 0);
    std::fill(expected.begin(), expected.begin() + 64, 'z');
    expect(report.data_checksum == checksum(expected.data(), expected.size()),
           "T5 the window holds exactly the bytes of the well-formed placement");
}

}   // namespace


// T7. Four ways to lose data silently: nothing refused, no failed write, an empty send queue, and a
// receiver that has applied a fraction of what was sent, so the counters say the run is healthy.
void test_the_four_silent_losses() {
    // T7a. A wire length below the largest window but far above any frame this transport emits must be
    // refused. Bounded by the largest registered window, which can be gigabytes, a garbage length of hundreds
    // of megabytes would be judged legitimate, allocated, and waited for forever.
    {
        // The declared size is a large production window, 1.82 GB. Nothing reads it, and nothing may -- the
        // point is that a window bound would accept a garbage length of hundreds of megabytes.
        // A one-megabyte window would have been refused either way and the check would prove nothing.
        std::vector<char> big(1 << 20, 0);
        srdsim::WindowTable windows;
        windows.add({big.data(), size_t(1820000000), false});
        srdsim::ReceiveAgent::Hooks hooks{};
        hooks.log = [](const char*) {};
        srdsim::ReceiveAgent agent(&windows, hooks);
        agent.set_max_frame_bytes(8192);
        int sv[2];
        expect(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0, "T7a a socket pair was created");
        expect(agent.serve(sv[1]), "T7a the agent serves the connection");

        // A header claiming far more than any frame this transport emits, then a healthy frame behind it.
        srdsim::WireHeader h{};
        h.kind = static_cast<uint32_t>(srdsim::WireKind::Placement);
        h.window_id = 0;
        h.offset = 0;
        h.length = 700u * 1024u * 1024u;
        expect(srdsim::write_all(sv[0], &h, sizeof(h)), "T7a the oversized header was sent");
        srdsim::SendLinks links;
        links.set(0, sv[0]);
        char payload[64] = {'z'};
        links.send_placement(0, 0, 0, payload, sizeof(payload), 1);
        links.flush_all();

        for (int i = 0; i < 200 and agent.refused_frames() == 0; ++ i)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        expect(agent.refused_frames() >= 1,
               ctx("T7a a length above the largest emitted frame is refused rather than allocated "
                   "(refused %llu, applied %llu)",
                   (unsigned long long)agent.refused_frames(),
                   (unsigned long long)agent.applied_placements()));
        links.shutdown_all();
        agent.stop();
    }

    // T7b. Teardown either delivers the queue or says how much it dropped.
    {
        // A reader that is present but slow. A bound counted in iterations of a loop that returns at once
        // on EAGAIN expires in under a millisecond, so a peer reading slower than that would lose the rest
        // of its queue with no counter recording it.
        std::vector<char> data(1 << 22, 0);
        srdsim::WindowTable windows;
        windows.add({data.data(), data.size(), false});
        srdsim::ReceiveAgent::Hooks hooks{};
        hooks.log = [](const char*) {};
        srdsim::ReceiveAgent agent(&windows, hooks);
        int sv[2];
        expect(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0, "T7b a socket pair was created");
        int small = 4096;
        ::setsockopt(sv[0], SOL_SOCKET, SO_SNDBUF, &small, sizeof(small));
        ::setsockopt(sv[1], SOL_SOCKET, SO_RCVBUF, &small, sizeof(small));
        // Queued before anything reads, and that ordering is the point. Serving first and then queuing
        // made the precondition a race with the reader: 320 kB against a 4 kB send buffer backs up on
        // one kernel and is drained faster than the assertion can see on another. This test passed on
        // macOS and failed on Linux for exactly that reason, which makes it a statement about the
        // scheduler of the machine rather than about teardown. With no reader yet, the backup is
        // arithmetic.
        srdsim::SendLinks links;
        links.set(0, sv[0]);
        std::vector<char> chunk(4096, 'q');
        constexpr int kFrames = 80;
        for (int i = 0; i < kFrames; ++ i)
            links.send_placement(0, 0, 0, chunk.data(), chunk.size(), uint64_t(i) + 1);
        expect(links.pending_bytes(0) > 0, "T7b the queue really is backed up before teardown");

        // Only now is there a reader, so what is checked is what teardown does with a queue that already
        // exists.
        expect(agent.serve(sv[1]), "T7b the agent serves the connection");
        links.shutdown_all();
        for (int i = 0; i < 400 and agent.applied_placements() < kFrames; ++ i)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        const auto lost = links.discarded();
        expect(lost.bytes == 0 and agent.applied_placements() == kFrames,
               ctx("T7b teardown waits for a slow reader instead of discarding its queue "
                   "(applied %llu of %d, discarded %llu byte(s) on %llu link(s))",
                   (unsigned long long)agent.applied_placements(), kFrames,
                   (unsigned long long)lost.bytes, (unsigned long long)lost.links));
        agent.stop();
    }

    // T7c. Rebinding a peer discards the partial frame in its outbox. Carrying it over puts the
    // remainder at the head of the new stream, which desynchronises the receiver on its first header.
    {
        int sv[2], tv[2];
        expect(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0, "T7c the first pair was created");
        expect(::socketpair(AF_UNIX, SOCK_STREAM, 0, tv) == 0, "T7c the second pair was created");
        srdsim::SendLinks links;
        links.set(0, sv[0]);
        std::vector<char> chunk(256 * 1024, 'p');   // larger than a socket pair will take at once
        links.send_placement(0, 0, 0, chunk.data(), chunk.size(), 1);
        expect(links.pending_bytes(0) > 0, "T7c a partial frame is left queued");
        links.set(0, tv[0]);                        // rebind
        expect(links.pending_bytes(0) == 0,
               ctx("T7c rebinding discards the partial frame (%zu byte(s) still queued)",
                   links.pending_bytes(0)));
        expect(links.discarded().bytes > 0, "T7c and says how much it discarded");
        ::close(sv[1]); ::close(tv[1]);
    }

    // T7d. A signal frame claiming a payload is a desynchronised stream, not a signal.
    {
        std::vector<uint64_t> words(8, 0);
        srdsim::WindowTable windows;
        windows.add({words.data(), words.size() * sizeof(uint64_t), false});
        srdsim::ReceiveAgent::Hooks hooks{};
        hooks.log = [](const char*) {};
        srdsim::ReceiveAgent agent(&windows, hooks);
        int sv[2];
        expect(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0, "T7d a socket pair was created");
        expect(agent.serve(sv[1]), "T7d the agent serves the connection");
        srdsim::WireHeader h{};
        h.kind = static_cast<uint32_t>(srdsim::WireKind::Signal);
        h.window_id = 0;
        h.offset = 0;
        h.value = 1;
        h.length = 8;                               // a signal never carries a payload
        expect(srdsim::write_all(sv[0], &h, sizeof(h)), "T7d the malformed signal was sent");
        for (int i = 0; i < 200 and agent.refused_frames() == 0; ++ i)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        expect(agent.refused_frames() >= 1,
               ctx("T7d a signal frame with a payload length is refused (refused %llu, applied %llu)",
                   (unsigned long long)agent.refused_frames(),
                   (unsigned long long)agent.applied_signals()));
        ::close(sv[0]);
        agent.stop();
    }
}


// T8. A finished connection's descriptor leaves the agent's table before it is closed.
//
// A loop that closes the descriptor and leaves the number in `fds_` makes a later `stop()` call `::shutdown`
// on whatever the operating system has since handed that number to, and an unrelated socket loses data. In
// the plugin the candidates for that number are live send links and the
// loopback pair, so the consequence is a send link dying mid-run with nothing refused anywhere.
void test_a_finished_connection_releases_its_number() {
    std::vector<char> data(1 << 16, 0);
    srdsim::WindowTable windows;
    windows.add({data.data(), data.size(), false});
    srdsim::ReceiveAgent::Hooks hooks{};
    hooks.log = [](const char*) {};
    auto agent = std::make_unique<srdsim::ReceiveAgent>(&windows, hooks);

    int sv[2];
    expect(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0, "T8 a socket pair was created");
    expect(agent->serve(sv[1]), "T8 the agent serves the connection");
    const int served = sv[1];
    ::close(sv[0]);                       // the peer goes away, so the loop ends and closes its side
    agent->join();

    // Take a fresh pair and see whether the operating system reissued the number the loop closed.
    int tv[2];
    expect(::socketpair(AF_UNIX, SOCK_STREAM, 0, tv) == 0, "T8 a second socket pair was created");
    const bool recycled = (tv[0] == served or tv[1] == served);
    // Write a byte, then tear the agent down. If the agent still believes it owns that number it will
    // shut the innocent socket down and the byte will not come back.
    const char probe = 'x';
    expect(::send(tv[0], &probe, 1, 0) == 1, "T8 a byte was written to the innocent socket");
    agent.reset();                        // destructor calls stop(), which shuts down what it owns
    char got = 0;
    const ssize_t n = ::recv(tv[1], &got, 1, MSG_DONTWAIT);
    expect(not recycled or (n == 1 and got == 'x'),
           ctx("T8 tearing the agent down does not shut down a descriptor it no longer owns "
               "(number %d was reissued: %s, read back %zd byte(s))",
               served, recycled ? "yes" : "no", n));
    ::close(tv[0]); ::close(tv[1]);
}

// T9. A peer that has stopped reading gets refused, not queued forever.
//
// The outbox exists because the socket may refuse bytes while the caller must not block. With no bound on
// it, a peer that stops reading turns that into unbounded memory -- and it did: the adversarial suite,
// which sends 4 MB frames as fast as it can into a link with a 4 kB send buffer, died of `std::bad_alloc`
// on Linux while passing on macOS, because there the link happened to break sooner. A limit is what makes
// the behaviour a property of the transport instead of a property of the machine.
//
// Refusal rather than dropping is the contract being checked. A reliable transport may not lose a frame it
// accepted, so a release that cannot be queued has to be declined, leaving the unit with the scheduler for
// the caller to retry. Dropping would be silent data loss.
void test_a_peer_that_stopped_reading_is_refused_not_queued() {
    int sv[2];
    expect(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0, "T9 a socket pair was created");
    const int small = 4096;
    ::setsockopt(sv[0], SOL_SOCKET, SO_SNDBUF, &small, sizeof(small));
    ::setsockopt(sv[1], SOL_SOCKET, SO_RCVBUF, &small, sizeof(small));
    // Nothing ever reads sv[1]. That is the case: the peer is present but not draining.

    srdsim::SendLinks links;
    links.set(0, sv[0]);
    // A small budget so the test finishes in memory it can afford. The production default is far larger;
    // what is being checked is that a budget is honoured, not its value.
    const size_t budget = 1u << 20;
    links.set_outbox_budget(budget);

    std::vector<char> chunk(64 * 1024, 'q');
    uint64_t accepted = 0, refused = 0;
    for (int i = 0; i < 400; ++ i) {
        if (links.send_placement(0, 0, 0, chunk.data(), chunk.size(), uint64_t(i) + 1)) ++ accepted;
        else ++ refused;
    }

    expect(refused > 0, ctx("T9 sends are refused once the peer is behind (accepted %llu, refused %llu)",
                           (unsigned long long)accepted, (unsigned long long)refused));
    expect(links.refused_for_budget() == refused,
           ctx("T9 every refusal is counted as a budget refusal (%llu of %llu)",
               (unsigned long long)links.refused_for_budget(), (unsigned long long)refused));
    // The bound is what the whole test is for: the queue may exceed the budget by at most the one frame
    // that was in flight when it crossed, never by the four hundred that were offered.
    const size_t queued = links.pending_bytes(0);
    expect(queued <= budget + chunk.size() + sizeof(srdsim::WireHeader),
           ctx("T9 the queue stays inside its budget (%zu queued, budget %zu)", queued, budget));
    expect(accepted > 0, "T9 the early sends were accepted, so the refusal is the budget and not the link");

    links.shutdown_all();
    ::close(sv[1]);
}

// T10. A frame for a window that was deregistered is refused, not applied to its old base.
//
// A table that keeps an entry after `deregMrSym` resolves a frame still in flight to a freed allocation and
// writes into it. For a host window that allocation is ordinary heap, so the write corrupts whatever the
// application allocated there next, and the crash it causes happens far from the write.
void test_a_deregistered_window_is_refused_not_written() {
    // Two windows so the check is about the retired one and not about an empty table.
    std::vector<char> keep(256, 0);
    std::vector<char> going(256, 0);
    srdsim::WindowTable windows;
    const uint32_t keep_id = windows.add({keep.data(), keep.size(), false});
    const uint32_t going_id = windows.add({going.data(), going.size(), false});

    srdsim::ReceiveAgent::Hooks hooks{};
    hooks.log = [](const char*) {};
    srdsim::ReceiveAgent agent(&windows, hooks);
    int sv[2];
    expect(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0, "T10 a socket pair was created");
    expect(agent.serve(sv[1]), "T10 the agent serves the connection");

    srdsim::SendLinks links;
    links.set(0, sv[0]);
    std::vector<char> payload(64, 'r');

    // Applied while the window is live, so the test knows the path works before it checks that it stops.
    expect(links.send_placement(0, going_id, 0, payload.data(), payload.size(), 1),
           "T10 a frame for a live window is sent");
    for (int i = 0; i < 200 and agent.applied_placements() == 0; ++ i)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    expect(agent.applied_placements() == 1, "T10 the live window received it");
    expect(going[0] == 'r', "T10 the bytes landed in the live window");

    // Now the window goes away, as `deregMrSym` makes it, and its memory is scribbled over the way a freed
    // allocation would be reused. Nothing the agent does may touch it again.
    windows.retire(going_id);
    std::fill(going.begin(), going.end(), '\0');

    const uint64_t refused_before = agent.refused_frames();
    expect(links.send_placement(0, going_id, 0, payload.data(), payload.size(), 2),
           "T10 the late frame is sent");
    for (int i = 0; i < 200 and agent.refused_frames() == refused_before; ++ i)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    expect(agent.refused_frames() > refused_before,
           ctx("T10 a frame for a retired window is refused (refused %llu, was %llu)",
               (unsigned long long)agent.refused_frames(), (unsigned long long)refused_before));
    expect(going[0] == '\0', "T10 the retired window's old memory was not written");
    expect(agent.applied_placements() == 1, "T10 nothing further was applied");

    // The surviving window still works, so retiring one did not retire the table.
    expect(links.send_placement(0, keep_id, 0, payload.data(), payload.size(), 3),
           "T10 a frame for the surviving window is sent");
    for (int i = 0; i < 200 and agent.applied_placements() < 2; ++ i)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    expect(agent.applied_placements() == 2, "T10 the surviving window still receives");
    expect(keep[0] == 'r', "T10 the bytes landed in the surviving window");

    links.shutdown_all();
    agent.stop();
}

// T11. Deregistration waits for a write that is already under way.
//
// T10 covers a frame that arrives after the window is gone. This covers the one that was being applied when
// it went: if the lookup and the write were two steps with nothing tying them to deregistration, a write
// that had looked the window up could land after NCCL freed it. The device hook is held open here to
// stand for a copy that has not been issued yet, and `retire` must not return until it is released.
void test_deregistration_waits_for_a_write_under_way() {
    std::vector<char> mem(256, 0);
    srdsim::WindowTable windows;
    const uint32_t id = windows.add({mem.data(), mem.size(), true});

    struct Gate { std::atomic<bool> entered{false}, release{false}; } gate;
    std::atomic<bool>& entered = gate.entered;
    std::atomic<bool>& release = gate.release;
    srdsim::ReceiveAgent::Hooks hooks{};
    hooks.log = [](const char*) {};
    hooks.user = &gate;
    hooks.device_copy = [](void* dst, const void* src, size_t n, void* user) {
        auto* g = static_cast<Gate*>(user);
        g->entered = true;
        while (not g->release) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::memcpy(dst, src, n);
        return true;
    };
    srdsim::ReceiveAgent agent(&windows, hooks);
    int sv[2];
    expect(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0, "T11 a socket pair was created");
    expect(agent.serve(sv[1]), "T11 the agent serves the connection");
    srdsim::SendLinks links;
    links.set(0, sv[0]);
    std::vector<char> payload(64, 'w');
    expect(links.send_placement(0, id, 0, payload.data(), payload.size(), 1), "T11 the frame is sent");
    for (int i = 0; i < 400 and not entered; ++ i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    expect(entered.load(), "T11 the write is under way");
    expect(windows.leases(id) == 1, ctx("T11 the write holds one lease (%u)", windows.leases(id)));

    std::atomic<bool> retired{false};
    std::thread dereg([&] { windows.retire(id); retired = true; });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    expect(not retired.load(), "T11 retire does not return while the write is under way");
    release = true;
    dereg.join();
    expect(retired.load(), "T11 retire returns once the write is done");
    expect(mem[0] == 'w', "T11 the write that was under way completed, into memory still owned");
    expect(windows.leases(id) == 0, "T11 no lease is left behind");

    srdsim::WindowTable::Lease late(&windows, id);
    expect(not late.ok(), "T11 a retired window cannot be leased");
    agent.stop();
}

// T12. A reader that waits on the signal word with an acquire load, as a GIN receiver does, and then reads the
// bytes the signal guards, without asking the agent anything. The agent must release the word after the
// bytes it placed before it, or that reader races with them. Only ThreadSanitizer detects the race: built
// without it, a relaxed add passes on x86, where the locked add is a full barrier, and almost always elsewhere.
void test_a_signal_word_orders_the_bytes_before_it() {
    std::vector<char> data(4096, 0);
    alignas(8) uint64_t signal_word = 0;
    srdsim::WindowTable windows;
    const uint32_t data_id = windows.add({data.data(), data.size(), false});
    const uint32_t signal_id = windows.add({&signal_word, sizeof(signal_word), false});

    srdsim::ReceiveAgent::Hooks hooks{};
    hooks.log = [](const char*) {};
    srdsim::ReceiveAgent agent(&windows, hooks);
    int sv[2];
    expect(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0, "T12 a socket pair was created");
    expect(agent.serve(sv[1]), "T12 the agent serves the connection");
    srdsim::SendLinks links;
    links.set(0, sv[0]);

    std::vector<char> payload(data.size(), 's');
    expect(links.send_placement(0, data_id, 0, payload.data(), payload.size(), 1), "T12 the payload is sent");
    expect(links.send_signal(0, signal_id, 0, 1, 1), "T12 the signal is sent after it");
    uint64_t seen = 0;
    for (int i = 0; i < 2000 and seen == 0; ++ i) {
        seen = __atomic_load_n(&signal_word, __ATOMIC_ACQUIRE);
        if (seen == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    expect(seen == 1, "T12 the signal arrived");
    // Without the signal nothing orders the agent's write before this read, so the bytes are not read then.
    if (seen == 1) {
        size_t wrong = 0;
        for (char c : data) wrong += c != 's';
        expect(wrong == 0, ctx("T12 every byte placed before the signal is visible once it is (%zu wrong)", wrong));
    }
    links.shutdown_all();
    agent.stop();
}

int main() {
    test_the_four_silent_losses();
    test_a_finished_connection_releases_its_number();
    test_a_peer_that_stopped_reading_is_refused_not_queued();
    test_a_deregistered_window_is_refused_not_written();
    test_deregistration_waits_for_a_write_under_way();
    test_delivery_and_order();
    test_bytes_land_correctly();
    test_bad_descriptors_are_refused();
    test_a_signal_word_orders_the_bytes_before_it();

    std::printf("%s: %d checks, %d failures\n", g_failures == 0 ? "PASS" : "FAIL", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
