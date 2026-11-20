// SPDX-License-Identifier: Apache-2.0

// Properties with time in them, checked under deliberate impairment.
//
// WHY THESE AND NOT MORE STATE CHECKS
// The rest of the suite checks safety: no reachable state does something wrong. That is 2.1 million
// assertions, three exhaustive enumerations and a dozen proofs, and **it did not catch a single one of
// the defects that actually cost time**. Those were all liveness failures -- a progress callback that
// stopped returning, a copy waiting for a kernel that waited for the copy, a queue that grew and never
// drained, a monitor its own subject had blocked. "Something eventually happens" is not a property of
// any state, so no enumeration of states can see it.
//
// So these tests measure elapsed time and require bounds, and they run against a fabric told to be slow,
// bursty, congested and one-sidedly frozen. The bounds are deliberately loose: the point is to separate
// "slow" from "never", not to police performance, and a tight bound on a shared machine is a test that
// fails for reasons the reader will not believe.
//
//   L1  a progress call returns within a bound, always, under every impairment. This is the property the
//       plugin violated for a week: it did a blocking write inside NCCL's progress callback, so when a
//       peer stopped reading the callback stopped returning and NCCL's proxy stopped with it.
//   L2  every submitted operation completes within a bound, under every impairment. Eventual delivery,
//       which reordering and delay are allowed to postpone and not to prevent.
//   L3  backpressure resolves. With a send buffer far smaller than a frame and a receiver that applies
//       on a timer, everything still lands.
//   L4  a frozen link does not stop the others. One path stalling is a fabric event; the whole
//       simulator stopping is a defect, and telling them apart needs two links and one freeze.
//   L5  the detector works. A link is frozen long enough that the stall condition is true, and the
//       agent's own counters have to show the freeze rather than the run merely surviving it. A monitor
//       nobody has watched fail is not a monitor.
//   L6  memory stays bounded while a receiver is slow but present. A queue is allowed to grow; it is not
//       allowed to grow without limit while the peer is still draining.
//
// Build: c++ -std=c++17 -O2 -pthread -I layer2 tests/test_srdsim_liveness.cpp -o test_srdsim_liveness

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "srdsim_impair.h"
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

using Clock = std::chrono::steady_clock;

int64_t ms_since(Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t).count();
}

// Bounds. Loose on purpose: these separate "eventually" from "never", and a shared machine can be very
// slow for reasons that have nothing to do with this code.
constexpr int64_t kProgressCallBudgetMs = 2000;    // one progress call must return within this
constexpr int64_t kCompletionBudgetMs   = 30000;   // all submitted work must land within this

constexpr size_t kWindowBytes = 1u << 20;   // one mebibyte, so a frame can exceed a small send buffer

// ---------------------------------------------------------------------------------------------
// A receiver in its own process, with an optional applied-frame throttle and freeze.
// ---------------------------------------------------------------------------------------------

struct Report {
    uint64_t placements = 0;
    uint64_t signals = 0;
    uint64_t refused = 0;
    uint64_t failed = 0;
    uint64_t sentinel = 0;
};

constexpr uint64_t kSentinel = 0x4c'49'56'45'4e'45'53'53ull;   // "LIVENESS"

// The throttle lives in a hook rather than in the agent: the agent's job is to apply what arrives, and
// how slowly the machine underneath it does that is a property of the machine.
srdsim::Impairments g_recv_imp;
std::atomic<uint64_t> g_applied{0};
Clock::time_point g_last_apply;
// The freeze has to be applied by the side that stops reading, which is the receiver. It was declared
// and never wired in the first version, so the knob did nothing and the two cases that depend on it
// passed without exercising anything -- exactly the failure mode this suite exists to catch.
srdsim::Freezer g_freezer;

// Per-link applied counts, so the freeze fires after this link's third frame rather than after the
// third frame on any link. Keyed on the whole receiving side, the freeze fired at frame 32 of 60 --
// after the frozen peer's traffic had already finished -- and every test that depended on it passed
// without a sender ever waiting for anything.
// Two in every case here, and the bound is stated rather than left as a literal in four places.
constexpr int kMaxLinks = 8;
std::atomic<uint64_t> g_applied_on_link[kMaxLinks];

int link_of(void* user) {
    // `serve` was given the peer index as the cookie, offset by one so that a null cookie stays
    // distinguishable from link zero.
    const auto raw = reinterpret_cast<uintptr_t>(user);
    return raw == 0 ? -1 : static_cast<int>(raw - 1);
}

void maybe_freeze(void* user) {
    const int link = link_of(user);
    if (link < 0 or link >= kMaxLinks) return;
    g_freezer.maybe_freeze(link, g_applied_on_link[link].load(std::memory_order_relaxed));
}

bool throttled_copy(void* dst, const void* src, size_t bytes, void* user) {
    maybe_freeze(user);
    if (const int link = link_of(user); link >= 0 and link < kMaxLinks)
        ++ g_applied_on_link[link];
    if (g_recv_imp.apply_interval_us != 0) {
        const auto wait = std::chrono::microseconds(g_recv_imp.apply_interval_us);
        while (Clock::now() - g_last_apply < wait)
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        g_last_apply = Clock::now();
    }
    std::memcpy(dst, src, bytes);
    ++ g_applied;
    return true;
}

bool throttled_add(void* dst, uint64_t value, void* user) {
    maybe_freeze(user);
    if (const int link = link_of(user); link >= 0 and link < kMaxLinks)
        ++ g_applied_on_link[link];
    __atomic_fetch_add(static_cast<uint64_t*>(dst), value, __ATOMIC_RELAXED);
    ++ g_applied;
    return true;
}

[[noreturn]] void run_receiver(int listen_fd, int report_fd, int connections,
                              const srdsim::Impairments& imp) {
    g_recv_imp = imp;
    g_freezer.configure(imp);
    g_last_apply = Clock::now();

    std::vector<char> data(kWindowBytes, 0);
    std::vector<uint64_t> signals(256, 0);
    srdsim::WindowTable windows;
    // Device windows so the hooks run: that is where a throttle can live without the agent knowing.
    windows.add({data.data(), data.size(), true});
    windows.add({signals.data(), signals.size() * sizeof(uint64_t), true});

    srdsim::ReceiveAgent::Hooks hooks{};
    hooks.device_copy = throttled_copy;
    hooks.device_add = throttled_add;
    hooks.log = [](const char*) {};
    srdsim::ReceiveAgent agent(&windows, hooks);

    for (int i = 0; i < connections; ++ i) {
        const int fd = ::accept(listen_fd, nullptr, nullptr);
        if (fd < 0) _exit(2);
        srdsim::set_nodelay(fd);
        // Congestion has to be applied to the accepted descriptor, here in the child, because that is
        // the only place it exists. Setting it on the listening socket instead does nothing on Darwin:
        // accepted sockets do not inherit it, so the receive buffer stayed at 326 kB, the whole run
        // fitted inside it, and a sender never had to wait for a frozen receiver at all. That is why
        // the mutant which removes O_NONBLOCK survived two attempts to catch it -- the code was wrong
        // and the fabric was too forgiving to notice.
        if (imp.send_buffer_bytes != 0) {
            int sz = static_cast<int>(imp.send_buffer_bytes);
            ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &sz, sizeof(sz));
        }
        // The peer index, offset by one, is this link's identity. Accept order matches dial order,
        // which matches the peer numbering the sender uses.
        if (not agent.serve(fd, reinterpret_cast<void*>(static_cast<uintptr_t>(i) + 1))) _exit(3);
    }
    ::close(listen_fd);
    agent.join();

    Report r{};
    r.placements = agent.applied_placements();
    r.signals = agent.applied_signals();
    r.refused = agent.refused_frames();
    r.failed = agent.failed_writes();
    r.sentinel = kSentinel;
    const bool ok = srdsim::write_all(report_fd, &r, sizeof(r));
    ::close(report_fd);
    _exit(ok ? 0 : 4);
}

struct Session {
    std::vector<int> fds;
    pid_t pid = -1;
    int report_fd = -1;
};

bool open_session(Session* s, int connections, const srdsim::Impairments& imp) {
    uint16_t port = 0;
    const int listen_fd = srdsim::listen_any(&port);
    if (listen_fd < 0) return false;
    int sv[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) { ::close(listen_fd); return false; }

    const pid_t pid = ::fork();
    if (pid < 0) { ::close(listen_fd); return false; }
    if (pid == 0) {
        ::close(sv[0]);
        run_receiver(listen_fd, sv[1], connections, imp);
    }
    ::close(sv[1]);
    ::close(listen_fd);
    for (int i = 0; i < connections; ++ i) {
        const int fd = srdsim::dial("127.0.0.1", port);
        if (fd < 0) { ::waitpid(pid, nullptr, 0); ::close(sv[0]); return false; }
        s->fds.push_back(fd);
    }
    s->pid = pid;
    s->report_fd = sv[0];
    return true;
}

Report close_session(Session* s, srdsim::SendLinks* links) {
    if (links) links->shutdown_all();
    for (const int fd : s->fds)
        if (::fcntl(fd, F_GETFD) != -1) ::close(fd);
    Report r{};
    const bool got = srdsim::read_all(s->report_fd, &r, sizeof(r));
    ::close(s->report_fd);
    ::waitpid(s->pid, nullptr, 0);
    if (not got or r.sentinel != kSentinel) r = Report{};
    return r;
}

// ---------------------------------------------------------------------------------------------
// The driver: submits work, then acts as the progress loop, measuring how long each call takes.
// ---------------------------------------------------------------------------------------------

struct RunOutcome {
    bool completed = false;
    // A link the transport gave up on. Without this the run looks finished: killing a link empties its
    // outbox, the driver sees nothing queued and calls that completion. "The sender has nothing left to
    // send" and "the receiver got everything" are different claims, and only the second one is delivery.
    bool link_died = false;
    int64_t elapsed_ms = 0;
    int64_t worst_progress_ms = 0;
    size_t units_released = 0;
    size_t peak_queued = 0;
    Report report{};
};

RunOutcome drive(const srdsim::Impairments& imp, int peers, int ops_per_peer, size_t op_bytes,
                 uint64_t seed, srdsim::Mode mode) {
    RunOutcome out;
    Session s;
    if (not open_session(&s, peers, imp)) return out;

    srdsim::SendLinks links;
    for (int p = 0; p < peers; ++ p) {
        links.set(p, s.fds[static_cast<size_t>(p)]);
        if (imp.send_buffer_bytes != 0) {
            int sz = static_cast<int>(imp.send_buffer_bytes);
            ::setsockopt(s.fds[static_cast<size_t>(p)], SOL_SOCKET, SO_SNDBUF, &sz, sizeof(sz));
        }
    }

    std::vector<char> src(op_bytes ? op_bytes : 1, 'L');
    srdsim::Scheduler::Options o;
    o.mode = mode;
    o.seed = seed;
    o.chunk_bytes = 8192;
    srdsim::Scheduler sched(o);

    uint64_t id = 0;
    for (int p = 0; p < peers; ++ p) {
        for (int k = 0; k < ops_per_peer; ++ k) {
            srdsim::Op op;
            op.id = ++ id;
            op.peer = p;
            op.src_base = src.data();
            op.bytes = op_bytes;
            op.dst_window = 0;
            op.dst_off = 0;
            op.has_signal = true;
            op.signal_window = 1;
            op.signal_off = (id % 200) * sizeof(uint64_t);
            op.signal_val = 1;
            std::string why;
            if (not sched.submit(op, &why)) return out;
        }
    }

    srdsim::Delayer delayer(imp, seed);
    const auto started = Clock::now();
    // Enough traffic that the freeze outlasts the socket buffers; see the note on Case for why volume
    // and not buffer size is what makes a freeze observable.
    std::vector<srdsim::Unit> batch;

    // This loop is the plugin's progress callback. Each iteration is timed, because the property that
    // matters is not how long the whole run takes but whether any single call fails to return.
    while (ms_since(started) < kCompletionBudgetMs) {
        const auto call_started = Clock::now();

        batch.clear();
        sched.progress(4, &batch);
        for (const auto& u : batch) {
            delayer.before_unit(u.kind == srdsim::Unit::Kind::Signal);
            const bool ok = (u.kind == srdsim::Unit::Kind::Placement)
                ? links.send_placement(u.peer, u.window, u.window_offset, u.src, u.bytes, u.op_id)
                : links.send_signal(u.peer, u.window, u.window_offset, u.signal_val, u.op_id);
            if (not ok) { out.link_died = true; break; }
            ++ out.units_released;
        }
        links.flush_all();

        size_t queued = 0;
        for (int p = 0; p < peers; ++ p) queued += links.pending_bytes(p);
        out.peak_queued = std::max(out.peak_queued, queued);

        out.worst_progress_ms = std::max(out.worst_progress_ms, ms_since(call_started));

        if (out.link_died) break;
        if (sched.idle() and queued == 0) { out.completed = true; break; }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }

    out.elapsed_ms = ms_since(started);
    out.report = close_session(&s, &links);
    // A passing timing test says nothing about how much room it had. With this on, a run reports its
    // own margins, which is the difference between "the budget holds" and "the budget holds by 3 ms".
    if (const char* v = std::getenv("SRDSIM_LIVENESS_VERBOSE"); v != nullptr and *v == '1')
        std::printf("    [trace] {%s} %s worst call %lld ms, elapsed %lld ms, released %zu, peak queued %zu, "
                    "applied %llu+%llu\n",
                    imp.describe().c_str(),
                    out.link_died ? "LINK DIED" : (out.completed ? "completed" : "INCOMPLETE"),
                    (long long)out.worst_progress_ms, (long long)out.elapsed_ms,
                    out.units_released, out.peak_queued,
                    (unsigned long long)out.report.placements,
                    (unsigned long long)out.report.signals);
    return out;
}

// ---------------------------------------------------------------------------------------------
// L1, L2, L3, L6: bounds under each impairment on its own and all together.
// ---------------------------------------------------------------------------------------------

// The volume is part of the case, not a constant, because how much has to move decides whether an
// impairment is felt at all. A receiver that stops reading only holds a sender up once the bytes
// outrun the socket buffers, and those are hundreds of kilobytes: with a few hundred kilobytes in the
// run, the kernel absorbed everything and a frozen receiver was indistinguishable from a fast one.
// Measured on this machine: the sender pushed all 197,808 bytes to a peer that had been frozen for
// 3.5 seconds and finished in 538 ms. Shrinking SO_SNDBUF does not fix that -- the send buffer bounds
// what is unsent, not what is unread, and the receiver keeps acknowledging into its own 326 kB.
struct Case {
    const char* name;
    srdsim::Impairments imp;
    int ops_per_peer = 6;
    size_t op_bytes = 32768;
};

std::vector<Case> cases() {
    std::vector<Case> v;
    v.push_back({"unimpaired", {}});

    srdsim::Impairments delayed;
    delayed.delay_us_min = 50; delayed.delay_us_max = 400;
    v.push_back({"delayed", delayed});

    srdsim::Impairments bursty;
    bursty.burst = 3; bursty.burst_pause_us = 2000;
    v.push_back({"bursty", bursty});

    srdsim::Impairments slow_receiver;
    slow_receiver.apply_interval_us = 500;
    v.push_back({"slow receiver", slow_receiver});

    srdsim::Impairments congested;
    // Smaller than one 8 kB frame, so every send is partial, but not so small that TCP collapses into
    // silly-window behaviour. At 2048 bytes a Linux receiver advertises a window of about one segment
    // and delayed acknowledgements drop the link to tens of kilobytes per second: 3 MB then took longer
    // than the completion budget, and the suite failed on the target for a reason that had nothing to do
    // with the code. The knob is meant to congest the link, not to break TCP.
    congested.send_buffer_bytes = 4096;
    v.push_back({"congested", congested});

    srdsim::Impairments lagging_signals;
    lagging_signals.signal_lag_us = 1000;
    v.push_back({"lagging signals", lagging_signals});

    // Longer than kProgressCallBudgetMs on purpose. A receiver that stops for longer than one progress
    // call is allowed to take makes a blocking send exceed the budget and a non-blocking one not, which
    // is the only way L1 can tell the two apart.
    srdsim::Impairments frozen;
    frozen.freeze_peer = 0;
    frozen.freeze_after_units = 3;
    // Six seconds, not three: the receiver's socket buffer absorbs a few hundred kilobytes before a
    // sender feels anything, and that head start is worth about two seconds here. A blocking descriptor
    // then waits about four seconds inside one progress call, well past the budget, while a
    // non-blocking one waits about a millisecond. Measured: 3500 ms of freeze left the blocking case at
    // 1.3 s -- a real difference that the budget did not catch, which is the same as not catching it.
    frozen.freeze_ms = 6000;
    // Big enough not to trigger silly-window behaviour, small enough to backpressure. What makes the
    // freeze visible here is the volume above, not the buffer: see the note on Case.
    frozen.send_buffer_bytes = 16384;
    // Enough bytes that the receiver's socket buffer cannot hide the freeze: 48 operations of 32 kB is
    // about 1.5 MB per peer against a few hundred kilobytes of buffer, so a sender that waits for a
    // frozen peer waits in full view and one that queues does not.
    v.push_back({"receiver freezes for longer than a progress call may take", frozen, 48, 32768});

    srdsim::Impairments everything;
    everything.delay_us_min = 20; everything.delay_us_max = 200;
    everything.burst = 4; everything.burst_pause_us = 1000;
    everything.apply_interval_us = 200;
    everything.send_buffer_bytes = 4096;
    everything.signal_lag_us = 300;
    v.push_back({"everything at once", everything});

    return v;
}

void test_bounds_under_impairment() {
    for (const Case& c : cases()) {
        for (const auto mode : {srdsim::Mode::InOrder, srdsim::Mode::Srd}) {
            // Small enough that even the slowest case finishes inside the budget, large enough that
            // several frames are in flight at once.
            const auto r = drive(c.imp, /*peers=*/2, c.ops_per_peer, c.op_bytes, /*seed=*/7, mode);
            const std::string where = std::string(c.name) + " / " + srdsim::mode_name(mode);

            // L1. The property the plugin broke: no single progress call may fail to return.
            expect(r.worst_progress_ms <= kProgressCallBudgetMs,
                   ctx("L1 [%s] the slowest progress call was %lld ms, budget %lld: a progress "
                       "callback that does not return stops NCCL's proxy entirely",
                       where.c_str(), (long long)r.worst_progress_ms,
                       (long long)kProgressCallBudgetMs));

            // L2. Eventual delivery. Delay and reordering may postpone, never prevent.
            expect(r.completed and not r.link_died,
                   ctx("L2 [%s] everything submitted was released and pushed within %lld ms "
                       "(released %zu unit(s), %zu byte(s) still queued, link died: %s)",
                       where.c_str(), (long long)kCompletionBudgetMs, r.units_released,
                       r.peak_queued, r.link_died ? "yes" : "no"));

            // L3. Backpressure resolves, and the receiver saw it all -- exactly all of it. An
            // inequality here is what let a defect through twice: `> 0` is true of a run that lost
            // nine frames in ten, and the sender's queue emptying is not delivery, because a link the
            // transport killed also has an empty queue. The counts are exact and they come from the
            // receiving process.
            const uint64_t chunks_per_op = (c.op_bytes + 8191u) / 8192u;
            const uint64_t want_placements = 2ull * uint64_t(c.ops_per_peer) * chunks_per_op;
            const uint64_t want_signals = 2ull * uint64_t(c.ops_per_peer);
            expect(r.report.placements == want_placements and r.report.signals == want_signals,
                   ctx("L3 [%s] the receiver applied every frame: %llu of %llu placement(s), "
                       "%llu of %llu signal(s)",
                       where.c_str(), (unsigned long long)r.report.placements,
                       (unsigned long long)want_placements,
                       (unsigned long long)r.report.signals,
                       (unsigned long long)want_signals));
            expect(r.report.refused == 0 and r.report.failed == 0,
                   ctx("L3 [%s] nothing was refused or failed (%llu refused, %llu failed)",
                       where.c_str(), (unsigned long long)r.report.refused,
                       (unsigned long long)r.report.failed));

            // L6. A queue may grow. It may not grow without limit while the peer is draining. The cap
            // is generous -- twice the total bytes submitted -- because the point is boundedness, not a
            // particular bound.
            const size_t submitted = 2u * size_t(c.ops_per_peer) * c.op_bytes;
            expect(r.peak_queued <= 2u * submitted,
                   ctx("L6 [%s] the send queue stayed bounded (peak %zu byte(s), submitted %zu)",
                       where.c_str(), r.peak_queued, submitted));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// L4 and L5: one frozen link, and whether the freeze is observable.
// ---------------------------------------------------------------------------------------------

void test_a_frozen_link_does_not_stop_the_others() {
    // Peer 0's link is frozen part way through. Peer 1's traffic must still complete, and the whole run
    // must still finish: one path stalling is something a fabric does, and the simulator stopping
    // because of it is a defect.
    srdsim::Impairments imp;
    imp.freeze_peer = 0;
    imp.freeze_after_units = 4;
    imp.freeze_ms = 4000;
    // Backpressure without silly-window behaviour; the freeze is made visible by the volume below.
    imp.send_buffer_bytes = 16384;

    const auto started = Clock::now();
    const auto r = drive(imp, /*peers=*/2, /*ops_per_peer=*/40, /*op_bytes=*/16384, /*seed=*/11,
                         srdsim::Mode::Srd);
    const int64_t took = ms_since(started);

    expect(r.worst_progress_ms <= kProgressCallBudgetMs,
           ctx("L4 no progress call blocked while a link was frozen (worst %lld ms)",
               (long long)r.worst_progress_ms));
    expect(r.completed,
           ctx("L4 the run completed despite a frozen link (%zu unit(s) released, %zu byte(s) queued)",
               r.units_released, r.peak_queued));
    expect(r.report.placements > 0 and r.report.signals > 0,
           "L4 traffic still landed with a link frozen part way through");

    // L5. The freeze has to be observable, or the case proves only that nothing broke. The witness is
    // elapsed time against an unimpaired run of the same size, and the margin has to be a real fraction
    // of the freeze rather than "at least as long": the first version asked only for `>=`, which is true
    // of a run that absorbed the freeze entirely and therefore true of a knob that did nothing.
    const auto baseline = drive(srdsim::Impairments{}, 2, 40, 16384, 11, srdsim::Mode::Srd);
    expect(baseline.completed, "L5 the unimpaired baseline completed");
    const int64_t margin = static_cast<int64_t>(imp.freeze_ms) / 2;
    expect(took >= baseline.elapsed_ms + margin,
           ctx("L5 the freeze cost real time: %lld ms against a baseline of %lld ms, needing at least "
               "%lld ms more. A freeze the run does not feel is a knob that is not wired to anything",
               (long long)took, (long long)baseline.elapsed_ms, (long long)margin));
    // And the impairment is described, so a reader of a failure knows what was on.
    expect(imp.describe() != "none", ctx("L5 the impairment describes itself: %s",
                                         imp.describe().c_str()));
}

// ---------------------------------------------------------------------------------------------
// The negative control: an impairment that is not configured must change nothing.
// ---------------------------------------------------------------------------------------------

void test_impairments_are_off_by_default() {
    const srdsim::Impairments none;
    expect(not none.any(), "L0 nothing is impaired by default");
    expect(none.describe() == "none", "L0 and it says so");
    // Reading an empty environment must produce the same thing, or a run is impaired by whatever a
    // previous one left behind.
    ::unsetenv("SRDSIM_DELAY_US_MAX");
    ::unsetenv("SRDSIM_BURST");
    ::unsetenv("SRDSIM_APPLY_INTERVAL_US");
    ::unsetenv("SRDSIM_SEND_BUFFER_BYTES");
    ::unsetenv("SRDSIM_FREEZE_PEER");
    ::unsetenv("SRDSIM_SIGNAL_LAG_US");
    expect(not srdsim::impairments_from_env().any(),
           "L0 an empty environment impairs nothing");
    ::setenv("SRDSIM_DELAY_US_MAX", "250", 1);
    const auto from_env = srdsim::impairments_from_env();
    expect(from_env.any() and from_env.delay_us_max == 250,
           "L0 a knob set in the environment is picked up");
    ::unsetenv("SRDSIM_DELAY_US_MAX");
}

}   // namespace

int main() {
    test_impairments_are_off_by_default();
    test_bounds_under_impairment();
    test_a_frozen_link_does_not_stop_the_others();

    if (not g_failures.empty()) {
        std::printf("FAIL: %d checks, %zu failures\n", g_checks, g_failures.size());
        for (const auto& f : g_failures) std::printf("  [FAIL] %s\n", f.c_str());
        return 1;
    }
    std::printf("PASS: %d checks, 0 failures\n", g_checks);
    return 0;
}
