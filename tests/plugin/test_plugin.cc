// SPDX-License-Identifier: Apache-2.0
// Drives the plugin through its op table without NCCL and without a GPU, the way NCCL's GIN host proxy
// calls it (harness.h).
//
//   test_plugin PLUGIN_SO TABLE_VERSION [--scenario NAME] [--mode MODE] [--seed N]
//
// Each case (scenario, mode, seed) runs in fresh processes, one per rank, so the plugin's process-wide
// state is per rank as under NCCL. Scenarios assert only what NCCL or a GPU kernel could observe: bytes
// in a window, a signal word, whether a request completed or retired.
//
// A case fails when a rank reports an assertion or an error, when nothing observable changes for
// kStallMs while a rank waits, or when it runs past the driver's wall clock. Known defects are listed in
// kExpected with the exact failure they produce; such a case is reported XFAIL, and the run fails if a
// listed case passes (remove its entry with the fix) or fails differently. A FAIL line carries the
// arguments that replay that one case.
#include "harness.h"
#if defined(HARNESS_CUDA)
#include <cuda.h>
#endif
#include "srdsim_schedule.h"

#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif

#include <cerrno>
#include <cstdlib>
#include <new>
#include <utility>

extern char** environ;

namespace {

using namespace harness;

constexpr int kDriverWallMs = 20000;   // the driver kills a case that runs longer
constexpr int kMaxWallKills = 3;       // after this many, the remaining cases are not run

// Signal words a case leaves set on the rank that checks them; every other word must still be zero.
void expect_signals(Rank& R, std::initializer_list<std::pair<size_t, uint64_t>> want) {
    for (size_t i = 0; i < kSignalWords; ++ i) {
        uint64_t v = 0;
        for (const auto& w : want) if (w.first == i) v = w.second;
        if (R.signal_word(i) != v)
            fail("assert", "signal-words", "word %zu is %llu, expected %llu", i,
                 static_cast<unsigned long long>(R.signal_word(i)), static_cast<unsigned long long>(v));
    }
}

// The bytes either side of a destination range that nothing may write.
void expect_guards(Rank& R, uint64_t dst, size_t n) {
    if (size_t bad = nonzero(R.at(dst - kGuard), kGuard) + nonzero(R.at(dst + n), kGuard))
        fail("assert", "outside-range", "%zu byte(s) written next to [%llu, +%zu)", bad,
             static_cast<unsigned long long>(dst), n);
}

Op put(int context, int peer, uint64_t src, size_t n, uint64_t dst) {
    Op o; o.kind = Op::Put; o.context = context; o.peer = peer; o.srcOff = src; o.size = n; o.dstOff = dst; return o;
}
Op put_signal(int context, int peer, uint64_t src, size_t n, uint64_t dst, size_t word, bool strong) {
    Op o = put(context, peer, src, n, dst); o.kind = Op::PutSignal; o.word = word; o.strong = strong; return o;
}
Op signal_only(int context, int peer, size_t word, bool strong) {
    Op o; o.kind = Op::Signal; o.context = context; o.peer = peer; o.word = word; o.strong = strong; return o;
}
Op get(int context, int peer, uint64_t remote, size_t n, uint64_t local) {
    Op o; o.kind = Op::Get; o.context = context; o.peer = peer; o.srcOff = remote; o.size = n; o.dstOff = local; return o;
}
Op flush(int context, int peer) { Op o; o.kind = Op::Flush; o.context = context; o.peer = peer; return o; }

// Offsets and sizes are not multiples of the chunk size, and source and destination offsets differ, so
// an ignored offset, a swapped pair or a dropped tail shows.

// A put of many chunks, then a signal of size 0 on the same context, submitted in the same proxy
// iteration. The sender runs that one iteration (which releases at most a few chunks) and then holds: for
// a weak signal until the target has looked, for a strong one up to kHoldMs, since the target cannot look
// before the signal arrives.
//
// strong: the signal is ordered behind every byte of the put, so it cannot arrive during the hold and the
// target sees the whole put when it does. weak, under reverse-ops (newest operation first): the signal
// overtakes the put and arrives during the hold, so the target must see most of the put missing. The two
// differ only in the flag; the weak case fails if the check cannot see what the flag guards.
constexpr int kHoldMs = 300;
void put_then_signal(Rank& R, bool strong, int salt) {
    constexpr size_t n = (1 << 20) - 256 + 13, src = 4099, dst = (1 << 20) + kGuard;
    if (g_rank == 0) {
        fill(R.at(src), n, 0, salt);
        R.pause();
        const Ticket* p = R.issue(put(0, 1, src, n, dst));
        const Ticket* s = R.issue(signal_only(0, 1, 0, strong));
        R.run_steps(1);
        if (strong) {
            // The flag cannot come before the signal, and the signal cannot come before the put.
            set_where("hold");
            const auto start = Clock::now();
            while (g_shared->flag[0].load(std::memory_order_acquire) == 0 and
                   Clock::now() - start < std::chrono::milliseconds(kHoldMs))
                std::this_thread::sleep_for(std::chrono::microseconds(100));
        } else {
            wait_flag(0, "target-looked");
        }
        R.resume();
        R.wait_retired(p, "put");
        R.wait_retired(s, "signal");
        set_flag(1);
    } else {
        R.wait_peer([&] { return R.signal_word(0) >= 1; }, "signal-word");
        const size_t bad = mismatches(R.at(dst), n, 0, salt);
        set_flag(0);
        if (strong and bad != 0) fail("assert", "strong-orders-earlier-put", "%zu bytes missing", bad);
        if (not strong and bad == 0) fail("assert", "control-missed", "%s", "the weak signal arrived after its put");
        R.wait_peer([&] { return mismatches(R.at(dst), n, 0, salt) == 0; }, "put-complete");
        wait_flag(1, "sender-done");
        expect_guards(R, dst, n);
        expect_signals(R, {{0, 1}});
    }
}
void put_then_strong_signal(Rank& R) { put_then_signal(R, true, 1); }
void weak_signal_control(Rank& R) { put_then_signal(R, false, 2); }

// Contexts are independent: a strong signal on one context is not ordered behind an earlier put on another
// context to the same peer. Under reverse-ops the newest operation goes first, so the signal arrives while part
// of the put on the other context is still missing. The sender holds after two proxy iterations (at most eight
// chunks each, of the put's 128) until the target has looked.
// Run with the put on context 0 and the signal on 1, and the other way round, so neither entry point's context
// can be dropped without a failure.
void strong_signal_other_context(Rank& R, int put_ctx, int sig_ctx) {
    constexpr size_t n = (1 << 20) - 256 + 13, src = 4099, dst = (1 << 20) + kGuard;
    if (g_rank == 0) {
        fill(R.at(src), n, 0, 19);
        // The put is submitted in one proxy iteration and the signal in the next, so the put is the earlier
        // operation whichever context the proxy visits first.
        R.pause();
        const Ticket* p = R.issue(put(put_ctx, 1, src, n, dst));
        R.run_steps(1);
        const Ticket* s = R.issue(signal_only(sig_ctx, 1, 7, /*strong=*/true));
        R.run_steps(1);
        wait_flag(0, "target-looked");
        R.resume();
        R.wait_retired(p, "put");
        R.wait_retired(s, "signal");
        set_flag(1);
    } else {
        R.wait_peer([&] { return R.signal_word(7) >= 1; }, "signal-word");
        const size_t bad = mismatches(R.at(dst), n, 0, 19);
        set_flag(0);
        if (bad == 0) fail("assert", "ordered-across-contexts", "%s", "the strong signal waited for the other context's put");
        R.wait_peer([&] { return mismatches(R.at(dst), n, 0, 19) == 0; }, "put-complete");
        wait_flag(1, "sender-done");
        expect_guards(R, dst, n);
        expect_signals(R, {{7, 1}});
    }
}

void strong_signal_other_context_01(Rank& R) { strong_signal_other_context(R, 0, 1); }
void strong_signal_other_context_10(Rank& R) { strong_signal_other_context(R, 1, 0); }

// I1 over many rounds, alternating contexts: when a weak signal's word reaches its value, the put it was
// bundled with has landed. The source is overwritten as soon as the put retires, so a plugin that still
// reads it after reporting completion corrupts the destination.
void i1_holds(Rank& R) {
    // A fresh destination each round: the barrier that orders one round's reads before the next round's
    // writes runs between processes, where ThreadSanitizer cannot see it.
    constexpr size_t n = (64 << 10) + 13, src = 4099, base = 1 << 20;
    constexpr int rounds = 12;
    for (int r = 0; r < rounds; ++ r) {
        const size_t dst = base + kGuard + size_t(r) * (n + kGuard);
        if (g_rank == 0) {
            fill(R.at(src), n, 0, r);
            R.wait_retired(R.issue(put_signal(r % kContexts, 1, src, n, dst, 1, false)), "put-with-signal");
        } else {
            R.wait_peer([&] { return R.signal_word(1) >= uint64_t(r + 1); }, "signal-word");
            if (size_t bad = mismatches(R.at(dst), n, 0, r)) fail("assert", "i1", "round %d: %zu bytes", r, bad);
        }
        barrier("round");
    }
    if (g_rank == 1) {
        for (int r = 0; r < rounds; ++ r) expect_guards(R, base + kGuard + size_t(r) * (n + kGuard), n);
        expect_signals(R, {{1, uint64_t(rounds)}});
    }
}

// The negative control for i1_holds: under violate-i1 the sender's one proxy iteration releases the
// signal and the first chunks, and holds the rest until the target has looked.
void violate_control(Rank& R) {
    constexpr size_t n = (1 << 20) - 256 + 13, src = 4099, dst = (1 << 20) + kGuard;
    if (g_rank == 0) {
        fill(R.at(src), n, 0, 3);
        R.pause();
        const Ticket* t = R.issue(put_signal(0, 1, src, n, dst, 2, false));
        R.run_steps(1);
        wait_flag(0, "target-looked");
        R.resume();
        R.wait_retired(t, "put-with-signal");
    } else {
        R.wait_peer([&] { return R.signal_word(2) >= 1; }, "signal-word");
        const size_t bad = mismatches(R.at(dst), n, 0, 3);
        set_flag(0);
        if (bad == 0) fail("assert", "control-missed", "%s", "the signal arrived with its whole put");
        R.wait_peer([&] { return mismatches(R.at(dst), n, 0, 3) == 0; }, "put-complete");
        expect_guards(R, dst, n);
        expect_signals(R, {{2, 1}});
    }
}

// Waits for `t` to retire, without failing either way, and gives up only once both kStopMs have passed
// and the rank's own proxy has run kStopSteps iterations: a starved proxy is not a get that waits.
constexpr int kStopMs = 2000;
constexpr uint64_t kStopSteps = 1000;
void wait_retired_while_stopped(Rank& R, const Ticket* t) {
    set_where("peer-stopped");
    const auto start = Clock::now();
    const uint64_t first = R.steps.load();
    while (t->retired.load(std::memory_order_acquire) == 0 and
           (Clock::now() - start < std::chrono::milliseconds(kStopMs) or R.steps.load() - first < kStopSteps))
        std::this_thread::sleep_for(std::chrono::microseconds(100));
}

// A get completes when its bytes are in the local window. The owner's whole process is stopped while the
// get is issued and for up to kStopMs after, so a get that retires in that time must already hold its
// data: a one-sided read has it, and a reply from the owner cannot have come. One that does not retire is
// given the owner back and must have its data when it retires.
void get_completes_with_data(Rank& R) {
    constexpr size_t n = (4 << 10) + 13, remote = (256 << 10) + 7, local = (300 << 10) + kGuard;
    if (g_rank == 1) fill(R.at(remote), n, 1, 5);
    barrier("owner filled");
    if (g_rank == 0) {
        stop_peer(1);
        const Ticket* g = R.issue(get(0, 1, remote, n, local));
        wait_retired_while_stopped(R, g);
        const bool early = g->retired.load(std::memory_order_acquire) != 0;
        const size_t bad_early = early ? mismatches(R.at(local), n, 1, 5) : 0;
        continue_peer(1);
        if (bad_early) fail("assert", "data-at-early-completion", "%zu bytes", bad_early);
        R.wait_retired(g, "get");
        if (size_t bad = mismatches(R.at(local), n, 1, 5)) fail("assert", "data-at-completion", "%zu bytes", bad);
        expect_guards(R, local, n);
        set_flag(0);
    } else {
        wait_flag(0, "requester-done");
    }
}

// A get needs nothing from the owner: the owner's proxy never runs while the requester waits for the
// get to retire with its bytes in place. One wait covers both, so fixing completion alone leaves the
// failure as it is.
// Run with a small get and with one larger than a socket's send buffer, so that part of the reply has to wait
// in the owner's outbox while the owner's proxy is not running.
void get_owner_idle_n(Rank& R, size_t n, uint64_t remote, uint64_t local) {
    if (g_rank == 1) { fill(R.at(remote), n, 1, 6); R.pause(); }
    barrier("owner filled");
    if (g_rank == 0) {
        const Ticket* g = R.issue(get(0, 1, remote, n, local));
        R.wait_until([&] { return g->retired.load(std::memory_order_acquire) != 0 and
                                  mismatches(R.at(local), n, 1, 6) == 0; }, "get");
        set_flag(0);
    } else {
        wait_flag(0, "requester-done");
        R.resume();
    }
}
void get_owner_idle(Rank& R) { get_owner_idle_n(R, (4 << 10) + 13, (256 << 10) + 7, (300 << 10) + kGuard); }
void get_owner_idle_large(Rank& R) { get_owner_idle_n(R, (1536 << 10) + 13, 7, kGuard); }

// A flush issued after a get completes, and the get's bytes are in place when it does. With
// `data_first` the flush is issued only once every byte of the get has landed, so nothing is left for it
// to fence and it must complete; without, it is issued as soon as the get retires and has to wait for
// the reply. Run with a get of one chunk and of several, and with no get at all, so a failure can be
// attributed to the size of the get rather than to the flush.
void get_then_flush(Rank& R, size_t n, int salt, bool data_first) {
    const size_t remote = (768 << 10) + 7, local = (900 << 10) + kGuard;
    if (g_rank == 1) fill(R.at(remote), n, 1, salt);
    barrier("owner filled");
    if (g_rank == 0) {
        if (n > 0) R.wait_retired(R.issue(get(0, 1, remote, n, local)), "get");
        if (data_first)
            R.wait_until([&] { return mismatches(R.at(local), n, 1, salt) == 0; }, "get-data",
                         [&] { return uint64_t(n - mismatches(R.at(local), n, 1, salt)); });
        R.wait_retired(R.issue(flush(0, 1)), "flush");
        if (size_t bad = mismatches(R.at(local), n, 1, salt)) fail("assert", "data-at-flush", "%zu bytes", bad);
        if (n > 0) expect_guards(R, local, n);
        set_flag(0);
    } else {
        wait_flag(0, "requester-done");
    }
}
void get_one_chunk_flush(Rank& R) { get_then_flush(R, (4 << 10) + 13, 7, true); }
void get_multichunk_flush(Rank& R) { get_then_flush(R, (64 << 10) + 13, 8, true); }
// A flush issued straight after a get, with the owner stopped: the flush must not complete before the
// whole reply has landed, and the data must be in place when it does. For a reply of several chunks this is
// the case that catches a fence released when its count passes through zero part way.
void get_early_flush(Rank& R, size_t n) {
    constexpr size_t remote = (768 << 10) + 7, local = (900 << 10) + kGuard;
    if (g_rank == 1) fill(R.at(remote), n, 1, 11);
    barrier("owner filled");
    if (g_rank == 0) {
        stop_peer(1);
        R.issue(get(0, 1, remote, n, local));
        const Ticket* f = R.issue(flush(0, 1));
        wait_retired_while_stopped(R, f);
        const bool early = f->retired.load(std::memory_order_acquire) != 0;
        const size_t bad_early = early ? mismatches(R.at(local), n, 1, 11) : 0;
        continue_peer(1);
        if (bad_early) fail("assert", "early-flush-without-data", "%zu bytes", bad_early);
        R.wait_retired(f, "flush");
        if (size_t bad = mismatches(R.at(local), n, 1, 11)) fail("assert", "data-at-flush", "%zu bytes", bad);
        expect_guards(R, local, n);
        set_flag(0);
    } else {
        wait_flag(0, "requester-done");
    }
}
void get_one_chunk_early_flush(Rank& R) { get_early_flush(R, (4 << 10) + 13); }

// Two gets of different sizes outstanding together on one context, then a flush: it completes only once
// both replies have landed. And a get of zero bytes, which has no reply to wait for, followed by a flush.
void two_gets_flush(Rank& R) {
    constexpr size_t a = (64 << 10) + 13, b = (20 << 10) + 5, ra = (768 << 10) + 7, rb = ra + a + 100,
                     la = (1200 << 10) + kGuard, lb = la + a + 2 * kGuard;
    if (g_rank == 1) { fill(R.at(ra), a, 1, 14); fill(R.at(rb), b, 1, 15); }
    barrier("owner filled");
    if (g_rank == 0) {
        R.pause();
        R.issue(get(0, 1, ra, a, la));
        R.issue(get(0, 1, rb, b, lb));
        const Ticket* f = R.issue(flush(0, 1));
        R.resume();
        R.wait_retired(f, "flush");
        if (size_t bad = mismatches(R.at(la), a, 1, 14) + mismatches(R.at(lb), b, 1, 15))
            fail("assert", "data-at-flush", "%zu bytes", bad);
        expect_guards(R, la, a);
        expect_guards(R, lb, b);
        set_flag(0);
    } else {
        wait_flag(0, "requester-done");
    }
}
void zero_byte_get_flush(Rank& R) {
    constexpr size_t local = (900 << 10) + kGuard;
    barrier("owner ready");
    if (g_rank == 0) {
        R.wait_retired(R.issue(get(0, 1, 7, 0, local)), "get");
        R.wait_retired(R.issue(flush(0, 1)), "flush");
        if (size_t bad = nonzero(R.at(local - kGuard), 2 * kGuard)) fail("assert", "zero-byte-get-wrote", "%zu bytes", bad);
        set_flag(0);
    } else {
        wait_flag(0, "requester-done");
    }
}
void get_multichunk_early_flush(Rank& R) { get_early_flush(R, (64 << 10) + 13); }
void flush_without_get(Rank& R) { get_then_flush(R, 0, 9, false); }

// Under SRDSIM_DRY_RUN=1 every operation completes without moving data. Two puts, the second waited for
// first, tell "the first request never completes" apart from "no dry-run request completes".
void dry_run(Rank& R) {
    constexpr size_t n = 4096 + 13, src = 4099, dst = (512 << 10) + kGuard;
    if (g_rank == 0) {
        fill(R.at(src), 2 * n, 0, 10);
        R.pause();
        const Ticket* a = R.issue(put(0, 1, src, n, dst));
        const Ticket* b = R.issue(put(0, 1, src + n, n, dst + n));
        R.resume();
        R.wait_done(b, "put-second");
        R.wait_done(a, "put-first");
        set_flag(0);
    } else {
        wait_flag(0, "sender-done");
        if (size_t bad = nonzero(R.at(dst - kGuard), 2 * n + 2 * kGuard)) fail("assert", "dry-run-wrote", "%zu bytes", bad);
    }
}

// Windows registered and deregistered on the main thread while puts with signals are in flight on the
// proxy thread, as NCCL registers user buffers while its progress thread runs.
void register_during_traffic(Rank& R) {
    constexpr size_t n = (64 << 10) + 13, src = 4099, base = 1 << 20;
    constexpr int rounds = 8;
    // A source per put: each is overwritten when its own put retires, while the others may still be read.
    // The sender's proxy is slowed so that the puts are still in flight while the windows are registered.
    if (g_rank == 0) {
        R.step_delay_us.store(20000);
        for (int r = 0; r < rounds; ++ r) {
            fill(R.at(src + size_t(r) * n), n, 0, 12 + r);
            R.issue(put_signal(r % kContexts, 1, src + size_t(r) * n, n, base + kGuard + size_t(r) * (n + kGuard), 4, false));
        }
    }
    const Ticket* last = nullptr;
    if (g_rank == 0) last = R.last_ticket();
    for (int k = 0; k < 4; ++ k) {
        R.register_and_release((64 << 10) + 4096 * size_t(k));
        // The case is about registration while traffic is in flight; if all of it has already retired, it
        // tested nothing that setup does not.
        if (k == 0 and last and last->retired.load(std::memory_order_acquire) != 0)
            fail("harness", "no-overlap", "%s", "every put retired before the first registration returned");
    }
    R.step_delay_us.store(0);
    if (g_rank == 1) {
        R.wait_peer([&] { return R.signal_word(4) >= uint64_t(rounds); }, "signal-word");
        for (int r = 0; r < rounds; ++ r)
            if (size_t bad = mismatches(R.at(base + kGuard + size_t(r) * (n + kGuard)), n, 0, 12 + r))
                fail("assert", "data", "round %d: %zu bytes", r, bad);
        for (int r = 0; r < rounds; ++ r) expect_guards(R, base + kGuard + size_t(r) * (n + kGuard), n);
        expect_signals(R, {{4, uint64_t(rounds)}});
    }
}

// An operation the plugin refuses is reported through queryLastError and its request completes, and the
// slot that request used serves the next, real operation, which must complete only once delivered and
// with its data. The refused put is larger than the largest single write the simulator models.
void refused_then_real(Rank& R) {
    constexpr size_t n = (16 << 10) + 13, src = 4099, dst = (512 << 10) + kGuard;
    if (g_rank == 0) {
        fill(R.at(src), n, 0, 13);
        // One byte over the largest single write the scheduler models.
        Op big = put(0, 1, src, srdsim::Scheduler::Options{}.max_op_bytes + 1, dst);
        big.refused = true;
        const Ticket* bad = R.issue(big);
        R.wait_retired(bad, "refused");
        if (not R.last_error()) fail("assert", "refusal-reported", "%s", "queryLastError reports nothing");
        R.wait_retired(R.issue(put_signal(0, 1, src, n, dst, 5, false)), "put-with-signal");
    } else {
        R.wait_peer([&] { return R.signal_word(5) >= 1; }, "signal-word");
        if (size_t bad = mismatches(R.at(dst), n, 0, 13)) fail("assert", "data", "%zu bytes", bad);
        expect_guards(R, dst, n);
        expect_signals(R, {{5, 1}});
    }
}

// A context destroyed with a get outstanding: the get still finishes when tested, its late reply does not
// count against the context created in its place, and a get and flush on the new context complete with
// their data.
void get_across_context_rebuild(Rank& R) {
    constexpr size_t n = (16 << 10) + 13, ra = (256 << 10) + 7, rb = (512 << 10) + 7,
                     la = (1200 << 10) + kGuard, lb = (1400 << 10) + kGuard;
    if (g_rank == 1) { fill(R.at(ra), n, 1, 16); fill(R.at(rb), n, 1, 17); }
    barrier("owner filled");
    if (g_rank == 0) {
        stop_peer(1);
        const Ticket* a = R.issue(get(0, 1, ra, n, la));
        std::this_thread::sleep_for(std::chrono::milliseconds(50));   // its request leaves
        R.recreate_context();
        continue_peer(1);
        R.wait_retired(a, "aborted-get");
        const Ticket* b = R.issue(get(0, 1, rb, n, lb));
        const Ticket* f = R.issue(flush(0, 1));
        R.wait_retired(b, "get");
        R.wait_retired(f, "flush");
        if (size_t bad = mismatches(R.at(lb), n, 1, 17)) fail("assert", "data-on-new-context", "%zu bytes", bad);
        expect_guards(R, lb, n);
        set_flag(0);
    } else {
        wait_flag(0, "requester-done");
    }
}

// More than the transport's 64 MiB outbox budget, put to a target whose process is stopped: nothing is lost.
// Units the link cannot take must stay with the plugin until it can, so while the target is stopped no
// error is reported, and once it continues every put lands and its signal counts once.
void backpressure_target_stopped(Rank& R) {
    constexpr size_t n = (1 << 20) - 256 + 13, src = 4099, dst = (1 << 20) + kGuard;
    constexpr int puts = 80;
    if (g_rank == 0) {
        fill(R.at(src), n, 0, 18);
        stop_peer(1);
        std::vector<const Ticket*> t;
        for (int i = 0; i < puts; ++ i) {
            Op o = put_signal(0, 1, src, n, dst, 6, false);
            o.keep_source = true;
            t.push_back(R.issue(o));
        }
        set_where("target-stopped");
        const auto start = Clock::now();
        const uint64_t first = R.steps.load();
        while (Clock::now() - start < std::chrono::milliseconds(2000) or R.steps.load() - first < 1000)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const bool lost = R.last_error();
        continue_peer(1);
        if (lost) fail("assert", "backpressure-lost", "%s", "queryLastError reported a failed send");
        for (const Ticket* x : t) R.wait_retired(x, "put-with-signal");
        set_flag(0);
    } else {
        R.wait_peer([&] { return R.signal_word(6) >= uint64_t(puts); }, "signal-word");
        if (size_t bad = mismatches(R.at(dst), n, 0, 18)) fail("assert", "data", "%zu bytes", bad);
        wait_flag(0, "sender-done");
        expect_guards(R, dst, n);
        expect_signals(R, {{6, uint64_t(puts)}});
    }
}

// An operation on a context index the RMA context does not have is reported through queryLastError, and still
// carried out: the put lands and retires, and a valid operation after it does too.
void out_of_range_context(Rank& R) {
    constexpr size_t n = (16 << 10) + 13, src = 4099, dst = (512 << 10) + kGuard;
    if (g_rank == 0) {
        fill(R.at(src), n, 0, 21);
        Op bad = put(0, 1, src, n, dst);
        bad.plugin_context = kContexts;
        R.wait_retired(R.issue(bad), "put-bad-context");
        if (not R.last_error()) fail("assert", "context-reported", "%s", "queryLastError reports nothing");
        R.wait_retired(R.issue(signal_only(0, 1, 9, true)), "signal");
    } else {
        R.wait_peer([&] { return R.signal_word(9) >= 1; }, "signal-word");
        R.wait_peer([&] { return mismatches(R.at(dst), n, 0, 21) == 0; }, "put-data");
        expect_guards(R, dst, n);
        expect_signals(R, {{9, 1}});
    }
}

#if defined(HARNESS_CUDA)
// A signal into a device window registered without NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET is an add to whatever the
// word holds, because the application may reset it: the target sets the word back to zero between two signals
// of 1, and must then read 1, not the running total 2. With the flag the plugin is the only writer and keeps
// the total, so the same sequence reads 2: that run shows the flag selects the stored total. Needs a GPU; skipped
// without one, unless HARNESS_REQUIRE_GPU is set.
void device_signal(Rank& R, uint64_t flags) {
    int n = 0;
    if (cuInit(0) != CUDA_SUCCESS or cuDeviceGetCount(&n) != CUDA_SUCCESS or n < 1) {
        if (std::getenv("HARNESS_REQUIRE_GPU")) fail("error", "no-gpu", "%s", "HARNESS_REQUIRE_GPU is set");
        fail("skip", "no-gpu", "%s", "no CUDA device");
    }
    CUdevice dev{};
    CUcontext c{};
    CUdeviceptr w{};
    constexpr size_t bytes = 64 * sizeof(uint64_t);
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS or cuDevicePrimaryCtxRetain(&c, dev) != CUDA_SUCCESS or
        cuCtxSetCurrent(c) != CUDA_SUCCESS or cuMemAlloc(&w, bytes) != CUDA_SUCCESS or cuMemsetD8(w, 0, bytes) != CUDA_SUCCESS)
        fail("error", "cuda-setup", "%s", "could not allocate the device window");
    void* mh = R.register_window(reinterpret_cast<void*>(w), bytes, NCCL_PTR_CUDA, flags);
    barrier("device window registered");
    const auto word = [&] { uint64_t v = 0; cuMemcpyDtoH(&v, w, sizeof v); return v; };
    if (g_rank == 0) {
        for (int round = 0; round < 2; ++ round) {
            Op o = signal_only(0, 1, 0, false);
            o.signal_mh = mh;
            R.wait_retired(R.issue(o), "signal");
            set_flag(2 * round);
            wait_flag(2 * round + 1, "target-read");
        }
    } else {
        for (int round = 0; round < 2; ++ round) {
            wait_flag(2 * round, "signal-sent");
            R.wait_peer([&] { return word() != 0; }, "device-word");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));   // a second store, if any, lands
            const uint64_t v = word();
            const uint64_t want = (flags & NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET) ? uint64_t(round + 1) : 1;
            if (v != want) fail("assert", want == 1 ? "add-not-store" : "store-with-never-reset",
                                "round %d: the word is %llu, expected %llu", round, static_cast<unsigned long long>(v),
                                static_cast<unsigned long long>(want));
            cuMemsetD8(w, 0, bytes);   // the application resets it
            set_flag(2 * round + 1);
        }
    }
    barrier("rounds done");
    R.deregister_window(mh);
    cuMemFree(w);
}
// I1 on the atomic-add path: a put with a signal into a device window registered without the flag. When the
// word changes, every byte of the put must already be in the window.
void device_put_signal(Rank& R) {
    int n = 0;
    if (cuInit(0) != CUDA_SUCCESS or cuDeviceGetCount(&n) != CUDA_SUCCESS or n < 1) {
        if (std::getenv("HARNESS_REQUIRE_GPU")) fail("error", "no-gpu", "%s", "HARNESS_REQUIRE_GPU is set");
        fail("skip", "no-gpu", "%s", "no CUDA device");
    }
    CUdevice dev{};
    CUcontext c{};
    CUdeviceptr w{};
    constexpr size_t bytes = 2u << 20, len = (1u << 20) + 13, off = 4096, word_off = 0;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS or cuDevicePrimaryCtxRetain(&c, dev) != CUDA_SUCCESS or
        cuCtxSetCurrent(c) != CUDA_SUCCESS or cuMemAlloc(&w, bytes) != CUDA_SUCCESS or cuMemsetD8(w, 0, bytes) != CUDA_SUCCESS)
        fail("error", "cuda-setup", "%s", "could not allocate the device window");
    void* mh = R.register_window(reinterpret_cast<void*>(w), bytes, NCCL_PTR_CUDA, 0);
    barrier("device window registered");
    if (g_rank == 0) {
        fill(R.at(4099), len, 0, 22);
        Op o = put_signal(0, 1, 4099, len, off, 0, false);
        o.dst_mh = mh;
        o.signal_mh = mh;
        R.wait_retired(R.issue(o), "put-with-signal");
    } else {
        const auto word = [&] { uint64_t v = 0; cuMemcpyDtoH(&v, w + word_off, sizeof v); return v; };
        R.wait_peer([&] { return word() != 0; }, "device-word");
        std::vector<unsigned char> got(len);
        cuMemcpyDtoH(got.data(), w + off, len);
        if (size_t bad = mismatches(got.data(), len, 0, 22)) fail("assert", "i1-device", "%zu bytes missing at the signal", bad);
    }
    barrier("checked");
    R.deregister_window(mh);
    cuMemFree(w);
}

void device_signal_reset(Rank& R) { device_signal(R, 0); }
void device_signal_never_reset(Rank& R) { device_signal(R, NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET); }
#endif

// With SRDSIM_CHUNK_BYTES above the transport's default frame limit of 4 MiB, a put of one chunk larger than
// that lands: the receiver accepts the frames the sender is configured to send.
// One full chunk of exactly the limit and a tail, so the frame at the limit is accepted.
void large_chunk(Rank& R) {
    constexpr size_t n = (7u << 20) + 13, src = 4099, dst = kGuard;
    if (g_rank == 0) {
        fill(R.at(src), n, 0, 20);
        R.wait_retired(R.issue(put_signal(0, 1, src, n, dst, 8, false)), "put-with-signal");
    } else {
        R.wait_peer([&] { return R.signal_word(8) >= 1; }, "signal-word");
        if (size_t bad = mismatches(R.at(dst), n, 0, 20)) fail("assert", "data", "%zu bytes", bad);
        expect_guards(R, dst, n);
        expect_signals(R, {{8, 1}});
    }
}

// A get the owner cannot serve (it reads past the owner's window) fails instead of hanging: the get retires, a
// flush after it completes, queryLastError reports it, and a valid get after it completes with its data.
void get_refused_by_owner(Rank& R) {
    constexpr size_t n = (16 << 10) + 13, remote = (512 << 10) + 7, local = (1200 << 10) + kGuard;
    if (g_rank == 1) fill(R.at(remote), n, 1, 23);
    barrier("owner filled");
    if (g_rank == 0) {
        R.wait_retired(R.issue(get(0, 1, kDataBytes - 100, n, local)), "refused-get");
        R.wait_retired(R.issue(flush(0, 1)), "flush-after-refused");
        if (not R.last_error()) fail("assert", "refusal-reported", "%s", "queryLastError reports nothing");
        R.wait_retired(R.issue(get(0, 1, remote, n, local)), "get");
        if (size_t bad = mismatches(R.at(local), n, 1, 23)) fail("assert", "data", "%zu bytes", bad);
        expect_guards(R, local, n);
        set_flag(0);
    } else {
        wait_flag(0, "requester-done");
    }
}

// A placement the target refuses (it runs past the target's window) does not let its own signal apply: the signal
// would announce bytes that never landed. A put with a signal after it on the same link applies both, which shows
// the earlier signal frame was seen and withheld, and that only the operation that lost a placement is affected.
void signal_after_dropped_placement(Rank& R) {
    constexpr size_t n = (16 << 10) + 13, src = 4099, good = (512 << 10) + kGuard;
    if (g_rank == 0) {
        fill(R.at(src), n, 0, 24);
        Op bad = put_signal(0, 1, src, n, kDataBytes - 100, 10, false);
        bad.keep_source = true;   // the good put after it reads the same source
        R.wait_retired(R.issue(bad), "bad-put-with-signal");
        R.wait_retired(R.issue(put_signal(0, 1, src, n, good, 11, false)), "good-put-with-signal");
    } else {
        R.wait_peer([&] { return R.signal_word(11) == 1; }, "good-signal");
        if (size_t bad = mismatches(R.at(good), n, 0, 24)) fail("assert", "good-put-data", "%zu bytes", bad);
        if (R.signal_word(10) != 0) fail("assert", "signal-without-data", "%s", "the signal of a dropped placement applied");
        if (size_t bad = nonzero(R.at(kDataBytes - 100), 100)) fail("assert", "dropped-placement-wrote", "%zu bytes", bad);
    }
}

// A negative count in the environment gives the default rather than wrapping to a very large number.
void negative_count_is_default(Rank& R) {
    (void)R;
    std::lock_guard<std::mutex> g(g_log_mu);
    if (g_init_line.find("chunk_bytes=8192 ") == std::string::npos)
        fail("assert", "negative-count", "SRDSIM_CHUNK_BYTES=-4096 gave '%s'", g_init_line.c_str());
}

// The impairments from the environment are applied. Each case puts 16 chunks of 8 KiB with a signal and checks the
// data, and the time the put takes bounds the impairment from below.
void impaired_put(Rank& R, long min_ms, bool measure_on_target) {
    constexpr size_t n = 16 * 8192, src = 4099, dst = (1 << 20) + kGuard;
    if (g_rank == 0) {
        fill(R.at(src), n, 0, 25);
        barrier("ready");
        const auto t0 = Clock::now();
        R.wait_retired(R.issue(put_signal(0, 1, src, n, dst, 12, false)), "put-with-signal");
        const long ms = static_cast<long>(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count());
        if (not measure_on_target and ms < min_ms) fail("assert", "impairment-not-applied", "the put took %ld ms, want >= %ld", ms, min_ms);
    } else {
        barrier("ready");
        const auto t0 = Clock::now();
        R.wait_peer([&] { return R.signal_word(12) == 1; }, "signal-word");
        const long ms = static_cast<long>(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count());
        if (size_t bad = mismatches(R.at(dst), n, 0, 25)) fail("assert", "data", "%zu bytes", bad);
        if (measure_on_target and ms < min_ms) fail("assert", "impairment-not-applied", "the signal took %ld ms, want >= %ld", ms, min_ms);
    }
}
void impaired_delay(Rank& R) { impaired_put(R, 16 * 2, false); }          // 2 ms before each of 17 units
void impaired_freeze(Rank& R) { impaired_put(R, 400, false); }            // frozen for 500 ms after 4 units
void impaired_apply_interval(Rank& R) { impaired_put(R, 16 * 5, true); }  // 5 ms before each applied frame
void impaired_signal_lag(Rank& R) { impaired_put(R, 150, false); }        // 200 ms before the signal

struct Scenario {
    const char* name;
    void (*run)(Rank&);
    int nranks;
    std::vector<const char*> modes;
    std::vector<int> seeds;
    const char* extra_env;   // NAME=VALUE pairs, separated by spaces, set in every rank; or nullptr
};

const std::vector<const char*> kKeepI1 = {"srd", "inorder", "peer-seq", "reverse-ops", "signal-last", "defer-one"};

const std::vector<Scenario>& scenarios() {
    static const std::vector<Scenario> s = {
        {"put_then_strong_signal", put_then_strong_signal, 2, kKeepI1, {1, 2, 3}, nullptr},
        {"weak_signal_control", weak_signal_control, 2, {"reverse-ops"}, {1, 2, 3}, nullptr},
        {"strong_signal_other_context_01", strong_signal_other_context_01, 2, {"reverse-ops"}, {1}, nullptr},
        {"strong_signal_other_context_10", strong_signal_other_context_10, 2, {"reverse-ops"}, {1}, nullptr},
        {"i1_holds", i1_holds, 2, kKeepI1, {1, 2, 3}, nullptr},
        {"violate_control", violate_control, 2, {"violate-i1"}, {1, 2, 3}, nullptr},
        {"get_completes_with_data", get_completes_with_data, 2, {"srd"}, {1}, nullptr},
        {"get_owner_idle", get_owner_idle, 2, {"srd"}, {1}, nullptr},
        {"get_owner_idle_large", get_owner_idle_large, 2, {"srd"}, {1}, nullptr},
        {"get_one_chunk_flush", get_one_chunk_flush, 2, {"srd"}, {1}, nullptr},
        {"get_multichunk_flush", get_multichunk_flush, 2, {"srd"}, {1}, nullptr},
        {"get_one_chunk_early_flush", get_one_chunk_early_flush, 2, {"srd"}, {1}, nullptr},
        {"get_multichunk_early_flush", get_multichunk_early_flush, 2, {"srd"}, {1, 2, 3}, nullptr},
        {"flush_without_get", flush_without_get, 2, {"srd"}, {1}, nullptr},
        {"two_gets_flush", two_gets_flush, 2, {"srd", "reverse-ops"}, {1, 2}, nullptr},
        {"zero_byte_get_flush", zero_byte_get_flush, 2, {"srd"}, {1}, nullptr},
        {"register_during_traffic", register_during_traffic, 2, {"srd"}, {1, 2, 3}, nullptr},
        {"refused_then_real", refused_then_real, 2, {"srd"}, {1}, nullptr},
        {"get_across_context_rebuild", get_across_context_rebuild, 2, {"srd"}, {1}, nullptr},
        {"backpressure_target_stopped", backpressure_target_stopped, 2, {"srd", "inorder"}, {1}, nullptr},
        {"out_of_range_context", out_of_range_context, 2, {"srd"}, {1}, nullptr},
#if defined(HARNESS_CUDA)
        {"device_signal_reset", device_signal_reset, 2, {"srd"}, {1}, "SRDSIM_DELIVERY_CONTEXT=primary"},
        {"device_signal_never_reset", device_signal_never_reset, 2, {"srd"}, {1}, "SRDSIM_DELIVERY_CONTEXT=primary"},
        {"device_put_signal", device_put_signal, 2, {"srd", "reverse-ops"}, {1, 2}, "SRDSIM_DELIVERY_CONTEXT=primary"},
#endif
        {"large_chunk", large_chunk, 2, {"srd"}, {1}, "SRDSIM_CHUNK_BYTES=5242880"},
        {"get_refused_by_owner", get_refused_by_owner, 2, {"srd"}, {1}, nullptr},
        {"signal_after_dropped_placement", signal_after_dropped_placement, 2, {"inorder"}, {1}, nullptr},
        {"negative_count_is_default", negative_count_is_default, 2, {"srd"}, {1}, "SRDSIM_CHUNK_BYTES=-4096"},
        {"plus_sign_count_is_default", negative_count_is_default, 2, {"srd"}, {1}, "SRDSIM_CHUNK_BYTES=+4096"},
        {"overflowing_count_is_default", negative_count_is_default, 2, {"srd"}, {1}, "SRDSIM_CHUNK_BYTES=99999999999999999999"},
        {"impaired_delay", impaired_delay, 2, {"srd"}, {1}, "SRDSIM_DELAY_US_MIN=2000 SRDSIM_DELAY_US_MAX=2000"},
        {"impaired_freeze", impaired_freeze, 2, {"srd"}, {1},
         "SRDSIM_FREEZE_PEER=1 SRDSIM_FREEZE_AFTER_UNITS=4 SRDSIM_FREEZE_MS=500"},
        {"impaired_apply_interval", impaired_apply_interval, 2, {"srd"}, {1}, "SRDSIM_APPLY_INTERVAL_US=5000"},
        {"impaired_signal_lag", impaired_signal_lag, 2, {"srd"}, {1}, "SRDSIM_SIGNAL_LAG_US=200000"},
        {"dry_run", dry_run, 2, {"srd"}, {1}, "SRDSIM_DRY_RUN=1"},
    };
    return s;
}

// Known defects (#26): the case fails today with exactly this status, "kind:id". A mode of "*" and a
// table of 0 match any. A fix removes its entry; the run fails if a listed case passes or fails differently.
struct Expected { const char* scenario; const char* mode; int table; const char* status; const char* issue; };
// None today.
const std::vector<Expected> kExpected = {};

const Expected* expected_for(const Scenario& sc, const char* mode, int table) {
    for (const auto& e : kExpected)
        if (std::strcmp(e.scenario, sc.name) == 0 and (std::strcmp(e.mode, "*") == 0 or std::strcmp(e.mode, mode) == 0) and
            (e.table == 0 or e.table == table))
            return &e;
    return nullptr;
}

struct Result {
    int rank = -1;           // the rank whose status decided it; -1 when the driver decided
    std::string status;      // "ok", or "kind:id detail"
    bool wall_clock = false;
};

void kill_all(const std::vector<pid_t>& pids, const std::vector<bool>& exited) {
    for (size_t q = 0; q < pids.size(); ++ q) if (not exited[q]) kill(-pids[q], SIGKILL);
}

// Runs one case in fresh processes and returns the first rank's non-ok status, or "ok".
Result run_case(const std::string& plugin, int table, const Scenario& sc, const char* mode, int seed) {
    void* mem = mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) return {-1, "harness:mmap errno " + std::to_string(errno)};
    auto* shared = new (mem) Shared();
    const int nranks = sc.nranks;
    std::vector<pid_t> pids;
    std::vector<bool> exited;
    std::fflush(nullptr);                       // or each rank flushes its inherited copy of the buffers
    for (int r = 0; r < nranks; ++ r) {
        const pid_t pid = fork();
        if (pid < 0) {
            const int err = errno;
            kill_all(pids, exited);
            for (pid_t p : pids) waitpid(p, nullptr, 0);
            munmap(mem, sizeof(Shared));
            return {-1, "harness:fork errno " + std::to_string(err)};
        }
        if (pid == 0) {
            setpgid(0, 0);
#if defined(__linux__)
            prctl(PR_SET_PDEATHSIG, SIGKILL);   // a rank never outlives the driver
#endif
            g_shared = shared; g_rank = r; g_nranks = nranks;
            shared->pid[r] = getpid();          // published to the peers by the first barrier
            setenv("SRDSIM_MODE", mode, 1);
            setenv("SRDSIM_SEED", std::to_string(seed).c_str(), 1);
            if (sc.extra_env) {   // NAME=VALUE pairs separated by spaces
                std::string all = sc.extra_env;
                for (size_t a = 0; a < all.size();) {
                    size_t b = all.find(' ', a);
                    if (b == std::string::npos) b = all.size();
                    const std::string kv = all.substr(a, b - a);
                    const size_t eq = kv.find('=');
                    if (eq != std::string::npos) setenv(kv.substr(0, eq).c_str(), kv.substr(eq + 1).c_str(), 1);
                    a = b + 1;
                }
            }
            Rank R;
            R.setup(plugin, table, mode, seed);
            sc.run(R);
            R.teardown();
            R.pass();
        }
        setpgid(pid, pid);                      // both sides, so an early kill(-pid) cannot miss
        pids.push_back(pid);
        exited.push_back(false);
    }
    Result result{-1, "ok"};
    const auto start = Clock::now();
    int left = nranks;
    while (left > 0) {
        for (int r = 0; r < nranks; ++ r) {
            if (exited[r]) continue;
            int st = 0;
            if (waitpid(pids[r], &st, WNOHANG) != pids[r]) continue;
            exited[r] = true; -- left;
            std::string s = shared->status[r];
            const bool clean = WIFEXITED(st) and WEXITSTATUS(st) == 0;
            if (s == "ok" and not clean) s = "exit:after-ok status " + std::to_string(st);
            if (s.empty()) s = WIFSIGNALED(st) ? "crash:signal " + std::to_string(WTERMSIG(st))
                                               : "exit:no-status code " + std::to_string(WEXITSTATUS(st));
            if (s != "ok" and result.status == "ok") {
                result = {r, s};
                kill_all(pids, exited);         // the peers would wait until their own wall clock
            }
        }
        if (left > 0 and Clock::now() - start > std::chrono::milliseconds(kDriverWallMs) and not result.wall_clock) {
            std::string where;
            for (int q = 0; q < nranks; ++ q)
                if (not exited[q]) where += " rank" + std::to_string(q) + "@" + shared->where[q];
            kill_all(pids, exited);
            if (result.status == "ok") result = {-1, "killed:wall-clock" + where};
            result.wall_clock = true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    munmap(mem, sizeof(Shared));
    return result;
}

// "kind:id", without the detail.
std::string status_key(const std::string& s) { return s.substr(0, s.find(' ')); }

bool parse_int(const char* s, int* out) {
    char* end = nullptr;
    errno = 0;
    const long v = std::strtol(s, &end, 10);
    if (errno != 0 or end == s or *end != '\0' or v < 0 or v > 1000000) return false;
    *out = int(v);
    return true;
}

// A case's settings come from the case alone, not from whatever the caller's environment holds.
void clear_srdsim_env() {
    std::vector<std::string> names;
    for (char** e = environ; *e; ++ e)
        if (std::strncmp(*e, "SRDSIM_", 7) == 0) names.emplace_back(*e, std::strcspn(*e, "="));
    for (const auto& n : names) unsetenv(n.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    const char* usage = "usage: %s PLUGIN_SO TABLE_VERSION [--scenario NAME] [--mode MODE] [--seed N]\n";
    if (argc < 3) { std::fprintf(stderr, usage, argv[0]); return 2; }
    const std::string plugin = argv[1];
    int table = 0;
    if (not parse_int(argv[2], &table)) { std::fprintf(stderr, "TABLE_VERSION '%s' is not a number\n", argv[2]); return 2; }
    const char *only = nullptr, *only_mode = nullptr;
    int only_seed = -1;
    for (int i = 3; i < argc; i += 2) {
        if (i + 1 >= argc) { std::fprintf(stderr, usage, argv[0]); return 2; }
        if (std::strcmp(argv[i], "--scenario") == 0) only = argv[i + 1];
        else if (std::strcmp(argv[i], "--mode") == 0) only_mode = argv[i + 1];
        else if (std::strcmp(argv[i], "--seed") == 0 and parse_int(argv[i + 1], &only_seed)) {}
        else { std::fprintf(stderr, usage, argv[0]); return 2; }
    }
    clear_srdsim_env();
    g_verbose = std::getenv("HARNESS_VERBOSE") != nullptr;

    int passed = 0, xfailed = 0, failed = 0, wall_kills = 0, ran = 0, not_run = 0, skipped = 0;
    for (const auto& sc : scenarios()) {
        if (only and std::strcmp(only, sc.name) != 0) continue;
        for (const char* mode : sc.modes) {
            if (only_mode and std::strcmp(only_mode, mode) != 0) continue;
            for (int seed : sc.seeds) {
                if (only_seed >= 0 and only_seed != seed) continue;
                if (wall_kills >= kMaxWallKills) { ++ not_run; continue; }
                ++ ran;
                const Result r = run_case(plugin, table, sc, mode, seed);
                if (r.wall_clock) ++ wall_kills;
                char where[256];
                std::snprintf(where, sizeof(where), "%s mode=%s seed=%d table=v%d", sc.name, mode, seed, table);
                char replay[256];
                std::snprintf(replay, sizeof(replay), "--scenario %s --mode %s --seed %d", sc.name, mode, seed);
                const std::string who = r.rank >= 0 ? "rank " + std::to_string(r.rank) + ": " : "";
                const Expected* e = expected_for(sc, mode, table);
                if (status_key(r.status).rfind("skip:", 0) == 0) {
                    ++ skipped; std::printf("SKIP   %s: %s\n", where, r.status.c_str());
                } else if (e == nullptr) {
                    if (r.status == "ok") { ++ passed; std::printf("PASS   %s\n", where); }
                    else { ++ failed; std::printf("FAIL   %s: %s%s (replay: %s %d %s)\n", where, who.c_str(), r.status.c_str(), argv[1], table, replay); }
                } else if (status_key(r.status) == e->status) {
                    ++ xfailed; std::printf("XFAIL  %s: %s%s (%s)\n", where, who.c_str(), r.status.c_str(), e->issue);
                } else if (r.status == "ok") {
                    ++ failed; std::printf("XPASS  %s: passes; remove it from kExpected (%s)\n", where, e->issue);
                } else {
                    ++ failed; std::printf("FAIL   %s: %s%s, expected %s (replay: %s %d %s)\n", where, who.c_str(), r.status.c_str(), e->status, argv[1], table, replay);
                }
                std::fflush(stdout);
            }
        }
    }
    if (wall_kills >= kMaxWallKills)
        std::printf("ABORT  %d cases hit the wall clock; the rest were not run\n", wall_kills);
    std::printf("%d cases: %d passed, %d expected failures, %d failed, %d not run, %d skipped\n", ran + not_run, passed,
                xfailed, failed, not_run, skipped);
    if (ran == 0) { std::printf("no case matched the filter\n"); return 1; }
    return failed == 0 and not_run == 0 ? 0 : 1;
}
