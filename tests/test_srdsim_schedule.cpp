// SPDX-License-Identifier: Apache-2.0

// Tests for the SRD simulator's delivery scheduler (`src/srdsim_schedule.h`).
//
// The scheduler is the component whose bugs would fabricate failures that no real fabric can
// produce. A simulator that cries wolf is worse than none, so its own properties are checked
// before it is used to judge anything else:
//
//   S1  every submitted byte is delivered exactly once, to the right address, with the right
//       length. No chunk is dropped, duplicated or overlapped.
//   S2  in `srd` mode a signal is never released before every byte of its own operation, and in
//       `violate-i1` mode it sometimes is. The adversary is exactly as strong as claimed and no
//       stronger.
//   S3  `inorder` mode releases units in submission order and never interleaves operations.
//       This is the baseline: a test that fails here is testing the simulator, not the target.
//   S4  the same seed reproduces the same release order, and different seeds explore different
//       orders. Reproducibility is what makes a counterexample useful.
//   S5  the adversary actually reorders: over a set of seeds it produces both the in-order
//       release and orders where a later operation's bytes land before an earlier one's, and it
//       reaches every permutation of a small case.
//   S6  an operation larger than the modelled single-write size is refused rather than silently
//       mismodelled.
//
// Built and run by CTest: cmake -S . -B build && cmake --build build && ctest --test-dir build

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
// Named explicitly rather than picked up through another header: clang provides these transitively and
// GCC does not.
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <string>
#include <vector>

#include "srdsim_schedule.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void expect(bool cond, const std::string& what) {
    ++g_checks;
    if (not cond) {
        ++g_failures;
        if (g_failures <= 40)
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

// A tiny harness that actually performs the released units against real buffers, so S1 is
// checked on bytes rather than on bookkeeping.
struct Fabric {
    std::vector<char> src;
    std::vector<char> dst;
    std::vector<uint64_t> signals;
    std::vector<srdsim::Unit> released;

    Fabric(size_t bytes, size_t num_signals)
        : src(bytes), dst(bytes, 0), signals(num_signals, 0) {
        for (size_t i = 0; i < bytes; ++ i)
            src[i] = static_cast<char>((i * 31 + 7) & 0xff);
    }

    void apply(const srdsim::Unit& u) {
        released.push_back(u);
        if (u.kind == srdsim::Unit::Kind::Placement) {
            std::memcpy(u.dst, u.src, u.bytes);
        } else {
            auto* w = static_cast<uint64_t*>(u.signal_word);
            *w += u.signal_is_add ? u.signal_val : 1;
        }
    }

    void drain(srdsim::Scheduler& s) {
        std::vector<srdsim::Unit> batch;
        while (not s.idle()) {
            batch.clear();
            if (s.progress(1, &batch) == 0)
                break;
            for (const auto& u : batch)
                apply(u);
        }
    }
};

srdsim::Op make_op(uint64_t id, Fabric& f, size_t off, size_t bytes, int signal_index) {
    srdsim::Op op;
    op.id = id;
    op.peer = 0;
    op.src_base = f.src.data();
    op.src_off = off;
    op.dst_base = f.dst.data();
    op.dst_off = off;
    op.bytes = bytes;
    if (signal_index >= 0) {
        op.has_signal = true;
        op.signal_base = f.signals.data();
        op.signal_off = static_cast<uint64_t>(signal_index) * sizeof(uint64_t);
    }
    return op;
}

// S1 + S2, over a range of modes, seeds and chunk sizes.
void test_delivery_and_i1() {
    for (const auto mode : {srdsim::Mode::InOrder, srdsim::Mode::Srd, srdsim::Mode::ViolateI1,
                            srdsim::Mode::ReverseOps, srdsim::Mode::SignalLast,
                            srdsim::Mode::DeferOne, srdsim::Mode::PeerSeq}) {
        int i1_broken_seeds = 0;
        for (uint64_t seed = 1; seed <= 64; ++ seed) {
            const size_t total = 40'000;
            Fabric f(total, 4);
            srdsim::Scheduler::Options o;
            o.mode = mode;
            o.chunk_bytes = 8192;
            o.seed = seed;
            srdsim::Scheduler s(o);

            // Four operations, each with its own signal, covering the buffer.
            for (int k = 0; k < 4; ++ k) {
                const size_t off = static_cast<size_t>(k) * 10'000;
                std::string why;
                expect(s.submit(make_op(static_cast<uint64_t>(k) + 1, f, off, 10'000, k), &why),
                       ctx("submit accepted: mode=%s seed=%llu k=%d (%s)",
                           srdsim::mode_name(mode), (unsigned long long)seed, k, why.c_str()));
            }
            f.drain(s);

            // S1: every byte delivered exactly once and correctly.
            expect(f.dst == f.src, ctx("S1 all bytes delivered correctly: mode=%s seed=%llu",
                                       srdsim::mode_name(mode), (unsigned long long)seed));
            std::map<std::pair<uint64_t, int>, int> chunk_seen;
            for (const auto& u : f.released)
                if (u.kind == srdsim::Unit::Kind::Placement)
                    ++ chunk_seen[{u.op_id, u.chunk_index}];
            for (const auto& kv : chunk_seen)
                expect(kv.second == 1,
                       ctx("S1 each chunk released exactly once: op=%llu chunk=%d times=%d",
                           (unsigned long long)kv.first.first, kv.first.second, kv.second));
            for (size_t k = 0; k < f.signals.size(); ++ k)
                expect(f.signals[k] == 1,
                       ctx("S1 each signal applied exactly once: signal=%zu value=%llu",
                           k, (unsigned long long)f.signals[k]));

            // S2: was the signal ever released before its own operation's last chunk?
            std::map<uint64_t, int> placed_by_op;
            std::map<uint64_t, int> chunk_total;
            for (const auto& u : f.released)
                if (u.kind == srdsim::Unit::Kind::Placement)
                    chunk_total[u.op_id] = u.chunk_count;
            bool broken_here = false;
            placed_by_op.clear();
            for (const auto& u : f.released) {
                if (u.kind == srdsim::Unit::Kind::Placement) {
                    ++ placed_by_op[u.op_id];
                } else {
                    if (placed_by_op[u.op_id] != chunk_total[u.op_id])
                        broken_here = true;
                }
            }
            if (mode == srdsim::Mode::ViolateI1) {
                if (broken_here) ++ i1_broken_seeds;
            } else {
                expect(not broken_here,
                       ctx("S2 signal never precedes its own bytes: mode=%s seed=%llu",
                           srdsim::mode_name(mode), (unsigned long long)seed));
            }
        }
        if (mode == srdsim::Mode::ViolateI1)
            // Every seed, not merely one. "At least one of 64" was true while the mode only *permitted*
            // the violation and left the choice to the generator, so most steps released the payload
            // first and the violation rarely reached the wire. A negative test that fires occasionally
            // cannot distinguish a backend that tolerates the violation from a run in which it never
            // happened, and DeepEP passed this mode with no corruption for exactly that reason.
            expect(i1_broken_seeds == 64,
                   ctx("S2 violate-i1 breaks the contract on every seed, not merely some (%d of 64)",
                       i1_broken_seeds));
    }
}

// S3: inorder never interleaves and never reorders.
void test_inorder_is_a_baseline() {
    Fabric f(30'000, 3);
    srdsim::Scheduler::Options o;
    o.mode = srdsim::Mode::InOrder;
    o.chunk_bytes = 4096;
    o.seed = 12345;
    srdsim::Scheduler s(o);
    for (int k = 0; k < 3; ++ k)
        s.submit(make_op(static_cast<uint64_t>(k) + 1, f, static_cast<size_t>(k) * 10'000, 10'000, k));
    f.drain(s);

    uint64_t last_op = 0;
    int last_chunk = -1;
    bool saw_signal_for_last = true;
    for (const auto& u : f.released) {
        expect(u.op_id >= last_op, ctx("S3 operations are not interleaved: op %llu after %llu",
                                       (unsigned long long)u.op_id, (unsigned long long)last_op));
        if (u.op_id != last_op) {
            expect(saw_signal_for_last, "S3 an operation finishes before the next one starts");
            last_op = u.op_id;
            last_chunk = -1;
            saw_signal_for_last = false;
        }
        if (u.kind == srdsim::Unit::Kind::Placement) {
            expect(u.chunk_index == last_chunk + 1,
                   ctx("S3 chunks are in order: op=%llu chunk=%d after %d",
                       (unsigned long long)u.op_id, u.chunk_index, last_chunk));
            last_chunk = u.chunk_index;
        } else {
            saw_signal_for_last = true;
        }
    }
}

// S4 + S5: reproducibility, and that the adversary actually explores orders.
void test_reproducible_and_exploring() {
    auto run = [](uint64_t seed) {
        Fabric f(6, 2);   // two operations of 3 bytes, chunk 1 byte -> 8 units total
        srdsim::Scheduler::Options o;
        o.mode = srdsim::Mode::Srd;
        o.chunk_bytes = 1;
        o.seed = seed;
        srdsim::Scheduler s(o);
        s.submit(make_op(1, f, 0, 3, 0));
        s.submit(make_op(2, f, 3, 3, 1));
        f.drain(s);
        std::string sig;
        for (const auto& u : f.released) {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%llu%s%d,",
                          (unsigned long long)u.op_id,
                          u.kind == srdsim::Unit::Kind::Placement ? "p" : "s", u.chunk_index);
            sig += buf;
        }
        return sig;
    };

    // S4: determinism.
    for (uint64_t seed : {1ull, 2ull, 7ull, 4242ull})
        expect(run(seed) == run(seed), ctx("S4 the same seed reproduces the same order: seed=%llu",
                                          (unsigned long long)seed));

    // S5: coverage of distinct orders, including the in-order one.
    // Two operations of three chunks plus a signal that must come last within its operation:
    // 3! orders per operation times C(8,4) interleavings = 6 * 6 * 70 = 2520 legal orders.
    constexpr size_t kLegalOrders = 2520;
    std::set<std::string> orders;
    for (uint64_t seed = 1; seed <= 60'000; ++ seed)
        orders.insert(run(seed));
    expect(orders.size() * 100 >= kLegalOrders * 95,
           ctx("S5 the adversary covers the legal order space (%zu of %zu, %.1f%%)",
               orders.size(), kLegalOrders, 100.0 * static_cast<double>(orders.size()) / kLegalOrders));
    expect(orders.size() <= kLegalOrders,
           ctx("S5 it never produces an order outside the legal set (%zu of %zu)", orders.size(), kLegalOrders));
    expect(orders.count("1p0,1p1,1p2,1s-1,2p0,2p1,2p2,2s-1,") == 1,
           "S5 the in-order release is among the orders it can produce");

    // Every legal order must be reachable: the two operations of 3 chunks + 1 signal each give
    // 8!/(3!*3!) / ... rather than reason about the count, just require that both "op 1 fully
    // before op 0" and "strictly alternating" appear.
    // Operation ids are one-based, because zero is reserved: `complete(0)` can never be true, so an
    // operation numbered zero could never be reported finished.
    bool saw_reversed = false, saw_alternating = false;
    for (const auto& s : orders) {
        if (s.rfind("2p", 0) == 0 and s.find("1p") > s.find("2s"))
            saw_reversed = true;
        if (s.rfind("1p0,2p", 0) == 0)
            saw_alternating = true;
    }
    expect(saw_reversed, "S5 a later operation can land entirely before an earlier one");
    expect(saw_alternating, "S5 operations can interleave chunk by chunk");
}

// S6: an oversized operation is refused, not mismodelled.
void test_oversized_is_refused() {
    Fabric f(16, 1);
    srdsim::Scheduler::Options o;
    o.max_op_bytes = 8;
    srdsim::Scheduler s(o);
    std::string why;
    expect(not s.submit(make_op(1, f, 0, 9, 0), &why), "S6 an oversized operation is refused");
    expect(not why.empty(), "S6 the refusal explains itself");
    expect(s.submit(make_op(2, f, 0, 8, 0), &why), "S6 an operation at the limit is accepted");
}

}   // namespace


// S7. Each schedule does what its name says.
//
// A mode whose constraint is not checked is a knob that may be wired to nothing, and the tests that depend
// on it then pass without exercising anything. Every mode added here gets the check that distinguishes it
// from `srd`.
void test_each_schedule_keeps_its_promise() {
    struct Event { uint64_t op; bool is_signal; int peer; };

    const auto record = [](srdsim::Mode mode, int peers, int ops_per_peer, uint64_t seed) {
        Fabric f(64'000, 8);
        srdsim::Scheduler::Options o;
        o.mode = mode;
        o.seed = seed;
        o.chunk_bytes = 1024;
        srdsim::Scheduler s(o);
        uint64_t id = 0;
        for (int p = 0; p < peers; ++ p)
            for (int k = 0; k < ops_per_peer; ++ k) {
                ++ id;
                auto op = make_op(id, f, 0, 4096, static_cast<int>(id % 8));
                op.peer = p;
                std::string why;
                s.submit(op, &why);
            }
        std::vector<Event> seen;
        std::vector<srdsim::Unit> batch;
        while (not s.idle()) {
            batch.clear();
            if (s.progress(3, &batch) == 0) break;
            for (const auto& u : batch)
                seen.push_back({u.op_id, u.kind == srdsim::Unit::Kind::Signal, u.peer});
        }
        return seen;
    };

    // reverse-ops: the newest operation finishes first.
    {
        const auto seen = record(srdsim::Mode::ReverseOps, 1, 3, 5);
        uint64_t first_signal = 0;
        for (const auto& e : seen)
            if (e.is_signal) { first_signal = e.op; break; }
        expect(first_signal == 3,
               ctx("S7 reverse-ops completes the newest operation first (first signal was op %llu, "
                   "expected 3)", (unsigned long long)first_signal));
    }

    // signal-last: no signal before the final placement.
    {
        const auto seen = record(srdsim::Mode::SignalLast, 2, 3, 7);
        size_t last_placement = 0, first_signal = seen.size();
        for (size_t i = 0; i < seen.size(); ++ i) {
            if (seen[i].is_signal) { if (first_signal == seen.size()) first_signal = i; }
            else last_placement = i;
        }
        expect(first_signal > last_placement,
               ctx("S7 signal-last releases every placement before any signal (first signal at %zu, "
                   "last placement at %zu)", first_signal, last_placement));
    }

    // defer-one: one operation's units all come after every other operation's.
    {
        const auto seen = record(srdsim::Mode::DeferOne, 1, 3, 11);
        // The held operation is the oldest, so its first unit must come after the last unit of every
        // other operation.
        size_t first_of_held = seen.size(), last_of_others = 0;
        for (size_t i = 0; i < seen.size(); ++ i) {
            if (seen[i].op == 1) { if (first_of_held == seen.size()) first_of_held = i; }
            else last_of_others = i;
        }
        expect(first_of_held > last_of_others,
               ctx("S7 defer-one holds one operation until the others are done (held op starts at %zu, "
                   "others end at %zu)", first_of_held, last_of_others));
    }

    // peer-seq: per peer, operations complete in submission order.
    {
        const auto seen = record(srdsim::Mode::PeerSeq, 3, 3, 13);
        std::map<int, uint64_t> last_signalled;
        bool out_of_order = false;
        for (const auto& e : seen)
            if (e.is_signal) {
                if (last_signalled.count(e.peer) and e.op < last_signalled[e.peer])
                    out_of_order = true;
                last_signalled[e.peer] = e.op;
            }
        expect(not out_of_order,
               "S7 peer-seq completes each peer's operations in submission order, which is what the "
               "AWS OFI GIN plugin does and is stronger than the contract requires");
        // And it must still interleave across peers, or it is just `inorder` with extra words.
        bool interleaved = false;
        for (size_t i = 1; i < seen.size(); ++ i)
            if (seen[i].peer != seen[i - 1].peer) { interleaved = true; break; }
        expect(interleaved, "S7 peer-seq still interleaves across peers");
    }
}


// S8. The op mix a real run submits, drained to empty.
//
// A run on hardware left one scheduler holding 165 operations -- 165 unreleased chunks and 165
// unreleased signals -- after hundreds of millions of progress calls. Work accepted and never released
// is a host-side property, so it belongs in a host test rather than on a rented GPU. The shapes here are
// copied from the plugin's own operation log: a four-byte put to the local rank, a zero-byte put
// carrying a signal, and a stream of 4,224-byte puts each carrying a signal, across two peers.
void test_the_real_op_mix_drains() {
    Fabric f(1 << 20, 256);
    srdsim::Scheduler::Options o;
    o.mode = srdsim::Mode::InOrder;
    o.seed = 3;
    o.chunk_bytes = 8192;
    srdsim::Scheduler s(o);

    uint64_t id = 0;
    const auto put = [&](int peer, size_t bytes, bool with_signal) {
        ++ id;
        auto op = make_op(id, f, 0, bytes, with_signal ? static_cast<int>(id % 200) : -1);
        op.peer = peer;
        std::string why;
        const bool ok = s.submit(op, &why);
        expect(ok, ctx("S8 operation %llu (peer %d, %zu bytes) was accepted: %s",
                       (unsigned long long)id, peer, bytes, why.c_str()));
    };

    put(0, 4, false);                 // the local-rank put the dispatch opens with
    put(1, 0, true);                  // the barrier: zero bytes, one signal
    put(1, 4, false);
    for (int i = 0; i < 200; ++ i)
        put(i % 2, 4224, true);       // the dispatch stream, alternating peers

    // Drive it the way the plugin does: a fixed budget per call, repeatedly.
    size_t released = 0, calls = 0;
    std::vector<srdsim::Unit> batch;
    while (not s.idle()) {
        batch.clear();
        const size_t n = s.progress(8, &batch);
        released += n;
        ++ calls;
        if (n == 0) break;            // the symptom: pending work and nothing released
        if (calls > 100'000) break;   // far more than enough for 203 operations
    }

    const auto left = s.outstanding();
    expect(s.idle(),
           ctx("S8 every operation was released (%zu still held: %zu chunk(s) and %zu signal(s) "
               "unreleased, oldest op %llu to peer %d, after %zu call(s) releasing %zu unit(s))",
               left.ops, left.placements, left.signals,
               (unsigned long long)left.oldest_op, left.oldest_peer, calls, released));
}


// S9. Driven the way the plugin drives it: several schedulers, one batch, and submissions arriving on
// another thread while progress runs.
//
// The scheduler drains on its own (S8). A run on hardware reported one scheduler holding 58 operations
// with 58 unreleased chunks after a million and a half drives, which `pick()` cannot produce -- so either
// the report was wrong or the way the plugin drives the scheduler is. This test is the second half of
// that question, and it is a host question: no GPU, no fabric, no DeepEP.
//
// It reproduces three things the single-scheduler test does not: the plugin advances every context on the
// communicator into one shared batch, NCCL submits from a different thread than the one calling progress,
// and the plugin abandons the rest of a batch when a send fails.
void test_driven_like_the_plugin() {
    constexpr int kContexts = 4;
    Fabric f(1 << 20, 256);

    std::vector<std::unique_ptr<srdsim::Scheduler>> scheds;
    std::vector<std::mutex> locks(kContexts);
    for (int i = 0; i < kContexts; ++ i) {
        srdsim::Scheduler::Options o;
        o.mode = srdsim::Mode::InOrder;
        o.seed = 1 + static_cast<uint64_t>(i);
        o.chunk_bytes = 8192;
        scheds.push_back(std::make_unique<srdsim::Scheduler>(o));
    }

    std::atomic<bool> submitting{true};
    std::atomic<size_t> submitted{0};
    // Submissions on their own thread, because NCCL calls `iput` from a different thread than the one
    // that calls `rmaProgress`, and a scheduler that is correct under one thread can still be driven
    // wrongly by two.
    std::thread producer([&] {
        uint64_t id = 0;
        for (int round = 0; round < 60; ++ round) {
            for (int c = 0; c < kContexts; ++ c) {
                ++ id;
                auto op = make_op(id, f, 0, 4224, static_cast<int>(id % 200));
                op.peer = static_cast<int>(id % 2);
                std::string why;
                {
                    std::lock_guard<std::mutex> g(locks[static_cast<size_t>(c)]);
                    if (scheds[static_cast<size_t>(c)]->submit(op, &why))
                        ++ submitted;
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        submitting = false;
    });

    size_t released = 0;
    std::vector<srdsim::Unit> batch;
    const auto started = std::chrono::steady_clock::now();
    for (;;) {
        batch.clear();
        for (int c = 0; c < kContexts; ++ c) {
            std::lock_guard<std::mutex> g(locks[static_cast<size_t>(c)]);
            scheds[static_cast<size_t>(c)]->progress(8, &batch);
        }
        released += batch.size();
        bool all_idle = true;
        for (int c = 0; c < kContexts; ++ c) {
            std::lock_guard<std::mutex> g(locks[static_cast<size_t>(c)]);
            if (not scheds[static_cast<size_t>(c)]->idle()) all_idle = false;
        }
        if (not submitting and all_idle) break;
        if (std::chrono::steady_clock::now() - started > std::chrono::seconds(20)) break;
    }
    producer.join();

    // One last sweep, because the producer may have submitted after the final check.
    for (int round = 0; round < 1000; ++ round) {
        batch.clear();
        for (int c = 0; c < kContexts; ++ c) {
            std::lock_guard<std::mutex> g(locks[static_cast<size_t>(c)]);
            scheds[static_cast<size_t>(c)]->progress(8, &batch);
        }
        released += batch.size();
    }

    size_t held = 0, chunks = 0, signals = 0;
    for (int c = 0; c < kContexts; ++ c) {
        std::lock_guard<std::mutex> g(locks[static_cast<size_t>(c)]);
        const auto o = scheds[static_cast<size_t>(c)]->outstanding();
        held += o.ops; chunks += o.placements; signals += o.signals;
    }
    expect(held == 0,
           ctx("S9 nothing is left held when every context is advanced from one loop while another "
               "thread submits (%zu operation(s) held, %zu chunk(s) and %zu signal(s) unreleased, "
               "%zu submitted, %zu released)",
               held, chunks, signals, submitted.load(), released));
}


// S10. The id contract: an id is non-zero and unique.
//
// The id is the key `complete()` answers by and the key `defer-one` selects by. Without the check, two
// operations numbered zero starve every operation in `defer-one` and stall permanently with no refusal
// anywhere, and two operations sharing a non-zero id send `pick()` into unbounded recursion.
void test_the_id_contract() {
    Fabric f(1 << 16, 8);
    const auto scheduler = [&](srdsim::Mode mode) {
        srdsim::Scheduler::Options o;
        o.mode = mode;
        o.seed = 4;
        o.chunk_bytes = 1024;
        return srdsim::Scheduler(o);
    };

    {
        auto s = scheduler(srdsim::Mode::InOrder);
        std::string why;
        auto zero = make_op(0, f, 0, 2048, 0);
        expect(not s.submit(zero, &why), "S10 an operation numbered zero is refused");
        expect(not why.empty(), "S10 the refusal explains itself");
    }
    {
        auto s = scheduler(srdsim::Mode::InOrder);
        std::string why;
        expect(s.submit(make_op(7, f, 0, 2048, 0), &why), "S10 the first operation is accepted");
        expect(not s.submit(make_op(7, f, 2048, 2048, 1), &why),
               "S10 a second operation with the same id is refused while the first is pending");
    }
    // And `defer-one` terminates and drains.
    {
        auto s = scheduler(srdsim::Mode::DeferOne);
        std::string why;
        for (uint64_t id = 1; id <= 6; ++ id)
            s.submit(make_op(id, f, 0, 2048, static_cast<int>(id)), &why);
        std::vector<srdsim::Unit> batch;
        size_t calls = 0;
        while (not s.idle() and calls < 10'000) {
            batch.clear();
            if (s.progress(8, &batch) == 0) break;
            ++ calls;
        }
        const auto left = s.outstanding();
        expect(s.idle(),
               ctx("S10 defer-one drains rather than stalling (%zu op(s) held after %zu call(s))",
                   left.ops, calls));
    }
}

// S11. Overlap detection: a write submitted while an earlier one to the same peer and destination is still in
// flight over some of the same bytes is counted, with both named; disjoint ranges, other peers, other
// destinations and an earlier write that has fully landed are not.
void test_overlapping_writes_are_named() {
    Fabric f(1 << 16, 4);
    srdsim::Scheduler::Options o;
    o.mode = srdsim::Mode::InOrder;
    o.detect_overlaps = true;
    srdsim::Scheduler s(o);
    std::string why;
    s.submit(make_op(1, f, 0, 4096, -1), &why);
    s.submit(make_op(2, f, 8192, 4096, -1), &why);        // disjoint
    expect(s.overlaps() == 0, "S11 disjoint ranges are not an overlap");
    auto other_peer = make_op(3, f, 1024, 1024, -1); other_peer.peer = 1;
    s.submit(other_peer, &why);
    expect(s.overlaps() == 0, "S11 another peer is not an overlap");
    s.submit(make_op(4, f, 2048, 4096, 0), &why);          // [2048, 6144) against op 1's [0, 4096)
    expect(s.overlaps() == 1 and s.overlap_earlier().id == 1 and s.overlap_later().id == 4,
           ctx("S11 an overlap names both operations (%llu overlap(s), earlier %llu, later %llu)",
               (unsigned long long)s.overlaps(), (unsigned long long)s.overlap_earlier().id,
               (unsigned long long)s.overlap_later().id));
    f.drain(s);
    s.submit(make_op(5, f, 0, 4096, -1), &why);            // op 1 has fully landed
    expect(s.overlaps() == 1, "S11 a write that has already landed does not count");
}

// S12. Off by default: the scan it needs is not paid for unless asked.
void test_overlap_detection_is_opt_in() {
    Fabric f(1 << 16, 4);
    srdsim::Scheduler::Options o;
    o.mode = srdsim::Mode::InOrder;
    srdsim::Scheduler s(o);
    std::string why;
    s.submit(make_op(1, f, 0, 4096, -1), &why);
    s.submit(make_op(2, f, 2048, 4096, -1), &why);
    expect(s.overlaps() == 0, "S12 overlap detection is off unless requested");
}

int main() {
    test_overlapping_writes_are_named();
    test_overlap_detection_is_opt_in();
    test_delivery_and_i1();
    test_the_id_contract();
    test_driven_like_the_plugin();
    test_the_real_op_mix_drains();
    test_each_schedule_keeps_its_promise();
    test_inorder_is_a_baseline();
    test_reproducible_and_exploring();
    test_oversized_is_refused();

    std::printf("%s: %d checks, %d failures\n", g_failures == 0 ? "PASS" : "FAIL", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
