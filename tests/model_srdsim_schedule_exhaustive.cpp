// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// Proves the scheduler's ordering contract by enumerating every release sequence it can produce, for
// configurations small enough to enumerate completely.
//
// WHY NOT MORE SEEDS
// `test_srdsim_schedule.cpp` samples: it fixes a seed, follows one path through the choice tree, and
// checks the contract along it. Over 60,000 seeds it happens to cover all 2,520 legal orders of one
// configuration, which is a pleasant accident and not a proof. The contract is a universal claim --
// *no* release sequence violates I1 -- and a universal claim is answered by enumeration or not at all.
//
// WHAT IS ENUMERATED
// The scheduler's only nondeterminism is which option `pick()` returns. So the state is
// (per-operation placed set, per-operation signal pending) and the transitions are the options `pick`
// would offer. Enumerating the reachable states and the edges between them covers every sequence the
// scheduler can produce for that configuration, for every seed that exists and every seed that does
// not.
//
// This re-implements the choice rule rather than driving the real `Scheduler`, for one reason worth
// stating plainly: a seeded PRNG cannot be asked for "all choices". The re-implementation is the part
// that could drift, so `model_matches_scheduler` cross-checks it against the real class -- every
// sequence the real scheduler produces from a seed must be one the model admits.
//
// PROPERTIES
//   P-I1   in `srd` and `inorder`, no reachable state releases an operation's signal while any of its
//          placements is unreleased. Reported as the number of states inspected.
//   P-I1N  in `violate-i1` at least one reachable sequence does break it, so the negative mode is a
//          real adversary rather than a mode that happens to behave.
//   P-ONCE every reachable terminal state has released each unit exactly once, and every enumerated
//          sequence has the same length as every other for that configuration.
//   P-DONE every path from the initial state reaches a terminal state: the scheduler cannot wedge.
//   P-ORD  in `inorder`, exactly one sequence exists, which is what "submission order" means.
//
// Build: c++ -std=c++17 -O2 -I layer2 tests/model_srdsim_schedule_exhaustive.cpp -o m

#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "srdsim_schedule.h"

namespace {

int g_failures = 0;

void fail(const std::string& what) {
    ++ g_failures;
    std::printf("  [FAIL] %s\n", what.c_str());
}

// One operation's shape: how many placement units, and whether it carries a signal.
struct OpShape {
    int chunks = 1;
    bool has_signal = true;
};

// The state of a run: for each operation, how many of its chunks are placed (a count is enough,
// because the chunks of one operation are interchangeable in the contract) and whether its signal is
// still pending.
struct State {
    std::vector<int> placed;
    std::vector<bool> signal_pending;

    bool operator<(const State& o) const {
        if (placed != o.placed) return placed < o.placed;
        return signal_pending < o.signal_pending;
    }
    bool operator==(const State& o) const {
        return placed == o.placed and signal_pending == o.signal_pending;
    }
};

// A move: which operation, and whether it releases a chunk or the signal.
struct Move {
    int op = 0;
    bool is_signal = false;
};

// The choice rule, mirroring `Scheduler::pick`. `inorder` never looks past the oldest operation with
// anything left; the other modes offer every pending unit.
std::vector<Move> options(const std::vector<OpShape>& shapes, const State& s, srdsim::Mode mode) {
    std::vector<Move> out;
    for (size_t i = 0; i < shapes.size(); ++ i) {
        const bool chunks_left = s.placed[i] < shapes[i].chunks;
        const bool signal_left = s.signal_pending[i];
        if (chunks_left) out.push_back({int(i), false});
        if (signal_left) {
            const bool releasable = (mode == srdsim::Mode::ViolateI1)
                                        ? true
                                        : s.placed[i] == shapes[i].chunks;
            if (releasable) out.push_back({int(i), true});
        }
        if (mode == srdsim::Mode::InOrder and not out.empty())
            break;
    }
    if (mode == srdsim::Mode::InOrder and out.size() > 1)
        out.resize(1);    // the real scheduler takes options.front()
    return out;
}

State apply(const std::vector<OpShape>&, const State& s, const Move& m) {
    State n = s;
    if (m.is_signal) n.signal_pending[size_t(m.op)] = false;
    else             ++ n.placed[size_t(m.op)];
    return n;
}

bool terminal(const std::vector<OpShape>& shapes, const State& s) {
    for (size_t i = 0; i < shapes.size(); ++ i) {
        if (s.placed[i] < shapes[i].chunks) return false;
        if (s.signal_pending[i]) return false;
    }
    return true;
}

struct Result {
    size_t states = 0;
    size_t edges = 0;
    size_t sequences = 0;          // distinct complete release sequences
    size_t i1_violating_states = 0;
    size_t terminal_states = 0;
    size_t dead_ends = 0;          // non-terminal states with no options: a wedge
    std::vector<size_t> sequence_lengths;
};

// Depth-first enumeration with memoised sequence counts. The state space for the configurations used
// here is small (thousands), so the memo is for the sequence count only; states and edges are counted
// by visiting.
Result enumerate(const std::vector<OpShape>& shapes, srdsim::Mode mode) {
    Result r;
    State init;
    init.placed.assign(shapes.size(), 0);
    init.signal_pending.resize(shapes.size());
    for (size_t i = 0; i < shapes.size(); ++ i) init.signal_pending[i] = shapes[i].has_signal;

    std::map<State, size_t> seq_count;      // number of complete sequences from this state
    std::set<State> seen;

    // Iterative post-order would be fiddly; the depth is bounded by the total unit count, which is
    // small here, so recursion through a lambda is clearer and safe.
    struct Walker {
        const std::vector<OpShape>& shapes;
        srdsim::Mode mode;
        Result& r;
        std::map<State, size_t>& seq_count;
        std::set<State>& seen;

        size_t visit(const State& s, size_t depth) {
            const auto memo = seq_count.find(s);
            if (memo != seq_count.end()) return memo->second;
            seen.insert(s);

            // I1 is a property of a *state*: a state in which a signal has been released while a
            // placement of the same operation has not.
            for (size_t i = 0; i < shapes.size(); ++ i) {
                const bool signal_released = shapes[i].has_signal and not s.signal_pending[i];
                if (signal_released and s.placed[i] < shapes[i].chunks) {
                    ++ r.i1_violating_states;
                    break;
                }
            }

            if (terminal(shapes, s)) {
                ++ r.terminal_states;
                r.sequence_lengths.push_back(depth);
                seq_count[s] = 1;
                return 1;
            }
            const auto opts = options(shapes, s, mode);
            if (opts.empty()) {
                ++ r.dead_ends;
                seq_count[s] = 0;
                return 0;
            }
            size_t total = 0;
            for (const Move& m : opts) {
                ++ r.edges;
                total += visit(apply(shapes, s, m), depth + 1);
            }
            seq_count[s] = total;
            return total;
        }
    } walker{shapes, mode, r, seq_count, seen};

    r.sequences = walker.visit(init, 0);
    r.states = seen.size();
    return r;
}

// Every sequence the real scheduler produces must be one this model admits. Without this the model
// could be self-consistent and wrong about the code.
void model_matches_scheduler(const std::vector<OpShape>& shapes, srdsim::Mode mode, size_t chunk_bytes) {
    for (uint64_t seed = 1; seed <= 200; ++ seed) {
        srdsim::Scheduler::Options o;
        o.mode = mode;
        o.seed = seed;
        o.chunk_bytes = chunk_bytes;
        srdsim::Scheduler sched(o);

        std::vector<char> src(chunk_bytes * 16, 0);
        std::vector<uint64_t> sig(shapes.size(), 0);
        for (size_t i = 0; i < shapes.size(); ++ i) {
            srdsim::Op op;
            op.id = i + 1;
            op.src_base = src.data();
            op.dst_base = src.data();
            op.bytes = chunk_bytes * size_t(shapes[i].chunks);
            op.has_signal = shapes[i].has_signal;
            op.signal_base = sig.data();
            op.signal_off = i * sizeof(uint64_t);
            std::string why;
            if (not sched.submit(op, &why)) { fail("the scheduler refused a modelled operation: " + why); return; }
        }

        State s;
        s.placed.assign(shapes.size(), 0);
        s.signal_pending.resize(shapes.size());
        for (size_t i = 0; i < shapes.size(); ++ i) s.signal_pending[i] = shapes[i].has_signal;

        std::vector<srdsim::Unit> batch;
        while (not sched.idle()) {
            batch.clear();
            if (sched.progress(1, &batch) == 0) break;
            for (const auto& u : batch) {
                const int op_index = int(u.op_id) - 1;
                const Move m{op_index, u.kind == srdsim::Unit::Kind::Signal};
                const auto opts = options(shapes, s, mode);
                const bool admitted = std::any_of(opts.begin(), opts.end(), [&](const Move& o2) {
                    return o2.op == m.op and o2.is_signal == m.is_signal;
                });
                if (not admitted) {
                    char b[220];
                    std::snprintf(b, sizeof(b),
                                  "the scheduler released op %d %s in mode %s (seed %llu) from a state "
                                  "the model does not offer it in",
                                  m.op, m.is_signal ? "signal" : "chunk", srdsim::mode_name(mode),
                                  (unsigned long long)seed);
                    fail(b);
                    return;
                }
                s = apply(shapes, s, m);
            }
        }
        if (not terminal(shapes, s)) {
            fail(std::string("the real scheduler stopped short of a terminal state in mode ") +
                 srdsim::mode_name(mode));
            return;
        }
    }
}

std::string shape_name(const std::vector<OpShape>& shapes) {
    std::string s;
    for (const auto& o : shapes) {
        char b[32];
        std::snprintf(b, sizeof(b), "%dc%s ", o.chunks, o.has_signal ? "+s" : "");
        s += b;
    }
    if (not s.empty()) s.pop_back();
    return s;
}

}   // namespace

int main() {
    // Configurations chosen to be enumerable and to cover the shapes the contract talks about: one
    // operation, several operations, operations of unequal size, and an operation without a signal.
    const std::vector<std::vector<OpShape>> configs = {
        {{1, true}},
        {{3, true}},
        {{2, true}, {2, true}},
        {{1, true}, {3, true}},
        {{2, true}, {2, false}},
        {{2, true}, {1, true}, {1, true}},
    };

    size_t total_states = 0, total_sequences = 0;
    std::printf("%-22s %-11s %8s %8s %10s %8s\n", "configuration", "mode", "states", "edges",
                "sequences", "len");
    for (const auto& shapes : configs) {
        for (const auto mode : {srdsim::Mode::InOrder, srdsim::Mode::Srd, srdsim::Mode::ViolateI1}) {
            const Result r = enumerate(shapes, mode);
            total_states += r.states;
            total_sequences += r.sequences;

            size_t len = r.sequence_lengths.empty() ? 0 : r.sequence_lengths.front();
            std::printf("%-22s %-11s %8zu %8zu %10zu %8zu\n", shape_name(shapes).c_str(),
                        srdsim::mode_name(mode), r.states, r.edges, r.sequences, len);

            // P-DONE: no reachable non-terminal state has nothing to offer.
            if (r.dead_ends != 0)
                fail(shape_name(shapes) + " " + srdsim::mode_name(mode) + ": the scheduler can wedge");

            // P-ONCE: every complete sequence has the same length, which is the total unit count.
            size_t units = 0;
            for (const auto& o : shapes) units += size_t(o.chunks) + (o.has_signal ? 1u : 0u);
            for (const size_t l : r.sequence_lengths)
                if (l != units) {
                    fail(shape_name(shapes) + " " + srdsim::mode_name(mode) +
                         ": a sequence released a different number of units than the configuration has");
                    break;
                }

            // P-I1 and P-I1N.
            if (mode == srdsim::Mode::ViolateI1) {
                if (r.i1_violating_states == 0)
                    fail(shape_name(shapes) +
                         " violate-i1: no reachable state breaks I1, so the negative mode proves nothing");
            } else {
                if (r.i1_violating_states != 0)
                    fail(shape_name(shapes) + " " + srdsim::mode_name(mode) + ": " +
                         std::to_string(r.i1_violating_states) + " reachable state(s) release a signal "
                         "before the operation's placements");
            }

            // P-ORD: submission order means exactly one sequence.
            if (mode == srdsim::Mode::InOrder and r.sequences != 1)
                fail(shape_name(shapes) + " inorder: " + std::to_string(r.sequences) +
                     " sequences exist, but submission order admits one");

            model_matches_scheduler(shapes, mode, /*chunk_bytes=*/64);
        }
    }

    std::printf("\n%zu states enumerated across %zu configurations x 3 modes, %zu complete release "
                "sequences\n", total_states, configs.size(), total_sequences);
    if (g_failures != 0) {
        std::printf("FAIL: %d property violation(s)\n", g_failures);
        return 1;
    }
    std::printf("PASS: every reachable state satisfies the ordering contract\n");
    return 0;
}
