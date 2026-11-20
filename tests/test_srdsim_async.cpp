// SPDX-License-Identifier: Apache-2.0

// The send queue's two promises: per peer, units leave in the order they were released, whatever becomes
// ready first; and an operation is held until every one of its units has left.
//
//   Q1  a unit that is not ready holds every later unit to the same peer, including a signal
//   Q2  it does not hold another peer's units
//   Q3  an operation is held while any of its units is queued, and released when the last one leaves
//   Q4  a failed send still removes the unit, so one bad link cannot stop the queue for good
//   Q5  a randomised schedule of readiness never lets a signal leave before a placement released ahead of it
//   Q6  peer 1 of one communicator is not peer 1 of another: their units do not wait on each other
//   Q7  a context going away takes exactly its own units with it, and releases their operations

#include "srdsim_async.h"

#include <cstdio>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

int g_checks = 0, g_failures = 0;
void expect(bool ok, const std::string& what) {
    ++ g_checks;
    if (not ok) { ++ g_failures; std::printf("  FAIL %s\n", what.c_str()); }
}

struct Unit { int peer = 0; bool is_signal = false; int id = 0; };
using Queue = srdsim::SendQueue<Unit>;

Queue::Entry entry(int peer, bool is_signal, int id, uint64_t op, const void* owner = nullptr,
                   const void* lane = nullptr) {
    Queue::Entry e;
    e.unit = Unit{peer, is_signal, id};
    e.owner = owner;
    e.lane = lane;
    e.op_id = op;
    e.slot = is_signal ? -1 : id;
    return e;
}

void test_order_and_isolation() {
    Queue q;
    std::set<int> ready;
    std::vector<int> sent;
    q.push(entry(0, false, 1, 10));
    q.push(entry(0, false, 2, 10));
    q.push(entry(0, true, 3, 10));
    q.push(entry(1, false, 4, 11));
    ready = {2, 3, 4};   // unit 1 not staged yet
    const auto is_ready = [&](const Queue::Entry& e) { return e.unit.is_signal or ready.count(e.unit.id); };
    const auto send = [&](const Queue::Entry& e) { sent.push_back(e.unit.id); return true; };
    q.drain(is_ready, send);
    expect(sent == std::vector<int>{4}, "Q1/Q2 peer 0 is held behind unit 1, peer 1 is not");
    expect(q.holds(nullptr, 10), "Q3 operation 10 is held while its units are queued");
    expect(not q.holds(nullptr, 11), "Q3 operation 11 is released once its only unit left");
    ready.insert(1);
    q.drain(is_ready, send);
    expect(sent == std::vector<int>{4, 1, 2, 3}, "Q1 peer 0's units leave in release order, the signal last");
    expect(not q.holds(nullptr, 10) and q.empty(), "Q3 operation 10 is released when its last unit leaves");
}

void test_failed_send_is_removed() {
    Queue q;
    q.push(entry(0, false, 1, 1));
    q.push(entry(0, true, 2, 1));
    size_t attempts = 0;
    q.drain([](const Queue::Entry&) { return true; }, [&](const Queue::Entry&) { ++ attempts; return false; });
    expect(attempts == 2 and q.empty(), "Q4 failed sends are removed and the queue does not stop");
}

void test_owners_are_separate() {
    Queue q;
    int a = 0, b = 0;
    q.push(entry(0, false, 1, 7, &a));
    expect(q.holds(&a, 7) and not q.holds(&b, 7), "Q3 the same op id in another context is not held");
    expect(q.holds_any_of(&a) and not q.holds_any_of(&b), "Q3 a flush of one context sees only its own units");
}

void test_randomised_signals_never_overtake() {
    std::mt19937 rng(1234);
    for (int trial = 0; trial < 2000; ++ trial) {
        Queue q;
        std::vector<Queue::Entry> released;
        const int peers = 1 + int(rng() % 3), n = 2 + int(rng() % 12);
        for (int i = 0; i < n; ++ i) {
            const int peer = int(rng() % unsigned(peers));
            released.push_back(entry(peer, rng() % 3 == 0, i, uint64_t(i / 3)));
            q.push(released.back());
        }
        std::set<int> ready;
        std::vector<Queue::Entry> sent;
        while (not q.empty()) {
            ready.insert(int(rng() % unsigned(n)));
            q.drain([&](const Queue::Entry& e) { return e.unit.is_signal or ready.count(e.unit.id); },
                    [&](const Queue::Entry& e) { sent.push_back(e); return true; });
        }
        // Per peer, the order sent is the order released.
        bool ok = true;
        for (int p = 0; p < peers; ++ p) {
            std::vector<int> r, s;
            for (const auto& e : released) if (e.unit.peer == p) r.push_back(e.unit.id);
            for (const auto& e : sent) if (e.unit.peer == p) s.push_back(e.unit.id);
            ok = ok and r == s;
        }
        expect(ok, "Q5 trial " + std::to_string(trial) + ": per-peer order is release order");
    }
}

void test_lanes_are_separate() {
    Queue q;
    int comm_a = 0, comm_b = 0;
    q.push(entry(1, false, 1, 1, nullptr, &comm_a));   // not ready
    q.push(entry(1, false, 2, 2, nullptr, &comm_b));   // ready
    std::vector<int> sent;
    q.drain([](const Queue::Entry& e) { return e.unit.id == 2; },
            [&](const Queue::Entry& e) { sent.push_back(e.unit.id); return true; });
    expect(sent == std::vector<int>{2}, "Q6 another communicator's peer 1 is not held behind this one's");
    expect(not q.empty_for(&comm_a, 1) and q.empty_for(&comm_b, 1), "Q6 emptiness is per connection");
}

void test_remove_owner() {
    Queue q;
    int a = 0, b = 0;
    q.push(entry(0, false, 1, 5, &a));
    q.push(entry(0, false, 2, 6, &b));
    q.push(entry(0, true, 3, 5, &a));
    std::vector<int> dropped;
    const size_t n = q.remove_owner(&a, [&](const Queue::Entry& e) { dropped.push_back(e.unit.id); });
    expect(n == 2 and dropped == std::vector<int>{1, 3}, "Q7 exactly the owner's units are removed, in order");
    expect(not q.holds(&a, 5) and q.holds(&b, 6) and q.depth() == 1, "Q7 the other context's unit stays");
}

}  // namespace

int main() {
    test_order_and_isolation();
    test_failed_send_is_removed();
    test_owners_are_separate();
    test_randomised_signals_never_overtake();
    test_lanes_are_separate();
    test_remove_owner();
    std::printf("%s: %d checks, %d failures\n", g_failures == 0 ? "PASS" : "FAIL", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
