// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// The send side's queue for units whose device source is still being staged.
//
// When the plugin delivers through CUDA copies from its own context, waiting for a copy costs a time slice
// while the application's kernels are resident (2.4 ms on H100), and submitting one costs microseconds. So a
// unit whose source is on the device is not sent when it is released: its device-to-host copy is submitted,
// and the unit waits here until that copy has landed. Everything else released for the same peer waits
// behind it, so the wire order for each peer stays the order the scheduler released -- which is what keeps
// a signal behind its own placements, and what keeps `violate-i1` violating in exactly the way it chose.
//
// Operation completion has to see this queue too. An operation the scheduler has released entirely is not
// finished while one of its units is still here: its source has not been read yet, and reporting it done
// would let NCCL reuse that source.
//
// No CUDA here. Readiness and sending are supplied by the caller, so the ordering and the accounting can be
// tested on a laptop.

#ifndef SRDSIM_ASYNC_H_
#define SRDSIM_ASYNC_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <utility>

namespace srdsim {

template <typename Unit>
class SendQueue {
public:
    // No device source to stage.
    static constexpr int kNoSlot = -1;
    // A device source waiting for a free staging slot. It holds its place in the peer's order meanwhile.
    static constexpr int kUnstaged = -2;

    struct Entry {
        Unit unit;
        const void* lane = nullptr;    // the connection the unit travels on; peer numbers are per communicator
        const void* owner = nullptr;   // the context the operation belongs to; op ids are per context
        uint64_t op_id = 0;
        int slot = kNoSlot;            // the staging slot holding the unit's bytes, or one of the two above
        bool failed = false;           // its staging copy failed; it is removed and reported, not sent
    };

    void push(Entry e) {
        ++ outstanding_[{e.owner, e.op_id}];
        ++ size_;
        peers_[key_of(e)].push_back(std::move(e));
    }

    bool empty() const { return size_ == 0; }
    size_t size() const { return size_; }

    bool empty_for(const void* lane, int peer) const {
        auto it = peers_.find({lane, peer});
        return it == peers_.end() or it->second.empty();
    }

    // Every queued entry, each peer's in order, for the caller to assign staging slots to the unstaged
    // ones. `f` returns false to stop the whole walk (no slot left).
    template <typename F>
    void for_each(F f) {
        for (auto& kv : peers_)
            for (auto& e : kv.second)
                if (not f(e)) return;
    }

    // Removes every entry of `owner`, handing each to `drop` first, for a context that is going away with
    // units it could not send. Returns how many were removed.
    template <typename Drop>
    size_t remove_owner(const void* owner, Drop drop) {
        size_t removed = 0;
        for (auto& kv : peers_) {
            auto& q = kv.second;
            for (auto it = q.begin(); it != q.end();) {
                if (it->owner != owner) { ++ it; continue; }
                drop(*it);
                forget(*it);
                it = q.erase(it);
                ++ removed;
            }
        }
        return removed;
    }

    size_t depth() const { return size_; }

    // True while any unit of the operation is still queued.
    bool holds(const void* owner, uint64_t op_id) const {
        auto it = outstanding_.find({owner, op_id});
        return it != outstanding_.end() and it->second != 0;
    }

    bool holds_any_of(const void* owner) const {
        for (const auto& kv : outstanding_)
            if (kv.first.first == owner and kv.second != 0) return true;
        return false;
    }

    // Sends, for every peer, each head whose `ready(entry)` is true, stopping that peer at the first head
    // that is not. `send(entry)` writes the unit and returns false if the link failed; a failed unit is
    // still removed, because leaving it at the head would stop the peer's queue for good, and the caller
    // records the failure. Returns the number of units removed.
    template <typename Ready, typename Send>
    size_t drain(Ready ready, Send send) {
        size_t removed = 0;
        for (auto& kv : peers_) {
            auto& q = kv.second;
            while (not q.empty() and ready(q.front())) {
                Entry e = std::move(q.front());
                q.pop_front();
                send(e);
                forget(e);
                ++ removed;
            }
        }
        return removed;
    }

private:
    using Key = std::pair<const void*, int>;
    static Key key_of(const Entry& e) { return {e.lane, e.unit.peer}; }

    void forget(const Entry& e) {
        auto it = outstanding_.find({e.owner, e.op_id});
        if (it != outstanding_.end() and -- it->second == 0) outstanding_.erase(it);
        -- size_;
    }

    std::map<Key, std::deque<Entry>> peers_;
    std::map<std::pair<const void*, uint64_t>, uint32_t> outstanding_;
    size_t size_ = 0;
};

}  // namespace srdsim

#endif  // SRDSIM_ASYNC_H_
