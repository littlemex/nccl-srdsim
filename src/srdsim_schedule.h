// SPDX-License-Identifier: Apache-2.0

// The delivery scheduler of the SRD simulator: the part that decides, for a stream of
// submitted RMA operations, in what order the bytes land at the target and when each
// operation's signal becomes observable.
//
// It is deliberately free of NCCL, CUDA and sockets so that it can be unit tested on its own
// (`tests/test_srdsim_schedule.cpp`). A scheduler bug would manufacture failures that no real
// fabric can produce, which is worse than having no simulator, so this file is the one that
// has to be right before anything else is believed.
//
// MODEL
// An operation is split into `ceil(bytes / chunk_bytes)` placement units plus, if it carries a
// signal, one signal unit. Placement units of one operation are mutually unordered and
// operations are mutually unordered: that is what SRD not preserving order means, at the packet
// and at the message granularity respectively.
//
// The one ordering constraint a real EFA-GDA backend does guarantee is that an operation's
// signal is observable only once every byte of that operation is placed. `Mode::Srd` enforces
// it. `Mode::ViolateI1` deliberately breaks it, which is a backend conformance negative test,
// not a test of DeepEP.
//
// GET
// A get operation (`Op::is_get_request`) has no local source -- the whole point of a get is that
// the bytes are not here yet -- so it is not chunked by `bytes`: it releases exactly one
// `Unit::Kind::GetRequest`, a small control message naming (peer, window, offset, length) to read
// and where the answer belongs. The caller turns that into a real reply once it arrives at the
// peer, by submitting an ordinary placement op (`Op::is_get_response`) back on *its own*
// scheduler -- so the reply is chunked and reordered exactly as any other placement is, which is
// where the bytes-level behaviour this simulator exists to reproduce actually lives. A get
// request's own local completion, like a put's, means only "handed to the wire"; nothing here
// models the fence a real NIC needs between posting a read and the result being visible, because
// DeepEP's own contract for this is `iflush`/`flushAsync`, not the get's own request.
//
// CALIBRATION RULE
// Do whatever SRD does not promise; keep whatever it does. Widening the adversary past that
// line produces false positives; narrowing it hides the window this simulator exists to open.

#ifndef SRDSIM_SCHEDULE_H_
#define SRDSIM_SCHEDULE_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <cstdio>
#include <string>
#include <vector>

namespace srdsim {

// The schedules. `Srd` is the default and the widest one that stays inside the contract; the four
// after it are deterministic shapes, added because a random schedule cannot answer "which reordering
// breaks it". A random run that fails tells you only that some interleaving fails.
//
// All of them except `ViolateI1` keep the one guarantee EFA gives: a bundled signal is released only
// after every byte of its own operation. NCCL states that as the weak-signal contract, and DeepEP's
// unordered kernels say they need nothing more.
enum class Mode {
    InOrder,     // submission order; the sanity baseline. If a test fails here, the test is wrong.
    Srd,         // reorder placements and operations; keep signal-after-placement. The default.
    ViolateI1,   // may release a signal before the operation's remaining chunks
    // Complete logical operations, newest first. Legal -- independent operations may complete in
    // either order -- and the simplest schedule that is not submission order.
    ReverseOps,
    // Every placement of every pending operation, then the signals. Legal for the same reason, and it
    // separates payload from signal as far as the contract allows, which is where a reader that peeks
    // at a counter before its data would fail.
    SignalLast,
    // One operation held back until every other pending operation has been released in full. The
    // shape a fabric produces when a single path is slow, and the one a receiver that assumes rough
    // progress across sources would fail on.
    DeferOne,
    // Per peer, strictly in submission order; chunks within an operation may still be reordered, and
    // peers are independent. This is what the AWS OFI GIN plugin actually does -- it retires received
    // operations in per-peer sequence order (`nccl_ofi_gin.h`, `rx_consumed`) -- so it is *stronger*
    // than the documented contract. Having it separates "DeepEP relies on more than the contract
    // gives" from "DeepEP relies on more than AWS's plugin gives", and only the second would be a
    // defect this emulator should be reproducing.
    PeerSeq,
};

inline const char* mode_name(Mode m) {
    switch (m) {
        case Mode::InOrder:   return "inorder";
        case Mode::Srd:       return "srd";
        case Mode::ViolateI1: return "violate-i1";
        case Mode::ReverseOps: return "reverse-ops";
        case Mode::SignalLast: return "signal-last";
        case Mode::DeferOne:  return "defer-one";
        case Mode::PeerSeq:   return "peer-seq";
    }
    return "?";
}

inline bool mode_from_string(const std::string& s, Mode* out) {
    if (s == "inorder")    { *out = Mode::InOrder;   return true; }
    if (s == "srd")        { *out = Mode::Srd;       return true; }
    if (s == "violate-i1") { *out = Mode::ViolateI1; return true; }
    if (s == "reverse-ops") { *out = Mode::ReverseOps; return true; }
    if (s == "signal-last") { *out = Mode::SignalLast; return true; }
    if (s == "defer-one")  { *out = Mode::DeferOne;  return true; }
    if (s == "peer-seq")   { *out = Mode::PeerSeq;   return true; }
    return false;
}

// One submitted operation, in the terms the RMA plugin receives it.
struct Op {
    uint64_t id = 0;
    int      peer = 0;
    // The RMA context index within the plugin context (`ncclRmaConfig::nContexts`). Ordering is scoped to a
    // context and a peer: nothing orders operations on different contexts, so a strong signal waits only for
    // earlier operations on its own context to its own peer.
    int      context = 0;
    // Placement: `bytes` from (src_base + src_off) to (dst_base + dst_off) at the peer.
    void*    src_base = nullptr;
    uint64_t src_off = 0;
    // Whether `src_base` is device memory. A socket cannot read from a device pointer, so a device
    // source has to be staged through host memory before it can be written to the wire. The
    // scheduler only carries the flag; the staging belongs to whoever owns the CUDA calls.
    bool     src_is_device = false;
    // Two ways to name the destination, because there are two transports. A same-host mapped
    // window is addressed by pointer; a peer on another host is addressed by (window id, offset)
    // and the receiving side resolves the id to its own base. The scheduler only carries them.
    void*    dst_base = nullptr;      // mapped-peer path: resolved by the caller
    uint64_t dst_off = 0;
    uint32_t dst_window = 0;          // wire path: which registered window
    size_t   bytes = 0;
    // Signal: an add/inc applied to a uint64 at (signal_base + signal_off) at the peer.
    bool     has_signal = false;
    void*    signal_base = nullptr;   // mapped-peer path
    uint64_t signal_off = 0;
    uint32_t signal_window = 0;       // wire path
    uint64_t signal_val = 1;
    bool     signal_is_add = false;
    // A strong signal orders every earlier put and signal on this context and peer before itself, not
    // just its own payload. NCCL states the two forms separately (`plugins/rma/README.md`), and EFA has
    // no strong signal -- but refusing one is worse than implementing it: a non-success return from a
    // submit entry point makes NCCL's GIN progress thread exit permanently (`src/gin/gin_host.cc`), and
    // the consumed descriptor is never retried. So the scheduler enforces it instead.
    bool     signal_is_strong = false;   // false: increment by one
    bool     is_strong = false;       // what the caller asked for, recorded for diagnostics

    // Get: this op asks the peer to read ITS OWN memory and reply with it, rather than placing bytes
    // this rank already has. `bytes` is how much to fetch and `peer` is who owns it, same as a put.
    // `src_base`/`src_off`/`src_is_device` are unused (there is nothing local to read yet) and
    // `dst_window`/`dst_off` are unused (the peer resolves where to read, not where to write) -- the
    // four fields below replace them for this kind.
    bool     is_get_request = false;
    uint32_t get_src_window = 0;     // which window, on the peer, to read
    uint64_t get_src_off = 0;
    uint32_t get_dst_window = 0;     // which window, locally, the reply must land in
    uint64_t get_dst_off = 0;
    int      requester_rank = 0;     // this rank's own id, so the peer knows who to answer

    // Get: this op is the owner's reply to a GetRequest it received. To the scheduler it is an
    // ordinary placement -- same chunking, same reordering against other pending work to that peer --
    // only its wire kind differs, so the requester can tell an answer to its own get() apart from an
    // unrelated put and release the right `iflush`.
    bool     is_get_response = false;
    uint64_t get_response_token = 0;   // opaque; echoes what the GetRequest's wire frame carried
};

// A unit of work the caller must perform when the scheduler releases it.
struct Unit {
    enum class Kind { Placement, Signal, GetRequest } kind = Kind::Placement;
    uint64_t op_id = 0;
    int      peer = 0;
    // Placement units carry a byte range; signal units carry the target word.
    const void* src = nullptr;
    void*       dst = nullptr;
    size_t      bytes = 0;
    void*       signal_word = nullptr;
    uint64_t    signal_val = 0;
    bool        signal_is_add = false;
    int         chunk_index = 0;
    int         chunk_count = 0;
    // The same destination expressed for the wire transport. `window` names a registered window and
    // `window_offset` is the byte offset into it, so no pointer from the peer's address space is
    // needed and no null-pointer arithmetic is involved. For a GetRequest unit these two name the
    // window+offset *on the peer* to read, the same fields a placement uses for its destination.
    uint32_t    window = 0;
    uint64_t    window_offset = 0;
    bool        src_is_device = false;

    // Get: only meaningful when `kind == Placement` and this chunk answers an earlier GetRequest. It
    // must go out tagged GetResponse, carrying the requester's token instead of this op's own
    // locally-scoped id, so the requester can match it to the right outstanding get().
    bool     is_get_response = false;
    uint64_t get_response_token = 0;
    // Get: only meaningful when `kind == GetRequest`. `window`/`window_offset` above name where on the
    // peer to read; these name where, in this rank's own numbering, the reply must land, and which
    // rank is asking.
    uint32_t get_dst_window = 0;
    uint64_t get_dst_off = 0;
    uint32_t get_requester_rank = 0;
};

// Deterministic PRNG: the exact sequence has to be reproducible from a seed so that a
// counterexample can be replayed. Keeping it in the file rather than using <random> avoids
// depending on a standard-library implementation detail for reproducibility.
class Rng {
public:
    explicit Rng(uint64_t seed) : state_(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    uint64_t next() {
        state_ ^= state_ >> 12;
        state_ ^= state_ << 25;
        state_ ^= state_ >> 27;
        return state_ * 0x2545F4914F6CDD1Dull;
    }
    // Uniform in [0, n).
    size_t below(size_t n) { return n ? static_cast<size_t>(next() % n) : 0; }

private:
    uint64_t state_;
};

class Scheduler {
public:
    struct Options {
        Mode     mode = Mode::Srd;
        size_t   chunk_bytes = 8192;   // stand-in for the fabric's segmentation granularity
        uint64_t seed = 1;
        // Refuse operations larger than a single EFA RDMA write. EFA-GDA splits them; this
        // simulator does not model the split, so it detects the case instead of pretending.
        size_t   max_op_bytes = size_t{1} << 30;
        bool     trace = false;
        // Which operation `defer-one` holds: the oldest of any kind, or the oldest of one kind. The
        // kinds are what separates "a receiver that needs an earlier unsignalled put" from "one that
        // needs an earlier signal" when `defer-one` fails and `signal-last` passes.
        enum class DeferKind { Any, Unsignaled, Signaled, SignalOnly } defer_kind = DeferKind::Any;
        // Count submissions that overlap an earlier unreleased write to the same peer (`overlaps()`). Off by
        // default: it scans the pending operations on every submit.
        bool detect_overlaps = false;
    };

    explicit Scheduler(Options opt) : opt_(opt), rng_(opt.seed) {}

    // Returns false if the operation is outside what the simulator claims to model. The caller
    // must surface that as an error rather than proceeding.
    bool submit(const Op& op, std::string* why = nullptr) {
        // Refused here rather than in the constructor. A constructor that cannot fail is easier to use
        // correctly than one that can, and the caller is already handling refusals at this point.
        // Without this the chunk arithmetic below divides by zero, which every direct caller can reach
        // and which the environment path happens to hide by clamping zero to the default.
        if (opt_.chunk_bytes == 0) {
            if (why)
                *why = "chunk_bytes is zero, so the chunk arithmetic has no meaning; set "
                       "Options::chunk_bytes to the fabric's segmentation granularity";
            return false;
        }
        // The id is a key, in two places: `complete()` answers by it and `defer-one` selects by it. A key
        // has to be unique and it has to be distinguishable from "none", and neither was checked. An id of
        // zero can never be reported complete (`complete()` refuses it), so accepting one would queue work
        // whose completion the caller can never observe. A duplicate id makes `complete()` answer for the
        // wrong operation. Two lines here remove a class of stall that took a long time to rule out.
        if (op.id == 0) {
            if (why)
                *why = "operation id 0 is reserved: `complete(0)` is always false, so an operation "
                       "numbered zero could never be reported finished";
            return false;
        }
        for (const Pending& q : pending_)
            if (q.op.id == op.id) {
                if (why)
                    *why = "an operation with this id is already pending; ids are the key this "
                           "scheduler answers completion and deferral by, and they must be unique";
                return false;
            }
        if (op.bytes > opt_.max_op_bytes) {
            if (why)
                *why = "operation exceeds the single-write size this simulator models; EFA-GDA "
                       "would split it and the signal accounting would differ";
            return false;
        }
        Pending p;
        p.op = op;
        // A get request has no bytes to chunk -- there is nothing local to split -- so it is always
        // exactly one control unit regardless of how much it asks the peer to fetch. The fetch itself
        // is chunked normally, as an ordinary placement, once the peer answers.
        p.chunk_count = op.is_get_request ? 1
                      : op.bytes == 0 ? 0
                      : static_cast<int>((op.bytes + opt_.chunk_bytes - 1) / opt_.chunk_bytes);
        p.placed.assign(static_cast<size_t>(p.chunk_count), false);
        p.remaining = p.chunk_count;
        p.init_unplaced();
        p.signal_pending = op.has_signal;
        if (op.id > largest_issued_id_) largest_issued_id_ = op.id;
        if (opt_.detect_overlaps) note_overlap(op);
        // An operation with no chunks and no signal has nothing to release. Queuing it would leave it
        // pending forever, because `reap` only runs after a unit is released and `pick` never returns
        // one for it: the scheduler would report `idle() == false` and `complete(id) == false` with no
        // way to make progress. Recording it as issued and dropping it is the honest answer -- there
        // is genuinely nothing to do -- and it keeps `idle()` meaning what it says.
        if (p.chunk_count == 0 and not p.signal_pending)
            return true;
        pending_.push_back(std::move(p));
        return true;
    }

    // Releases up to `budget` units and appends them to `out`. Returns the number released.
    // The caller performs the memory writes for each returned unit, in the returned order.
    // What the link to each peer can take now. A unit for a peer that has no room is not released: it stays
    // pending, and so does its operation, which is backpressure. Without it the caller would have to drop a
    // unit it could not send after the scheduler had already counted it as released.
    struct Room {
        virtual ~Room() = default;
        // Room for one more unit of up to `bytes` of payload to `peer`. Within one `progress` call the answer for
        // a peer may go from yes to no but not back, so the schedule does not depend on when a socket drains.
        virtual bool fits(int peer, size_t bytes) = 0;
        virtual void take(int peer, size_t bytes) = 0;  // a unit of `bytes` was released to this peer
    };

    size_t progress(size_t budget, std::vector<Unit>* out, Room* room = nullptr) {
        room_ = room;
        struct Clear { Room** r; ~Clear() { *r = nullptr; } } clear{&room_};
        size_t released = 0;
        while (released < budget) {
            const Choice c = pick();
            if (c.pending_index < 0)
                break;
            Pending& p = pending_[static_cast<size_t>(c.pending_index)];
            Unit u;
            if (c.chunk_index >= 0 and p.op.is_get_request) {
                // One control unit, not a byte range: see the GET note at the top of this file.
                u.kind = Unit::Kind::GetRequest;
                u.window = p.op.get_src_window;
                u.window_offset = p.op.get_src_off;
                u.bytes = p.op.bytes;
                u.get_dst_window = p.op.get_dst_window;
                u.get_dst_off = p.op.get_dst_off;
                u.get_requester_rank = static_cast<uint32_t>(p.op.requester_rank);
                // The token the owner's reply must echo back, not this op's own `id`: see the GetRequest
                // branch of `wire_unit`/`send_unit` (`nccl_rma_srdsim.cc`) for why `op_id` is the wrong
                // value here.
                u.get_response_token = p.op.get_response_token;
                u.chunk_index = 0;
                u.chunk_count = 1;
                p.mark_placed(0);
            } else if (c.chunk_index >= 0) {
                const size_t off = static_cast<size_t>(c.chunk_index) * opt_.chunk_bytes;
                const size_t len = (off + opt_.chunk_bytes > p.op.bytes) ? p.op.bytes - off : opt_.chunk_bytes;
                u.kind = Unit::Kind::Placement;
                u.src = static_cast<const char*>(p.op.src_base) + p.op.src_off + off;
                u.src_is_device = p.op.src_is_device;
                u.dst = static_cast<char*>(p.op.dst_base) + p.op.dst_off + off;
                u.bytes = len;
                u.chunk_index = c.chunk_index;
                u.chunk_count = p.chunk_count;
                u.window = p.op.dst_window;
                u.window_offset = p.op.dst_off + off;
                // A reply to someone else's get travels as an ordinary placement here; only its wire
                // kind and the id it carries differ, both decided by the caller from these two fields.
                u.is_get_response = p.op.is_get_response;
                u.get_response_token = p.op.get_response_token;
                p.mark_placed(c.chunk_index);
            } else {
                u.kind = Unit::Kind::Signal;
                u.signal_word = static_cast<char*>(p.op.signal_base) + p.op.signal_off;
                u.signal_val = p.op.signal_is_add ? p.op.signal_val : 1;
                u.signal_is_add = p.op.signal_is_add;
                u.chunk_index = -1;              // a signal unit is not one of the chunks
                u.chunk_count = p.chunk_count;
                u.window = p.op.signal_window;
                u.window_offset = p.op.signal_off;
                p.signal_pending = false;
            }
            u.op_id = p.op.id;
            u.peer = p.op.peer;
            if (opt_.trace)
                std::fprintf(stderr, "[srdsim] op %llu %s chunk %d/%d\n",
                             static_cast<unsigned long long>(u.op_id),
                             u.kind == Unit::Kind::Placement ? "place" :
                             u.kind == Unit::Kind::GetRequest ? "get-request" : "signal",
                             u.chunk_index, u.chunk_count);
            if (room_) room_->take(u.peer, u.kind == Unit::Kind::Placement ? u.bytes : 0);
            out->push_back(u);
            ++ released;
            ++ units_released_;
            reap();
        }
        return released;
    }

    // True once every submitted operation has had all its units released.
    bool idle() const { return pending_.empty(); }

    // What is still held, for a report rather than for a decision. `idle()` answers yes or no, and a
    // stall needs the shape: how many operations, how many units of each kind, and the oldest one's id.
    // Without this, "the scheduler is holding work" and "the wire lost it" look the same from outside.
    struct Outstanding {
        size_t ops = 0;
        size_t placements = 0;      // chunks not yet released
        size_t signals = 0;         // bundled signals not yet released
        uint64_t oldest_op = 0;
        int oldest_peer = -1;
    };

    Outstanding outstanding() const {
        Outstanding o;
        o.ops = pending_.size();
        for (const Pending& p : pending_) {
            for (int c = 0; c < p.chunk_count; ++ c)
                if (not p.placed[static_cast<size_t>(c)]) ++ o.placements;
            if (p.signal_pending) ++ o.signals;
        }
        if (not pending_.empty()) {
            o.oldest_op = pending_.front().op.id;
            o.oldest_peer = pending_.front().op.peer;
        }
        return o;
    }

    // An operation is complete (locally and remotely, in this same-host simulator) once all of
    // its units have been released.
    //
    // An id that was never submitted answers false, not true. The obvious implementation -- absent
    // from the pending list means finished -- gives the same answer for "this finished" and "this was
    // never asked for", so a caller that passes the wrong id is told its work is done. That is the
    // shape of a silent failure, and it is worth one comparison to avoid.
    bool complete(uint64_t op_id) const {
        if (op_id == 0 or op_id > largest_issued_id_)
            return false;
        for (const auto& p : pending_)
            if (p.op.id == op_id)
                return false;
        return true;
    }

    // The largest id submitted so far. Ids are chosen by the caller, so this is the only thing the
    // scheduler can say about which ids exist; a caller that needs more can keep its own record.
    uint64_t issued_ids() const { return largest_issued_id_; }

    // Operations submitted while an earlier one to the same peer, not yet released in full, writes some of the
    // same destination bytes. The final contents then depend on which lands last, which only an ordered fabric
    // decides; under SRD it is a write-after-write race. "Released" is the scheduler's knowledge: with delivery
    // that completes asynchronously a released write can still be landing, so this undercounts rather than
    // overcounts. Counted, not refused, and only with `Options::detect_overlaps`.
    uint64_t overlaps() const { return overlaps_; }
    const Op& overlap_earlier() const { return overlap_earlier_; }
    const Op& overlap_later() const { return overlap_later_; }

    uint64_t units_released() const { return units_released_; }

private:
    struct Pending {
        Op   op;
        int  chunk_count = 0;
        std::vector<bool> placed;
        int  remaining = 0;
        bool signal_pending = false;
        // A Fenwick tree over the chunks not yet placed (1 = still to place), so the k-th of them is found in
        // O(log n) instead of by scanning `placed`.
        std::vector<int> unplaced;

        void init_unplaced() {
            unplaced.assign(static_cast<size_t>(chunk_count) + 1, 0);
            for (int i = 1; i <= chunk_count; ++ i) {
                unplaced[static_cast<size_t>(i)] += 1;
                const int j = i + (i & -i);
                if (j <= chunk_count) unplaced[static_cast<size_t>(j)] += unplaced[static_cast<size_t>(i)];
            }
        }
        void mark_placed(int c) {
            placed[static_cast<size_t>(c)] = true;
            -- remaining;
            for (int i = c + 1; i <= chunk_count; i += i & -i) -- unplaced[static_cast<size_t>(i)];
        }
        // The chunk index of the k-th (from 0) chunk not yet placed, in index order.
        int kth_unplaced(int k) const {
            int pos = 0;
            int step = 1;
            while (step * 2 <= chunk_count) step *= 2;
            for (; step > 0; step /= 2) {
                const int next = pos + step;
                if (next <= chunk_count and unplaced[static_cast<size_t>(next)] <= k) {
                    pos = next;
                    k -= unplaced[static_cast<size_t>(next)];
                }
            }
            return pos;   // 1-based position pos + 1, as a 0-based index
        }
    };

    struct Choice {
        int pending_index = -1;
        int chunk_index = -1;   // -1 means the signal unit
    };

    // Which operation `defer-one` is holding. The escape at the bottom of `pick()` tests `have_deferred_`,
    // never `deferred_id_ != 0`: guarded by the id it would never fire for id 0, and `defer-one` would
    // stall with nothing refused. Ids are also refused on submit if zero or already pending.
    uint64_t deferred_id_ = 0;
    bool     have_deferred_ = false;

    bool defer_kind_matches(const Op& op) const {
        using K = Options::DeferKind;
        switch (opt_.defer_kind) {
            case K::Any:        return true;
            case K::Unsignaled: return not op.has_signal;
            case K::Signaled:   return op.has_signal and op.bytes != 0;
            case K::SignalOnly: return op.has_signal and op.bytes == 0;
        }
        return true;
    }

    bool signal_releasable(const Pending& p) const {
        if (not p.signal_pending)
            return false;
        if (opt_.mode == Mode::ViolateI1)
            return true;                 // the whole point of this mode
        if (p.remaining != 0)
            return false;                // I1: every byte of its own operation placed first
        if (not p.op.signal_is_strong)
            return true;
        // A strong signal additionally waits for every earlier operation to this peer to be fully
        // released. That is the difference between the two forms, and it is enforced here rather than
        // refused at the door.
        for (const Pending& q : pending_) {
            if (q.op.id >= p.op.id) continue;
            if (q.op.peer != p.op.peer or q.op.context != p.op.context) continue;
            if (q.remaining != 0 or q.signal_pending) return false;
        }
        return true;
    }

    Choice pick() {
        // `pending_` is in submission order, so "the oldest operation for this context and peer" is the first
        // one found for them. `peer-seq` needs that per context and peer rather than globally.
        std::map<std::pair<int, int>, size_t> oldest_for_peer;
        if (opt_.mode == Mode::PeerSeq)
            for (size_t i = 0; i < pending_.size(); ++ i)
                oldest_for_peer.emplace(std::make_pair(pending_[i].op.context, pending_[i].op.peer), i);

        // `defer-one` holds the oldest operation back. Which one is held has to be decided once and
        // stay decided: choosing afresh each call would hold whichever happens to be oldest now, which
        // is a different schedule and not the one being tested.
        if (opt_.mode == Mode::DeferOne and not have_deferred_) {
            for (const Pending& p : pending_) {
                if (not defer_kind_matches(p.op)) continue;
                deferred_id_ = p.op.id;
                have_deferred_ = true;
                break;
            }
        }

        // `signal-last` may release no signal while any placement anywhere is still unreleased. Placements held for
        // room on another peer's link do not count, or one full link would hold every other peer's signals.
        bool any_placement_left = false;
        if (opt_.mode == Mode::SignalLast)
            for (const Pending& p : pending_)
                if (p.remaining != 0 and has_room(p)) { any_placement_left = true; break; }
        bool skipped_for_room = false;

        // What each operation can release now, in submission order: its chunks not yet placed, in index order,
        // then its signal if releasable. The choice below is over that sequence of options, without building it,
        // so a pick costs O(operations + log chunks) rather than O(chunks pending); the options and the random
        // draws are the same as when the sequence was built (`test_srdsim_schedule` S12 pins the order).
        struct Avail { size_t i; int chunks; bool signal; };
        std::vector<Avail> avail;
        size_t total = 0;
        for (size_t i = 0; i < pending_.size(); ++ i) {
            const Pending& p = pending_[i];
            if (opt_.mode == Mode::PeerSeq and oldest_for_peer[std::make_pair(p.op.context, p.op.peer)] != i)
                continue;                // this context and peer have older work outstanding
            if (not has_room(p)) {
                skipped_for_room = true;
                continue;                // the link to this peer is full: nothing for it is released now
            }
            if (opt_.mode == Mode::DeferOne and have_deferred_ and p.op.id == deferred_id_ and
                pending_.size() > 1)
                continue;                // held until it is the only one left
            const bool sig = signal_releasable(p) and not (opt_.mode == Mode::SignalLast and any_placement_left);
            if (p.remaining == 0 and not sig) continue;
            avail.push_back({i, p.remaining, sig});
            total += static_cast<size_t>(p.remaining) + (sig ? 1 : 0);
            if (opt_.mode == Mode::InOrder)
                break;                   // never look past the oldest operation that can be released now
        }
        const auto first_of = [&](const Avail& a) -> Choice {
            const Pending& p = pending_[a.i];
            return a.chunks > 0 ? Choice{static_cast<int>(a.i), p.kth_unplaced(0)} : Choice{static_cast<int>(a.i), -1};
        };
        if (total == 0) {
            // The deferred operation is the only thing left: stop deferring it and look once more, in a
            // loop rather than by recursion, so the scheduler never reports itself idle with work
            // outstanding and no sequence of ids can make it re-choose the same operation without bound.
            // Not while something else is only waiting for room: the deferred operation is then not the only one
            // left, and releasing it would change which operation the mode holds.
            if (opt_.mode == Mode::DeferOne and have_deferred_ and not skipped_for_room) {
                have_deferred_ = false;
                deferred_id_ = 0;
                for (size_t i = 0; i < pending_.size(); ++ i) {
                    const Pending& p = pending_[i];
                    if (not has_room(p)) continue;
                    if (p.remaining > 0) return {static_cast<int>(i), p.kth_unplaced(0)};
                    if (signal_releasable(p)) return {static_cast<int>(i), -1};
                }
            }
            return {};
        }
        if (opt_.mode == Mode::InOrder)
            return first_of(avail.front());
        if (opt_.mode == Mode::ReverseOps) {
            // The newest pending operation, and within it the lowest-numbered thing available (the signal counts
            // as -1), so the schedule is a function of the queue and not of the seed.
            const Avail& a = avail.back();
            return a.signal ? Choice{static_cast<int>(a.i), -1} : first_of(a);
        }
        // `violate-i1` prefers the signal, deterministically, and that is the difference between a
        // negative test and a coin toss. Making the signal merely *releasable* left the choice to the
        // generator, so on most steps the payload went first and the violation never reached the wire:
        // every ordering mode passed with no corruption, including this one, which is what a negative
        // test that does not fire looks like from outside. A mode named for breaking a guarantee has to
        // break it whenever it can.
        if (opt_.mode == Mode::ViolateI1)
            for (const Avail& a : avail)
                if (a.signal) return {static_cast<int>(a.i), -1};
        // A free choice among the options, inside whatever constraint the mode applied above.
        size_t k = rng_.below(total);
        for (const Avail& a : avail) {
            if (k < static_cast<size_t>(a.chunks)) return {static_cast<int>(a.i), pending_[a.i].kth_unplaced(static_cast<int>(k))};
            k -= static_cast<size_t>(a.chunks);
            if (a.signal) {
                if (k == 0) return {static_cast<int>(a.i), -1};
                -- k;
            }
        }
        return {};   // not reached: k < total
    }

    void reap() {
        size_t w = 0;
        for (size_t i = 0; i < pending_.size(); ++ i) {
            const Pending& p = pending_[i];
            const bool done = (p.remaining == 0) and (not p.signal_pending);
            if (not done) {
                if (w != i)
                    pending_[w] = std::move(pending_[i]);
                ++ w;
            }
        }
        pending_.resize(w);
    }

    Options opt_;
    Rng rng_;
    Room* room_ = nullptr;   // set for the duration of one `progress` call

    bool has_room(const Pending& p) {
        if (room_ == nullptr) return true;
        // Only the signal is left once every placement is released, and a signal carries no payload.
        const size_t unit = p.op.is_get_request or p.op.bytes == 0 or p.remaining == 0
                          ? 0 : std::min(opt_.chunk_bytes, p.op.bytes);
        return room_->fits(p.op.peer, unit);
    }
    std::vector<Pending> pending_;
    uint64_t units_released_ = 0;
    uint64_t largest_issued_id_ = 0;
    uint64_t overlaps_ = 0;
    Op overlap_earlier_{}, overlap_later_{};

    static bool same_destination(const Op& a, const Op& b) {
        if (a.dst_base != nullptr or b.dst_base != nullptr) return a.dst_base == b.dst_base;
        return a.dst_window == b.dst_window;
    }

    void note_overlap(const Op& op) {
        // A get request names nothing in `dst_window`/`dst_off` -- it is not a local write at all, the
        // reply is -- so treating it as one here would manufacture an overlap against window 0.
        if (op.bytes == 0 or op.is_get_request) return;
        for (const Pending& q : pending_) {
            if (q.op.peer != op.peer or q.op.bytes == 0 or not same_destination(q.op, op)) continue;
            if (q.remaining == 0) continue;   // every unit of it has been released
            const uint64_t a0 = q.op.dst_off, a1 = q.op.dst_off + q.op.bytes;
            const uint64_t b0 = op.dst_off, b1 = op.dst_off + op.bytes;
            if (a0 < b1 and b0 < a1) {
                ++ overlaps_;
                overlap_earlier_ = q.op;
                overlap_later_ = op;
                return;
            }
        }
    }
};

}   // namespace srdsim

#endif   // SRDSIM_SCHEDULE_H_
