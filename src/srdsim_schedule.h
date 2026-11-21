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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <cstdio>
#include <string>
#include <unordered_set>
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
        // Tests only: choose with `pick_reference`, the list-building pick, instead of `pick`.
        bool     reference_pick = false;
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
        if (pending_ids_.count(op.id)) {
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
        compact();
        const size_t slot = pending_.size();
        pending_.push_back(std::move(p));
        weight_.push_back(0);
        fw_append();
        pending_ids_.insert(op.id);
        ++ live_count_;
        peer_slots_[op.peer].insert(slot);
        key_slots_[std::make_pair(op.context, op.peer)].insert(slot);
        refresh(slot);
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
        // a peer may go from yes to no but not back, so the schedule does not depend on when a socket drains. Room for
        // `bytes` is room for anything smaller.
        virtual bool fits(int peer, size_t bytes) = 0;
        virtual void take(int peer, size_t bytes) = 0;  // a unit of `bytes` was released to this peer
    };

    size_t progress(size_t budget, std::vector<Unit>* out, Room* room = nullptr) {
        struct Clear {
            Scheduler* s;
            ~Clear() { s->room_ = nullptr; s->clear_excluded(); }
        } clear{this};
        room_ = room;
        compact();
        size_t released = 0;
        while (released < budget) {
            const Choice c = opt_.reference_pick ? pick_reference()
                           : opt_.mode == Mode::Srd ? pick_srd() : pick();
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
            refresh(static_cast<size_t>(c.pending_index));
            reap(static_cast<size_t>(c.pending_index));
        }
        return released;
    }

    // Tests only: no operation that is done is still pending, and `pending_ids_` holds exactly the pending ids. Every
    // index the release tree keeps matches a recount from the operations alone: the weights against `pick`'s rule for
    // what an operation can release now, the trees against the weights, and the per-peer and per-key sets.
    bool consistent() const {
        if (pending_ids_.size() != live_count_) return false;
        size_t live = 0, dead = 0;
        std::map<int, int64_t> peer_w;
        std::map<int, std::multiset<size_t>> units;
        std::map<int, std::set<size_t>> peers;
        std::map<std::pair<int, int>, std::set<size_t>> keys;
        int64_t total = 0;
        for (size_t i = 0; i < pending_.size(); ++ i) {
            const Pending& p = pending_[i];
            int w = 0;
            if (p.live) {
                ++ live;
                if ((p.remaining == 0 and not p.signal_pending) or pending_ids_.count(p.op.id) == 0) return false;
                bool sig = p.signal_pending and p.remaining == 0;
                if (sig and p.op.signal_is_strong)
                    for (size_t j = 0; j < i; ++ j) {
                        const Pending& q = pending_[j];
                        if (q.live and q.op.peer == p.op.peer and q.op.context == p.op.context) sig = false;
                    }
                w = p.remaining + (sig ? 1 : 0);
                peers[p.op.peer].insert(i);
                keys[std::make_pair(p.op.context, p.op.peer)].insert(i);
                if (const size_t u = unit_of(p)) units[p.op.peer].insert(u);
                if (w) peer_w[p.op.peer] += w;
            } else {
                ++ dead;
            }
            if (weight_[i] != w or p.in_units != (p.live and unit_of(p) != 0)) return false;
            total += w;
        }
        if (live != live_count_ or dead != dead_count_ or total != fw_total_) return false;
        if (fw_.size() != pending_.size() + 1 and not (fw_.empty() and pending_.empty())) return false;
        for (size_t j = 1; j < fw_.size(); ++ j) {
            int64_t sum = 0;
            for (size_t i = j - (j & (~j + 1)) + 1; i <= j; ++ i) sum += weight_[i - 1];
            if (fw_[j] != sum) return false;
        }
        std::map<int, int64_t> kept_w;
        for (const auto& e : peer_weight_) if (e.second) kept_w.insert(e);
        return peers == peer_slots_ and keys == key_slots_ and units == peer_units_ and peer_w == kept_w and ex_.empty() and
               excluded_.empty();
    }

    // Tests only: how many times the slots of finished operations were dropped.
    uint64_t compactions() const { return compactions_; }

    // True once every submitted operation has had all its units released.
    bool idle() const { return live_count_ == 0; }

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
        o.ops = live_count_;
        bool first = true;
        for (const Pending& p : pending_) {
            if (not p.live) continue;
            for (int c = 0; c < p.chunk_count; ++ c)
                if (not p.placed[static_cast<size_t>(c)]) ++ o.placements;
            if (p.signal_pending) ++ o.signals;
            if (first) { o.oldest_op = p.op.id; o.oldest_peer = p.op.peer; first = false; }
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
        return pending_ids_.count(op_id) == 0;
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
        bool live = true;        // false once every unit is released; the slot stays until `compact`
        bool in_units = false;   // its unit size is in `peer_units_`
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
            assert(k >= 0 and k < remaining);
            int pos = 0;
            int step = 1;
            while (step <= chunk_count / 2) step *= 2;
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
        // A strong signal additionally waits for every operation submitted before it on its context and peer to be
        // fully released. That is the difference between the two forms, and it is enforced here rather than refused
        // at the door. Earlier is by submission, the order `pending_` keeps, not by id: ids are the caller's, and an
        // order by id would disagree with `peer-seq`, which keeps submission order, and hold both operations forever.
        for (const Pending& q : pending_) {
            if (&q == &p) break;
            if (not q.live or q.op.peer != p.op.peer or q.op.context != p.op.context) continue;
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
                if (pending_[i].live) oldest_for_peer.emplace(std::make_pair(pending_[i].op.context, pending_[i].op.peer), i);

        // `defer-one` holds the oldest operation back. Which one is held has to be decided once and
        // stay decided: choosing afresh each call would hold whichever happens to be oldest now, which
        // is a different schedule and not the one being tested.
        if (opt_.mode == Mode::DeferOne and not have_deferred_) {
            for (const Pending& p : pending_) {
                if (not p.live or not defer_kind_matches(p.op)) continue;
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
                if (p.live and p.remaining != 0 and has_room(p)) { any_placement_left = true; break; }
        bool skipped_for_room = false;

        // What each operation can release now, in submission order: its chunks not yet placed, in index order,
        // then its signal if releasable. The choice below is over that sequence of options, without building it,
        // so a pick costs O(operations + log chunks) rather than O(chunks pending); the options and the random
        // draws are the same as when the sequence was built (`test_srdsim_schedule` S12 pins the order).
        std::vector<Avail>& avail = avail_scratch_;
        avail.clear();
        size_t total = 0;
        for (size_t i = 0; i < pending_.size(); ++ i) {
            const Pending& p = pending_[i];
            if (not p.live) continue;
            if (opt_.mode == Mode::PeerSeq and oldest_for_peer[std::make_pair(p.op.context, p.op.peer)] != i)
                continue;                // this context and peer have older work outstanding
            if (not has_room(p)) {
                skipped_for_room = true;
                continue;                // the link to this peer is full: nothing for it is released now
            }
            if (opt_.mode == Mode::DeferOne and have_deferred_ and p.op.id == deferred_id_ and
                live_count_ > 1)
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
                    if (not p.live or not has_room(p)) continue;
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

    // The list-building pick this scheduler used before `pick` counted its options instead: kept only as the
    // oracle `test_srdsim_schedule` S13 checks `pick` against, choice by choice (`Options::reference_pick`).
    Choice pick_reference() {
        std::vector<Choice> options;
        // `pending_` is in submission order, so "the oldest operation for this context and peer" is the first
        // one found for them. `peer-seq` needs that per context and peer rather than globally.
        std::map<std::pair<int, int>, size_t> oldest_for_peer;
        if (opt_.mode == Mode::PeerSeq)
            for (size_t i = 0; i < pending_.size(); ++ i)
                if (pending_[i].live) oldest_for_peer.emplace(std::make_pair(pending_[i].op.context, pending_[i].op.peer), i);

        // `defer-one` holds the oldest operation back. Which one is held has to be decided once and
        // stay decided: choosing afresh each call would hold whichever happens to be oldest now, which
        // is a different schedule and not the one being tested.
        if (opt_.mode == Mode::DeferOne and not have_deferred_) {
            for (const Pending& p : pending_) {
                if (not p.live or not defer_kind_matches(p.op)) continue;
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
                if (p.live and p.remaining != 0 and has_room(p)) { any_placement_left = true; break; }
        bool skipped_for_room = false;

        for (size_t i = 0; i < pending_.size(); ++ i) {
            const Pending& p = pending_[i];
            if (not p.live) continue;
            if (opt_.mode == Mode::PeerSeq and oldest_for_peer[std::make_pair(p.op.context, p.op.peer)] != i)
                continue;                // this context and peer have older work outstanding
            if (not has_room(p)) {
                skipped_for_room = true;
                continue;                // the link to this peer is full: nothing for it is released now
            }
            if (opt_.mode == Mode::DeferOne and have_deferred_ and p.op.id == deferred_id_ and
                live_count_ > 1)
                continue;                // held until it is the only one left
            for (int c = 0; c < p.chunk_count; ++ c)
                if (not p.placed[static_cast<size_t>(c)])
                    options.push_back({static_cast<int>(i), c});
            if (signal_releasable(p) and
                not (opt_.mode == Mode::SignalLast and any_placement_left))
                options.push_back({static_cast<int>(i), -1});
            if (opt_.mode == Mode::InOrder and not options.empty())
                break;                   // never look past the oldest operation that can be released now
        }
        if (options.empty()) {
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
                    if (not p.live or not has_room(p)) continue;
                    for (int c = 0; c < p.chunk_count; ++ c)
                        if (not p.placed[static_cast<size_t>(c)])
                            return {static_cast<int>(i), c};
                    if (signal_releasable(p))
                        return {static_cast<int>(i), -1};
                }
            }
            return {};
        }
        if (opt_.mode == Mode::InOrder)
            return options.front();
        if (opt_.mode == Mode::ReverseOps) {
            // The newest pending operation, and within it the lowest-numbered thing available, so the
            // schedule is a function of the queue and not of the seed.
            Choice best = options.front();
            for (const Choice& c : options)
                if (c.pending_index > best.pending_index or
                    (c.pending_index == best.pending_index and c.chunk_index < best.chunk_index))
                    best = c;
            return best;
        }
        // `violate-i1` prefers the signal, deterministically, and that is the difference between a
        // negative test and a coin toss. Making the signal merely *releasable* left the choice to the
        // generator, so on most steps the payload went first and the violation never reached the wire:
        // every ordering mode passed with no corruption, including this one, which is what a negative
        // test that does not fire looks like from outside. A mode named for breaking a guarantee has to
        // break it whenever it can.
        if (opt_.mode == Mode::ViolateI1)
            for (const Choice& c : options)
                if (c.chunk_index < 0) return c;
        if (opt_.mode == Mode::SignalLast or opt_.mode == Mode::DeferOne or
            opt_.mode == Mode::PeerSeq)
            return options[rng_.below(options.size())];   // free choice inside the constraint
        return options[rng_.below(options.size())];
    }

    // Only the operation a unit was just released from can have become done, so only it is checked; the others
    // keep their order. `submit` and this are the only writers of `pending_` and `pending_ids_`, and a unit's
    // release (`progress`) is the only change to an operation's `remaining` and `signal_pending`.
    // The slot stays, marked not live, until `compact` removes it, so the indices of the other operations, and of the
    // release tree, do not move while a `progress` call is choosing.
    void reap(size_t i) {
        assert(i < pending_.size());
        Pending& p = pending_[i];
        if (not p.live or p.remaining != 0 or p.signal_pending) return;
        pending_ids_.erase(p.op.id);
        p.live = false;
        -- live_count_;
        ++ dead_count_;
        refresh(i);
        remove_slot(peer_slots_, p.op.peer, i);
        const auto key = std::make_pair(p.op.context, p.op.peer);
        auto ks = key_slots_.find(key);
        const bool was_oldest = ks != key_slots_.end() and *ks->second.begin() == i;
        if (ks != key_slots_.end()) {
            ks->second.erase(i);
            if (ks->second.empty()) key_slots_.erase(ks);
        }
        // A strong signal waits for every earlier operation on its context and peer: the next oldest may now go.
        if (was_oldest) {
            ks = key_slots_.find(key);
            if (ks != key_slots_.end()) refresh(*ks->second.begin());
        }
        p.placed = {};
        p.unplaced = {};
    }

    // ---- The release tree ----------------------------------------------------------------------------------------
    // `pick` scans every pending operation, which is what the ordering modes need. `Mode::Srd` draws uniformly among
    // every unit any operation can release now, so it keeps, per slot of `pending_`, how many units that operation
    // can release (its chunks not yet placed, plus one for its signal when that is releasable), and a Fenwick tree
    // over those weights: a draw is then found in O(log n). The options are in the order `pick` lists them and are
    // drawn with the same generator, so without a Room, or with one whose answer does not depend on the unit's size,
    // the schedule is `pick`'s (`test_srdsim_schedule` S12 and S13).
    //
    // Room is asked once per pick per peer with something to release, for the largest unit that peer has pending; a
    // peer that has none is left out of the draw for the rest of the `progress` call. The Room contract (yes may become
    // no within a call, never back; room for a unit is room for a smaller one) makes that sound. `pick` asks per
    // operation, so with a Room whose answer depends on the size, `srd` can hold a peer's small units back while its
    // largest does not fit, where `pick` would release them; both release only units that fit.

    // The largest unit operation `p` can still release, in bytes of payload: 0 once only its signal is left.
    size_t unit_of(const Pending& p) const {
        if (not p.live or p.op.is_get_request or p.op.bytes == 0 or p.remaining == 0) return 0;
        return std::min(opt_.chunk_bytes, p.op.bytes);
    }
    // The units operation `i` can release now under `Mode::Srd`'s rules.
    int weight(size_t i) const {
        const Pending& p = pending_[i];
        if (not p.live) return 0;
        bool sig = p.signal_pending and p.remaining == 0;
        if (sig and p.op.signal_is_strong) {
            // The oldest live operation on its context and peer, as in `signal_releasable`.
            const auto ks = key_slots_.find(std::make_pair(p.op.context, p.op.peer));
            sig = ks != key_slots_.end() and *ks->second.begin() == i;
        }
        return p.remaining + (sig ? 1 : 0);
    }
    // Brings slot `i`'s weight, its peer's total and its peer's unit sizes up to date with the operation.
    void refresh(size_t i) {
        Pending& p = pending_[i];
        const size_t u = unit_of(p);
        if (p.in_units != (u != 0)) {
            if (u != 0) {
                peer_units_[p.op.peer].insert(u);
            } else {
                auto it = peer_units_.find(p.op.peer);
                it->second.erase(it->second.find(std::min(opt_.chunk_bytes, p.op.bytes)));
                if (it->second.empty()) peer_units_.erase(it);
            }
            p.in_units = u != 0;
        }
        const int w = weight(i);
        const int64_t d = int64_t(w) - weight_[i];
        if (d == 0) return;
        weight_[i] = w;
        tree_add(fw_, i, d);
        fw_total_ += d;
        auto pw = peer_weight_.find(p.op.peer);
        if (pw == peer_weight_.end()) pw = peer_weight_.emplace(p.op.peer, 0).first;
        pw->second += d;
        if (pw->second == 0) peer_weight_.erase(pw);
        if (ex_built_ and excluded_.count(p.op.peer)) tree_add(ex_, i, d);
    }
    static void tree_add(std::vector<int64_t>& t, size_t i, int64_t d) {
        for (size_t j = i + 1; j < t.size(); j += j & (~j + 1)) t[j] += d;
    }
    // Appends a slot of weight 0 to `fw_`: the new node covers the `lowbit` slots that end at it. Only between
    // `progress` calls, when there is no excluded tree to extend.
    void fw_append() {
        assert(ex_.empty());
        if (fw_.empty()) fw_.push_back(0);
        const size_t n = fw_.size();   // the new node's 1-based index
        int64_t sum = 0;
        for (size_t j = n - 1, stop = n - (n & (~n + 1)); j > stop; j -= j & (~j + 1)) sum += fw_[j];
        fw_.push_back(sum);
    }
    // A tree over `w`, keeping only the slots `keep` accepts.
    template <class Keep> static void tree_build(std::vector<int64_t>& t, const std::vector<int>& w, Keep keep) {
        t.assign(w.size() + 1, 0);
        for (size_t i = 1; i <= w.size(); ++ i) {
            if (keep(i - 1)) t[i] += w[i - 1];
            const size_t j = i + (i & (~i + 1));
            if (j <= w.size()) t[j] += t[i];
        }
    }
    static void remove_slot(std::map<int, std::set<size_t>>& m, int k, size_t i) {
        auto it = m.find(k);
        if (it == m.end()) return;
        it->second.erase(i);
        if (it->second.empty()) m.erase(it);
    }
    void clear_excluded() {
        excluded_.clear();
        ex_.clear();
        ex_built_ = false;
    }
    // Drops the slots of operations no longer live once they are as many as the live ones, so the cost of rebuilding
    // the indices, O(n log n), is spread over the reaps that made them. Never called while a `progress` call is
    // choosing.
    void compact() {
        if (dead_count_ == 0 or dead_count_ < live_count_ + 32) return;
        ++ compactions_;
        std::vector<Pending> kept;
        kept.reserve(live_count_);
        for (Pending& p : pending_) if (p.live) kept.push_back(std::move(p));
        pending_ = std::move(kept);
        dead_count_ = 0;
        peer_slots_.clear();
        key_slots_.clear();
        peer_units_.clear();
        peer_weight_.clear();
        for (size_t i = 0; i < pending_.size(); ++ i) {
            Pending& p = pending_[i];
            p.in_units = false;
            peer_slots_[p.op.peer].insert(i);
            key_slots_[std::make_pair(p.op.context, p.op.peer)].insert(i);
        }
        weight_.assign(pending_.size(), 0);
        fw_.assign(pending_.size() + 1, 0);
        fw_total_ = 0;
        for (size_t i = 0; i < pending_.size(); ++ i) refresh(i);
    }

    // `pick` for `Mode::Srd`, from the release tree.
    Choice pick_srd() {
        if (room_ != nullptr) {
            for (const auto& pw : peer_weight_) {
                if (excluded_.count(pw.first)) continue;
                const auto pu = peer_units_.find(pw.first);
                const size_t unit = pu == peer_units_.end() ? 0 : *pu->second.rbegin();
                if (not room_->fits(pw.first, unit)) { excluded_.insert(pw.first); ex_built_ = false; }
            }
        }
        int64_t excluded_weight = 0;
        for (int peer : excluded_) {
            const auto pw = peer_weight_.find(peer);
            if (pw != peer_weight_.end()) excluded_weight += pw->second;
        }
        const int64_t total = fw_total_ - excluded_weight;
        if (total <= 0) return {};
        const bool ex = not excluded_.empty();
        if (ex and not ex_built_) {
            tree_build(ex_, weight_, [&](size_t i) { return excluded_.count(pending_[i].op.peer) != 0; });
            ex_built_ = true;
        }
        size_t k = rng_.below(static_cast<size_t>(total));
        // The slot whose units hold draw `k`, descending the tree of weights less the excluded ones.
        size_t pos = 0;
        size_t step = 1;
        while (step * 2 < fw_.size()) step *= 2;
        for (; step > 0; step /= 2) {
            const size_t next = pos + step;
            if (next >= fw_.size()) continue;
            const int64_t v = fw_[next] - (ex ? ex_[next] : 0);
            if (v <= static_cast<int64_t>(k)) { pos = next; k -= static_cast<size_t>(v); }
        }
        const Pending& p = pending_[pos];   // 0-based slot
        if (k < static_cast<size_t>(p.remaining)) return {static_cast<int>(pos), p.kth_unplaced(static_cast<int>(k))};
        return {static_cast<int>(pos), -1};
    }

    Options opt_;
    Rng rng_;
    Room* room_ = nullptr;   // set for the duration of one `progress` call
    struct Avail { size_t i; int chunks; bool signal; };
    std::vector<Avail> avail_scratch_;   // reused by `pick`, so a pick does not allocate

    bool has_room(const Pending& p) {
        if (room_ == nullptr) return true;
        // Only the signal is left once every placement is released, and a signal carries no payload.
        const size_t unit = p.op.is_get_request or p.op.bytes == 0 or p.remaining == 0
                          ? 0 : std::min(opt_.chunk_bytes, p.op.bytes);
        return room_->fits(p.op.peer, unit);
    }
    std::vector<Pending> pending_;               // in submission order; a slot no longer live waits for `compact`
    std::unordered_set<uint64_t> pending_ids_;   // the ids of the live operations, for `complete` and the duplicate check
    size_t live_count_ = 0, dead_count_ = 0;
    std::map<int, std::set<size_t>> peer_slots_;                    // live slots per peer
    std::map<int, int64_t> peer_weight_;                            // per peer with any: the sum of its slots' weights
    std::map<int, std::multiset<size_t>> peer_units_;               // per peer: `unit_of` each live slot, where not 0
    std::map<std::pair<int, int>, std::set<size_t>> key_slots_;     // live slots per (context, peer), oldest first
    std::vector<int> weight_;                    // per slot: what `weight` was when last refreshed
    std::vector<int64_t> fw_, ex_;               // Fenwick trees over `weight_`, and over the excluded peers' part
    int64_t fw_total_ = 0;
    std::set<int> excluded_;                     // peers without room for the rest of this `progress` call
    bool ex_built_ = false;                      // `ex_` is over `excluded_` as it is now
    uint64_t compactions_ = 0;
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
            if (not q.live or q.op.peer != op.peer or q.op.bytes == 0 or not same_destination(q.op, op)) continue;
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
