// SPDX-License-Identifier: Apache-2.0

// The wire side of the SRD simulator: how a released unit actually lands in the peer's window.
//
// WHY THE RECEIVER WRITES, NOT THE SENDER
// The obvious design has the sender map the peer's window and copy into it. That needs two things the
// hardware may not give: GPU-to-GPU peer access, which is not always available, and a cuMem allocation
// exported across the process boundary (which works). Across hosts the first is impossible in principle.
//
// So the roles are inverted: the sender ships a descriptor and the bytes over a stream, and the
// **receiving** side's agent writes into its own window. That needs neither peer access nor an
// exported allocation, works identically within a host and across hosts, and keeps the interesting
// property intact -- the *order* in which units land is still decided entirely by the sender's
// scheduler, because a stream preserves the order in which the sender wrote to it.
//
// WHY A WINDOW ID INSTEAD OF AN ADDRESS
// `regMrSym` is collective, so every rank registers the same windows in the same order. Numbering
// them gives a name both sides agree on, and the receiver resolves the name to its own base
// address. The sender never needs to know where the peer's memory lives. This is also closer to
// what a real NIC does: a put names a remote key, not a remote pointer.
//
// WHAT THIS DOES NOT MODEL
// A stream is reliable and ordered, so nothing here reproduces loss or reordering by itself. That
// is deliberate: the reordering must come from the scheduler, where a test can control it and
// reproduce it from a seed. A transport that reordered on its own would make failures
// irreproducible, which is the opposite of what this is for.

#ifndef SRDSIM_TRANSPORT_H_
#define SRDSIM_TRANSPORT_H_

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <deque>
#include <map>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <condition_variable>
#include <poll.h>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <csignal>

#include <fcntl.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

namespace srdsim {

// Milliseconds on a monotonic clock, from an arbitrary origin. Only differences are meaningful.
//
// This exists because a frozen counter is ambiguous and a dated one is not. Two samples of
// "driven 254699 times" taken a hundred seconds apart say the count did not change, which is
// consistent with a thread that died, a thread blocked inside this code, and a report that is itself
// stale -- three different bugs with three different owners. "no progress call for 104312 ms" is
// consistent with only the first two, and pairing it with a mark taken on entry to each call
// separates those.
inline uint64_t mono_ms() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

// How long ago a mark was taken, or `never` when the mark was never set. Zero is a real time, so a
// zero mark has to mean "unset" separately from "just now".
inline uint64_t age_ms(uint64_t mark, uint64_t never = ~uint64_t(0)) {
    if (mark == 0) return never;
    const uint64_t now = mono_ms();
    return now >= mark ? now - mark : 0;
}

// ---------------------------------------------------------------------------
// Wire format
// ---------------------------------------------------------------------------

// GetRequest carries no payload: the requester does not have the bytes yet, that is the whole point of
// a get. GetResponse is the owner's reply, carrying the fetched bytes; it lands exactly like a
// Placement (same code path in the receive loop) and differs only in that landing it also has to
// release whichever `iflush` is waiting on it.
enum class WireKind : uint32_t { Placement = 1, Signal = 2, Shutdown = 3, GetRequest = 4, GetResponse = 5 };

// Fixed-size header, little-endian host order. Both ends are the same build on the same
// architecture, so no byte-order conversion: this is a test transport between peers that were
// launched together, not a protocol that has to survive a version skew.
//
// `length` is always how many payload bytes trail this one frame on the wire -- 0 for Signal and for
// GetRequest, which carries no payload either. A GetRequest's own fetch size travels in `value`
// instead, which every other kind leaves unused; keeping `length` uniform across kinds is what lets
// the one bound check at the top of the receive loop (how much to allocate for *this frame*) stay a
// single check instead of one more branch per kind.
struct WireHeader {
    uint32_t kind;              // WireKind
    uint32_t window_id;         // which registered window. GetRequest: the window *on the peer* to read
    uint32_t dest_window_id;    // GetRequest only: the window, in the requester's own numbering, the
                                 // reply must address
    uint32_t requester_rank;    // GetRequest only: which rank to send the reply to -- the receiving
                                 // side never otherwise learns who is on the other end of a connection
                                 // (see the comment on `serve()`), and a get is the first kind where it
                                 // has to answer somebody specific
    uint64_t offset;            // byte offset into `window_id`
    uint64_t length;            // payload bytes trailing this frame; 0 for Signal and GetRequest
    uint64_t value;             // add-by value for Signal; fetch size (bytes) for GetRequest
    uint64_t op_id;             // tracing only for Placement/Signal. For GetRequest, an opaque token
                                 // the requester chose (its own context, as a pointer-sized value
                                 // meaningful only to itself); the owner echoes it back verbatim in the
                                 // GetResponse so the requester can tell which get() to release
    uint64_t dest_offset;       // GetRequest only: the offset paired with `dest_window_id`
};
static_assert(sizeof(WireHeader) == 56,
             "WireHeader must stay tightly packed (4 x uint32 + 5 x uint64, uint32s grouped first so "
             "the uint64s need no padding to stay 8-byte aligned)");

// ---------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------

// One registered symmetric window, as this rank sees it. `base` is this rank's own pointer, which
// is all the receiving side needs.
struct Window {
    void*  base = nullptr;
    size_t bytes = 0;
    bool   is_device = false;
};

// The receiver writes through this. Host windows are a memcpy; device windows need a CUDA copy,
// which the plugin supplies so that this header stays free of CUDA.
using DeviceCopyFn = bool (*)(void* dst, const void* src, size_t bytes, void* user);
using DeviceAddFn  = bool (*)(void* dst_u64, uint64_t value, void* user);

// Called with the owner's own local source (resolved from `window_id`/`offset`, already bounds checked
// the same way a Placement's destination is) when a GetRequest arrives, so the plugin can submit the
// reply on its own scheduler. Returns whether it was accepted, for the same counting `device_copy`'s
// return does.
using GetRequestFn = bool (*)(const void* src, bool src_is_device, size_t length,
                              uint32_t dest_window, uint64_t dest_off, uint32_t requester_rank,
                              uint64_t owner_token, void* user);
// Called each time a frame of a GetResponse has actually landed in memory -- not merely been sent, which is
// all a placement's own completion ever meant -- with the bytes it carried, so the plugin can release the
// `iflush` request that is fencing on the get. A reply of several chunks lands as several frames.
// `owner_token` is the opaque token the requester put in the GetRequest's `op_id`, echoed back unchanged.
using GetResponseLandedFn = void (*)(uint64_t owner_token, uint64_t bytes, void* user);

class WindowTable {
public:
    // Returns the id assigned to this window. Registration is collective, so the same call order
    // on every rank yields the same ids without any exchange.
    uint32_t add(const Window& w) {
        std::lock_guard<std::mutex> g(mu_);
        windows_.push_back(w);
        in_use_.push_back(0);
        return static_cast<uint32_t>(windows_.size() - 1);
    }

    // A window held for one write. While any lease on a window is outstanding, `retire` does not return, so
    // a write that looked the window up before deregistration finishes before NCCL frees the memory.
    //
    // Looking a window up and then writing it was two steps with nothing between them and deregistration:
    // a frame whose payload was still being read off the socket when NCCL deregistered was then written
    // into memory that had been freed. The lease makes lookup-and-write one step as far as `retire` is
    // concerned. It is taken after the payload has been read, so a stalled peer cannot hold a
    // deregistration up.
    class Lease {
    public:
        Lease(WindowTable* t, uint32_t id) : t_(t), id_(id) { ok_ = t_->acquire(id_, &w_); }
        ~Lease() { if (ok_) t_->release(id_); }
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        bool ok() const { return ok_; }
        const Window& window() const { return w_; }
    private:
        WindowTable* t_;
        uint32_t id_;
        Window w_{};
        bool ok_ = false;
    };

    bool get(uint32_t id, Window* out) const {
        std::lock_guard<std::mutex> g(mu_);
        if (id >= windows_.size()) return false;
        // A retired window is not a window. Answering with its old base is how a frame that arrives after
        // deregistration gets written into memory somebody else now owns.
        if (windows_[id].base == nullptr) return false;
        *out = windows_[id];
        return true;
    }

    // Marks a window as gone, keeping its id occupied.
    //
    // The id space has to stay stable -- ids travel on the wire and both ranks number windows by
    // registration order -- so an entry is invalidated rather than erased.
    //
    // Without this the table outlives the memory it describes. NCCL deregisters a window, the allocation
    // is freed, and a frame still in flight for that id is applied to the old base. For a host window that
    // base is ordinary heap, so the write corrupts whatever the application allocated there next.
    // Applications such as vLLM register and deregister host windows during start-up.
    //
    // Waits for every outstanding lease on the window: a write already under way finishes first, and no new
    // one can start.
    void retire(uint32_t id) {
        std::unique_lock<std::mutex> g(mu_);
        if (id >= windows_.size()) return;
        windows_[id].base = nullptr;
        windows_[id].bytes = 0;
        released_.wait(g, [&] { return in_use_[id] == 0; });
    }

    uint32_t leases(uint32_t id) const {
        std::lock_guard<std::mutex> g(mu_);
        return id < in_use_.size() ? in_use_[id] : 0;
    }

    size_t size() const {
        std::lock_guard<std::mutex> g(mu_);
        return windows_.size();
    }

    // The largest window registered so far, which is the largest payload any legitimate frame can
    // carry. The receiver uses it as the bound on a wire-supplied length: a length above it cannot be
    // a real frame, and cannot be trusted to say how much of the stream to skip either.
    //
    // It is a maximum over all windows rather than the named window's size on purpose. The window id
    // is read from the same header as the length, so a desynchronised stream gets both wrong, and a
    // bound that depends on the id would be as untrustworthy as the id.
    size_t largest_bytes() const {
        std::lock_guard<std::mutex> g(mu_);
        size_t m = 0;
        for (const Window& w : windows_)
            if (w.bytes > m) m = w.bytes;
        return m;
    }

private:
    bool acquire(uint32_t id, Window* out) {
        std::lock_guard<std::mutex> g(mu_);
        if (id >= windows_.size() or windows_[id].base == nullptr) return false;
        ++ in_use_[id];
        *out = windows_[id];
        return true;
    }

    void release(uint32_t id) {
        {
            std::lock_guard<std::mutex> g(mu_);
            -- in_use_[id];
        }
        released_.notify_all();
    }

    mutable std::mutex mu_;
    std::condition_variable released_;
    std::vector<Window> windows_;
    std::vector<uint32_t> in_use_;   // outstanding leases per window, parallel to `windows_`
};

// True when the range [offset, offset + length) lies entirely inside `bytes`, without ever computing
// `offset + length`.
//
// The obvious form, `offset + length > bytes`, is not a bounds check. Both operands come off the wire,
// and `offset = 2^64 - 8` with `length = 16` wraps to 8, passes, and names a write far outside the
// window. Written this way there is no addition to overflow, so the property holds for every pair of
// 64-bit inputs rather than for the pairs somebody thought to try.
//
// At namespace scope rather than inside the agent so that it can be checked on its own; the boundary
// values are where a bounds check is wrong, and they are easier to state as a table than to provoke
// through a socket.
inline bool range_fits(uint64_t offset, uint64_t length, size_t bytes) {
    if (offset > bytes) return false;
    return length <= static_cast<uint64_t>(bytes) - offset;
}

// The same function under a name a test can call without implying it is part of the agent's surface.
inline bool range_fits_for_test(uint64_t offset, uint64_t length, size_t bytes) {
    return range_fits(offset, length, bytes);
}

// ---------------------------------------------------------------------------
// Socket helpers
// ---------------------------------------------------------------------------

// A peer that closes early must surface as a failed write, not as a signal that kills the process.
// Called from every entry point that can create a socket.
inline void ignore_sigpipe_once() {
    static const bool done = [] {
        std::signal(SIGPIPE, SIG_IGN);
        return true;
    }();
    (void)done;
}

// Bounded, because every send descriptor is deliberately non-blocking. Retrying EAGAIN with no wait and
// no limit is a busy spin that never ends if the reader has gone, and the caller holds a per-peer lock
// while it happens -- so every other thread with work for that peer stops too. On the loopback link the
// reader is a thread in this same process, which teardown may already have stopped.
// How many threads are inside `write_all` right now, and when the oldest of them entered. A blocking
// write inside a call the caller expects to return promptly is invisible from outside: the counters it
// would have advanced simply do not advance, which looks exactly like never being called. These two
// numbers tell those apart, and they are the only way to observe the one cycle this design admits --
// a sender parked on a full socket whose reader is a thread that has itself stopped draining.
struct WriteBlockState {
    uint64_t threads_inside = 0;
    uint64_t oldest_entered_ms = 0;
};
inline std::atomic<uint64_t>& write_inside_count() {
    static std::atomic<uint64_t> n{0};
    return n;
}
inline std::atomic<uint64_t>& write_oldest_entry_ms() {
    static std::atomic<uint64_t> t{0};
    return t;
}
inline WriteBlockState write_block_state() {
    WriteBlockState s{};
    s.threads_inside = write_inside_count().load(std::memory_order_relaxed);
    s.oldest_entered_ms = write_oldest_entry_ms().load(std::memory_order_relaxed);
    return s;
}

inline bool write_all(int fd, const void* buf, size_t bytes,
                      int64_t budget_ms = 2000) {
    const char* p = static_cast<const char*>(buf);
    const auto started = std::chrono::steady_clock::now();
    // Published only while a write is in progress. The entry stamp is set by the thread that finds the
    // counter at zero, so it names the oldest writer rather than the newest; a newest-writer stamp
    // would reset on every call and never show a thread that has been stuck for a minute.
    struct Mark {
        Mark() {
            if (write_inside_count().fetch_add(1, std::memory_order_acq_rel) == 0)
                write_oldest_entry_ms().store(mono_ms(), std::memory_order_relaxed);
        }
        ~Mark() {
            if (write_inside_count().fetch_sub(1, std::memory_order_acq_rel) == 1)
                write_oldest_entry_ms().store(0, std::memory_order_relaxed);
        }
    } mark;
    while (bytes > 0) {
        const ssize_t n = ::send(fd, p, bytes, 0);
        if (n > 0) { p += n; bytes -= static_cast<size_t>(n); continue; }
        if (n < 0 and errno == EINTR) continue;
        if (n < 0 and (errno == EAGAIN or errno == EWOULDBLOCK)) {
            const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - started).count();
            if (waited >= budget_ms) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        return false;
    }
    return true;
}

inline bool read_all(int fd, void* buf, size_t bytes) {
    char* p = static_cast<char*>(buf);
    while (bytes > 0) {
        const ssize_t n = ::recv(fd, p, bytes, 0);
        if (n > 0) { p += n; bytes -= static_cast<size_t>(n); continue; }
        if (n == 0) return false;                       // peer closed
        if (errno == EINTR or errno == EAGAIN) continue;
        return false;
    }
    return true;
}

inline void set_nodelay(int fd) {
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

// Listens on a chosen port, or on an ephemeral one if `port` is zero, and reports what was bound.
// Under NCCL the port travels in the connection handle and an ephemeral one is right; a peer that
// has no channel to learn it has to be told a fixed port in advance instead.
inline int listen_on(uint16_t port, uint16_t* out_port = nullptr) {
    ignore_sigpipe_once();
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons(port);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0 or ::listen(fd, 64) != 0) {
        ::close(fd);
        return -1;
    }
    if (out_port != nullptr) {
        sockaddr_in got{};
        socklen_t len = sizeof(got);
        if (::getsockname(fd, reinterpret_cast<sockaddr*>(&got), &len) != 0) {
            ::close(fd);
            return -1;
        }
        *out_port = ntohs(got.sin_port);
    }
    return fd;
}

inline int listen_any(uint16_t* out_port) { return listen_on(0, out_port); }

inline int dial(const char* ip, uint16_t port, int attempts = 600) {
    ignore_sigpipe_once();
    for (int i = 0; i < attempts; ++ i) {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return -1;
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons(port);
        if (::inet_pton(AF_INET, ip, &a.sin_addr) != 1) { ::close(fd); return -1; }
        if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0) {
            set_nodelay(fd);
            return fd;
        }
        ::close(fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return -1;
}

// ---------------------------------------------------------------------------
// The receiving agent
// ---------------------------------------------------------------------------

// One thread per inbound connection. It applies whatever arrives, in arrival order, which equals
// the order the sender wrote it. Applying is the only thing it does: every decision about order
// was already made on the other side.
//
// Connections are applied concurrently and nothing orders one connection's writes against another's,
// as on a real NIC: puts from different peers to overlapping bytes are the application's error.
class ReceiveAgent {
public:
    struct Hooks {
        DeviceCopyFn device_copy = nullptr;   // needed only for device windows
        DeviceAddFn  device_add = nullptr;
        GetRequestFn get_request = nullptr;
        GetResponseLandedFn get_response_landed = nullptr;
        void*        user = nullptr;
        void (*log)(const char*) = nullptr;
        // Called when the next frame has not arrived yet and the agent is about to wait for it. A writer
        // that returns before its bytes land settles them here, so nothing it issued is still pending
        // while the link is quiet -- which is when an application tears its memory down.
        void (*idle)(void* user) = nullptr;
    };

    ReceiveAgent(WindowTable* windows, Hooks hooks) : windows_(windows), hooks_(hooks) {}

    // The largest payload any single frame may carry. The sender chunks by `chunk_bytes`, so this is
    // that value; it is set rather than inferred because the agent has no other way to know it, and
    // inferring it from the windows is what made the check useless.
    void set_max_frame_bytes(uint64_t bytes) {
        if (bytes != 0) max_frame_bytes_ = bytes;
    }

    ~ReceiveAgent() { stop(); }

    // Returns false, having taken no ownership of `fd`, if this agent is stopping or has stopped.
    // Accepting then would start a thread that `join` has already passed, so nothing would ever join
    // it and the descriptor would be closed by a loop the caller cannot observe.
    // `user` identifies this link to the hooks. Without it a hook can tell that something arrived but
    // not on which link, so nothing that behaves differently per link -- a freeze on one path, a
    // throttle on one path -- can be expressed at all. `Hooks::user` remains the fallback for callers
    // that do not care, which is the plugin.
    bool serve(int fd, void* user = nullptr) {
        ignore_sigpipe_once();
        std::lock_guard<std::mutex> g(mu_);
        if (stopping_.load(std::memory_order_acquire))
            return false;
        fds_.push_back(fd);
        conns_.push_back(std::make_unique<Conn>());
        Conn* c = conns_.back().get();
        c->fd = fd;
        c->user = user;
        threads_.emplace_back([this, fd, c] { loop(fd, c); });
        return true;
    }

    // How many connections this agent has been given. A mesh that came up half-formed is otherwise
    // indistinguishable from a peer that never sent anything.
    size_t connections() const {
        std::lock_guard<std::mutex> g(mu_);
        return fds_.size();
    }

    // Waits for every connection to end on its own, which happens when the peer sends Shutdown or
    // closes. This is the normal path: the sender decides when it is done.
    void join() {
        std::vector<std::thread> ts;
        {
            std::lock_guard<std::mutex> g(mu_);
            ts.swap(threads_);
        }
        for (auto& t : ts)
            if (t.joinable()) t.join();
    }

    // Tears down without waiting for the peer. The flag alone is not enough because a thread can be
    // blocked in recv, so the sockets are shut down to unblock it first.
    void stop() {
        stopping_.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> g(mu_);
            for (const int fd : fds_)
                ::shutdown(fd, SHUT_RDWR);
        }
        join();
    }

    // Per-connection counts. Which link stopped is not derivable from the totals, and "one peer went
    // quiet" and "every peer went quiet" have different causes.
    struct ConnectionState {
        int fd = -1;
        uint64_t placements = 0;
        uint64_t signals = 0;
        uint64_t bytes = 0;
        bool     finished = false;
    };

    std::vector<ConnectionState> connection_states() const {
        std::lock_guard<std::mutex> g(mu_);
        std::vector<ConnectionState> out;
        out.reserve(conns_.size());
        for (const auto& c : conns_)
            out.push_back({c->fd, c->placements.load(std::memory_order_acquire),
                           c->signals.load(std::memory_order_acquire),
                           c->bytes.load(std::memory_order_acquire),
                           c->finished.load(std::memory_order_acquire)});
        return out;
    }

    // The counts of applied writes are released by the receive threads and acquired here, so a caller that
    // sees a count also sees the host-window bytes it counts. For a device window a count means only that
    // the copy hook returned.
    uint64_t applied_placements() const { return placements_.load(std::memory_order_acquire); }
    uint64_t applied_signals() const { return signals_.load(std::memory_order_acquire); }

    // Two kinds of failure with two different owners. A frame this agent declined is the peer's
    // mistake; a write that did not happen is local. Reporting one number for both means a run where
    // every CUDA copy failed reads the same as a run where the peer sent nonsense, and the two need
    // different people to look at them.
    // Per-window counts. "1929 placements applied" is true and useless when the question is whether the
    // one window a waiter polls received anything: an aggregate cannot distinguish a window that got
    // everything from a window that got nothing while its neighbour got everything.
    struct WindowActivity {
        uint64_t placements = 0;
        uint64_t signals = 0;
        uint64_t bytes = 0;
        uint64_t min_offset = ~uint64_t(0);
        uint64_t max_end = 0;
        // How much was added at each exact offset, for signals only. A total says that signals
        // arrived; it cannot say that the one counter a stalled channel is watching was among them.
        // NCCL lays indexed signals out as `(signalId + contextId * nSignalsPerContext) * 8`, so an
        // offset here names one (context, signal) pair and nothing else.
        std::map<uint64_t, uint64_t> added_at;
    };

    WindowActivity activity(uint32_t window_id) const {
        std::lock_guard<std::mutex> g(activity_mu_);
        const auto it = activity_.find(window_id);
        return it == activity_.end() ? WindowActivity{} : it->second;
    }

    std::vector<uint32_t> windows_touched() const {
        std::lock_guard<std::mutex> g(activity_mu_);
        std::vector<uint32_t> out;
        out.reserve(activity_.size());
        for (const auto& kv : activity_) out.push_back(kv.first);
        return out;
    }

    uint64_t refused_frames() const { return refused_.load(std::memory_order_relaxed); }
    uint64_t failed_writes() const { return failed_writes_.load(std::memory_order_relaxed); }
    uint64_t get_requests_answered() const { return get_requests_.load(std::memory_order_acquire); }
    uint64_t errors() const { return refused_frames() + failed_writes(); }

    // Whether this agent is still alive, and if it is stuck, where. Every field is a duration rather
    // than a count, because the question a stall poses is never "how many" but "how long ago", and a
    // count cannot answer it.
    struct Liveness {
        uint64_t since_last_header_ms = ~uint64_t(0);   // never, when no frame has ever arrived
        uint64_t inside_device_op_ms = 0;               // 0 when no thread is inside one
        uint64_t device_op_bytes = 0;                   // what that thread is applying
        uint64_t device_ops_entered = 0;
        uint64_t device_ops_left = 0;
    };
    Liveness liveness() const {
        Liveness l{};
        l.since_last_header_ms = age_ms(last_header_at_.load(std::memory_order_relaxed));
        const uint64_t entered = copy_started_at_.load(std::memory_order_relaxed);
        // Read the stamp before the count so a copy that finishes between the two reads is reported as
        // finished rather than as one that started at an unknown time.
        if (entered != 0 and copies_inside_.load(std::memory_order_relaxed) != 0) {
            l.inside_device_op_ms = age_ms(entered, 0);
            l.device_op_bytes = copy_bytes_.load(std::memory_order_relaxed);
        }
        l.device_ops_entered = copies_entered_.load(std::memory_order_relaxed);
        l.device_ops_left = copies_left_.load(std::memory_order_relaxed);
        return l;
    }

private:
    // Stamps the window during which a thread of this agent is inside a device hook. Entered and left
    // are counted separately rather than as one balance, because "entered 4102, left 4101" names the
    // stuck call while a balance of 1 does not say whether it is the first or the four thousandth.
    struct InsideCopy {
        ReceiveAgent* a;
        InsideCopy(ReceiveAgent* agent, uint64_t bytes) : a(agent) {
            a->copies_entered_.fetch_add(1, std::memory_order_relaxed);
            a->copy_bytes_.store(bytes, std::memory_order_relaxed);
            if (a->copies_inside_.fetch_add(1, std::memory_order_acq_rel) == 0)
                a->copy_started_at_.store(mono_ms(), std::memory_order_relaxed);
        }
        ~InsideCopy() {
            a->copies_left_.fetch_add(1, std::memory_order_relaxed);
            if (a->copies_inside_.fetch_sub(1, std::memory_order_acq_rel) == 1)
                a->copy_started_at_.store(0, std::memory_order_relaxed);
        }
    };
    std::atomic<uint64_t> last_header_at_{0};
    std::atomic<uint64_t> copy_started_at_{0};
    std::atomic<uint64_t> copy_bytes_{0};
    std::atomic<uint64_t> copies_inside_{0};
    std::atomic<uint64_t> copies_entered_{0};
    std::atomic<uint64_t> copies_left_{0};
    void say(const char* msg) {
        if (hooks_.log) hooks_.log(msg);
        else std::fprintf(stderr, "[srdsim/agent] %s\n", msg);
    }

    // One per connection, so a report can say which link went quiet rather than only that the total
    // stopped moving.
    struct Conn {
        int fd = -1;
        void* user = nullptr;   // this link's identity, as the caller chose to name it
        std::atomic<uint64_t> placements{0};
        std::atomic<uint64_t> signals{0};
        std::atomic<uint64_t> bytes{0};
        std::atomic<bool> finished{false};
    };

    void loop(int fd, Conn* conn) {
        std::vector<char> buf;
        // The exit condition is the peer closing or sending Shutdown. `stopping_` is only an abort
        // for teardown, checked after each message rather than before the first one, so that a
        // caller which tears down immediately after serving still processes what already arrived.
        for (;;) {
            WireHeader h{};
            if (hooks_.idle != nullptr) {
                pollfd pfd{fd, POLLIN, 0};
                if (::poll(&pfd, 1, 0) == 0) hooks_.idle(conn->user ? conn->user : hooks_.user);
            }
            if (not read_all(fd, &h, sizeof(h)))
                break;                                  // peer closed or failed
            // Dated on arrival, not on application. The two differ exactly when applying is what
            // stopped, which is the case this mark exists to name: a reader still taking frames off the
            // wire but parked in the device copy looks, from the applied counters alone, identical to a
            // reader that has stopped receiving.
            last_header_at_.store(mono_ms(), std::memory_order_relaxed);
            if (h.kind == static_cast<uint32_t>(WireKind::Shutdown))
                break;

            Window w{};
            const bool known_window = windows_->get(h.window_id, &w);

            // The length decides how much of the stream belongs to this frame, so it has to be
            // checked before it is used for anything at all -- including for deciding how much to
            // drain. A length no window could hold cannot be trusted to drain either, so the only
            // safe action is to stop reading this stream. A desynchronised stream can produce such a
            // length.
            // The bound is the largest frame this side can *emit*, not the largest window that exists.
            // Those can differ by orders of magnitude: a window may be gigabytes, so a garbage length of
            // hundreds of megabytes would pass a window bound, and the agent would allocate it and wait
            // forever for bytes that never come. The scheduler never emits a placement larger than
            // `chunk_bytes`, and `chunk_bytes` does not come from the wire, so it is the honest bound: a
            // desynchronised stream produces arbitrary lengths, and only lengths above the bound get caught.
            const uint64_t max_frame = max_frame_bytes_;
            if (h.length > max_frame) {
                char m[220];
                std::snprintf(m, sizeof(m),
                              "frame length %llu exceeds the largest frame this transport emits "
                              "(%llu); the stream cannot be resynchronised, closing the link",
                              (unsigned long long)h.length, (unsigned long long)max_frame);
                say(m);
                refused_.fetch_add(1, std::memory_order_relaxed);
                break;
            }

            // From here the length is bounded by a real window, so draining it is safe and the stream
            // can be kept framed across a refusal.
            const auto drain_payload = [&]() -> bool {
                if (h.length == 0) return true;
                buf.resize(static_cast<size_t>(h.length));
                return read_all(fd, buf.data(), static_cast<size_t>(h.length));
            };

            if (not known_window) {
                char m[160];
                std::snprintf(m, sizeof(m), "unknown window id %u (have %zu); dropping",
                              h.window_id, windows_->size());
                say(m);
                refused_.fetch_add(1, std::memory_order_relaxed);
                if (not drain_payload()) break;
                continue;
            }

            // A GetResponse lands exactly like a Placement -- same bounds check, same device/host
            // copy -- and only needs one more step once it has: see the trailing call below.
            if (h.kind == static_cast<uint32_t>(WireKind::Placement) or
                h.kind == static_cast<uint32_t>(WireKind::GetResponse)) {
                if (not range_fits(h.offset, h.length, w.bytes)) {
                    char m[220];
                    std::snprintf(m, sizeof(m),
                                  "placement out of range: window %u offset %llu length %llu size %zu",
                                  h.window_id, (unsigned long long)h.offset,
                                  (unsigned long long)h.length, w.bytes);
                    say(m);
                    refused_.fetch_add(1, std::memory_order_relaxed);
                    if (not drain_payload()) break;
                    continue;
                }
                if (not drain_payload()) break;
                // Held from here until the write has been issued. Taken after the payload is off the
                // socket, and the window re-read under it, because it may have been retired meanwhile.
                WindowTable::Lease lease(windows_, h.window_id);
                if (not lease.ok() or not range_fits(h.offset, h.length, lease.window().bytes)) {
                    char m[160];
                    std::snprintf(m, sizeof(m), "window %u was deregistered while a placement for it was "
                                  "being received; dropping", h.window_id);
                    say(m);
                    refused_.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                w = lease.window();
                char* dst = static_cast<char*>(w.base) + h.offset;
                bool ok = true;
                if (w.is_device) {
                    // Stamped across the hook because the hook synchronises the stream, and a
                    // synchronise that never returns is the one failure this whole class of design is
                    // exposed to: the device is occupied by resident kernels that are themselves
                    // waiting on what this copy would deliver. Without the stamp that state is
                    // indistinguishable from an idle agent.
                    InsideCopy inside(this, h.length);
                    ok = hooks_.device_copy and
                         hooks_.device_copy(dst, buf.data(), static_cast<size_t>(h.length),
                                            conn->user ? conn->user : hooks_.user);
                    if (not hooks_.device_copy)
                        say("a device window arrived but no device copy hook was installed");
                } else {
                    std::memcpy(dst, buf.data(), static_cast<size_t>(h.length));
                }
                // Counted only when the write actually happened. Counting attempts as applications
                // would make a run where every write failed look identical to a healthy one, and the
                // whole point of these counters is to tell "never sent" from "never landed".
                if (ok) { placements_.fetch_add(1, std::memory_order_release);
                          if (conn) { conn->placements.fetch_add(1, std::memory_order_release);
                                      conn->bytes.fetch_add(h.length, std::memory_order_release); }
                          note(h.window_id, false, h.offset, h.length);
                          // The one way a GetResponse differs from a Placement: landing it also has to
                          // tell whichever get() is waiting that its data is actually here now, not
                          // merely that it was sent -- the distinction `iflush`/`flushAsync` exist for.
                          if (h.kind == static_cast<uint32_t>(WireKind::GetResponse) and
                              hooks_.get_response_landed)
                              hooks_.get_response_landed(h.op_id, h.length, conn->user ? conn->user : hooks_.user); }
                else    failed_writes_.fetch_add(1, std::memory_order_relaxed);
            } else if (h.kind == static_cast<uint32_t>(WireKind::GetRequest)) {
                // No payload on the wire -- the requester does not have the bytes yet -- so the fetch
                // size travels in `value`, not `length`; `length` stays "bytes trailing this frame" for
                // every kind, which is the one thing the generic bound check above relies on.
                if (h.length != 0) {
                    char m[200];
                    std::snprintf(m, sizeof(m),
                                  "a get request claims a payload of %llu byte(s); the stream is "
                                  "desynchronised, closing the link",
                                  (unsigned long long)h.length);
                    say(m);
                    refused_.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
                const uint64_t fetch_bytes = h.value;
                if (not range_fits(h.offset, fetch_bytes, w.bytes)) {
                    char m[220];
                    std::snprintf(m, sizeof(m),
                                  "get request out of range: window %u offset %llu length %llu size %zu",
                                  h.window_id, (unsigned long long)h.offset,
                                  (unsigned long long)fetch_bytes, w.bytes);
                    say(m);
                    refused_.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                // Re-read under a lease, same as a Placement does, because the window this rank is
                // about to read could have been deregistered between the unlocked peek above and now.
                WindowTable::Lease lease(windows_, h.window_id);
                if (not lease.ok() or not range_fits(h.offset, fetch_bytes, lease.window().bytes)) {
                    char m[160];
                    std::snprintf(m, sizeof(m), "window %u was deregistered while a get request for it "
                                  "was being served; dropping", h.window_id);
                    say(m);
                    refused_.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                w = lease.window();
                const void* src = static_cast<const char*>(w.base) + h.offset;
                const bool ok = hooks_.get_request and
                                hooks_.get_request(src, w.is_device, static_cast<size_t>(fetch_bytes),
                                                   h.dest_window_id, h.dest_offset, h.requester_rank,
                                                   h.op_id, conn->user ? conn->user : hooks_.user);
                if (not hooks_.get_request)
                    say("a get request arrived but no get-request hook was installed");
                if (ok) get_requests_.fetch_add(1, std::memory_order_release);
                else    failed_writes_.fetch_add(1, std::memory_order_relaxed);
            } else if (h.kind == static_cast<uint32_t>(WireKind::Signal)) {
                // A signal frame has no payload, so a non-zero length means the stream is no longer
                // framed and the next header would be read out of somebody's data. There is no
                // resynchronising from that, so the link is closed and the refusal counted.
                // `tests/model_srdsim_framing.cpp` does not enumerate this case.
                if (h.length != 0) {
                    char m[200];
                    std::snprintf(m, sizeof(m),
                                  "a signal frame claims a payload of %llu byte(s); the stream is "
                                  "desynchronised, closing the link",
                                  (unsigned long long)h.length);
                    say(m);
                    refused_.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
                if (not range_fits(h.offset, sizeof(uint64_t), w.bytes)) {
                    char m[200];
                    std::snprintf(m, sizeof(m),
                                  "signal out of range: window %u offset %llu size %zu; dropping",
                                  h.window_id, (unsigned long long)h.offset, w.bytes);
                    say(m);
                    refused_.fetch_add(1, std::memory_order_relaxed);
                    continue;                            // a signal carries no payload to drain
                }
                WindowTable::Lease lease(windows_, h.window_id);
                if (not lease.ok() or not range_fits(h.offset, sizeof(uint64_t), lease.window().bytes)) {
                    char m[160];
                    std::snprintf(m, sizeof(m), "window %u was deregistered before a signal for it was "
                                  "applied; dropping", h.window_id);
                    say(m);
                    refused_.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                w = lease.window();
                void* dst = static_cast<char*>(w.base) + h.offset;
                bool ok = true;
                if (w.is_device) {
                    InsideCopy inside(this, sizeof(uint64_t));
                    ok = hooks_.device_add and
                         hooks_.device_add(dst, h.value, conn->user ? conn->user : hooks_.user);
                    if (not hooks_.device_add)
                        say("a device signal arrived but no device add hook was installed");
                } else {
                    // Release, so that a reader which acquire-loads the word sees every byte this
                    // thread placed before it. A relaxed add would leave a correct reader racing
                    // with those bytes; a reader that loads the word without acquire is still wrong,
                    // and is still reported as such.
                    __atomic_fetch_add(static_cast<uint64_t*>(dst), h.value, __ATOMIC_RELEASE);
                }
                if (ok) { signals_.fetch_add(1, std::memory_order_release);
                          if (conn) conn->signals.fetch_add(1, std::memory_order_release);
                          note(h.window_id, true, h.offset, sizeof(uint64_t), h.value); }
                else    failed_writes_.fetch_add(1, std::memory_order_relaxed);
            } else {
                say("unknown wire kind; closing the connection");
                refused_.fetch_add(1, std::memory_order_relaxed);
                break;
            }
            if (stopping_.load(std::memory_order_acquire))
                break;
        }
        if (conn) conn->finished.store(true, std::memory_order_release);
        // Out of the table before it is closed. Without this, `fds_` keeps a number the loop has already
        // closed, and a later `stop()` calls `::shutdown` on whatever the operating system has since
        // handed that number to, and an unrelated socket loses data. The close and the table entry have
        // to be owned by the same code.
        forget(fd);
        ::close(fd);
    }

    void forget(int fd) {
        std::lock_guard<std::mutex> g(mu_);
        for (auto& f : fds_)
            if (f == fd) f = -1;
    }

    WindowTable* windows_;
    Hooks hooks_;
    // A generous default: larger than any chunk size this project uses, small enough that a garbage
    // length is caught rather than allocated.
    uint64_t max_frame_bytes_ = 1u << 22;
    mutable std::mutex mu_;
    std::vector<std::thread> threads_;
    std::vector<int> fds_;
    std::vector<std::unique_ptr<Conn>> conns_;
    std::atomic<bool> stopping_{false};
    std::atomic<uint64_t> placements_{0};
    std::atomic<uint64_t> signals_{0};
    void note(uint32_t window_id, bool is_signal, uint64_t offset, uint64_t length,
              uint64_t added = 0) {
        std::lock_guard<std::mutex> g(activity_mu_);
        WindowActivity& a = activity_[window_id];
        if (is_signal) ++ a.signals; else ++ a.placements;
        a.bytes += length;
        if (offset < a.min_offset) a.min_offset = offset;
        if (offset + length > a.max_end) a.max_end = offset + length;
        if (is_signal) a.added_at[offset] += added;
    }

    mutable std::mutex activity_mu_;
    std::map<uint32_t, WindowActivity> activity_;
    std::atomic<uint64_t> refused_{0};
    std::atomic<uint64_t> failed_writes_{0};
    std::atomic<uint64_t> get_requests_{0};
};

// ---------------------------------------------------------------------------
// The sending side
// ---------------------------------------------------------------------------

// One stream per peer. Writes are serialised per peer so that the order the scheduler chose is the
// order that reaches the peer; that serialisation is the whole reason the simulator's control over
// ordering survives the wire.
// Set on a thread that appends several frames and then pushes every outbox once (`SendLinks::DeferDrain`).
inline thread_local bool t_defer_drain = false;

class SendLinks {
public:
    // Optional breadcrumb sink. The transport can block in exactly two places -- waiting for a peer's
    // send lock, and pushing bytes -- and telling them apart is the difference between "the socket is
    // full" and "another thread is holding the lock". A stall report that says only "in the transport"
    // sends the reader to the wrong half.
    using WhereFn = void (*)(const char*);
    void set_where_hook(WhereFn fn) { where_ = fn; }
    // Binds a descriptor to a peer, closing whatever was bound before. Leaking the previous one costs
    // a descriptor per rebind and eventually the process's whole table, and the leak is invisible
    // because everything keeps working until it suddenly does not.
    // Discards whatever a peer's outbox still holds, and says how much. Called when a link is replaced or
    // killed: a frame is only meaningful whole, so carrying a partial one into a new connection puts the
    // remainder of it at the head of the new stream and desynchronises the receiver immediately.
    void discard_outbox_locked(int peer) {
        const size_t i = static_cast<size_t>(peer);
        if (i >= outboxes_.size()) return;
        const size_t left = outboxes_[i].size();
        if (left == 0) return;
        outboxes_[i].bytes.clear();
        outboxes_[i].sent = 0;
        discarded_bytes_ += left;
        ++ discarded_links_;
    }

    void set(int peer, int fd) {
        // Every entry point that takes ownership of a descriptor, not only the ones that create one. The
        // comment on `ignore_sigpipe_once` claimed this already and it was not true: a descriptor from
        // `socketpair` handed straight to `set()` left SIGPIPE live, and a peer closing first killed the
        // process instead of failing the write. The plugin happened to survive because `connect` runs
        // `listen_on` first; a test that skipped that step died on the spot.
        ignore_sigpipe_once();
        int previous = -1;
        {
            std::lock_guard<std::mutex> g(mu_);
            if (static_cast<size_t>(peer) >= fds_.size()) {
                fds_.resize(static_cast<size_t>(peer) + 1, -1);
                locks_.resize(static_cast<size_t>(peer) + 1);
            }
            previous = fds_[static_cast<size_t>(peer)];
            fds_[static_cast<size_t>(peer)] = fd;
            if (outboxes_.size() < fds_.size()) outboxes_.resize(fds_.size());
            // A frame is only meaningful whole. Anything left over from the previous descriptor would
            // arrive as the head of the new stream.
            if (previous != fd) discard_outbox_locked(peer);
            // Non-blocking, because every send from here has to be able to return without having sent
            // anything. A blocking descriptor is what let a progress callback stop returning.
            const int flags = ::fcntl(fd, F_GETFL, 0);
            if (flags >= 0) ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        }
        // Closed outside the table lock, and only if it is not the same descriptor being rebound to
        // itself, which would close a descriptor the caller still expects to own.
        if (previous >= 0 and previous != fd)
            ::close(previous);
    }

    // `kind` defaults to Placement; a get's reply is sent through here too, tagged GetResponse instead,
    // since the two are identical on the wire apart from that one byte -- same window/offset/length
    // addressing, same framing.
    bool send_placement(int peer, uint32_t window_id, uint64_t offset, const void* src,
                        uint64_t length, uint64_t op_id, WireKind kind = WireKind::Placement, bool reserved = false) {
        WireHeader h{};
        h.kind = static_cast<uint32_t>(kind);
        h.window_id = window_id;
        h.offset = offset;
        h.length = length;
        h.op_id = op_id;
        return send_framed(peer, h, src, length, reserved);
    }

    // Pushes whatever is queued for a peer, as far as the socket will take it, and returns. Called from
    // the progress loop so that backpressure is drained by the caller that is allowed to be slow.
    bool flush(int peer) {
        std::mutex* lock = nullptr;
        {
            std::lock_guard<std::mutex> g(mu_);
            if (static_cast<size_t>(peer) >= fds_.size()) return true;
            lock = &locks_[static_cast<size_t>(peer)];
        }
        std::lock_guard<std::mutex> g(*lock);
        return drain_locked(peer);
    }

    // Holds room in the outbox for `peer` for frames of `bytes` in all (headers included) that the caller will
    // send with `reserved`, or returns false when it would not fit beside what is queued and reserved. A link that
    // is down grants it, so that the send fails as a link failure rather than waiting for room forever.
    bool reserve(int peer, size_t bytes) {
        std::lock_guard<std::mutex> g(mu_);
        if (static_cast<size_t>(peer) >= fds_.size() or fds_[static_cast<size_t>(peer)] < 0) return true;
        if (outboxes_.size() < fds_.size()) outboxes_.resize(fds_.size());
        if (reserved_.size() < fds_.size()) reserved_.resize(fds_.size(), 0);
        auto& r = reserved_[static_cast<size_t>(peer)];
        const size_t used = outboxes_[static_cast<size_t>(peer)].size() + r;
        if (used != 0 and bytes > outbox_budget_ - std::min(used, outbox_budget_)) return false;
        r += bytes;
        return true;
    }
    // Room reserved and not yet used, over every peer: zero whenever nothing released is waiting to be sent.
    size_t reserved_total() {
        std::lock_guard<std::mutex> g(mu_);
        size_t t = 0;
        for (size_t r : reserved_) t += r;
        return t;
    }
    // Returns room reserved and not used.
    void unreserve(int peer, size_t bytes) {
        std::lock_guard<std::mutex> g(mu_);
        if (static_cast<size_t>(peer) >= reserved_.size()) return;
        auto& r = reserved_[static_cast<size_t>(peer)];
        r -= std::min(r, bytes);
    }

    // Whether nothing is waiting in the outbox for `peer`: everything appended has reached the socket.
    bool outbox_empty(int peer) {
        std::lock_guard<std::mutex> g(mu_);
        if (static_cast<size_t>(peer) >= outboxes_.size()) return true;
        return outboxes_[static_cast<size_t>(peer)].size() == 0;
    }

    bool flush_all() {
        size_t n;
        {
            std::lock_guard<std::mutex> g(mu_);
            n = fds_.size();
        }
        bool ok = true;
        for (size_t i = 0; i < n; ++ i)
            ok = flush(static_cast<int>(i)) and ok;
        return ok;
    }

    // How many bytes are waiting for a peer. Reported rather than hidden: a queue that only grows is
    // the visible form of a receiver that has stopped reading, and it is the number to look at when a
    // run slows down instead of stopping.
    // How much one peer may have waiting before releases are refused. Generous relative to a chunk (the
    // scheduler releases 8 kB units) and small relative to the machine, so a healthy run never meets it
    // and a stuck peer meets it long before the allocator does.
    void set_outbox_budget(size_t bytes) { if (bytes != 0) outbox_budget_ = bytes; }
    uint64_t refused_for_budget() const {
        return refused_for_budget_.load(std::memory_order_relaxed);
    }

    size_t pending_bytes(int peer) const {
        std::lock_guard<std::mutex> g(mu_);
        if (static_cast<size_t>(peer) >= outboxes_.size()) return 0;
        return outboxes_[static_cast<size_t>(peer)].size();
    }

    size_t peers() const {
        std::lock_guard<std::mutex> g(mu_);
        return fds_.size();
    }

    // Per-peer send state. Summing the queues across peers hides the only thing worth knowing when a
    // run wedges: whether one link is stuck or all of them are. A total of three megabytes queued reads
    // the same whether it is one peer refusing everything or every peer slightly behind.
    struct PeerSendState {
        int fd = -1;
        size_t queued = 0;
        uint64_t frames_queued = 0;
        uint64_t bytes_sent = 0;
        uint64_t would_block = 0;   // times the socket refused bytes
    };

    // What teardown could not deliver. Zero on every healthy run; non-zero is a data loss this transport
    // is obliged to admit to.
    struct Discarded { uint64_t bytes = 0; uint64_t links = 0; };
    Discarded discarded() const {
        std::lock_guard<std::mutex> g(mu_);
        return {discarded_bytes_, discarded_links_};
    }

    void set_shutdown_budget_ms(int64_t ms) {
        std::lock_guard<std::mutex> g(mu_);
        if (ms >= 0) shutdown_budget_ms_ = ms;
    }

    PeerSendState send_state(int peer) const {
        std::lock_guard<std::mutex> g(mu_);
        PeerSendState st;
        const size_t i = static_cast<size_t>(peer);
        if (i < fds_.size()) st.fd = fds_[i];
        if (i < outboxes_.size()) {
            st.queued = outboxes_[i].size();
            st.frames_queued = outboxes_[i].frames;
            st.bytes_sent = outboxes_[i].pushed;
            st.would_block = outboxes_[i].blocked;
        }
        return st;
    }

    bool send_signal(int peer, uint32_t window_id, uint64_t offset, uint64_t value, uint64_t op_id,
                     bool reserved = false) {
        WireHeader h{};
        h.kind = static_cast<uint32_t>(WireKind::Signal);
        h.window_id = window_id;
        h.offset = offset;
        h.length = 0;
        h.value = value;
        h.op_id = op_id;
        return send_framed(peer, h, nullptr, 0, reserved);
    }

    // No payload: the requester does not have the bytes yet. `src_window_id`/`src_offset`/`fetch_bytes`
    // name what to read on the peer; `dest_window_id`/`dest_offset` name where, in the requester's own
    // numbering, the reply belongs; `requester_rank` is this rank's own id, since the peer otherwise has
    // no way to learn who asked (see the comment on `serve()`); `owner_token` is an opaque value the
    // requester chose, to be echoed back unchanged in the GetResponse.
    bool send_get_request(int peer, uint32_t src_window_id, uint64_t src_offset, uint64_t fetch_bytes,
                          uint32_t dest_window_id, uint64_t dest_offset, uint32_t requester_rank,
                          uint64_t owner_token, bool reserved = false) {
        WireHeader h{};
        h.kind = static_cast<uint32_t>(WireKind::GetRequest);
        h.window_id = src_window_id;
        h.offset = src_offset;
        h.length = 0;
        h.value = fetch_bytes;
        h.op_id = owner_token;
        h.dest_window_id = dest_window_id;
        h.dest_offset = dest_offset;
        h.requester_rank = requester_rank;
        return send_framed(peer, h, nullptr, 0, reserved);
    }

    // Sends a Shutdown to every peer and closes the link.
    //
    // Each peer is handled while holding *that peer's send lock*, for two reasons that both matter.
    // Writing the Shutdown frame without it can interleave with a sender mid-frame, so the peer reads
    // a Shutdown header out of the middle of a payload -- the same desynchronisation this file exists
    // to make impossible. And closing without it closes a descriptor another thread is writing to; the
    // number is then reusable, so a later write from that thread can land on an unrelated object.
    //
    // The table lock is taken and released per peer rather than held across the whole loop, because
    // the send lock has to be taken while the table lock is not held: a sender takes the table lock
    // first and the send lock second, and taking them in the other order here would deadlock.
    void shutdown_all() {
        size_t n;
        {
            std::lock_guard<std::mutex> g(mu_);
            n = fds_.size();
        }
        for (size_t i = 0; i < n; ++ i) {
            std::mutex* lock = nullptr;
            {
                std::lock_guard<std::mutex> g(mu_);
                if (i >= fds_.size() or fds_[i] < 0) continue;
                lock = &locks_[i];
            }
            std::lock_guard<std::mutex> sending(*lock);
            // Everything already accepted goes out before the link closes, or this side says how much it
            // dropped. The bound is time, not iterations: `drain_locked` returns immediately on EAGAIN, so
            // an iteration bound expires in under a millisecond and a peer that reads slowly would lose
            // the rest of its queue silently. The loop stops early only when it stops making progress.
            {
                const auto started = std::chrono::steady_clock::now();
                size_t last = pending_bytes(static_cast<int>(i));
                while (last > 0) {
                    if (not drain_locked(static_cast<int>(i))) break;
                    const size_t now_pending = pending_bytes(static_cast<int>(i));
                    if (now_pending == 0) break;
                    if (now_pending == last) {
                        const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - started).count();
                        if (waited >= shutdown_budget_ms_) break;
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                    last = now_pending;
                }
                const size_t dropped = pending_bytes(static_cast<int>(i));
                if (dropped > 0) {
                    std::lock_guard<std::mutex> g(mu_);
                    discarded_bytes_ += dropped;
                    ++ discarded_links_;
                }
            }
            int fd = -1;
            {
                std::lock_guard<std::mutex> g(mu_);
                if (i >= fds_.size()) continue;
                fd = fds_[i];
                fds_[i] = -1;               // claimed, so no later sender can pick it up
            }
            if (fd < 0) continue;
            WireHeader h{};
            h.kind = static_cast<uint32_t>(WireKind::Shutdown);
            write_all(fd, &h, sizeof(h));
            ::close(fd);
        }
    }

    int fd_for(int peer) const {
        std::lock_guard<std::mutex> g(mu_);
        if (static_cast<size_t>(peer) >= fds_.size()) return -1;
        return fds_[static_cast<size_t>(peer)];
    }

private:
    // Appends a whole frame to the peer's outbox and pushes as much of it as the socket will take.
    //
    // This never blocks, and that is the point. It is called from NCCL's progress callback, whose contract
    // is to make what progress it can and return. A blocking write to a peer that stopped reading would
    // not return, and NCCL's proxy would stop entirely, including the half of it that drains the other
    // direction.
    //
    // A frame is appended in one piece under the peer's lock, so the stream stays framed and ordered no
    // matter how many threads send or how often the socket refuses bytes.
    // `reserved`: the caller holds a reservation for this frame (`reserve`), which it consumes; such a frame is
    // never refused for budget. Any other frame is refused when it would not fit beside what is queued and what
    // is reserved.
    bool send_framed(int peer, const WireHeader& h, const void* payload, uint64_t length, bool reserved = false) {
        std::mutex* lock = nullptr;
        {
            std::lock_guard<std::mutex> g(mu_);
            if (static_cast<size_t>(peer) >= fds_.size()) return false;
            if (fds_[static_cast<size_t>(peer)] < 0) return false;
            lock = &locks_[static_cast<size_t>(peer)];
            if (outboxes_.size() < fds_.size()) outboxes_.resize(fds_.size());
            if (reserved_.size() < fds_.size()) reserved_.resize(fds_.size(), 0);
        }
        note("transport: waiting for the peer's send lock");
        std::lock_guard<std::mutex> g(*lock);
        note("transport: appending a frame to the outbox");
        {
            std::lock_guard<std::mutex> tg(mu_);
            auto& out = outboxes_[static_cast<size_t>(peer)];
            // Refused rather than queued once the peer is this far behind, and this is a bound the design
            // needs rather than a tidiness. The outbox exists because the socket may refuse bytes and the
            // caller must not block; with no limit, a peer that stops reading turns that into unbounded
            // memory and eventually `std::bad_alloc`. The adversarial suite creates the condition: 4 MB
            // frames into a 4 kB send buffer. With the bound, the outcome does not depend on how soon
            // the platform breaks the link.
            //
            // Refusing is the correct answer rather than dropping. A reliable transport may not lose a
            // frame it accepted, so the release is declined and the scheduler still holds the unit: the
            // caller retries, which is backpressure. Dropping would be silent data loss, which is the one
            // thing this emulator's calibration forbids outright.
            const size_t incoming = sizeof(h) + static_cast<size_t>(length);
            if (reserved) {
                auto& r = reserved_[static_cast<size_t>(peer)];
                r -= std::min(r, incoming);
            } else {
                const size_t queued = out.size() + reserved_[static_cast<size_t>(peer)];
                if (queued != 0 and incoming > outbox_budget_ - std::min(queued, outbox_budget_)) {
                    ++ refused_for_budget_;
                    return false;
                }
            }
            const auto* hb = reinterpret_cast<const char*>(&h);
            ++ out.frames;
            out.append(hb, sizeof(h));
            if (length > 0)
                out.append(static_cast<const char*>(payload), static_cast<size_t>(length));
        }
        if (t_defer_drain) return true;   // the caller pushes every outbox once it has appended its frames
        note("transport: pushing bytes to the socket");
        const bool ok = drain_locked(peer);
        note("transport: done");
        return ok;
    }

    // While one is alive, frames sent from this thread are appended to the outbox and not pushed; the caller
    // calls `flush_all` afterwards. The bytes on each stream are the same; only the number of system calls
    // changes, from one or more per frame to one or more per peer.
    struct DeferDrain {
        DeferDrain() { t_defer_drain = true; }
        ~DeferDrain() { t_defer_drain = false; }
        DeferDrain(const DeferDrain&) = delete;
        DeferDrain& operator=(const DeferDrain&) = delete;
    };

    // Called with the peer's send lock held. Pushes bytes until the socket refuses them.
    bool drain_locked(int peer) {
        for (;;) {
            int fd = -1;
            const char* p = nullptr;
            size_t n = 0;
            {
                std::lock_guard<std::mutex> g(mu_);
                if (static_cast<size_t>(peer) >= fds_.size()) return true;
                fd = fds_[static_cast<size_t>(peer)];
                if (static_cast<size_t>(peer) < outboxes_.size()) {
                    auto& out = outboxes_[static_cast<size_t>(peer)];
                    if (out.size() == 0) return true;
                    p = out.front();
                    n = out.size();
                }
            }
            if (fd < 0) return false;
            if (n == 0) return true;
            const ssize_t sent = ::send(fd, p, n, 0);
            if (sent > 0) {
                std::lock_guard<std::mutex> g(mu_);
                auto& ob = outboxes_[static_cast<size_t>(peer)];
                ob.pushed += static_cast<uint64_t>(sent);
                ob.consume(static_cast<size_t>(sent));
                continue;
            }
            if (sent < 0 and (errno == EAGAIN or errno == EWOULDBLOCK)) {
                std::lock_guard<std::mutex> g(mu_);
                if (static_cast<size_t>(peer) < outboxes_.size())
                    ++ outboxes_[static_cast<size_t>(peer)].blocked;
                return true;   // try again later
            }
            if (sent < 0 and errno == EINTR) continue;
            // A real failure. The outbox still holds a partial frame, so the stream cannot be
            // resynchronised and the link is closed rather than continued.
            kill_link(peer);
            return false;
        }
    }

    // Closes a link whose framing can no longer be trusted. Called with the peer's send lock held.
    void kill_link(int peer) {
        std::lock_guard<std::mutex> g(mu_);
        const size_t i = static_cast<size_t>(peer);
        if (i < fds_.size() and fds_[i] >= 0) {
            ::close(fds_[i]);
            fds_[i] = -1;
        }
        // The outbox holds a partial frame at this point -- that is why the link was killed. Keeping it
        // would hand the remainder to whatever descriptor takes this peer's slot next.
        discard_outbox_locked(peer);
    }

    void note(const char* where) const { if (where_) where_(where); }

    WhereFn where_ = nullptr;
    mutable std::mutex mu_;
    std::vector<int> fds_;
    // One outbox per peer, holding whole frames in the order they were submitted. A vector with a
    // consumed offset rather than a deque, because `send` needs a contiguous run of bytes and a deque
    // has none. The offset is compacted away once it dominates, so a long run does not keep bytes it
    // has already sent.
    // How long teardown will keep trying to push a peer's queue, and what it gave up on. Reported rather
    // than hidden: a transport that drops data on the way out has to say so, or the next reader of the
    // logs concludes that everything arrived.
    int64_t  shutdown_budget_ms_ = 2000;
    uint64_t discarded_bytes_ = 0;
    uint64_t discarded_links_ = 0;

    struct Outbox {
        std::vector<char> bytes;
        size_t sent = 0;
        uint64_t frames = 0;    // frames appended
        uint64_t pushed = 0;    // bytes the socket accepted
        uint64_t blocked = 0;   // times the socket refused
        size_t size() const { return bytes.size() - sent; }
        const char* front() const { return bytes.data() + sent; }
        void consume(size_t n) {
            sent += n;
            if (sent == bytes.size()) { bytes.clear(); sent = 0; }
            else if (sent > (1u << 20) and sent * 2 > bytes.size()) {
                bytes.erase(bytes.begin(), bytes.begin() + static_cast<long>(sent));
                sent = 0;
            }
        }
        void append(const char* p, size_t n) { bytes.insert(bytes.end(), p, p + n); }
    };
    std::vector<Outbox> outboxes_;
    size_t outbox_budget_ = 64u << 20;              // 64 MB per peer
    std::vector<size_t> reserved_;                  // per peer: room held for frames not yet appended (`reserve`)
    std::atomic<uint64_t> refused_for_budget_{0};
    std::deque<std::mutex> locks_;   // deque: a mutex is not movable, so the vector cannot grow it
};

}   // namespace srdsim

#endif   // SRDSIM_TRANSPORT_H_
