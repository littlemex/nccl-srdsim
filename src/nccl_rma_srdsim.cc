// SPDX-License-Identifier: Apache-2.0

// An NCCL RMA plugin that simulates EFA's SRD: reliable delivery with no ordering guarantee,
// with the one ordering property a real EFA-GDA backend does provide (a signal is observable
// only after every byte of the put that carried it has landed) either enforced or deliberately
// broken, under test control.
//
// WHY THIS IS THE RIGHT INTERPOSITION POINT
// NCCL's CPU-proxy GIN path routes the host data path through the RMA op-table
// (`ncclRmaPlugin_v14`). Indexed signals are not a separate message on that path: NCCL keeps the
// signal array in a registered window and turns a device-side signalling put into
// `iputSignal(..., signalOff, signalMhandle, ...)`. So a plugin sees, and therefore controls,
// both the placement of the payload and the moment the signal becomes visible. That is exactly
// the pair of freedoms an SRD simulator needs, and it needs no device code: the kernels are the
// real ones and run on real GPUs.
//
// WHAT IT CANNOT DO
// The EFA-GDA device path (WQE encoding, doorbell batching, the completion-counter wrap) is not
// exercised, because that path bypasses the host entirely. NCCL's internal proxy backend also
// reports `supportsStrongSignals = true` and `supportsVASignals = true` unconditionally, so this
// plugin cannot make the capability negotiation look like EFA's; it can only make the delivery
// behave like it. Neither limit matters for testing the unordered kernels, which set
// `ginStrongSignalsRequired = false` and `ginVaSignalsRequired = false` themselves, but both
// have to be stated so nobody reads a green run as EFA coverage.
//
// This file is the op table and bootstrap around the scheduler (`srdsim_schedule.h`) and the transport
// (`srdsim_transport.h`).
//
// Build. The plugin ABI headers (nccl_rma.h, nccl_net.h, nccl_common.h) are not shipped in the
// nvidia-nccl-cu13 wheel, so they come from an NCCL source checkout, assembled by
// `scripts/prepare_nccl_headers.sh`; then `cmake -S . -B build -DSRDSIM_BUILD_PLUGIN=ON
// -DNCCL_PLUGIN_INCLUDE_DIRS=<out>/nccl` (see README.md).
// Use: NCCL_RMA_PLUGIN=libnccl-rma-srdsim.so NCCL_GIN_TYPE=2, plus the SRDSIM_* settings (see README.md).

#include <csignal>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <random>
#include <unordered_map>
#include <vector>

#include <chrono>
#include <condition_variable>
#include <thread>

#include <execinfo.h>
#include <sys/socket.h>
#include <unistd.h>

#include "nccl_rma.h"

#include "srdsim_async.h"
#include "srdsim_mapping.h"
#include "srdsim_schedule.h"
#include "srdsim_transport.h"

#ifdef SRDSIM_WITH_CUDA
#include <cuda.h>
#include <nvrtc.h>
#endif

// GDRCopy is how the data path actually reaches device memory. It is separate from SRDSIM_WITH_CUDA
// because the two are needed for different things and a host without the gdrdrv module can still build
// and run everything else: CUDA is still used to register windows and to build the fallback kernel, while
// every byte of payload and every signal goes through a BAR mapping.
#ifdef SRDSIM_WITH_GDRCOPY
#include <gdrapi.h>
#endif

namespace {

// ---------------------------------------------------------------------------
// Logging and configuration
// ---------------------------------------------------------------------------

ncclDebugLogger_t g_log = nullptr;

bool log_ops();   // defined below; needed here so logf can honour it

// Routed through NCCL's logger when there is one, and always to stderr as well once observation is
// asked for. That second half is not redundant: NCCL's logger drops NCCL_LOG_INFO unless
// NCCL_DEBUG=INFO, so an operator who sets SRDSIM_LOG_OPS=1 and reads nothing would conclude the
// plugin is never called, when in fact the messages were filtered two layers away. Asking to see the
// operations is an explicit request and should not depend on someone else's verbosity setting.
void logf(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (g_log)
        g_log(NCCL_LOG_INFO, NCCL_ALL, __FILE__, __LINE__, "%s", buf);
    if (g_log == nullptr or log_ops()) {
        std::fprintf(stderr, "[srdsim] %s\n", buf);
        std::fflush(stderr);
    }
}

// Registration, deregistration and teardown, always on stderr: they are rare, and whether a window's memory
// was deregistered before it was freed is the question a crash in the delivery path turns on.
void lifecycle(const char* fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "[srdsim] %s\n", buf);
    std::fflush(stderr);
}

std::string env_str(const char* name, const char* dflt) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string(dflt);
}

// Not for the data path. `getenv` walks `environ` without a lock, and the application changes its environment
// while NCCL's threads run -- DeepEP calls `setenv("NCCL_ELASTIC_BUFFER_REGISTER", ...)` when it allocates a
// buffer -- so a `getenv` on a proxy thread can read a table being reallocated and crash. Every read on a path
// that runs repeatedly is a function-local static, read once.
size_t env_size(const char* name, size_t dflt) {
    const char* v = std::getenv(name);
    if (not v) return dflt;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(v, &end, 10);
    return (end and *end == '\0' and parsed > 0) ? static_cast<size_t>(parsed) : dflt;
}

// Two switches make the plugin usable as a pure observer, to see whether an application's puts reach it
// at all:
//   SRDSIM_LOG_OPS=1  log every iput / iputSignal with its arguments
//   SRDSIM_DRY_RUN=1  accept operations and complete them immediately without moving any data
// A dry run cannot produce correct results; the point is to watch the call sequence without hanging NCCL.
bool log_ops() {
    static const bool v = env_size("SRDSIM_LOG_OPS", 0) != 0;
    return v;
}

bool dry_run() {
    static const bool v = env_size("SRDSIM_DRY_RUN", 0) != 0;
    return v;
}

// A crash inside a plugin loaded by dlopen, called from a CUDA proxy thread, inside a process that
// torch spawned, surfaces to the user as "process terminated with signal SIGSEGV" and nothing else.
// The frame that matters is three layers below anything the harness can see, and the boxes this runs
// on have no debugger installed. So the plugin prints its own backtrace: whatever is wrong, the
// operator gets a line to start from rather than a signal number.
//
// Chaining to the previous handler matters. Overwriting it would hide crashes that are not ours and
// turn a CUDA fault into an apparent plugin fault.
// The handler the host had before this plugin was loaded, so it can be run afterwards rather than
// discarded. Three slots because three signals are hooked, and a shared slot would send SIGABRT to
// whatever was registered for SIGSEGV.
struct PreviousHandler { int sig; void (*fn)(int); };
PreviousHandler g_previous[3] = {{SIGSEGV, SIG_DFL}, {SIGBUS, SIG_DFL}, {SIGABRT, SIG_DFL}};

void crash_handler(int sig) {
    void* frames[64];
    const int n = ::backtrace(frames, 64);
    std::fprintf(stderr, "\n[srdsim] caught signal %d; %d frames follow\n", sig, n);
    ::backtrace_symbols_fd(frames, n, 2);
    std::fflush(stderr);
    // Hand the signal back to whoever had it. In a host like vLLM that is Python's faulthandler, and
    // it prints the Python traceback, which is the half of the story this plugin cannot see; going
    // straight to SIG_DFL would throw that away.
    for (const auto& p : g_previous) {
        if (p.sig != sig) continue;
        if (p.fn != SIG_DFL and p.fn != SIG_ERR and p.fn != SIG_IGN) {
            ::signal(sig, p.fn);
            p.fn(sig);
        }
        break;
    }
    ::signal(sig, SIG_DFL);
    ::raise(sig);
}

// Off unless asked for. A library loaded into somebody else's process should not take that process's
// fatal signals by default: doing so turns the host's own crash report into this plugin's, and a
// plugin that appears in every backtrace looks like the cause of every crash. It is enabled by the
// same switch that turns on the plugin's other reporting.
void install_crash_handler_once() {
    static const bool done = [] {
        if (env_size("SRDSIM_NO_BACKTRACE", 0) != 0)
            return true;
        if (env_size("SRDSIM_BACKTRACE", 0) == 0 and not log_ops())
            return true;
        for (auto& p : g_previous)
            p.fn = ::signal(p.sig, crash_handler);
        return true;
    }();
    (void)done;
}

srdsim::Scheduler::Options scheduler_options() {
    srdsim::Scheduler::Options o;
    const std::string mode = env_str("SRDSIM_MODE", "srd");
    if (not srdsim::mode_from_string(mode, &o.mode)) {
        logf("unknown SRDSIM_MODE '%s'; falling back to srd", mode.c_str());
        o.mode = srdsim::Mode::Srd;
    }
    o.seed = env_size("SRDSIM_SEED", 1);
    o.chunk_bytes = env_size("SRDSIM_CHUNK_BYTES", 8192);
    o.trace = env_size("SRDSIM_TRACE", 0) != 0;
    o.detect_overlaps = env_size("SRDSIM_DETECT_OVERLAP", 0) != 0;
    const std::string kind = env_str("SRDSIM_DEFER_KIND", "any");
    using K = srdsim::Scheduler::Options::DeferKind;
    if (kind == "unsignaled") o.defer_kind = K::Unsignaled;
    else if (kind == "signaled") o.defer_kind = K::Signaled;
    else if (kind == "signal-only") o.defer_kind = K::SignalOnly;
    else if (kind != "any") logf("unknown SRDSIM_DEFER_KIND '%s'; holding the oldest of any kind", kind.c_str());
    return o;
}

// ---------------------------------------------------------------------------
// Where each thread last was
// ---------------------------------------------------------------------------

// A stall is only diagnosable if something can report during it, and the previous attempt could not:
// the report was printed from the progress loop, and the progress loop was the thing that had stopped
// returning. So the monitor is its own thread, and every thread that can block leaves a breadcrumb
// before it does.
//
// The breadcrumb is a plain pointer to a string literal, published with a relaxed atomic store. It
// costs a store on the data path and it is the difference between "it froze" and "it froze in
// write_all to peer 1".
class Checkpoints {
public:
    void set(const char* where) {
        auto& slot = mine();
        slot.where.store(where, std::memory_order_relaxed);
        // Dated, because the breadcrumb alone cannot distinguish the two states it is asked about. A
        // thread sitting at "rmaProgress" is either inside a call that will not return or between calls
        // and about to be given another; the sequence number does not separate those either, since a
        // stuck thread and an unused thread both stop advancing it. The moment the breadcrumb was left
        // does separate them, and it is one store.
        slot.at.store(srdsim::mono_ms(), std::memory_order_relaxed);
        slot.seq.fetch_add(1, std::memory_order_relaxed);
    }

    // Prints every thread that has ever left a breadcrumb, with a sequence number so a reader can tell
    // a thread that is stuck from one that is merely idle between calls, and with the age of the
    // breadcrumb so a reader can tell that without having to take two samples and compare them.
    void report(void (*out)(const char*, ...)) {
        std::lock_guard<std::mutex> g(mu_);
        for (const auto& kv : slots_) {
            const char* w = kv.second->where.load(std::memory_order_relaxed);
            const uint64_t age =
                srdsim::age_ms(kv.second->at.load(std::memory_order_relaxed), 0);
            out("  thread %llu: %s (%llu visit(s), last moved %llu ms ago)",
                static_cast<unsigned long long>(kv.first),
                w ? w : "(nowhere yet)",
                static_cast<unsigned long long>(kv.second->seq.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(age));
        }
    }

private:
    struct Slot {
        std::atomic<const char*> where{nullptr};
        std::atomic<uint64_t> seq{0};
        std::atomic<uint64_t> at{0};                    // when the breadcrumb was last moved
    };

    Slot& mine() {
        static thread_local Slot* cached = nullptr;
        if (cached) return *cached;
        auto slot = std::make_unique<Slot>();
        cached = slot.get();
        const auto id = static_cast<uint64_t>(
            std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xffffull);
        std::lock_guard<std::mutex> g(mu_);
        slots_[id] = std::move(slot);
        return *cached;
    }

    std::mutex mu_;
    std::map<uint64_t, std::unique_ptr<Slot>> slots_;
};

// Never destroyed, like every global that NCCL's threads or the plugin's own touch on the data path. A
// process that exits without destroying its communicators runs static destructors while those threads still
// run, so a destroyed `g_where` would be written by NCCL's GIN proxy thread after `exit` had freed it (a
// heap-use-after-free under ASan; otherwise glibc's `malloc_consolidate(): unaligned fastbin chunk detected`).
Checkpoints& g_where = *new Checkpoints;

// Marks a scope, so a thread's breadcrumb says what it is inside rather than what it last finished.
struct Mark {
    explicit Mark(const char* where) { g_where.set(where); }
    ~Mark() { g_where.set("(returned)"); }
};

// ---------------------------------------------------------------------------
// Symmetric memory
//
// Exporting a cuMem allocation across a process boundary works, but giving a *different* GPU access to
// it needs peer access, which is not always available and which cannot exist across hosts at all.
//
// So this plugin does not reach into the peer's window. `regMrSym` is collective, so numbering the
// windows in registration order gives both sides a name they agree on, and the receiving side
// resolves the name to its own base address. A put then travels as (window id, offset, bytes) and
// the peer's agent writes it locally. No peer access, no exported allocation, and the same code
// path within a host and across hosts.
// ---------------------------------------------------------------------------

struct MrHandle {
    void*    local_base = nullptr;
    size_t   bytes = 0;
    int      ptr_type = 0;                // NCCL_PTR_HOST or NCCL_PTR_CUDA
    uint32_t window_id = 0;               // the name both sides agree on
    uint64_t mr_flags = 0;                // what NCCL asked for, kept so it can be reported
};

// ---------------------------------------------------------------------------
// Bootstrap and connection state
// ---------------------------------------------------------------------------

// The connection handle NCCL exchanges between ranks: where to reach this rank's agent.
struct ConnHandle {
    char     ip[46] = {0};
    uint16_t port = 0;
};

struct ListenComm {
    int fd = -1;
    uint16_t port = 0;
};

struct RmaCtx;   // declared below; CollComm keeps the contexts created on it so a report can ask them

// Sends the replies to gets this rank serves, on a thread of its own, so a reply never waits for this rank's
// NCCL progress thread and never blocks the receive thread that read its request. The bytes are copied when
// the request is read, under the transport's lease on the window. The thread sends each reply in frames of
// `chunk_bytes` as the link takes them, retrying what the outbox refuses, and keeps pushing a peer's outbox
// until it is empty. Replies to one peer go out in the order their requests arrived; peers are served in turn,
// so a requester that stops reading holds up only its own replies.
class GetResponder {
public:
    explicit GetResponder(srdsim::SendLinks* links) : links_(links) {}
    ~GetResponder() { stop(); }

    // False once teardown has begun: the reply is not sent.
    bool enqueue(int peer, uint32_t dest_window, uint64_t dest_off, uint64_t token, std::vector<char> bytes,
                 size_t chunk) {
        std::lock_guard<std::mutex> g(mu_);
        if (stopping_) return false;
        queues_[peer].push_back({dest_window, dest_off, token, std::move(bytes), chunk, 0});
        if (not thread_.joinable()) thread_ = std::thread([this] { loop(); });
        cv_.notify_one();
        return true;
    }

    // Sends what it can for up to kDrainMs, then gives up on the rest and reports it.
    void stop() {
        {
            std::lock_guard<std::mutex> g(mu_);
            if (stopping_) return;
            stopping_ = true;
            deadline_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(kDrainMs);
            cv_.notify_one();
        }
        if (thread_.joinable()) thread_.join();
        std::lock_guard<std::mutex> g(mu_);
        size_t left = 0;
        for (const auto& q : queues_) left += q.second.size();
        if (left != 0) logf("%zu get reply(ies) not sent at teardown", left);
        queues_.clear();
    }

private:
    static constexpr int kDrainMs = 1000;
    struct Reply {
        uint32_t dest_window;
        uint64_t dest_off;
        uint64_t token;
        std::vector<char> bytes;
        size_t chunk;
        size_t sent;      // bytes appended to the link so far
    };

    bool past_deadline() const { return stopping_ and std::chrono::steady_clock::now() >= deadline_; }

    void loop() {
        std::unique_lock<std::mutex> g(mu_);
        for (;;) {
            bool any = not flushing_.empty();
            for (const auto& q : queues_) any = any or not q.second.empty();
            if (not any and stopping_) break;
            if (not any) { cv_.wait(g); continue; }
            if (past_deadline()) break;
            bool moved = false;
            for (auto& q : queues_) {
                if (q.second.empty()) continue;
                const int peer = q.first;
                Reply& r = q.second.front();   // only this thread pops, and a deque keeps references on push
                g.unlock();
                while (r.sent < r.bytes.size()) {
                    const size_t n = std::min(r.chunk, r.bytes.size() - r.sent);
                    if (not links_->send_placement(peer, r.dest_window, r.dest_off + r.sent, r.bytes.data() + r.sent,
                                                   n, r.token, srdsim::WireKind::GetResponse)) break;
                    r.sent += n;
                    moved = true;
                }
                const bool appended = r.sent == r.bytes.size();
                g.lock();
                if (appended) { q.second.pop_front(); flushing_.insert(peer); }
            }
            // A reply fully appended may still sit in the outbox; push it until the socket has taken it.
            for (auto it = flushing_.begin(); it != flushing_.end();) {
                const int peer = *it;
                g.unlock();
                links_->flush(peer);
                const bool empty = links_->outbox_empty(peer);
                g.lock();
                it = empty ? flushing_.erase(it) : std::next(it);
            }
            if (not moved) cv_.wait_for(g, std::chrono::microseconds(200));   // the links are full: retry
        }
    }

    srdsim::SendLinks* links_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::map<int, std::deque<Reply>> queues_;   // by requesting rank
    std::set<int> flushing_;                    // peers whose outbox still holds a reply
    std::thread thread_;
    bool stopping_ = false;
    std::chrono::steady_clock::time_point deadline_{};
};

struct CollComm {
    int nranks = 0;
    int rank = 0;
    srdsim::WindowTable windows;                       // this rank's own windows, by id
    srdsim::SendLinks links;                           // one stream per peer
    // Before `agent`, so that on destruction the receive threads, which hand it replies, stop first.
    GetResponder responder{&links};                    // sends the replies to gets this rank serves
    std::unique_ptr<srdsim::ReceiveAgent> agent;       // applies what peers send us
    // How many peers actually connected to this rank's agent. A mesh that came up half-formed is
    // indistinguishable from a peer that never sent anything unless this is counted.
    int accepted_connections = 0;
    // Each window's size and residence, kept so a report can say "the 768 byte host window" rather
    // than "window 7". Which window a waiter polls is the question that matters when nothing arrives,
    // and a bare number cannot answer it.
    struct WindowShape { size_t bytes = 0; bool is_device = false; };
    // The first error this communicator hit, so the stall report can lead with it.
    std::string last_error;
    std::vector<WindowShape> window_shapes;
    bool     window_report_done = false;
    uint64_t last_reported_placements = ~0ull;
    uint64_t last_reported_signals = ~0ull;
    std::vector<std::unique_ptr<MrHandle>> mrs;
    // Every context created on this communicator, so a report can ask each scheduler what it is still
    // holding. Raw pointers: NCCL owns the lifetime through `createContext` / `destroyContext`, and a
    // destroyed context removes itself.
    std::vector<RmaCtx*> rma_contexts;
    // Signal counters this rank has sent to, by (peer, window, offset), and how: bit 1 with a put, bit 2 alone.
    // Per communicator, under `mu`, so it goes with the communicator; see `note_signal_use`.
    std::map<std::tuple<int, uint32_t, uint64_t>, uint8_t> signal_uses;
    std::mutex mu;
};

struct RmaCtx {
    CollComm* coll = nullptr;
    int contexts = 0;
    // Per context, because the process-wide counters cannot answer "was this scheduler ever driven".
    // A scheduler holding 165 operations and a scheduler that was never asked look identical without
    // these two numbers, and that ambiguity cost a full measurement cycle.
    std::atomic<uint64_t> progress_calls{0};
    std::atomic<uint64_t> units_released{0};
    std::atomic<uint64_t> ops_submitted{0};
    // Units this context released and could not send. Counted because nothing else records them: the
    // scheduler has already retired them and the receiving side never hears of them.
    std::atomic<uint64_t> send_failures{0};
    // Gets this context has issued whose reply has not yet landed. `iflush`'s request completes only
    // once this is back to zero as well as the send queue being empty: a put's `iflush` fences bytes
    // this rank sent, a get's fences bytes this rank is waiting to receive, and the same entry point
    // serves both. Approximate across ranks the same way `send_holds_any_of` already is -- `iflush`'s
    // own `rank` argument is accepted and ignored below, as it always was.
    // Counted in bytes, not gets: a reply of several chunks lands as several frames, each reported with its
    // own length, so a count of gets would reach zero at the first frame and then go negative.
    std::atomic<int64_t> pending_get_bytes{0};
    std::unique_ptr<srdsim::Scheduler> sched;
    uint64_t next_op_id = 1;
    bool has_error = false;
    std::string error;
    std::mutex mu;
};

struct Request {
    uint64_t magic = 0;
    uint64_t op_id = 0;
    RmaCtx*  ctx = nullptr;
    // A flush is not an operation, it is a question about all of them: "is everything I submitted
    // settled?". Encoded as an operation id, a flush issued before anything was submitted would get id 0,
    // which `complete` refuses as never issued, and the caller would wait forever. The distinction
    // belongs in the request.
    bool     is_flush = false;
    // Complete when it was handed back: a dry-run operation, or one the scheduler refused. It names no
    // operation, so `test` must not ask the scheduler about it.
    bool     done_at_issue = false;
    // A get: its entry in `g_gets`, which says how many of its bytes have still to land. Zero otherwise.
    uint64_t get_id = 0;
};

// Requests are handed back to NCCL as opaque pointers and tested until done, so the address of a
// request has to stay valid for as long as NCCL might still hold it.
//
// A `std::vector` cannot provide that: growing it reallocates and invalidates every pointer already given
// to NCCL, and the next `test()` reads a freed `Request` and locks a mutex at whatever address that
// produces, which shows up as a SIGSEGV inside `pthread_mutex_lock`, far from this file. A deque gives stable
// element addresses, which is the property this needs and the only reason to prefer it here.
struct RequestSlab {
    // Non-zero while a request is live, so a stale or foreign pointer is caught and reported instead
    // of being followed. Cheap, and the alternative is the crash described above.
    static constexpr uint64_t kMagic = 0x5344'5253'494d'0001ull;

    std::deque<Request> slots;
    std::vector<Request*> free_list;
    mutable std::mutex mu;

    Request* acquire() {
        std::lock_guard<std::mutex> g(mu);
        Request* r;
        if (free_list.empty()) {
            slots.emplace_back();
            r = &slots.back();
        } else {
            r = free_list.back();
            free_list.pop_back();
        }
        // Every mutable field, not only the magic. Leaving `is_flush` set meant the first put after any
        // completed flush inherited it, and `test()` then answered `sched->idle()` for a put -- which is
        // either "never done", freezing NCCL's completion index until the ring fills and the device blocks,
        // or "done at a momentary lull", telling NCCL a put finished that has not been sent. The
        // conformance suite completed a flush and then issued a put on the very next line, so it walked
        // through this on every run and passed, because it never asked whether that put completed.
        r->magic = kMagic;
        r->op_id = 0;
        r->ctx = nullptr;
        r->is_flush = false;
        r->done_at_issue = false;
        r->get_id = 0;
        return r;
    }

    // Called once a request has been reported done. NCCL does not hand the request back, so this is
    // the only point at which reuse is safe.
    void release(Request* r) {
        std::lock_guard<std::mutex> g(mu);
        r->magic = 0;
        free_list.push_back(r);
    }

    // Under the lock. A stale read here returns an error from `test()`, and an error from `test()` makes
    // NCCL's GIN progress thread exit permanently -- so the cost of a false negative is the whole run.
    bool live(const Request* r) const {
        if (r == nullptr) return false;
        std::lock_guard<std::mutex> g(mu);
        return r->magic == kMagic;
    }
};

// Never destroyed; see `g_where`.
RequestSlab& g_requests = *new RequestSlab;

// ---------------------------------------------------------------------------
// The data path
// ---------------------------------------------------------------------------

// Hands one released unit to the peer. The unit's order was decided by the scheduler; a stream
// preserves it, and the peer's agent applies in arrival order. Nothing about ordering is decided
// here, which is the point.
#ifdef SRDSIM_WITH_CUDA
void prebuild_signal_kernel();   // defined below, next to the signal kernel it builds
// The plugin's private non-blocking stream, shared by every device access it makes. Declared here
// because the sending path needs it before the class that owns it is defined.
CUstream device_stream();
#endif

// Pinned host memory to copy out of, one buffer per thread that applies frames.
//
// This is the second half of the same deadlock the non-blocking stream above addresses. `cuMemcpyHtoDAsync`
// from a *pageable* source is not asynchronous: the driver has to stage the bytes through pinned
// memory it owns, and that staging path waits on the device. So the submit itself blocks, and the device
// never drains when what occupies it is resident kernels (DeepEP's forwarding CTAs, for example) spinning
// for the bytes this copy would deliver. The kernel waits for the copy and
// the copy waits for the kernel, and the non-blocking stream does not help: the stream governs
// ordering between streams, not whether the driver's staging pool can be refilled.
//
// A pinned source removes the staging step, so the copy becomes a genuine DMA that a copy engine
// performs whether or not any SM is available. That is also what the hardware being modelled does: a
// NIC DMAs out of registered memory and does not ask the GPU's schedulers for permission.
//
// One buffer per thread rather than one shared: the agent runs a thread per connection, and a shared
// buffer would need a lock held across the copy, which serialises peers for no reason.
#ifdef SRDSIM_WITH_GDRCOPY
// Device windows, mapped into this process's address space so the CPU can read and write them with
// ordinary stores.
//
// WHY A BAR MAPPING, NOT A CUDA COPY
// While kernels are resident and spinning for data, a host-issued transfer in the same context does not
// complete, whatever its size, stream, source allocation or pinning: it blocks until the kernel times out,
// then returns CUDA_ERROR_LAUNCH_FAILED and the context is lost. The kernel waits for bytes the copy would
// deliver and the copy waits for the kernel to stop.
//
// A BAR mapping breaks that cycle: a CPU store through it is visible to a resident kernel at once. The
// write submits nothing to CUDA, so there is nothing for the kernel to be ahead of in a queue.
//
// This is also what the backend being simulated does: `aws-ofi-nccl` maps device memory with GDRCopy for
// exactly this class of write (`src/nccl_ofi_gdrcopy.cpp`). Using it makes the emulator more faithful,
// not less.
//
// WHAT IS PINNED, AND WHEN
// Everything, once, at registration. `gdr_pin_buffer` commits BAR aperture rather than deferring it to
// `gdr_map`, so pinning a subrange is the only way to use less -- and pinning on the data path would
// replace one risky driver operation in the hot path with another. This assumes a BAR aperture much larger
// than the windows (128 GB against a 1.82 GB window on the card this was sized for); where it is not, pin
// subranges.
class GdrWindows {
public:
    // Opened lazily and once. A host without the module gets a single message and the CUDA fallback.
    bool ready() {
        std::lock_guard<std::mutex> lock(mu_);
        if (not tried_open_) {
            tried_open_ = true;
            // A runtime off switch, so a control run can keep the plugin loaded and DeepEP running while
            // no window is pinned, no BAR is mapped and no byte is written through one. Without it the
            // only way to remove the registration path is to rebuild without GDRCopy, which changes the
            // binary as well as the behaviour and so cannot isolate one from the other.
            if (env_size("SRDSIM_NO_GDRCOPY", 0) != 0) {
                logf("SRDSIM_NO_GDRCOPY is set: no window will be pinned or mapped, and device access "
                     "falls back to CUDA copies, which do not complete while a resident kernel occupies "
                     "the device");
                return false;
            }
            g_ = gdr_open();
            if (g_ == nullptr) {
                logf("gdr_open failed; falling back to CUDA copies, which do not complete while a "
                     "resident kernel occupies the device");
            } else {
                int rt_major = 0, rt_minor = 0, drv_major = 0, drv_minor = 0;
                gdr_runtime_get_version(&rt_major, &rt_minor);
                gdr_driver_get_version(g_, &drv_major, &drv_minor);
                logf("gdrcopy is available: runtime %d.%d, driver %d.%d", rt_major, rt_minor,
                     drv_major, drv_minor);
            }
        }
        return g_ != nullptr;
    }

    // Pins and maps one registered window. Called from `regMrSym`, before any of DeepEP's kernels exist.
    void add(void* base, size_t bytes) {
        if (not ready()) return;
        // The pinned range has to start on a GPU page or `gdr_pin_buffer` refuses it with EINVAL and no
        // explanation. NCCL rounds its allocations to the VMM granularity so this holds in practice; it is
        // checked because the failure is otherwise indistinguishable from an unsupported allocation.
        const uintptr_t addr = reinterpret_cast<uintptr_t>(base);
        if ((addr & (GPU_PAGE_SIZE - 1)) != 0) {
            logf("a device window at %p is not GPU-page aligned, so it cannot be mapped; this window "
                 "will use CUDA copies", base);
            return;
        }
        // Already mapped ranges are reused rather than pinned again. NCCL registers the same window on
        // more than one communicator, and pinning and mapping it afresh for each registration spends BAR
        // aperture on a range that is already resident, doubles the kernel's bookkeeping, and leaves two host
        // mappings for one piece of device memory, of which the lookup can only ever use the first.
        {
            std::lock_guard<std::mutex> lock(mu_);
            for (const Region& existing : regions_) {
                if (existing.base != addr) continue;
                if (bytes <= existing.bytes) {
                    logf("a device window at %p is already mapped (%zu byte(s)); reusing that mapping "
                         "for this %zu byte registration", base, existing.bytes, bytes);
                    return;
                }
                // A larger registration at the same base is a different proposition: the existing
                // mapping does not cover it. Said out loud rather than silently pinned twice, because
                // the lookup would keep answering with the shorter one.
                logf("a device window at %p is mapped as %zu byte(s) but has been registered again as "
                     "%zu; the larger range is not mapped and will use CUDA copies", base,
                     existing.bytes, bytes);
                return;
            }
        }
        Region r{};
        r.base = addr;
        r.bytes = bytes;
        if (gdr_pin_buffer(g_, addr, bytes, 0, 0, &r.handle) != 0) {
            logf("could not pin %zu byte(s) at %p for mapping (errno %d); this window will use CUDA "
                 "copies", bytes, base, errno);
            return;
        }
        // The pinned range starts at the GPU page *containing* the window, which can be below the window
        // base. Every access therefore carries a lead offset -- and the mapping has to be long enough to
        // hold that lead as well as the window, or a write at the far end runs off the end of the mapping
        // by exactly the lead. Mapping only `bytes` is correct only when the lead is zero.
        //
        // The size is asked for before the map, so `gdr_get_info` comes first: it is valid after the pin.
        gdr_info_t info{};
        if (gdr_get_info(g_, r.handle, &info) != 0) {
            logf("could not read the mapping info for the window at %p", base);
            gdr_unpin_buffer(g_, r.handle);
            return;
        }
        if (info.va > addr) {
            // The pinned range should never start above the window. If it does, the offset arithmetic
            // below would underflow, so the window is left to the CUDA path rather than mapped wrongly.
            logf("the pinned range for the window at %p starts above it (va %llx); not mapping it", base,
                 static_cast<unsigned long long>(info.va));
            gdr_unpin_buffer(g_, r.handle);
            return;
        }
        // From `srdsim_mapping.h`, so the suite tests the arithmetic this line depends on rather than a
        // copy of it. `GPU_PAGE_SIZE` comes from gdrapi and the header names the same number
        // independently; they are checked against each other here so a change to either is caught at
        // compile time rather than by a wrong offset at run time.
        static_assert(srdsim::kGpuPageBytes == GPU_PAGE_SIZE,
                      "srdsim_mapping.h and gdrapi.h disagree about the GPU page size");
        const auto shape = srdsim::mapping_shape(addr, static_cast<uintptr_t>(info.va), bytes);
        const size_t lead = shape.lead;
        const size_t map_bytes = shape.map_bytes;
        if (gdr_map(g_, r.handle, &r.bar, map_bytes) != 0) {
            logf("could not map %zu byte(s) for the %zu byte window at %p (errno %d)", map_bytes, bytes,
                 base, errno);
            gdr_unpin_buffer(g_, r.handle);
            return;
        }
        r.mapped_base = info.va;
        r.mapped_bytes = map_bytes;
        if (lead != 0)
            logf("the window at %p sits %zu byte(s) into its pinned page; %zu byte(s) mapped to cover it",
                 base, lead, map_bytes);
        std::lock_guard<std::mutex> lock(mu_);
        regions_.push_back(r);
        logf("mapped a %zu byte device window at %p through the BAR (mapping at %p, va %llx)",
             bytes, base, r.bar, static_cast<unsigned long long>(info.va));
    }

    // True when every device access can go through a mapping. Reported once so an operator knows which
    // mechanism a run actually used, since the two differ by ninety-three seconds.
    bool covers(const void* device_ptr, size_t bytes) {
        std::lock_guard<std::mutex> lock(mu_);
        return find_locked(device_ptr, bytes) != nullptr;
    }

    bool write(void* device_dst, const void* src, size_t bytes) {
        void* target = nullptr;
        gdr_mh_t handle{};
        {
            std::lock_guard<std::mutex> lock(mu_);
            const Region* r = find_locked(device_dst, bytes);
            if (r == nullptr) return false;
            size_t offset = 0;
            // Bounded against the *window*, not the mapping. Page rounding makes the mapping longer, and
            // writing into that slack would put bytes past the memory NCCL registered.
            if (not srdsim::mapping_offset(r->base, r->bytes, r->mapped_base,
                                          reinterpret_cast<uintptr_t>(device_dst), bytes, &offset))
                return false;
            target = static_cast<char*>(r->bar) + offset;
            handle = r->handle;
        }
        // No lock across the copy. The mapping is fixed for the life of the process, and holding a lock
        // here would serialise every peer's agent thread through one critical section for no reason.
        return gdr_copy_to_mapping(handle, target, src, bytes) == 0;
    }

    // Drops the mapping for a window that is going away.
    //
    // Without this the registry outlives the memory it describes, and that is not a leak but a wrong
    // answer: NCCL frees a window and a later allocation can return the same address with a different
    // size, after which every write for the new window goes through the old mapping. The bytes land in
    // memory nobody is watching, or in memory somebody else now owns. Applications such as vLLM build and
    // tear down communicators during start-up.
    //
    // Unmapping and unpinning here is safe in a way that CUDA calls on this path are not: both are ioctls
    // to the gdrdrv module and neither waits for the context to go quiet.
    void forget(void* base) {
        gdr_mh_t handle{};
        void* bar = nullptr;
        size_t window_bytes = 0;
        size_t mapped_bytes = 0;
        {
            std::lock_guard<std::mutex> lock(mu_);
            const uintptr_t addr = reinterpret_cast<uintptr_t>(base);
            for (size_t i = 0; i < regions_.size(); ++ i) {
                if (regions_[i].base != addr) continue;
                handle = regions_[i].handle;
                bar = regions_[i].bar;
                window_bytes = regions_[i].bytes;
                mapped_bytes = regions_[i].mapped_bytes;
                regions_.erase(regions_.begin() + static_cast<ptrdiff_t>(i));
                break;
            }
        }
        // Outside the lock, and only if a region was actually found: `deregMrSym` is called for host
        // windows too, which were never mapped.
        if (bar == nullptr) return;
        // The *mapped* length, not the registered one. They differ whenever the window sits partway into
        // its pinned page, and unmapping the shorter length would leave part of the mapping behind.
        gdr_unmap(g_, handle, bar, mapped_bytes);
        gdr_unpin_buffer(g_, handle);
        logf("unmapped the %zu byte device window at %p (%zu byte(s) of mapping)", window_bytes, base,
             mapped_bytes);
    }

    bool read(void* dst, const void* device_src, size_t bytes) {
        const void* source = nullptr;
        gdr_mh_t handle{};
        {
            std::lock_guard<std::mutex> lock(mu_);
            const Region* r = find_locked(device_src, bytes);
            if (r == nullptr) return false;
            size_t offset = 0;
            if (not srdsim::mapping_offset(r->base, r->bytes, r->mapped_base,
                                          reinterpret_cast<uintptr_t>(device_src), bytes, &offset))
                return false;
            source = static_cast<const char*>(r->bar) + offset;
            handle = r->handle;
        }
        return gdr_copy_from_mapping(handle, dst, source, bytes) == 0;
    }

private:
    struct Region {
        uintptr_t base = 0;          // the window as this rank sees it
        uintptr_t mapped_base = 0;   // the GPU page the mapping starts at, which may be lower
        size_t bytes = 0;            // the window as registered
        size_t mapped_bytes = 0;     // what was actually mapped, which covers the lead as well
        void* bar = nullptr;
        gdr_mh_t handle{};
    };

    // Linear, because there are a handful of windows and a tree would be more code than the search it
    // replaces. The range check avoids computing `offset + bytes`, so a length off the wire cannot wrap
    // it -- the same reasoning as `range_fits` in the transport.
    const Region* find_locked(const void* p, size_t bytes) const {
        const uintptr_t a = reinterpret_cast<uintptr_t>(p);
        for (const Region& r : regions_) {
            if (a < r.base) continue;
            const uintptr_t offset = a - r.base;
            if (offset > r.bytes) continue;
            if (bytes <= r.bytes - offset) return &r;
        }
        return nullptr;
    }

    std::mutex mu_;
    bool tried_open_ = false;
    gdr_t g_ = nullptr;
    std::vector<Region> regions_;
};

// Never destroyed; see `g_where`.
GdrWindows& g_gdr = *new GdrWindows;
#endif  // SRDSIM_WITH_GDRCOPY

#ifdef SRDSIM_WITH_CUDA
// The first CUDA error this process saw, and when.
//
// Every error this plugin has reported so far has been 719, `CUDA_ERROR_LAUNCH_FAILED`, which is the
// sticky code a context returns after something has already gone wrong on it. A sticky code names the
// symptom and hides the cause, and the order matters more than the code: if the device faults before
// this plugin's copies stop returning, the plugin is downstream of the fault and its own machinery is
// not at issue. That ordering cannot be recovered from the logs afterwards, because the first thing to
// notice a dead context is whatever calls into it next.
//
// `cuStreamQuery` is the right probe because it does not block. `cuCtxSynchronize` would wait for the
// resident kernel, and a probe that hangs cannot report that something is hanging.
struct FirstDeviceError {
    std::atomic<uint64_t> at{0};
    std::atomic<int> code{0};
    std::atomic<const char*> where{nullptr};

    // Records only the first, so a flood of later 719s cannot overwrite the one that explains them.
    void note(CUresult rc, const char* what) {
        if (rc == CUDA_SUCCESS or rc == CUDA_ERROR_NOT_READY) return;
        uint64_t expected = 0;
        if (at.compare_exchange_strong(expected, srdsim::mono_ms(), std::memory_order_acq_rel)) {
            code.store(static_cast<int>(rc), std::memory_order_relaxed);
            where.store(what, std::memory_order_relaxed);
            const char* name = nullptr;
            cuGetErrorName(rc, &name);
            logf("FIRST CUDA ERROR on this process: %s (%d) seen by %s", name ? name : "unknown",
                 int(rc), what);
        }
    }
};
// Never destroyed; see `g_where`.
FirstDeviceError& g_first_device_error = *new FirstDeviceError;


// The largest frame either side can put on the wire, and therefore the size of every pinned staging
// slot. Read from the same variable the scheduler chunks by, so a change to one cannot leave the other
// refusing every frame and silently falling back to pageable copies.
size_t staging_capacity() {
    static const size_t v = env_size("SRDSIM_CHUNK_BYTES", 8192);
    return v;
}

// Pinned host memory for every copy this plugin performs, reserved in one block before any of DeepEP's
// kernels exist, and never allocated or freed again.
//
// Both halves of that sentence are load-bearing.
//
// *Never freed*: `cuMemFreeHost` can wait on the context, and so never return, while resident kernels
// spin for the bytes these buffers carry, and other threads' driver calls then wait behind it on a libcuda
// lock; growing a buffer by reallocation would call it on the data path.
//
// *Reserved early*: NVIDIA lists page-locked host allocation among the operations that implicitly
// synchronize the context, so allocating pinned memory waits for the context to go quiet. The context
// holds DeepEP's forwarding kernels, which are resident and spinning for the bytes these buffers carry,
// so that wait can never end. Allocating lazily on first use only narrows the window from every frame
// to the first frame per thread; it does not close it. The window closes only if the allocation happens
// before anything is resident, which is why this is reserved from `regMrSym` -- registration is
// collective and precedes every launch.
//
// A slot per thread, claimed by an atomic increment, because claiming must not itself call CUDA.
bool pinned_staging();

// True when every delivery in this process is a CUDA copy from the plugin's own context, and so copies,
// signals and send-side staging are submitted without waiting. Decided once. With GDRCopy the payload and
// signals are CPU stores, which nothing orders against queued CUDA writes, so any CUDA copy is then a
// fallback and is waited for at once. Pageable staging, a deliberate reproduction mode, waits too.
bool delivery_async() {
    static const bool v = [] {
#ifdef SRDSIM_WITH_GDRCOPY
        if (g_gdr.ready()) return false;
#endif
        // SRDSIM_ASYNC_DELIVERY=0 keeps every copy waited for, as before, for comparisons.
        return pinned_staging() and env_str("SRDSIM_ASYNC_DELIVERY", "1") != "0";
    }();
    return v;
}

class PinnedPool {
public:
    // Called once from window registration, on the thread NCCL registers from, before any kernel runs.
    // Idempotent, because registration happens once per window and there are several windows.
    void reserve() {
        std::lock_guard<std::mutex> g(mu_);
        if (tried_) return;
        tried_ = true;
        slot_bytes_ = staging_capacity();
        // The ring is the number of copies a thread may have outstanding. Only a run that delivers
        // asynchronously has more than one outstanding; see `delivery_async`.
        // 128 slots of 8 KiB is 1 MiB outstanding per thread: at one wait per 2.4 ms time slice, about
        // 430 MB/s per peer. Bounded so a configuration cannot pin more than `kMaxPinnedBytes`.
        size_t ring = delivery_async() ? std::max<size_t>(2, env_size("SRDSIM_STAGING_RING", 128)) : 2;
        const size_t per_ring = kGroups * slot_bytes_;
        if (per_ring == 0 or ring > kMaxPinnedBytes / per_ring) {
            const size_t capped = per_ring == 0 ? 2 : std::max<size_t>(2, kMaxPinnedBytes / per_ring);
            logf("a staging ring of %zu would pin more than %zu bytes; using %zu", ring, kMaxPinnedBytes, capped);
            ring = capped;
        }
        shape_ = srdsim::StagingShape{kGroups, ring};
        const size_t kSlots = shape_.slots();
        // The layout comes from `srdsim_mapping.h`, where the suite asserts for *every* slot that it
        // cannot touch the signal's bytes.
        layout_ = srdsim::PoolLayout{kSlots, slot_bytes_};
        const size_t total = layout_.total_bytes();
        void* p = nullptr;
        // Allocated inside the registration's `DeliveryContext` scope, so it is pinned for the context
        // every copy that reads it runs in. PORTABLE would pin it for the application's context too,
        // which is the context whose lock the plugin is avoiding.
        if (cuMemAllocHost(&p, total) != CUDA_SUCCESS) {
            logf("could not reserve %zu byte(s) of pinned staging memory; every copy will run from "
                 "pageable memory, which blocks in the driver while a resident kernel occupies the "
                 "device", total);
            return;
        }
        base_ = static_cast<char*>(p);
        logf("reserved %zu byte(s) of pinned staging memory: %zu group(s) of %zu slot(s) of %zu, plus %zu "
             "signal entries; delivery is %s", total, shape_.groups, shape_.ring, slot_bytes_,
             layout_.signal_entries(), delivery_async() ? "asynchronous" : "synchronous (every copy waited for)");
    }

    size_t ring() const { return shape_.ring; }

    // The calling thread's slot, or nullptr when the pool could not be reserved or is exhausted. No CUDA
    // call on this path at all, which is the property that matters.
    void* mine(size_t bytes, size_t i = 0) {
        if (base_ == nullptr or bytes > slot_bytes_) return nullptr;
        static thread_local int group = -1;
        if (group < 0) {
            const int taken = next_.fetch_add(1, std::memory_order_relaxed);
            if (taken >= static_cast<int>(shape_.receive_groups())) {
                // Named once per thread rather than per frame: a flood would bury the report that says
                // why the copies became slow.
                logf("the pinned staging pool is exhausted (%zu receiving threads); this thread will copy "
                     "from pageable memory", shape_.receive_groups());
                group = -2;
            } else {
                group = taken;
            }
        }
        return group >= 0 ? base_ + layout_.staging_offset(shape_.slot(static_cast<size_t>(group), i)) : nullptr;
    }

    // Slot `i` of the send group, which no receiving thread can claim. The caller serialises its use.
    void* send_slot(size_t bytes, size_t i) {
        if (base_ == nullptr or bytes > slot_bytes_) return nullptr;
        return base_ + layout_.staging_offset(shape_.slot(shape_.send_group(), i));
    }

    // Signal entry `k`, in turn; the caller serialises its use and waits before reusing one.
    void* signal_entry(size_t k) {
        if (base_ == nullptr) return nullptr;
        return base_ + layout_.signal_entry_offset(k);
    }
    size_t signal_entries() const { return base_ != nullptr ? layout_.signal_entries() : 0; }
    bool reserved() const { return base_ != nullptr; }

    // Eight bytes for the signal totals, from the last slot, which no staging thread can claim. Separate
    // because the signal path holds a lock across its copy and must not share a buffer with a payload.
    void* signal_slot() {
        if (base_ == nullptr) return nullptr;
        // The extra slot, past every staging slot, so no payload can share these bytes.
        return base_ + layout_.signal_offset();
    }

private:
    // One per receiving connection plus the sending thread, with room to spare. Two ranks use three; a
    // larger mesh uses one per peer, and the pool is sized so that growing the mesh does not reintroduce
    // an allocation on the data path.
    // One group per receiving connection plus the send group, with room to spare. Two ranks use three.
    static constexpr size_t kGroups = 32;
    static constexpr size_t kMaxPinnedBytes = size_t(1) << 30;
    srdsim::StagingShape shape_{};
    srdsim::PoolLayout layout_{};
    std::mutex mu_;
    bool tried_ = false;
    size_t slot_bytes_ = 0;
    char* base_ = nullptr;
    std::atomic<int> next_{0};
};

PinnedPool g_pinned;

#endif  // SRDSIM_WITH_CUDA

#ifdef SRDSIM_WITH_CUDA
// The agent runs on a thread this plugin created, and a fresh thread has no current CUDA context.
// Every driver-API call from it then fails with CUDA_ERROR_INVALID_CONTEXT, which the agent could only
// report as "the write failed": no data moved, the peer waited forever, and nothing said why. So every
// plugin thread binds a context before its first copy.
//
// Which context is the part that matters. A module load or a free in a context takes a lock that every
// other driver call in *that context* needs, and holds it until the context's resident kernels finish:
// while one thread is inside `cuModuleLoadData` or `cuMemFree` with a spinning kernel resident, even
// `cuEventCreate`, which submits no GPU work, waits until the kernel gives up. DeepEP's JIT loads a module
// inside dispatch, while its earlier kernels spin waiting for this plugin's bytes, so copies issued from the
// primary context deadlock against it. A second context in the same process is not held by that lock.
//
// So the plugin copies from a context of its own, on the device of the windows it writes. The windows
// must be VMM allocations (`ncclMemAlloc`), whose mappings belong to the process rather than to a context,
// so the same addresses are valid there.
// Registration refuses any other kind of device window rather than let a copy into it fail later
// (`check_delivery_window`). Sources of puts are registered windows too, so the same check covers them.
// SRDSIM_DELIVERY_CONTEXT=primary uses the device's primary context instead, to reproduce the deadlock
// on purpose; it accepts any allocation, as the plugin did before.
//
// SRDSIM_DELIVERY_CONTEXT=green uses a green context over the smallest SM partition the device allows,
// made at start-up. The cost of the own context is that its copies wait for a time slice while the
// application's kernels are resident: 2,283 us per copy with a wait, where a green context's took 8.9 us,
// the same as with no kernel resident (probe `green` cells, RTX PRO 6000, all 188 SMs spinning). But a
// green context shares the primary context's lock: a module load in the application held its copy until
// the kernel gave up, 20 s, as it holds the primary context's. So it is the fast choice only for an
// application that loads no module and frees nothing while its kernels wait for this plugin.
//
// Not destroyed: the context lives as long as the process, like the pinned pool and the stream made in
// it, and tearing it down before them would be the ordering bug to worry about.
struct Delivery {
    CUcontext ctx = nullptr;
    CUstream stream = nullptr;   // made in `ctx`, so no copy can reach it from another context
    CUdevice device = -1;
    bool primary = false;
};

bool delivery_uses_primary() {
    static const bool v = [] {
        const char* e = std::getenv("SRDSIM_DELIVERY_CONTEXT");
        return e != nullptr and std::strcmp(e, "primary") == 0;
    }();
    return v;
}

bool delivery_uses_green() {
    static const bool v = [] {
        const char* e = std::getenv("SRDSIM_DELIVERY_CONTEXT");
        return e != nullptr and std::strcmp(e, "green") == 0;
    }();
    return v;
}

#if !defined(CUDA_VERSION) || CUDA_VERSION >= 12040
// A green context and a stream in it. Current on no thread when it returns.
bool open_green(CUdevice dev, CUcontext* ctx, CUstream* stream, unsigned int* sms) {
    CUdevResource all{}, part{}, rest{};
    unsigned int groups = 1;
    CUdevResourceDesc desc{};
    CUgreenCtx green{};
    // The partition is a minimum for the split, not a claim about the copies, which need a copy engine and no SM:
    // 8 was the smallest group the RTX PRO 6000 accepted. The context is made at start-up and kept.
    constexpr unsigned int kGreenMinSms = 8;
    CUresult rc = cuDeviceGetDevResource(dev, &all, CU_DEV_RESOURCE_TYPE_SM);
    if (rc == CUDA_SUCCESS) rc = cuDevSmResourceSplitByCount(&part, &groups, &all, &rest, 0, kGreenMinSms);
    if (rc == CUDA_SUCCESS) rc = cuDevResourceGenerateDesc(&desc, &part, 1);
    if (rc == CUDA_SUCCESS) rc = cuGreenCtxCreate(&green, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM);
    if (rc == CUDA_SUCCESS) rc = cuCtxFromGreenCtx(ctx, green);
    if (rc == CUDA_SUCCESS) rc = cuGreenCtxStreamCreate(stream, green, CU_STREAM_NON_BLOCKING, 0);
    if (rc != CUDA_SUCCESS) {
        if (green != nullptr) cuGreenCtxDestroy(green);
        logf("could not create a green delivery context on device %d (%d)", int(dev), int(rc));
        return false;
    }
    *sms = part.sm.smCount;
    return true;
}
#endif

std::mutex g_delivery_mu;
Delivery g_delivery;

// Creates the delivery context on `dev` the first time, and afterwards only checks that `dev` is the same.
// One device per process is the shape this runs in; a second device is refused by name.
void on_process_exit();   // defined with `DeliveryContext`, which it guards

bool open_delivery(CUdevice dev) {
    std::lock_guard<std::mutex> g(g_delivery_mu);
    if (g_delivery.ctx != nullptr) {
        if (g_delivery.device == dev) return true;
        logf("a device window on device %d, but delivery was opened on device %d; one device per process "
             "is supported", int(dev), int(g_delivery.device));
        return false;
    }
    Delivery d;
    d.device = dev;
    d.primary = delivery_uses_primary();
    if (delivery_uses_green()) {
#if !defined(CUDA_VERSION) || CUDA_VERSION >= 12040
        unsigned int sms = 0;
        if (not open_green(dev, &d.ctx, &d.stream, &sms)) return false;
        g_delivery = d;
        std::atexit(on_process_exit);
        lifecycle("delivering from a green CUDA context over %u SMs of device %d (SRDSIM_DELIVERY_CONTEXT=green): "
                  "copies do not wait for a time slice, but do wait behind the application's module loads and "
                  "frees", sms, int(dev));
        return true;
#else
        logf("SRDSIM_DELIVERY_CONTEXT=green needs CUDA 12.4 or later");
        return false;
#endif
    }
    CUresult rc;
    if (d.primary) {
        rc = cuDevicePrimaryCtxRetain(&d.ctx, dev);
        if (rc != CUDA_SUCCESS) { logf("cuDevicePrimaryCtxRetain failed (%d)", int(rc)); return false; }
    } else {
#if CUDA_VERSION >= 13000
        rc = cuCtxCreate(&d.ctx, nullptr, 0, dev);
#else
        rc = cuCtxCreate(&d.ctx, 0, dev);
#endif
        if (rc != CUDA_SUCCESS) {
            logf("could not create the plugin's own CUDA context on device %d (%d)", int(dev), int(rc));
            return false;
        }
        // cuCtxCreate made it current on this thread, which is one of the application's. The stream is
        // made while it is current, then the thread is put back as it was.
    }
    if (d.primary and cuCtxPushCurrent(d.ctx) != CUDA_SUCCESS) {
        logf("cuCtxPushCurrent failed; the delivery stream cannot be made");
        return false;
    }
    rc = cuStreamCreate(&d.stream, CU_STREAM_NON_BLOCKING);
    CUcontext popped = nullptr;
    const CUresult pop_rc = cuCtxPopCurrent(&popped);
    if (rc != CUDA_SUCCESS) { logf("could not create the delivery stream (%d)", int(rc)); return false; }
    if (pop_rc != CUDA_SUCCESS or popped != d.ctx) {
        logf("restoring the registering thread's context failed (%d); refusing to continue with the "
             "application's thread in the plugin's context", int(pop_rc));
        return false;
    }
    g_delivery = d;
    std::atexit(on_process_exit);
    logf(d.primary ? "delivering from the primary CUDA context of device %d (SRDSIM_DELIVERY_CONTEXT=primary): "
                     "copies will wait behind any module load or free in the application"
                   : "delivering from the plugin's own CUDA context on device %d, so the application's "
                     "module loads and frees do not hold its copies",
         int(dev));
    return true;
}

Delivery delivery() {
    std::lock_guard<std::mutex> g(g_delivery_mu);
    return g_delivery;
}

// Called at registration for every device window: opens delivery on the window's device, and refuses a
// window the delivery context could not write.
bool check_delivery_window(void* base) {
    int ordinal = -1;
    CUresult rc = cuPointerGetAttribute(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                        reinterpret_cast<CUdeviceptr>(base));
    if (rc != CUDA_SUCCESS) {
        logf("cannot tell which device window %p is on (%d)", base, int(rc));
        return false;
    }
    CUdevice dev = 0;
    if (cuDeviceGet(&dev, ordinal) != CUDA_SUCCESS or not open_delivery(dev)) return false;
    if (delivery_uses_primary()) return true;
    CUmemGenericAllocationHandle h{};
    rc = cuMemRetainAllocationHandle(&h, base);
    if (rc != CUDA_SUCCESS) {
        logf("device window %p is not a VMM allocation (%d), so a copy from the plugin's own context "
             "cannot address it; allocate windows with ncclMemAlloc (NCCL_CUMEM_ENABLE=1), or set "
             "SRDSIM_DELIVERY_CONTEXT=primary and accept that copies wait behind the application's module "
             "loads", base, int(rc));
        return false;
    }
    cuMemRelease(h);
    return true;
}

// Process exit tears the CUDA driver down from the main thread's exit handlers, while NCCL's threads and the
// plugin's agents may still be running. A driver call made then lands in structures being freed, and either
// crashes the calling thread or corrupts the heap, and glibc then aborts in the driver's own teardown with
// `malloc_consolidate(): unaligned fastbin chunk detected`. This happens in processes that exit without
// destroying their communicators.
//
// So the plugin registers an exit handler when it first opens delivery. Handlers run in reverse order of
// registration, and the plugin is loaded after the CUDA runtime has initialised, so this one runs before the
// driver's teardown. It marks the process as exiting, waits briefly for driver calls already under way in a
// `DeliveryContext` to leave, and from then on every scope refuses, so no plugin thread enters the driver
// again.
std::atomic<bool> g_exiting{false};
std::atomic<int> g_in_driver{0};

void on_process_exit() {
    g_exiting.store(true, std::memory_order_seq_cst);
    for (int i = 0; i < 2000 and g_in_driver.load(std::memory_order_seq_cst) != 0; ++ i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

// Makes the delivery context current for one scope and restores whatever was current before, on every
// thread alike -- including threads with no context, because a thread without one may still be NCCL's,
// and leaving the plugin's context on it would move NCCL's later calls into it.
class DeliveryContext {
public:
    DeliveryContext() {
        // Counted before the check, so the exit handler either sees this scope and waits for it, or this
        // scope sees the flag and refuses.
        g_in_driver.fetch_add(1, std::memory_order_seq_cst);
        counted_ = true;
        if (g_exiting.load(std::memory_order_seq_cst)) return;
        d_ = delivery();
        if (d_.ctx == nullptr) {
            // Rate-limited rather than silent: every failed copy after this returns false for the same
            // reason, and a report without the reason is the failure this plugin keeps trying to end.
            static std::atomic<int> said{0};
            if (said.fetch_add(1) < 3) logf("no delivery context: no device window has been registered");
            return;
        }
        CUcontext current = nullptr;
        if (cuCtxGetCurrent(&current) == CUDA_SUCCESS and current == d_.ctx) { ok_ = true; return; }
        const CUresult rc = cuCtxPushCurrent(d_.ctx);
        if (rc != CUDA_SUCCESS) {
            logf("cuCtxPushCurrent failed (%d); device windows cannot be written from this thread", int(rc));
            return;
        }
        pushed_ = ok_ = true;
    }
    ~DeliveryContext() {
        restore();
        if (counted_) g_in_driver.fetch_sub(1, std::memory_order_seq_cst);
    }
    void restore() {
        if (not pushed_) return;
        pushed_ = false;
        CUcontext popped = nullptr;
        const CUresult rc = cuCtxPopCurrent(&popped);
        if (rc != CUDA_SUCCESS or popped != d_.ctx)
            logf("restoring this thread's CUDA context failed (%d, popped %p, expected %p); its later CUDA "
                 "calls may run in the wrong context", int(rc), static_cast<void*>(popped),
                 static_cast<void*>(d_.ctx));
    }
    DeliveryContext(const DeliveryContext&) = delete;
    DeliveryContext& operator=(const DeliveryContext&) = delete;
    bool ok() const { return ok_; }
    // Whether a delivery context exists. Without one there is no delivery stream, so "not ok" then means
    // there is nothing to wait for rather than a failure.
    bool exists() const { return d_.ctx != nullptr; }
    CUstream stream() const { return d_.stream; }

private:
    Delivery d_;
    bool pushed_ = false;
    bool ok_ = false;
    bool counted_ = false;
};

bool pinned_staging() {
    static const bool v = env_size("SRDSIM_PAGEABLE_STAGING", 0) == 0;
    return v;
}

#endif  // SRDSIM_WITH_CUDA

// Defined with the rest of get handling, below `srdsim_iget`; forward-declared so the `ReceiveAgent`
// construction that wires them in as hooks does not have to sit after everything that uses `RmaCtx`.
bool srdsim_handle_get_request(const void* src, bool src_is_device, size_t length,
                               uint32_t dest_window, uint64_t dest_off, uint32_t requester_rank,
                               uint64_t owner_token, void* user);
void srdsim_handle_get_response_landed(uint64_t owner_token, uint64_t bytes, void* user);

#ifdef SRDSIM_WITH_CUDA
// Creates `n` events into `out`, or none: a partly built ring would hand out null events later.
bool make_events(std::vector<CUevent>* out, size_t n) {
    std::vector<CUevent> made;
    made.reserve(n);
    for (size_t i = 0; i < n; ++ i) {
        CUevent e{};
        if (cuEventCreate(&e, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS) {
            for (CUevent m : made) cuEventDestroy(m);
            return false;
        }
        made.push_back(e);
    }
    *out = std::move(made);
    return true;
}

// Once a copy's completion cannot be established, no staging slot can be trusted again: every asynchronous
// path refuses from then on, and the report says why.
std::atomic<bool> g_delivery_poisoned{false};

void poison_delivery(const char* why, CUresult rc) {
    g_first_device_error.note(rc, why);
    if (not g_delivery_poisoned.exchange(true))
        logf("asynchronous delivery stopped: %s failed (%d); no staging slot is reused after this", why, int(rc));
}

// A slot's copy has landed (1), has not yet (0), or its completion cannot be established (-1).
int landed(CUevent e) {
    const CUresult rc = cuEventQuery(e);
    if (rc == CUDA_SUCCESS) return 1;
    if (rc == CUDA_ERROR_NOT_READY) return 0;
    poison_delivery("querying a staging copy", rc);
    return -1;
}

// The send side when delivery is asynchronous (`delivery_async`). A placement whose source is on the device
// joins its connection's queue at once, holding its place in the release order, and has its device-to-host
// copy submitted as soon as a send slot is free; it leaves once that copy has landed, and every later unit
// for the same connection leaves behind it, so each connection's wire order is the scheduler's release order.
// Nothing here waits: slots are reclaimed by querying, and a unit that finds none stays unstaged until a
// later call finds one. The transport's outbox copies a frame's bytes (`SendLinks::send_framed`), so a slot
// is free again as soon as its unit has been written.
struct SendStaging {
    std::mutex mu;
    srdsim::SendQueue<srdsim::Unit> queue;
    std::vector<CUevent> done;
    std::vector<bool> busy;
    bool events_failed = false;
    uint64_t unstaged_waits = 0;   // units that found no free slot on arrival, for the report
    // Operations with a unit that could not be sent: their later units are withheld, so a signal never
    // follows a placement that did not go.
    std::set<std::pair<const void*, uint64_t>> broken;
};
// Never destroyed; see `g_where`. Progress reaches this queue until the process is gone.
SendStaging& g_send = *new SendStaging;

using SendQ = srdsim::SendQueue<srdsim::Unit>;
using SendEntry = SendQ::Entry;

void probe_records(const char* bytes, size_t n, const char* where, int peer, uint64_t op_id);   // below, with its options

bool wire_unit(CollComm* cc, const srdsim::Unit& u, const void* src) {
    if (u.kind == srdsim::Unit::Kind::Placement) {
        probe_records(static_cast<const char*>(src), u.bytes, "sender, on the wire (queued path)", u.peer, u.op_id);
        // A get's reply is an ordinary placement here except for its wire kind and the id it carries:
        // the requester cannot use this rank's own locally-scoped op id for anything, so the token it
        // chose at `iget()` time goes out instead.
        if (u.is_get_response)
            return cc->links.send_placement(u.peer, u.window, u.window_offset, src, u.bytes,
                                             u.get_response_token, srdsim::WireKind::GetResponse, /*reserved=*/true);
        return cc->links.send_placement(u.peer, u.window, u.window_offset, src, u.bytes, u.op_id,
                                        srdsim::WireKind::Placement, /*reserved=*/true);
    }
    if (u.kind == srdsim::Unit::Kind::GetRequest)
        // Same reasoning as the response two lines up: the peer only ever echoes this value back, so
        // what travels is the token `iget()` chose (this rank's own `RmaCtx*`), not `u.op_id` -- an
        // id scoped to this rank's own scheduler and meaningless, and collision-prone, on the other end.
        return cc->links.send_get_request(u.peer, u.window, u.window_offset, u.bytes,
                                          u.get_dst_window, u.get_dst_off, u.get_requester_rank,
                                          u.get_response_token, /*reserved=*/true);
    return cc->links.send_signal(u.peer, u.window, u.window_offset, u.signal_is_add ? u.signal_val : 1, u.op_id,
                                 /*reserved=*/true);
}

void note_send_failure(RmaCtx* ctx, const srdsim::Unit& u);
void release_reservation(CollComm* cc, const srdsim::Unit& u);


// Assigns free slots to unstaged units, in each connection's order. Called with `g_send.mu` held and the
// delivery context current.
void stage_pending_locked(CUstream stream) {
    if (g_delivery_poisoned.load()) return;
    size_t next_free = 0;
    g_send.queue.for_each([&](SendEntry& e) {
        if (e.slot != SendQ::kUnstaged) return true;
        while (next_free < g_send.busy.size() and g_send.busy[next_free]) ++ next_free;
        if (next_free == g_send.busy.size()) { ++ g_send.unstaged_waits; return false; }   // later calls retry
        const size_t i = next_free;
        void* slot = g_pinned.send_slot(e.unit.bytes, i);
        CUresult rc = slot == nullptr ? CUDA_ERROR_INVALID_VALUE
                                      : cuMemcpyDtoHAsync(slot, reinterpret_cast<CUdeviceptr>(const_cast<void*>(e.unit.src)),
                                                          e.unit.bytes, stream);
        if (rc == CUDA_SUCCESS) rc = cuEventRecord(g_send.done[i], stream);
        if (rc != CUDA_SUCCESS) {
            // The copy may have been submitted before the record failed, so the slot stays held.
            poison_delivery("submitting a send-side staging copy", rc);
            g_send.busy[i] = true;
            e.failed = true;
            e.slot = static_cast<int>(i);
            return false;
        }
        g_send.busy[i] = true;
        e.slot = static_cast<int>(i);
        return true;
    });
}

// Sends every head that is ready. Called with `g_send.mu` held and the delivery context current.
void apply_snapshot(const void* owner, const srdsim::Unit& u, void* staged);   // with the snapshots, below
bool snapshot_source();

size_t drain_sends_locked() {
    return g_send.queue.drain(
        [](const SendEntry& e) {
            if (e.failed) return true;
            if (e.slot == SendQ::kNoSlot) return true;
            if (e.slot == SendQ::kUnstaged) return false;
            return landed(g_send.done[static_cast<size_t>(e.slot)]) != 0;   // a failed query is removed below
        },
        [](const SendEntry& e) {
            auto* ctx = static_cast<RmaCtx*>(const_cast<void*>(e.owner));
            const auto key = std::make_pair(e.owner, e.op_id);
            bool ok = not e.failed and not g_delivery_poisoned.load() and not g_send.broken.count(key);
            if (ok) {
                const void* src = e.slot >= 0 ? g_pinned.send_slot(e.unit.bytes, static_cast<size_t>(e.slot))
                                              : e.unit.src;
                if (e.slot >= 0 and snapshot_source())
                    apply_snapshot(e.owner, e.unit, const_cast<void*>(src));
                ok = wire_unit(ctx->coll, e.unit, src);
            }
            // A slot whose copy could not be confirmed is never handed out again.
            if (e.slot >= 0 and not e.failed and not g_delivery_poisoned.load())
                g_send.busy[static_cast<size_t>(e.slot)] = false;
            if (not ok) {
                g_send.broken.insert(key);
                note_send_failure(ctx, e.unit);
            }
            return ok;
        });
}

// Stages, sends what is ready, and stages again into the slots that sending freed. Queries only.
void pump_sends_locked(CUstream stream) {
    stage_pending_locked(stream);
    if (drain_sends_locked() != 0) stage_pending_locked(stream);
}

void drain_sends() {
    std::lock_guard<std::mutex> g(g_send.mu);
    if (g_send.queue.empty()) return;
    DeliveryContext in_delivery_context;
    if (not in_delivery_context.ok()) return;
    pump_sends_locked(in_delivery_context.stream());
}

// For deregistration (`send` true): everything queued for `owner` (or everyone, with nullptr) is staged,
// waited for and sent while the memory is still the application's, and anything that cannot go is removed and
// reported. For a context or communicator going away (`send` false): nothing is sent, only removed. By then
// the application may already have freed what the units point at -- a buffer torn down without `destroy()`
// frees its host windows without deregistering them -- and a unit sent now would carry stale heap from the
// sender into freed heap on the receiver. Returns false if something had to be dropped.
bool finish_sends(const void* owner, bool send) {
    std::lock_guard<std::mutex> g(g_send.mu);
    if (g_send.queue.empty()) return true;
    DeliveryContext in_delivery_context;
    bool ok = in_delivery_context.ok();
    for (int round = 0; send and ok and round < 1024; ++ round) {
        if (owner != nullptr ? not g_send.queue.holds_any_of(owner) : g_send.queue.empty()) break;
        pump_sends_locked(in_delivery_context.stream());
        const CUresult rc = cuStreamSynchronize(in_delivery_context.stream());
        if (rc != CUDA_SUCCESS) { poison_delivery("waiting for send-side staging at teardown", rc); ok = false; }
        pump_sends_locked(in_delivery_context.stream());
        if (g_delivery_poisoned.load()) ok = false;
    }
    const auto drop = [](const SendEntry& e) {
        g_send.broken.insert({e.owner, e.op_id});
        release_reservation(static_cast<CollComm*>(const_cast<void*>(e.lane)), e.unit);
        note_send_failure(static_cast<RmaCtx*>(const_cast<void*>(e.owner)), e.unit);
    };
    size_t dropped = 0;
    if (owner != nullptr) {
        dropped = g_send.queue.remove_owner(owner, drop);
    } else {
        std::set<const void*> owners;
        g_send.queue.for_each([&](SendEntry& e) { owners.insert(e.owner); return true; });
        for (const void* o : owners) dropped += g_send.queue.remove_owner(o, drop);
    }
    if (dropped != 0) logf("dropped %zu unsent unit(s) %s", dropped, send ? "at deregistration" : "at teardown");
    return ok and dropped == 0;
}

// A communicator going away: every queued unit that would travel on it is removed and reported, none sent.
void drop_sends_on(const void* lane) {
    std::set<const void*> owners;
    {
        std::lock_guard<std::mutex> g(g_send.mu);
        g_send.queue.for_each([&](SendEntry& e) { if (e.lane == lane) owners.insert(e.owner); return true; });
    }
    for (const void* o : owners) finish_sends(o, /*send=*/false);
}

bool send_holds(const void* owner, uint64_t op_id) {
    std::lock_guard<std::mutex> g(g_send.mu);
    return g_send.queue.holds(owner, op_id);
}

bool send_holds_any_of(const void* owner) {
    std::lock_guard<std::mutex> g(g_send.mu);
    return g_send.queue.holds_any_of(owner);
}

std::string send_queue_report() {
    std::lock_guard<std::mutex> g(g_send.mu);
    size_t busy = 0;
    for (bool b : g_send.busy) busy += b;
    char line[200];
    std::snprintf(line, sizeof(line), "send queue: %zu unit(s) queued, %zu of %zu slot(s) held, %llu arrival(s) "
                  "found no free slot, %zu operation(s) withheld after a failure%s", g_send.queue.size(), busy,
                  g_send.busy.size(), static_cast<unsigned long long>(g_send.unstaged_waits), g_send.broken.size(),
                  g_delivery_poisoned.load() ? "; asynchronous delivery stopped" : "");
    return line;
}

// Returns false only when the unit could not be accepted at all; a unit that is queued and fails later is
// reported through `note_send_failure`.
bool send_unit_async(const srdsim::Unit& u, RmaCtx* owner) {
    std::lock_guard<std::mutex> g(g_send.mu);
    DeliveryContext in_delivery_context;
    if (not in_delivery_context.ok() or g_delivery_poisoned.load()) return false;
    const CUstream stream = in_delivery_context.stream();
    if (g_send.done.empty() and not g_send.events_failed) {
        if (make_events(&g_send.done, g_pinned.ring())) g_send.busy.assign(g_send.done.size(), false);
        else { g_send.events_failed = true; logf("could not create the send ring's events"); }
    }
    if (g_send.events_failed) return false;
    const auto key = std::make_pair(static_cast<const void*>(owner), u.op_id);
    if (g_send.broken.count(key)) return false;   // an earlier unit of this operation did not go
    SendEntry e;
    e.unit = u;
    e.lane = owner->coll;
    e.owner = owner;
    e.op_id = u.op_id;
    const bool stage = u.kind == srdsim::Unit::Kind::Placement and u.src_is_device;
    if (not stage and g_send.queue.empty_for(e.lane, u.peer))
        return wire_unit(owner->coll, u, u.src);   // nothing ahead of it on this connection
    e.slot = stage ? SendQ::kUnstaged : SendQ::kNoSlot;
    g_send.queue.push(std::move(e));
    pump_sends_locked(stream);
    return true;
}
#endif  // SRDSIM_WITH_CUDA

// SRDSIM_PROBE_RECORD_BYTES / SRDSIM_PROBE_STAMP_OFFSET: a diagnostic that knows one thing about the application's
// layout: records of a fixed size (as DeepEP's combine sends), in which the application has been instrumented to
// write a stamp (a small non-negative int) at a fixed offset. When an operation's bytes are a whole number of
// records, every record's stamp is looked at where the bytes are on the host: the sender's copy at
// submit (with SRDSIM_SNAPSHOT_SOURCE=1) and the receiver's before it writes them to the device. An operation in
// which some records carry a stamp and others do not is reported with where it was seen, so a record that is
// wrong on the wire tells a wrong read of the source apart from a destination overwritten after delivery.
struct RecordProbe { size_t rec = 0, off = 0; std::atomic<uint64_t> found{0}; };
RecordProbe& record_probe() {
    static RecordProbe* p = [] {
        auto* q = new RecordProbe;
        q->rec = env_size("SRDSIM_PROBE_RECORD_BYTES", 0);
        q->off = env_size("SRDSIM_PROBE_STAMP_OFFSET", 0);
        if (q->rec != 0 and q->off + sizeof(int) > q->rec) q->rec = 0;
        if (q->rec != 0)
            lifecycle("record probe active: %zu-byte records, stamp at offset %zu", q->rec, q->off);
        return q;
    }();
    return *p;
}

void probe_records(const char* bytes, size_t n, const char* where, int peer, uint64_t op_id) {
    RecordProbe& p = record_probe();
    if (p.rec == 0 or n < p.rec or n % p.rec != 0) return;
    const size_t count = n / p.rec;
    size_t good = 0, first_bad = count;
    int bad_value = 0;
    for (size_t r = 0; r < count; ++ r) {
        int v;
        std::memcpy(&v, bytes + r * p.rec + p.off, sizeof(v));
        if (v >= 0 and v < (1 << 24)) { ++ good; continue; }
        if (first_bad == count) { first_bad = r; bad_value = v; }
    }
    if (good == 0 or good == count) return;
    const uint64_t k = ++ p.found;
    if (k <= 32 or ((k & (k - 1)) == 0 and (__builtin_ctzll(k) % 2) == 0))
        lifecycle("record probe (%llu so far): %s, peer %d op %llu, %zu of %zu records carry no stamp, first at record %zu (%d)",
                  static_cast<unsigned long long>(k), where, peer, static_cast<unsigned long long>(op_id),
                  count - good, count, first_bad, bad_value);
}

// SRDSIM_SNAPSHOT_SOURCE=1: a diagnostic for a source reused too early. A put's source must stay unchanged
// until the put completes; the plugin reads it when each unit is released, which may be long after the submit.
// If the application rewrites the source before completion, a late unit carries the new bytes. With this set,
// the whole device source of each put is copied when it is submitted; when a unit is released, the bytes it
// would have read are compared with that copy, a difference is counted and named, and the copy is what is sent
// -- so a run that fails without it and passes with it, with differences counted, has shown the application
// reusing a source too early. Synchronous delivery only (GDRCopy, or SRDSIM_ASYNC_DELIVERY=0). A diagnostic, and a
// slow one: every device source is copied to the host on the submit path, under the context's lock.
bool snapshot_source() {
    static const bool v = env_size("SRDSIM_SNAPSHOT_SOURCE", 0) != 0;
    return v;
}

struct SourceSnapshots {
    struct Snap { const char* src = nullptr; std::vector<char> bytes; size_t unsent = 0; };
    std::mutex mu;
    std::map<std::pair<const void*, uint64_t>, Snap> snaps;
    std::atomic<uint64_t> taken{0}, changed_units{0}, changed_bytes{0};
};
SourceSnapshots& g_snapshots = *new SourceSnapshots;

#ifdef SRDSIM_WITH_CUDA
bool read_device(void* host, const void* src, size_t bytes);   // defined below, with the staging it shares
#endif

void take_snapshot(const void* owner, const srdsim::Op& op) {
#ifdef SRDSIM_WITH_CUDA
    // Only the synchronous send path compares and substitutes; with asynchronous delivery a copy would be taken
    // and never used, and "no source changed" would be reported for units nobody looked at.
    // With asynchronous delivery the copy is compared where the staged bytes are about to go on the wire
    // (`drain_sends_locked`). The submit-time read goes through the mapping only: a CUDA copy on the submit path
    // would wait on the resident kernel.
#ifdef SRDSIM_WITH_GDRCOPY
    const bool mapped = g_gdr.ready();
#else
    const bool mapped = false;
#endif
    if (delivery_async() and not mapped) {
        static std::atomic<bool> said{false};
        if (not said.exchange(true))
            lifecycle("SRDSIM_SNAPSHOT_SOURCE=1 is ignored: with asynchronous delivery it needs GDRCopy");
        return;
    }
    if (not op.src_is_device or op.bytes == 0 or op.src_base == nullptr) return;
    SourceSnapshots::Snap snap;
    snap.src = static_cast<const char*>(op.src_base) + op.src_off;
    snap.bytes.resize(op.bytes);
    snap.unsent = op.bytes;
    bool read = false;
#ifdef SRDSIM_WITH_GDRCOPY
    if (delivery_async()) read = g_gdr.read(snap.bytes.data(), snap.src, op.bytes);
#endif
    if (not delivery_async()) read = read_device(snap.bytes.data(), snap.src, op.bytes);
    if (not read) return;
    probe_records(snap.bytes.data(), op.bytes, "sender, at submit", op.peer, op.id);
    const uint64_t k = ++ g_snapshots.taken;
    if (k == 1) lifecycle("SRDSIM_SNAPSHOT_SOURCE=1: device sources are copied at submit and sent from the copy");
    std::lock_guard<std::mutex> g(g_snapshots.mu);
    g_snapshots.snaps[{owner, op.id}] = std::move(snap);
#else
    (void)owner; (void)op;
#endif
}

#ifdef SRDSIM_WITH_CUDA
// Replaces `staged` (the bytes read now) by the submit-time copy, and reports whether they differed.
void apply_snapshot(const void* owner, const srdsim::Unit& u, void* staged) {
    std::lock_guard<std::mutex> g(g_snapshots.mu);
    auto it = g_snapshots.snaps.find({owner, u.op_id});
    if (it == g_snapshots.snaps.end()) {
        static std::atomic<uint64_t> misses{0};
        const uint64_t m = ++ misses;
        if (m <= 8 or (m & (m - 1)) == 0)
            lifecycle("no submit-time copy for op %llu to peer %d (%llu so far): its unit is sent as read now, unchecked",
                      static_cast<unsigned long long>(u.op_id), u.peer, static_cast<unsigned long long>(m));
        return;
    }
    auto& snap = it->second;
    const uintptr_t at = reinterpret_cast<uintptr_t>(u.src), base = reinterpret_cast<uintptr_t>(snap.src);
    if (at < base or at - base + u.bytes > snap.bytes.size()) return;
    const size_t offset = static_cast<size_t>(at - base);
    const char* want = snap.bytes.data() + offset;
    if (std::memcmp(staged, want, u.bytes) != 0) {
        size_t n = 0;
        for (size_t i = 0; i < u.bytes; ++ i) n += static_cast<const char*>(staged)[i] != want[i];
        const uint64_t c = ++ g_snapshots.changed_units;
        g_snapshots.changed_bytes += n;
        if ((c & (c - 1)) == 0 and (__builtin_ctzll(c) % 2) == 0)
            lifecycle("source rewritten before its put completed (%llu unit(s) so far): op %llu to peer %d, "
                      "%zu of %zu byte(s) differ at offset %zu of the op -- sending the submit-time bytes",
                      static_cast<unsigned long long>(c), static_cast<unsigned long long>(u.op_id), u.peer, n,
                      u.bytes, offset);
        std::memcpy(staged, want, u.bytes);
    }
    snap.unsent = snap.unsent > u.bytes ? snap.unsent - u.bytes : 0;
    if (snap.unsent == 0) g_snapshots.snaps.erase(it);
}
#endif  // SRDSIM_WITH_CUDA

#ifdef SRDSIM_WITH_CUDA
// A synchronous read of device memory from the plugin's own context: the mapping when there is one, a copy on
// the delivery stream otherwise. Only the diagnostic above uses it on the submit path.
bool read_device(void* host, const void* src, size_t bytes) {
#ifdef SRDSIM_WITH_GDRCOPY
    if (g_gdr.read(host, src, bytes)) return true;
#endif
    DeliveryContext in_delivery_context;
    if (not in_delivery_context.ok()) return false;
    const CUstream stream = device_stream();
    CUresult rc = cuMemcpyDtoHAsync(host, reinterpret_cast<CUdeviceptr>(const_cast<void*>(src)), bytes, stream);
    if (rc == CUDA_SUCCESS) rc = cuStreamSynchronize(stream);
    return rc == CUDA_SUCCESS;
}
#endif

bool send_unit(CollComm* cc, const srdsim::Unit& u, [[maybe_unused]] RmaCtx* owner) {
#ifdef SRDSIM_WITH_CUDA
    // Asynchronous only with a reserved pool; without one, the synchronous path below stages as before.
    if (delivery_async() and g_pinned.reserved()) return send_unit_async(u, owner);
#endif
    if (u.kind == srdsim::Unit::Kind::Placement) {
        Mark mark("send_unit: placement");
        const void* src = u.src;
#ifdef SRDSIM_WITH_CUDA
        // A socket cannot read from device memory. Handing `send()` a device pointer fails with
        // EFAULT *after* the frame header has already been written, which desynchronises the stream
        // and turns every later frame into a nonsense window id. So a device source is staged through
        // host memory first. The buffer is per thread and grows to the largest unit seen, because the
        // alternative is an allocation on the data path.
        // Pinned, and for a sharper reason here than on the receiving side: this runs on the one
        // thread NCCL drives the whole GIN proxy with, so a copy that blocks here stops `test`, every
        // submit entry point and `rmaProgress` at once. A pageable destination makes
        // `cuMemcpyDtoHAsync` stage through the driver's own pinned pool and wait on the device, and
        // the device is occupied by the kernel that is waiting for these very bytes.
        // Outside the branch on purpose. `src` is read by `send_placement` below, so a buffer declared
        // inside the branch would be freed before the bytes were written -- the fallback path would
        // send freed memory, which is worse than the deadlock it exists to avoid.
        std::vector<char> pageable;
        if (u.src_is_device) {
            // The sending side needs a context for exactly the same reason the receiving side does:
            // this runs on an NCCL proxy thread, which has none of its own. Without it every copy
            // fails with CUDA_ERROR_INVALID_CONTEXT (201) and no payload ever leaves the rank.
            DeliveryContext in_delivery_context;
            if (not in_delivery_context.ok()) return false;
            // Same stream, same reason: a synchronous `cuMemcpyDtoH` runs on the legacy default stream
            // and waits for DeepEP's spinning kernel, which is waiting for the bytes this copy is
            // fetching. The sending path deadlocks exactly as the receiving path did.
            Mark staging_mark("send_unit: staging a device source through host memory");
            void* host = pinned_staging() ? g_pinned.mine(u.bytes) : nullptr;
            if (host == nullptr) {
                // Either pinning was refused or it was deliberately turned off to reproduce the
                // deadlock. Either way the buffer still has to exist, and saying so beats hanging.
                if (pinned_staging())
                    logf("could not pin %zu byte(s) of staging memory; copying into pageable memory, "
                         "which blocks in the driver while a resident kernel occupies the device",
                         u.bytes);
                pageable.resize(u.bytes);
                host = pageable.data();
            }
            // Reading has the same problem as writing: a `cuMemcpyDtoHAsync` out of the window does not
            // return while the kernel is resident. The census caught exactly this call in flight for 103
            // seconds with nothing else in the driver. So the mapping is tried first, and the CUDA path
            // below is only reached on a host without it.
            bool staged_through_mapping = false;
#ifdef SRDSIM_WITH_GDRCOPY
            staged_through_mapping = g_gdr.read(host, u.src, u.bytes);
#endif
            if (staged_through_mapping) {
                // Falls through to the one wire write below rather than returning here. Duplicating the
                // send would mean two places to keep in step, and the breadcrumb it sets is the one that
                // says a stall is in the socket rather than in the staging.
                src = host;
                if (snapshot_source()) apply_snapshot(owner, u, host);
            } else {
                const CUstream stream = device_stream();
                CUresult rc = cuMemcpyDtoHAsync(host,
                                                reinterpret_cast<CUdeviceptr>(const_cast<void*>(u.src)),
                                                u.bytes, stream);
                if (rc == CUDA_SUCCESS) rc = cuStreamSynchronize(stream);
                if (rc != CUDA_SUCCESS) {
                    g_first_device_error.note(rc, "the send-side staging copy");
                    logf("staging %zu bytes out of device memory failed (%d)", u.bytes, int(rc));
                    return false;
                }
                // Handed to the wire inside this scope, so the pinned buffer is still alive and
                // still owned by this thread when `send_placement` reads it.
                src = host;
                if (snapshot_source()) apply_snapshot(owner, u, host);
            }
        }
#endif
        Mark wire_mark("send_unit: writing a placement to the wire");
        probe_records(static_cast<const char*>(src), u.bytes, "sender, on the wire", u.peer, u.op_id);
        if (u.is_get_response)
            return cc->links.send_placement(u.peer, u.window, u.window_offset, src, u.bytes,
                                             u.get_response_token, srdsim::WireKind::GetResponse, /*reserved=*/true);
        return cc->links.send_placement(u.peer, u.window, u.window_offset, src, u.bytes, u.op_id,
                                        srdsim::WireKind::Placement, /*reserved=*/true);
    }
    if (u.kind == srdsim::Unit::Kind::GetRequest) {
        Mark get_mark("send_unit: writing a get request to the wire");
        // Same reasoning as the response case above: the token is this rank's own `RmaCtx*`, not `u.op_id`.
        return cc->links.send_get_request(u.peer, u.window, u.window_offset, u.bytes,
                                          u.get_dst_window, u.get_dst_off, u.get_requester_rank,
                                          u.get_response_token, /*reserved=*/true);
    }
    Mark signal_mark("send_unit: writing a signal to the wire");
    return cc->links.send_signal(u.peer, u.window, u.window_offset,
                                 u.signal_is_add ? u.signal_val : 1, u.op_id,
                                 /*reserved=*/true);
}

// The receiving side's writers. Host windows are handled inside the agent; these are installed for
// device windows, and they are the only CUDA in the plugin.
#ifdef SRDSIM_WITH_CUDA
// A private non-blocking stream for everything this plugin submits to the device.
//
// `cuMemcpyHtoD` is synchronous and runs on the legacy default stream, and the legacy default stream
// synchronises with every blocking stream in the context. DeepEP's dispatch kernel spins waiting for
// data, so a default-stream copy waits for that kernel to finish: the kernel waits for the bytes and
// the bytes wait for the kernel. Traffic flowed until the kernel started spinning and then froze
// completely, with the progress thread stuck inside the copy -- which is why even the stall watchdog
// never printed, because the thread that would have printed it was the blocked one.
//
// A stream created with CU_STREAM_NON_BLOCKING has no implicit dependency on the legacy default
// stream, so the copy proceeds while the kernel spins. This is not an optimisation; it is the
// difference between working and deadlocking.
// Made with the delivery context in `open_delivery`, so it belongs to the context every copy runs in.
CUstream device_stream() { return delivery().stream; }

// Asks the driver whether the context is still usable, without waiting for it. Called from the report,
// so the answer is dated relative to everything else there.
// A copy the resident kernel cannot possibly be involved in.
//
// The census established that exactly one CUDA call is in flight during the freeze -- a single
// `cuMemcpyDtoHAsync` out of DeepEP's window -- and nothing else. That rules out a lock holder, and
// leaves two possibilities which no amount of further watching can separate: either this context cannot
// service any transfer while DeepEP's kernels are resident, or there is something about copying out of
// *that* memory specifically.
//
// So the probe owns both ends: its own device allocation, made at reservation time before any kernel
// existed, and its own stream. Nothing DeepEP has ever seen. If this copy also fails to return, the
// context cannot move bytes at all while the kernel spins and the emulator's whole approach is
// unsupportable on this path. If it returns promptly, the hazard is the address, and the question
// narrows to what NCCL's symmetric window is that a plain allocation is not.
//
// Printed as two lines on purpose: a start and a finish. A call that never returns cannot report its own
// duration, so the absence of the second line is the measurement.
struct OwnBufferProbe {
    // Two sizes, and the large one goes first because the small one may not come back.
    //
    // The probe established that a copy out of this plugin's own allocation on its own stream blocks
    // exactly as DeepEP's does, so the hazard is not the address: while the kernel is resident this
    // context cannot move bytes. One mechanism explains that and is worth separating, because it decides
    // whether the emulator is fixable at all. Drivers implement *small* transfers with a kernel launch
    // rather than by the copy engine, and a kernel cannot be scheduled behind a kernel that never exits,
    // whereas a copy engine is independent of the SMs. If that is what is happening, a large copy returns
    // promptly while a small one hangs -- and the emulator's frames are 8 kB, which is small.
    //
    // If both hang, the size is irrelevant and no transfer can be issued from the host at all while
    // DeepEP spins, which would mean this interposition cannot work on this path in any form.
    static constexpr size_t kLarge = 1u << 20;

    CUdeviceptr device = 0;
    CUstream stream = nullptr;
    void* host = nullptr;
    bool ready = false;

    // Called where the pinned pool is reserved: before any launch, so neither the allocation nor the
    // stream creation can wait on a resident kernel.
    void reserve() {
        if (ready) return;
        if (cuMemAlloc(&device, kLarge) != CUDA_SUCCESS) return;
        if (cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) return;
        if (cuMemAllocHost(&host, kLarge) != CUDA_SUCCESS) return;
        ready = true;
    }

    void one(const char* label, size_t bytes) {
        logf("  probe: copying %zu byte(s) out of this plugin's own device memory on its own stream, "
             "which DeepEP has never seen (%s). No result line means it did not return.", bytes, label);
        const uint64_t started = srdsim::mono_ms();
        CUresult rc = cuMemcpyDtoHAsync(host, device, bytes, stream);
        if (rc == CUDA_SUCCESS) rc = cuStreamSynchronize(stream);
        const char* name = nullptr;
        cuGetErrorName(rc, &name);
        logf("  probe: the %s copy returned %s (%d) after %llu ms", label,
             name ? name : "unknown", int(rc),
             static_cast<unsigned long long>(srdsim::age_ms(started, 0)));
    }

    // Run at most once, because a call that hangs hangs the thread printing the report.
    void run_once() {
        static bool done = false;
        if (not ready or done) return;
        done = true;
        DeliveryContext in_delivery_context;   // its buffers were made in the delivery context
        if (not in_delivery_context.ok()) return;
        // Large first. If the small one went first and hung, the large one would never be attempted and
        // the comparison the probe exists for would be lost.
        one("large", kLarge);
        one("small", sizeof(uint64_t));
    }
};
OwnBufferProbe g_own_probe;

void probe_device_health(bool* idle, bool* busy) {
    if (idle != nullptr) *idle = false;
    if (busy != nullptr) *busy = false;
    DeliveryContext in_delivery_context;
    if (not in_delivery_context.ok()) return;
    const CUstream stream = in_delivery_context.stream();
    if (stream == nullptr) return;
    const CUresult rc = cuStreamQuery(stream);
    // Reported separately, because "no error" alone cannot distinguish a stream with nothing on it from
    // a stream whose work has not finished -- and those say opposite things about a thread that is
    // stuck inside a copy. An idle stream next to a blocked submit means the submit never reached the
    // stream at all, which points at the driver rather than at the device.
    if (idle != nullptr) *idle = (rc == CUDA_SUCCESS);
    if (busy != nullptr) *busy = (rc == CUDA_ERROR_NOT_READY);
    g_first_device_error.note(rc, "cuStreamQuery from the report");
}


// Deliberate contract violations, for negative controls. Every other impairment in this project slows the
// transport down without breaking its promise; these two break it on purpose, the way `violate-i1` breaks the
// ordering promise. They exist because a verification that has never been shown to fail is not known to
// verify anything: a check passing on a correct backend says nothing until it is also seen to fail on a
// backend that corrupts or loses a placement.
//
//   SRDSIM_CORRUPT_EVERY=N   flip one byte in the middle of every Nth placement applied on this rank
//   SRDSIM_DROP_EVERY=N      report every Nth placement as applied without writing it
//
// Counted per rank, over device placements only, and announced once so a run cannot carry them unnoticed.
struct NegativeControl {
    uint64_t corrupt_every = env_size("SRDSIM_CORRUPT_EVERY", 0);
    uint64_t drop_every = env_size("SRDSIM_DROP_EVERY", 0);
    std::atomic<uint64_t> placements{0};
    std::atomic<uint64_t> corrupted{0};
    std::atomic<uint64_t> dropped{0};
    std::atomic<bool> announced{false};
};
NegativeControl g_negative;

// A thread's receive ring: its group's slots in turn, each with an event recorded after the copy out of it.
// A slot is waited for only when it comes round again, so a thread has up to `ring` copies outstanding.
// Nothing else needs a wait: every copy and every signal goes to the one delivery stream, which runs them
// in submission order, and a signal is submitted after the data it covers (see `SignalShadow::add`).
struct ReceiveRing {
    std::vector<CUevent> done;
    std::vector<bool> pending;
    size_t next = 0;
};

// Set when this thread has submitted a copy that may not have landed; cleared when `delivery_idle` settles it.
thread_local bool t_unsettled = false;

// The agent is about to wait for its next frame. Whatever this thread submitted is waited for now, once per
// burst rather than once per frame, so no copy into a window is still pending while the link is quiet --
// which is when an application that skips its teardown frees its windows without deregistering them.
void delivery_idle(void* /*user*/) {
    if (not t_unsettled) return;
    t_unsettled = false;
    DeliveryContext in_delivery_context;
    if (not in_delivery_context.ok()) return;
    const CUresult rc = cuStreamSynchronize(in_delivery_context.stream());
    if (rc != CUDA_SUCCESS) poison_delivery("settling copies before waiting for the next frame", rc);
}

bool device_copy_async(void* dst, const void* src, size_t bytes, CUstream stream) {
    static thread_local ReceiveRing ring;
    if (g_delivery_poisoned.load()) return false;
    if (ring.done.empty()) {
        if (not make_events(&ring.done, g_pinned.ring())) {
            logf("could not create the receive ring's events");
            return false;
        }
        ring.pending.assign(ring.done.size(), false);
    }
    const size_t i = ring.next;
    if (ring.pending[i]) {
        // Queried first: a slot whose copy has landed needs no wait, and a wait from this context costs a
        // time slice while the application's kernels are resident.
        int state = landed(ring.done[i]);
        if (state == 0) {
            const CUresult w = cuEventSynchronize(ring.done[i]);
            state = w == CUDA_SUCCESS ? 1 : -1;
            if (w != CUDA_SUCCESS) poison_delivery("waiting for a receive staging slot", w);
        }
        if (state < 0) return false;
        ring.pending[i] = false;
    }
    void* pinned = g_pinned.mine(bytes, i);
    std::memcpy(pinned, src, bytes);
    CUresult rc = cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(dst), pinned, bytes, stream);
    if (rc == CUDA_SUCCESS) rc = cuEventRecord(ring.done[i], stream);
    if (rc != CUDA_SUCCESS) {
        // The copy may have been submitted before the record failed, so the slot cannot be trusted.
        poison_delivery("submitting a payload copy", rc);
        return false;
    }
    ring.pending[i] = true;
    ring.next = (i + 1) % ring.done.size();
    t_unsettled = true;
    return true;
}

// Placements applied into device windows, reported at 1, 4, 16, ... on stderr. Registering windows proves only
// that the plugin was loaded; an application that routes its traffic elsewhere registers them all the same,
// and then serves correct answers while the simulator carries nothing. This count is what tells the two apart.
std::atomic<uint64_t> g_device_placements{0};

bool device_copy(void* dst, const void* src, size_t bytes, void* /*user*/) {
    Mark mark("device_copy: applying a placement into a device window");
    probe_records(static_cast<const char*>(src), bytes, "receiver, before the device write", -1, 0);
    {
        const uint64_t k = ++ g_device_placements;
        if ((k & (k - 1)) == 0 and (__builtin_ctzll(k) % 2) == 0)
            lifecycle("device placements applied: %llu", static_cast<unsigned long long>(k));
    }
    if (g_negative.corrupt_every != 0 or g_negative.drop_every != 0) {
        // On stderr rather than through NCCL's logger, which drops INFO unless NCCL_DEBUG=INFO: a control
        // that is active and invisible is the thing this announcement exists to prevent.
        if (not g_negative.announced.exchange(true))
            lifecycle("NEGATIVE CONTROL ACTIVE: corrupt every %llu, drop every %llu placement(s). Results are "
                      "wrong by construction.", static_cast<unsigned long long>(g_negative.corrupt_every),
                      static_cast<unsigned long long>(g_negative.drop_every));
        const uint64_t n = ++ g_negative.placements;
        if (g_negative.drop_every != 0 and n % g_negative.drop_every == 0) {
            const uint64_t d = ++ g_negative.dropped;
            if ((d & (d - 1)) == 0)   // at 1, 2, 4, 8, ...: shows the control is applied, without a line each
                lifecycle("negative control: %llu placement(s) dropped of %llu", static_cast<unsigned long long>(d),
                          static_cast<unsigned long long>(n));
            return true;
        }
        if (g_negative.corrupt_every != 0 and n % g_negative.corrupt_every == 0 and bytes != 0) {
            static thread_local std::vector<char> scratch;
            scratch.assign(static_cast<const char*>(src), static_cast<const char*>(src) + bytes);
            scratch[bytes / 2] = static_cast<char>(scratch[bytes / 2] ^ 0x5a);
            src = scratch.data();
            const uint64_t c = ++ g_negative.corrupted;
            if ((c & (c - 1)) == 0)
                lifecycle("negative control: %llu placement(s) corrupted of %llu, this one %zu bytes",
                          static_cast<unsigned long long>(c), static_cast<unsigned long long>(n), bytes);
        }
    }
#ifdef SRDSIM_WITH_GDRCOPY
    // The BAR mapping first: a CPU store needs no driver call and no wait. Without it the copy below runs
    // from the plugin's own context (see `DeliveryContext`).
    if (g_gdr.write(dst, src, bytes)) return true;
#endif
    DeliveryContext in_delivery_context;
    if (not in_delivery_context.ok()) return false;
    const CUstream stream = device_stream();
    if (delivery_async() and g_pinned.mine(bytes) != nullptr) return device_copy_async(dst, src, bytes, stream);
    if (pinned_staging()) {
        if (void* pinned = g_pinned.mine(bytes)) {
            std::memcpy(pinned, src, bytes);
            src = pinned;
        } else {
            // Named rather than silently degraded. Falling back without a word would turn a deadlock
            // into "it hangs sometimes", which is the failure mode this whole report exists to end.
            logf("could not pin %zu byte(s) of staging memory; copying from pageable memory, which "
                 "blocks in the driver while a resident kernel occupies the device", bytes);
        }
    }
    // Asynchronous submit, then wait on this stream only. The source is a buffer this call owns for
    // its duration, so it must not be reused until the copy has landed; waiting here is what makes
    // that safe without a per-frame allocation.
    CUresult rc = cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(dst), src, bytes, stream);
    if (rc != CUDA_SUCCESS) {
        g_first_device_error.note(rc, "the payload copy submit");
        logf("cuMemcpyHtoDAsync of %zu bytes failed (%d)", bytes, int(rc));
        return false;
    }
    rc = cuStreamSynchronize(stream);
    if (rc != CUDA_SUCCESS) {
        g_first_device_error.note(rc, "the payload copy wait");
        logf("a device copy did not complete (%d)", int(rc));
    }
    return rc == CUDA_SUCCESS;
}

// Incrementing a signal word has to be atomic with respect to the GPU, and two earlier attempts were
// wrong in opposite directions.
//
// Reading the word back from the device, adding, and storing it lost increments: the acceptance rule
// aggregates across senders, there is one agent thread per peer, and two threads read-modify-writing
// the same eight bytes through the host lose each other's work.
//
// Keeping the running total on the host and storing an absolute value fixed that and broke something
// else. NCCL's own source says the GPU resets these counters -- "the GPU kernel is not allowed to reset
// the counter whilst there are outstanding operations", which is a rule about *when* it may, not
// whether. A host-side absolute store overwrites a reset the kernel has performed, and the kernel's
// `landed = readSignal - base` arithmetic then works from a number nothing agrees on.
//
// So the only correct mechanism is a device-side atomic add, which is also what the hardware being
// modelled does: a NIC's write to a counter is atomic with respect to the GPU. The kernel is compiled
// at runtime with NVRTC rather than at build time, so the plugin still builds with a host compiler and
// no CUDA toolchain.
class DeviceAtomicAdd {
public:
    // Exposed so the outcome can be established before anything else has touched the context. Public
    // because the point is to be called from outside the data path.
    bool ready_for_use() { return ensure_ready(); }

    // Compiled once, on first use, because a plugin that pays for this at load time would pay for it
    // even in runs that never touch a device window.
    bool add(void* dst_u64, uint64_t value) {
        if (not ensure_ready()) return false;
        CUdeviceptr dst = reinterpret_cast<CUdeviceptr>(dst_u64);
        void* args[] = {&dst, &value};
        // One lock for the launch, because a CUstream is not safe to submit to from several threads at
        // once and there is one agent thread per peer.
        std::lock_guard<std::mutex> g(launch_mu_);
        const CUresult rc = cuLaunchKernel(kernel_, 1, 1, 1, 1, 1, 1, 0, device_stream(), args,
                                           nullptr);
        if (rc != CUDA_SUCCESS) {
            logf("signal atomic add launch failed (%d)", int(rc));
            return false;
        }
        // Not synchronised here. Synchronising per signal made the run five times slower -- 378 units
        // applied where the previous design managed 1,931 in the same wall clock -- because every
        // signal waited for a stream while a DeepEP kernel occupied the GPU. Ordering is what the
        // reader needs, not a host-side wait: the launches go to one stream in the order the agents
        // released them, so the GPU applies them in that order, and `flush()` is where a caller that
        // needs them all visible waits once.
        //
        // This is a place where being faithful and being fast agree. A real NIC does not stop to
        // acknowledge each counter update either.
        ++ outstanding_;
        return true;
    }

    // Waits for every signal submitted so far. Called from the progress loop rather than from each
    // signal, so the cost is paid once per batch instead of once per word.
    bool flush() {
        std::lock_guard<std::mutex> g(launch_mu_);
        if (outstanding_ == 0) return true;
        DeliveryContext in_delivery_context;   // called from NCCL's progress thread
        if (not in_delivery_context.ok()) return false;
        const CUresult rc = cuStreamSynchronize(device_stream());
        if (rc != CUDA_SUCCESS) {
            logf("signal atomic adds did not complete (%d)", int(rc));
            return false;
        }
        outstanding_ = 0;
        return true;
    }

private:
    bool ensure_ready() {
        std::lock_guard<std::mutex> g(mu_);
        if (ready_) return true;
        if (failed_) return false;
        if (not build()) { failed_ = true; return false; }
        ready_ = true;
        return true;
    }

    bool build() {
        // `unsigned long long` rather than `uint64_t`: atomicAdd has an overload for the former and
        // the two are not the same type to the compiler on every platform.
        // System scope, and a system-scope fence before the add.
        //
        // NCCL registers the signals region with `NCCL_NET_MR_FLAG_FORCE_SO`, and its own source says
        // strong ordering there is "vital to ensure ordering between puts and signals"
        // (`src/gin/gin_host_proxy.cc`). The reader is DeepEP's kernel, which loads the counter with
        // `ld_acquire_sys`. An acquire at system scope pairs with a release at system scope; a plain
        // `atomicAdd` is device scope and a device-scope store is not what that load is waiting for.
        //
        // The fence orders every byte of the payload this signal was bundled with -- written by the copy
        // submitted earlier on the same stream -- before the counter becomes visible. That is the whole
        // contract this plugin exists to keep, expressed in the one place the GPU can enforce it.
        static const char* kSource =
            "extern \"C\" __global__ void srdsim_signal_add(unsigned long long* dst,\n"
            "                                              unsigned long long value) {\n"
            "    __threadfence_system();\n"
            "    atomicAdd_system(dst, value);\n"
            "}\n";
        nvrtcProgram prog{};
        if (nvrtcCreateProgram(&prog, kSource, "srdsim_signal_add.cu", 0, nullptr, nullptr) !=
            NVRTC_SUCCESS) {
            logf("could not create the signal kernel program");
            return false;
        }
        // Compiled for the architecture of the device actually present, asked of the driver rather
        // than assumed: a plugin that hard-coded one would silently produce a cubin this GPU cannot
        // load, which is the same class of mistake as assuming an instance type.
        int major = 0, minor = 0;
        CUdevice dev = 0;
        cuCtxGetDevice(&dev);
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
        // `sm_` and not `compute_`: asking for the real architecture yields a cubin the driver loads
        // directly, while `compute_` yields PTX the driver has to compile itself. That second step is
        // where this broke. On sm_120 the PTX produced by the workspace's NVRTC was newer than the
        // installed driver could accept, `cuModuleLoadData` refused it, and **every signal update
        // silently failed for the rest of the run** -- the first failure was reported and the `failed_`
        // flag made all the others quiet. The visible symptom was a DeepEP barrier waiting forever for
        // a counter nothing was writing.
        //
        // A cubin is also the honest artefact here: the plugin is emulating a NIC's write to a counter
        // on this GPU, not shipping a kernel for future ones.
        char arch[32];
        std::snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%d%d", major, minor);
        const char* opts[] = {arch};
        nvrtcResult compiled = nvrtcCompileProgram(prog, 1, opts);
        std::vector<char> image;
        bool have_cubin = false;
        if (compiled == NVRTC_SUCCESS) {
            size_t cubin_size = 0;
            if (nvrtcGetCUBINSize(prog, &cubin_size) == NVRTC_SUCCESS and cubin_size > 0) {
                image.assign(cubin_size, 0);
                have_cubin = nvrtcGetCUBIN(prog, image.data()) == NVRTC_SUCCESS;
            }
        }
        if (not have_cubin) {
            // Fall back to PTX rather than giving up: an NVRTC too old to emit a cubin for this
            // architecture can still emit PTX, and a driver newer than that NVRTC will load it. The
            // two failure modes are opposite, so keeping both paths covers both.
            size_t log_size = 0;
            nvrtcGetProgramLogSize(prog, &log_size);
            std::vector<char> log(log_size ? log_size : 1, 0);
            nvrtcGetProgramLog(prog, log.data());
            logf("no cubin for sm_%d%d (%s); falling back to PTX", major, minor,
                 log_size > 1 ? log.data() : "no compiler output");
            nvrtcDestroyProgram(&prog);
            nvrtcProgram ptx_prog{};
            if (nvrtcCreateProgram(&ptx_prog, kSource, "srdsim_signal_add.cu", 0, nullptr, nullptr) !=
                NVRTC_SUCCESS) {
                logf("could not create the signal kernel program for the PTX fallback");
                return false;
            }
            char virt[32];
            std::snprintf(virt, sizeof(virt), "--gpu-architecture=compute_%d%d", major, minor);
            const char* ptx_opts[] = {virt};
            compiled = nvrtcCompileProgram(ptx_prog, 1, ptx_opts);
            if (compiled != NVRTC_SUCCESS) {
                nvrtcGetProgramLogSize(ptx_prog, &log_size);
                std::vector<char> plog(log_size ? log_size : 1, 0);
                nvrtcGetProgramLog(ptx_prog, plog.data());
                logf("the signal kernel did not compile for compute_%d%d: %s", major, minor,
                     plog.data());
                nvrtcDestroyProgram(&ptx_prog);
                return false;
            }
            size_t ptx_size = 0;
            nvrtcGetPTXSize(ptx_prog, &ptx_size);
            image.assign(ptx_size, 0);
            nvrtcGetPTX(ptx_prog, image.data());
            nvrtcDestroyProgram(&ptx_prog);
        } else {
            nvrtcDestroyProgram(&prog);
        }

        // The error code is part of the message. Without it this line said only that the load failed,
        // and a load that fails because the driver cannot accept the PTX version looks exactly like one
        // that fails because the image is for the wrong architecture.
        const CUresult loaded = cuModuleLoadData(&module_, image.data());
        if (loaded != CUDA_SUCCESS) {
            const char* name = nullptr;
            cuGetErrorName(loaded, &name);
            logf("could not load the signal kernel module: %s (%d), image is %s of %zu bytes",
                 name ? name : "unknown", int(loaded), have_cubin ? "a cubin" : "PTX", image.size());
            return false;
        }
        if (cuModuleGetFunction(&kernel_, module_, "srdsim_signal_add") != CUDA_SUCCESS) {
            logf("the signal kernel is missing from its own module");
            return false;
        }
        // No stream of its own. Signals go on the same stream as the payload copies, and that is the
        // whole point rather than a saving.
        //
        // The contract this emulator exists to keep is that a signal becomes observable only after the
        // payload it was bundled with has landed. The scheduler releases them in that order and one TCP
        // stream per peer delivers them in that order, but *applying* them was split across two
        // independent CUDA streams -- so the GPU was free to run the signal's atomic add before the
        // payload's copy. Nothing in the local suite can see that: it proves the release order and the
        // wire order, and the reordering happened below both, inside the driver.
        //
        // One stream makes submission order the application order, and submission order is the order the
        // receiving agent read the frames off the wire.
        logf("signal updates use a device-side atomic add, %s for sm_%d%d, on the payload stream",
             have_cubin ? "a cubin" : "PTX", major, minor);
        return true;
    }

    std::mutex mu_;
    std::mutex launch_mu_;
    uint64_t outstanding_ = 0;
    bool ready_ = false;
    bool failed_ = false;
    CUmodule module_{};
    CUfunction kernel_{};
};

// Never destroyed; see `g_where`.
DeviceAtomicAdd& g_device_atomic_add = *new DeviceAtomicAdd;

// Compiles and loads the signal kernel now, reporting the outcome. Called when the first device window
// is registered, which is before any of DeepEP's kernels have run.
void prebuild_signal_kernel() {
    static std::once_flag once;
    std::call_once(once, [] {
        const bool ok = g_device_atomic_add.ready_for_use();
        logf("signal kernel prepared at window registration: %s", ok ? "yes" : "NO");
    });
}

// A counter written the way a NIC writes one: by DMA, from a value the transport already knows.
//
// This is the default; the kernel in `DeviceAtomicAdd` is used only with SRDSIM_SIGNAL_KERNEL=1, so that the
// two can be compared. A NIC does not launch a kernel to increment a counter; it writes eight bytes over the bus. Launching a kernel would give the emulator a dependency on GPU compute that no
// fabric has, and resident kernels (DeepEP's forwarding CTAs, for example) spin while they wait for exactly
// these counters -- so the emulator would need an SM at the moment the code under test occupies them all,
// and whether a run completes would depend on how many SMs the application leaves free.
//
// Correctness rests on one fact from NCCL: the signals region is registered with
// `NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET`, so the counters are cumulative for the life of the window and
// NCCL performs any logical reset by remembering an offset rather than by asking the plugin to zero
// memory (`src/gin/gin_host_proxy.cc`, `nccl_device/gin/proxy/gin_proxy.h`). The plugin is therefore the
// only writer, and an absolute store of a running total it keeps itself is exactly equivalent to an
// atomic add -- without needing the device to run anything.
//
// The store is eight aligned bytes, which is the granularity DeepEP's own comment relies on when it says
// a whole integer write is atomic.
class SignalShadow {
public:
    bool add(void* dst_u64, uint64_t value) {
        const CUstream stream = device_stream();
        // The running total is computed under the lock and copied out to a per-thread slot, so the
        // bytes the copy engine reads cannot be changed by another thread. A shared staging slot would
        // have been correct here too -- the totals only ever increase -- but only after an argument, and
        // a per-thread slot needs none.
        // The whole read-modify-store is under one lock, including the copy. Computing the total under the
        // lock and submitting the store outside it lets two agent threads -- there is one per peer, and one
        // counter is fed by every peer contributing to a part -- store their totals in the opposite order:
        // thread A takes 5, thread B takes 8, B stores first, A stores second, and the device ends at 5.
        // Later adds paper over it, but the last add of a phase does not, and a kernel waiting on that
        // count waits forever. That is the same lost update the contract document claims was already fixed;
        // it had only moved down a layer. The copy is eight bytes, so holding the lock across it costs
        // nothing worth measuring.
        std::lock_guard<std::mutex> g(mu_);
        const uint64_t staged = (totals_[dst_u64] += value);
        // Copied out of pinned memory, not off the stack. This is the same hazard as the payload path: with
        // a pageable source `cuMemcpyHtoDAsync` is not asynchronous,
        // the driver stages the bytes through a pool of its own and that path waits on a device which
        // resident kernels are occupying while they spin for these very counters. Eight bytes in flight
        // are still eight bytes the driver must stage.
        //
        // One slot rather than one per thread, because `mu_` is already held: the lock that makes the
        // totals ordered also makes the slot exclusive, so a second slot would buy nothing.
#ifdef SRDSIM_WITH_GDRCOPY
        // The counter goes through the mapping too, and the absolute total is what makes that safe: this
        // plugin is the only writer, because NCCL sets NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET and resets the
        // counter logically by remembering an offset rather than by touching the memory. So a plain store
        // of the running total is correct without any atomicity against the GPU, which a BAR write does
        // not have.
        if (g_gdr.write(dst_u64, &staged, sizeof(uint64_t))) return true;
#endif
        if (delivery_async() and g_pinned.signal_entries() != 0) return add_async(dst_u64, staged, stream);
        const void* source = &staged;
        if (void* pinned = pinned_slot()) {
            std::memcpy(pinned, &staged, sizeof(uint64_t));
            source = pinned;
        }
        // Synchronous with respect to this stream, for the same reason the payload copy is: the staged
        // value must not change under a copy that has not read it yet.
        CUresult rc = cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(dst_u64), source,
                                        sizeof(uint64_t), stream);
        if (rc == CUDA_SUCCESS) rc = cuStreamSynchronize(stream);
        if (rc != CUDA_SUCCESS) {
            g_first_device_error.note(rc, "the signal counter write");
            const char* name = nullptr;
            cuGetErrorName(rc, &name);
            logf("writing a signal counter failed: %s (%d)", name ? name : "unknown", int(rc));
            return false;
        }
        return true;
    }

    // Called with `mu_` held, so totals reach the stream in the order they were taken. Every contribution a
    // total includes was added by a thread that had already submitted that contribution's data to the same
    // stream (placements precede their signal on each link), so the stream lands the data first. An entry
    // is reused only after its copy has run; otherwise nothing waits.
    bool add_async(void* dst_u64, uint64_t staged, CUstream stream) {
        if (g_delivery_poisoned.load()) return false;
        if (entry_done_.empty()) {
            if (not make_events(&entry_done_, g_pinned.signal_entries())) {
                logf("could not create the signal entries' events");
                return false;
            }
            entry_pending_.assign(entry_done_.size(), false);
        }
        const size_t k = next_entry_;
        if (entry_pending_[k]) {
            int state = landed(entry_done_[k]);
            if (state == 0) {
                const CUresult w = cuEventSynchronize(entry_done_[k]);
                state = w == CUDA_SUCCESS ? 1 : -1;
                if (w != CUDA_SUCCESS) poison_delivery("waiting for a signal entry", w);
            }
            if (state < 0) return false;
            entry_pending_[k] = false;
        }
        void* entry = g_pinned.signal_entry(k);
        std::memcpy(entry, &staged, sizeof(uint64_t));
        CUresult rc = cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(dst_u64), entry, sizeof(uint64_t), stream);
        if (rc == CUDA_SUCCESS) rc = cuEventRecord(entry_done_[k], stream);
        if (rc != CUDA_SUCCESS) {
            poison_delivery("submitting a signal counter write", rc);
            return false;
        }
        entry_pending_[k] = true;
        next_entry_ = (k + 1) % entry_done_.size();
        t_unsettled = true;
        return true;
    }

    void forget_range(void* base, size_t bytes) {
        if (base == nullptr or bytes == 0) return;
        auto* lo = static_cast<char*>(base);
        char* hi = lo + bytes;
        std::lock_guard<std::mutex> g(mu_);
        for (auto it = totals_.begin(); it != totals_.end();) {
            auto* p = static_cast<char*>(it->first);
            it = (p >= lo and p < hi) ? totals_.erase(it) : std::next(it);
        }
    }

private:
    // Eight bytes out of the pool reserved before any kernel existed. Allocating them here instead, even
    // once, would put a page-locked allocation on the data path -- and that call implicitly synchronizes
    // the context, which cannot go quiet while DeepEP spins for the counter this write carries. Lazily
    // allocating narrows that window to the first signal; it does not close it.
    void* pinned_slot() {
        return pinned_staging() ? g_pinned.signal_slot() : nullptr;
    }

    std::mutex mu_;
    std::map<void*, uint64_t> totals_;   // the running total this plugin has written
    std::vector<CUevent> entry_done_;    // asynchronous delivery only: one per signal entry
    std::vector<bool> entry_pending_;
    size_t next_entry_ = 0;
};

// Never destroyed; see `g_where`. Agent threads apply signals until the process is gone.
SignalShadow& g_signal_shadow = *new SignalShadow;

// Clears the running totals for a region that is going away. Without this the totals outlive the memory:
// NCCL frees a context's signals and a later allocation can return the same address, which
// `createContext` then zeroes -- while the shadow still holds the old total, so the first signal on the
// new context stores a large stale absolute value and a barrier is satisfied before anything arrives.
void forget_signal_totals(void* base, size_t bytes) {
    g_signal_shadow.forget_range(base, bytes);
}

// Device windows registered with NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET. Only in these is the plugin the only
// writer of a signal word, so only here may a signal be stored as a running total. A signal into any other
// device window is an atomic add on the device, because the application may reset the word itself (aws-ofi-nccl
// likewise keeps its signal shadow only for windows with this flag).
struct NeverResetWindows {
    std::mutex mu;
    std::map<char*, size_t> ranges;   // base -> bytes
    void add(void* base, size_t bytes) { std::lock_guard<std::mutex> g(mu); ranges[static_cast<char*>(base)] = bytes; }
    void forget(void* base) { std::lock_guard<std::mutex> g(mu); ranges.erase(static_cast<char*>(base)); }
    bool covers(void* p) {
        auto* c = static_cast<char*>(p);
        std::lock_guard<std::mutex> g(mu);
        auto it = ranges.upper_bound(c);
        if (it == ranges.begin()) return false;
        -- it;
        return c + sizeof(uint64_t) <= it->first + it->second;
    }
};
NeverResetWindows& g_never_reset = *new NeverResetWindows;

bool device_add(void* dst_u64, uint64_t value, void* /*user*/) {
    Mark mark("device_add: applying a signal into a device window");
    DeliveryContext in_delivery_context;
    if (not in_delivery_context.ok()) return false;
    // The kernel path is kept because it is the only way to check the two against each other, and
    // because a deployment where the plugin is not the only writer would need it.
    static const bool signal_kernel = env_size("SRDSIM_SIGNAL_KERNEL", 0) != 0;   // read once; see `env_size`
    if (signal_kernel or not g_never_reset.covers(dst_u64))
        return g_device_atomic_add.add(dst_u64, value);
    return g_signal_shadow.add(dst_u64, value);
}
#endif

// Signal counters used two ways. A counter one sender increments both with puts and with bare signals holds two
// kinds of increment; a receiver that counts it as completions will count the bare ones too. What is recorded is
// only that the counter is shared, not what the receiver does with it. NCCL lays a signal out at
// (signalId + contextId * nSignalsPerContext) * 8, so the offset names one counter. Always on: the entries are
// bounded by the counters this rank signals, and the lookup is taken under the lock `submit` already holds for the
// signal-window check.
std::atomic<uint64_t> g_mixed_signal_counters{0};

void note_signal_use_locked(CollComm* cc, const srdsim::Op& op) {
    if (not op.has_signal) return;
    uint8_t& k = cc->signal_uses[{op.peer, op.signal_window, op.signal_off}];
    const uint8_t before = k;
    k |= op.bytes != 0 ? 1 : 2;
    if (before != 3 and k == 3) {
        const uint64_t n = ++ g_mixed_signal_counters;
        if (n <= 16)
            lifecycle("signal counter used both by puts and by bare signals: peer %d window %u offset %llu (slot %llu) "
                      "-- a receiver that counts it as completions would count the bare signals too (%llu such "
                      "counter(s) so far)", op.peer, op.signal_window, static_cast<unsigned long long>(op.signal_off),
                      static_cast<unsigned long long>(op.signal_off / sizeof(uint64_t)), static_cast<unsigned long long>(n));
    }
}

// SRDSIM_DETECT_OVERLAP=2: the overlap check above, across contexts. Each scheduler only knows its own context's
// operations, and an application that writes the same destination bytes through two contexts (two QPs) races
// just the same, since nothing orders operations on different contexts. Every placement still in flight is kept
// here with its context; another context's operation counts as in flight only when its scheduler, asked under its
// own lock, says so -- a busy lock skips the question rather than wait, so this can miss an overlap but never
// invents one. A diagnostic, and a slow one: every submit walks the operations in flight.
bool detect_cross_overlap() {
    static const bool v = env_size("SRDSIM_DETECT_OVERLAP", 0) >= 2;
    return v;
}

struct CrossOverlaps {
    struct InFlight { RmaCtx* ctx = nullptr; srdsim::Op op; };
    std::mutex mu;
    std::vector<InFlight> live;
    uint64_t found = 0;
};
CrossOverlaps& g_cross = *new CrossOverlaps;

// Every get whose reply has not yet fully landed, by an id this process never reuses. The id is the token a
// GetRequest carries and its GetResponse echoes back, so a reply finds its get, and through it the context,
// without trusting anything that crossed the socket: an id with no entry is dropped. A get completes when its
// entry has no bytes left to land. `destroyContext` marks the entries of the context it destroys as aborted,
// so a late reply for them is dropped and `test` finishes them without reporting their data as landed.
struct GetRecord {
    RmaCtx* ctx;         // null once the context was destroyed with the reply outstanding
    int64_t remaining;   // bytes of the reply that have not landed
};
// A leaf lock: nothing else is taken while it is held.
struct OutstandingGets {
    std::mutex mu;
    // Ids start at a random 32-bit epoch, so a reply addressed to another process, or to an earlier run of
    // this one, does not match a live get by accident.
    uint64_t next = (static_cast<uint64_t>(std::random_device{}()) << 32) | 1u;
    std::unordered_map<uint64_t, GetRecord> gets;
};
OutstandingGets& g_gets = *new OutstandingGets;

// Called with `ctx->mu` held, after the scheduler accepted `op`.
void note_cross_overlap(RmaCtx* ctx, const srdsim::Op& op) {
    if (op.bytes == 0) return;
    std::lock_guard<std::mutex> g(g_cross.mu);
    size_t kept = 0;
    for (size_t i = 0; i < g_cross.live.size(); ++ i) {
        const CrossOverlaps::InFlight e = g_cross.live[i];
        bool known = true, pending = false;
        if (e.ctx == ctx) {
            pending = not ctx->sched->complete(e.op.id);
        } else if (e.ctx->mu.try_lock()) {
            pending = not e.ctx->sched->complete(e.op.id);
            e.ctx->mu.unlock();
        } else {
            known = false;
        }
        if (known and not pending) continue;
        g_cross.live[kept ++] = e;
        if (not known or e.op.peer != op.peer or e.op.dst_window != op.dst_window or e.op.dst_base != op.dst_base)
            continue;
        const uint64_t a0 = e.op.dst_off, a1 = e.op.dst_off + e.op.bytes, b0 = op.dst_off, b1 = op.dst_off + op.bytes;
        if (not (a0 < b1 and b0 < a1)) continue;
        const uint64_t k = ++ g_cross.found;
        if (k <= 32 or ((k & (k - 1)) == 0 and (__builtin_ctzll(k) % 2) == 0))
            lifecycle("overlapping writes in flight across contexts (%llu so far): peer %d window %u, earlier op %llu "
                      "on context %p [%llu, +%zu)%s, later op %llu on context %p [%llu, +%zu)%s%s",
                      static_cast<unsigned long long>(k), op.peer, op.dst_window,
                      static_cast<unsigned long long>(e.op.id), static_cast<const void*>(e.ctx),
                      static_cast<unsigned long long>(e.op.dst_off), e.op.bytes, e.op.has_signal ? " signalled" : "",
                      static_cast<unsigned long long>(op.id), static_cast<const void*>(ctx),
                      static_cast<unsigned long long>(op.dst_off), op.bytes, op.has_signal ? " signalled" : "",
                      e.ctx == ctx ? " (same context)" : "");
    }
    g_cross.live.resize(kept);
    g_cross.live.push_back({ctx, op});
}

ncclResult_t submit(RmaCtx* ctx, const srdsim::Op& op, void** request) {
    std::lock_guard<std::mutex> g(ctx->mu);
    srdsim::Op with_id = op;
    with_id.id = ctx->next_op_id++;
    // The window ids are the part worth logging. Everything else can look right while a signal lands
    // in a window the peer is not watching, and then the only symptom is a wait that never ends.
    // NCCL lays indexed signals out as `(signalId + contextId * nSignalsPerContext) * sizeof(uint64_t)`
    // (`src/gin/gin_host_proxy.cc`), so a signal offset must be 8-byte aligned and must fit the window.
    // Checking it here turns "a signal went somewhere" into "a signal went to one nameable counter", and
    // an offset that fails either test is a mapping bug this plugin would otherwise carry out faithfully.
    if (with_id.has_signal) {
        size_t signal_window_bytes = 0;
        {
            std::lock_guard<std::mutex> g2(ctx->coll->mu);
            if (with_id.signal_window < ctx->coll->window_shapes.size())
                signal_window_bytes = ctx->coll->window_shapes[with_id.signal_window].bytes;
            note_signal_use_locked(ctx->coll, with_id);
        }
        if (with_id.signal_off % sizeof(uint64_t) != 0 or
            (signal_window_bytes != 0 and
             with_id.signal_off + sizeof(uint64_t) > signal_window_bytes)) {
            // Reported, not returned. An error out of a submit entry point makes NCCL's GIN progress
            // thread exit permanently, so a diagnostic would cost the whole run -- and the operation is
            // delivered anyway, because refusing it changes nothing about the mapping being wrong.
            // `queryLastError` is the channel the header nominates for exactly this.
            char detail[240];
            std::snprintf(detail, sizeof(detail),
                          "signal op %llu targets offset %llu, which is not an aligned 8-byte slot "
                          "inside window %u (%zu bytes); NCCL computes it as "
                          "(signalId + contextId * nSignalsPerContext) * 8, so this is a mapping error",
                          static_cast<unsigned long long>(with_id.id),
                          static_cast<unsigned long long>(with_id.signal_off), with_id.signal_window,
                          signal_window_bytes);
            logf("%s", detail);
            {
                std::lock_guard<std::mutex> g2(ctx->coll->mu);
                if (ctx->coll->last_error.empty()) ctx->coll->last_error = detail;
            }
        }
    }

    // The signal's own window, offset and value belong in this line. They were left out, and the
    // consequence was that a stall which is *about* signals could be watched for hours without seeing
    // which counter each one targeted. DeepEP's kernels read indexed signals by slot id; the only way to
    // check that slot ids map to distinct offsets is to print the offsets.
    if (log_ops()) {
        if (with_id.has_signal)
            logf("op %llu on context %p: peer=%d bytes=%zu srcOff=%llu dstWin=%u dstOff=%llu "
                 "| signal win=%u off=%llu %s %llu",
                 static_cast<unsigned long long>(with_id.id), static_cast<const void*>(ctx),
                 with_id.peer, with_id.bytes,
                 static_cast<unsigned long long>(with_id.src_off), with_id.dst_window,
                 static_cast<unsigned long long>(with_id.dst_off),
                 with_id.signal_window,
                 static_cast<unsigned long long>(with_id.signal_off),
                 with_id.signal_is_add ? "add" : "inc",
                 static_cast<unsigned long long>(with_id.signal_val));
        else
            logf("op %llu: peer=%d bytes=%zu srcOff=%llu dstWin=%u dstOff=%llu",
                 static_cast<unsigned long long>(with_id.id), with_id.peer, with_id.bytes,
                 static_cast<unsigned long long>(with_id.src_off), with_id.dst_window,
                 static_cast<unsigned long long>(with_id.dst_off));
    }
    if (with_id.has_signal and log_ops())
        logf("op %llu signal: win=%u off=%llu value=%llu op=%s strong=%d",
             static_cast<unsigned long long>(with_id.id), with_id.signal_window,
             static_cast<unsigned long long>(with_id.signal_off),
             static_cast<unsigned long long>(with_id.signal_val),
             with_id.signal_is_add ? "add" : "inc", static_cast<int>(with_id.is_strong));
    if (dry_run()) {
        // Hand back a request that is already complete. Nothing is moved, so results are wrong by
        // construction; this exists only to observe the call sequence.
        Request* r = g_requests.acquire();
        r->done_at_issue = true;
        r->ctx = ctx;
        *request = r;
        return ncclSuccess;
    }
    std::string why;
    ctx->ops_submitted.fetch_add(1, std::memory_order_relaxed);
    const uint64_t overlaps_before = ctx->sched->overlaps();
    const bool submitted = ctx->sched->submit(with_id, &why);
    // A get request names nothing in `dst_window`/`dst_off` -- see the note on `Scheduler::note_overlap`
    // -- so counting it here would manufacture a cross-context overlap against window 0 the same way.
    if (submitted and with_id.is_get_request) {
        ctx->pending_get_bytes.fetch_add(static_cast<int64_t>(with_id.bytes), std::memory_order_relaxed);
        if (log_ops())
            logf("op %llu on context %p is a get: fetch %zu byte(s) from peer %d window %u off %llu "
                 "into local window %u off %llu",
                 static_cast<unsigned long long>(with_id.id), static_cast<const void*>(ctx), with_id.bytes,
                 with_id.peer, with_id.get_src_window, static_cast<unsigned long long>(with_id.get_src_off),
                 with_id.get_dst_window, static_cast<unsigned long long>(with_id.get_dst_off));
    }
    // After the scheduler accepted it, so a refused operation leaves no copy behind; still under `ctx->mu`, so no
    // unit of it can have been released yet.
    if (submitted and snapshot_source()) take_snapshot(ctx, with_id);
    if (submitted and detect_cross_overlap() and not with_id.is_get_request) note_cross_overlap(ctx, with_id);
    if (submitted and ctx->sched->overlaps() != overlaps_before) {
        const uint64_t k = ctx->sched->overlaps();
        if ((k & (k - 1)) == 0 and (__builtin_ctzll(k) % 2) == 0) {
            const srdsim::Op& a = ctx->sched->overlap_earlier();
            const srdsim::Op& b = ctx->sched->overlap_later();
            lifecycle("overlapping writes in flight (%llu so far on context %p): peer %d window %u, earlier op %llu "
                      "[%llu, +%zu)%s, later op %llu [%llu, +%zu)%s -- the result depends on which lands last",
                      static_cast<unsigned long long>(k), static_cast<const void*>(ctx), b.peer, b.dst_window,
                      static_cast<unsigned long long>(a.id), static_cast<unsigned long long>(a.dst_off), a.bytes,
                      a.has_signal ? " signalled" : "", static_cast<unsigned long long>(b.id),
                      static_cast<unsigned long long>(b.dst_off), b.bytes, b.has_signal ? " signalled" : "");
        }
    }
    if (not submitted) {
        // Recorded and reported, not returned. See the note above: an error from here is fatal to NCCL's
        // progress thread, and a refused operation that the caller is never told about at least leaves the
        // rest of the run observable. The request is handed back already complete, because the operation
        // is not going to happen and a request that never completes freezes the caller's queue.
        ctx->has_error = true;
        ctx->error = why;
        logf("refusing operation: %s", why.c_str());
        {
            std::lock_guard<std::mutex> g2(ctx->coll->mu);
            if (ctx->coll->last_error.empty()) ctx->coll->last_error = "refused an operation: " + why;
        }
        Request* r = g_requests.acquire();
        r->done_at_issue = true;  // the operation will not happen; a request that never completes would freeze NCCL
        r->ctx = ctx;
        *request = r;
        return ncclSuccess;
    }
    Request* r = g_requests.acquire();
    r->op_id = with_id.id;
    r->ctx = ctx;
    *request = r;
    return ncclSuccess;
}

// ---------------------------------------------------------------------------
// Op table
// ---------------------------------------------------------------------------

ncclResult_t srdsim_init(void** context, uint64_t /*commId*/, ncclDebugLogger_t logFunction) {
    install_crash_handler_once();
    g_log = logFunction;
    *context = nullptr;
    const auto o = scheduler_options();
    logf("SRD simulator: mode=%s seed=%llu chunk_bytes=%zu log_ops=%d dry_run=%d",
         srdsim::mode_name(o.mode), static_cast<unsigned long long>(o.seed), o.chunk_bytes,
         static_cast<int>(log_ops()), static_cast<int>(dry_run()));
    if (dry_run())
        logf("DRY RUN: operations are accepted and completed without moving data. Results will be "
             "wrong. Use this only to observe which plugin entry points NCCL calls.");
    return ncclSuccess;
}

ncclResult_t srdsim_devices(int* ndev) {
    *ndev = 1;
    return ncclSuccess;
}

ncclResult_t srdsim_get_properties(int /*dev*/, ncclNetProperties_v12_t* props) {
    std::memset(props, 0, sizeof(*props));
    props->name = const_cast<char*>("srdsim");
    props->pciPath = nullptr;
    props->guid = 0;
    // Device windows are supported: the receiving agent writes them with a CUDA copy, so no peer
    // access and no exported allocation are needed.
#ifdef SRDSIM_WITH_CUDA
    props->ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA;
#else
    props->ptrSupport = NCCL_PTR_HOST;
#endif
    // NCCL_PTR_DMABUF is claimed too, for a window that spans more than one physical allocation
    // (NCCL's engram storage: a GPU segment plus a CPU segment registered together) rather than
    // because this plugin has its own dmabuf-fd machinery. `ncclGinRegister` (NCCL's
    // `src/gin/gin_host.cc`) refuses that "multi-segment" case outright unless every active GIN
    // connection reports this bit -- but the registration call it then makes is the ordinary
    // `regMrSym`, unconditionally, not a dmabuf-specific one. The one path that *can* reach for a
    // real fd is the GIN proxy backend's own `ncclGinProxyRegMrSym` (`gin_host_proxy.cc`), for a
    // CUDA-type registration: it asks CUDA for a dmabuf fd and calls `regMrSymDmaBuf`, falling back
    // to plain `regMrSym` on any failure. `srdsim_reg_mr_sym_dmabuf` below already discards the
    // fd/offset and does exactly the plain registration either way, so that call succeeds on the
    // first attempt and the fallback is never needed. Net effect: claiming this bit unlocks
    // multi-segment windows and changes nothing else this plugin has to do.
    props->ptrSupport |= NCCL_PTR_DMABUF;
    props->speed = 400'000;               // nominal; nothing here is a performance measurement
    props->latency = 0;
    props->port = 0;
    props->maxComms = 1024;
    props->maxRecvs = 1;
    // An RMA plugin has to declare itself a GIN proxy device, and this is not a preference: NCCL's
    // own v13 shim refuses the plugin outright otherwise, with
    //
    //   // RMA plugin must report GIN proxy type
    //   if (props.netDeviceType != NCCL_NET_DEVICE_GIN_PROXY) {
    //     WARN("RMA v13 (%s) requires GIN PROXY type, got netDeviceType %d", ...)
    //
    // and, less visibly, NCCL's internal GIN proxy backend reads its GIN type through *this*
    // function (its op table leaves `getProperties` NULL so the call lands on the RMA plugin). So
    // reporting HOST here does not merely mislabel the device, it leaves `ginType` and
    // `railedGinType` at none, which is the assertion DeepEP fails on before any kernel runs. The
    // symptom is three layers away from the cause, which is why it was worth a comment this long.
    props->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
    props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
    return ncclSuccess;
}

// The address to publish. Preferring an explicit override keeps this working when the box has
// several interfaces and NCCL picked a different one for its own bootstrap.
std::string local_ip() {
    const std::string forced = env_str("SRDSIM_LOCAL_IP", "");
    if (not forced.empty())
        return forced;
    // Fall back to the address a UDP socket would use to reach a peer, which resolves the default
    // route without sending anything.
    const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd >= 0) {
        sockaddr_in to{};
        to.sin_family = AF_INET;
        to.sin_port = htons(53);
        ::inet_pton(AF_INET, "10.255.255.255", &to.sin_addr);
        if (::connect(fd, reinterpret_cast<sockaddr*>(&to), sizeof(to)) == 0) {
            sockaddr_in me{};
            socklen_t len = sizeof(me);
            if (::getsockname(fd, reinterpret_cast<sockaddr*>(&me), &len) == 0) {
                char buf[46] = {0};
                if (::inet_ntop(AF_INET, &me.sin_addr, buf, sizeof(buf))) {
                    ::close(fd);
                    return buf;
                }
            }
        }
        ::close(fd);
    }
    return "127.0.0.1";
}

ncclResult_t srdsim_listen(void* /*ctx*/, int /*dev*/, void* handle, void** listenComm) {
    auto* lc = new ListenComm();
    lc->fd = srdsim::listen_any(&lc->port);
    if (lc->fd < 0) {
        delete lc;
        return ncclSystemError;
    }
    ConnHandle h{};
    const std::string ip = local_ip();
    std::snprintf(h.ip, sizeof(h.ip), "%s", ip.c_str());
    h.port = lc->port;
    static_assert(sizeof(ConnHandle) <= NCCL_NET_HANDLE_MAXSIZE,
                  "the connection handle must fit in what NCCL exchanges");
    std::memset(handle, 0, NCCL_NET_HANDLE_MAXSIZE);
    std::memcpy(handle, &h, sizeof(h));
    *listenComm = lc;
    logf("listening on %s:%u", h.ip, static_cast<unsigned>(h.port));
    return ncclSuccess;
}

void report_windows_once(CollComm* cc);   // defined below, next to the window bookkeeping

// The monitor thread. Independent of everything it watches, which is the whole point: a report printed
// from the progress loop prints nothing when the progress loop is what has stopped.
class StallMonitor {
public:
    void watch(CollComm* cc) {
        std::lock_guard<std::mutex> g(mu_);
        comms_.push_back(cc);
        if (thread_.joinable()) return;
        running_.store(true, std::memory_order_release);
        thread_ = std::thread([this] { loop(); });
    }

    // Called before a communicator is destroyed. Without it the monitor keeps a raw pointer and goes on
    // dereferencing `cc->agent`, `cc->links` and `cc->rma_contexts` every half second after `closeColl`
    // has deleted it, and a monitor reading freed memory can print anything. The monitor is armed only
    // when logging is on, so the hazard exists only in that configuration. `connect` watches the comm before a
    // dial can fail and delete it, so that path calls this too.
    void unwatch(CollComm* cc) {
        std::lock_guard<std::mutex> g(mu_);
        comms_.erase(std::remove(comms_.begin(), comms_.end(), cc), comms_.end());
    }

    void stop() {
        running_.store(false, std::memory_order_release);
        if (thread_.joinable()) thread_.join();
    }

    ~StallMonitor() { stop(); }

private:
    void loop() {
        static const int64_t quiet_ms = static_cast<int64_t>(env_size("SRDSIM_STALL_MS", 10000));
        // How often to repeat the report while a stall persists. Often enough that a trend is visible
        // inside one run, rarely enough that the log stays readable.
        static const int64_t kRepeatMs = static_cast<int64_t>(env_size("SRDSIM_STALL_REPEAT_MS", 5000));
        uint64_t last_total = ~0ull;
        auto last_change = std::chrono::steady_clock::now();
        bool reported = false;
        while (running_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            uint64_t total = 0;
            std::vector<CollComm*> comms;
            {
                std::lock_guard<std::mutex> g(mu_);
                comms = comms_;
            }
            for (CollComm* cc : comms)
                if (cc and cc->agent)
                    total += cc->agent->applied_placements() + cc->agent->applied_signals();
            const auto now = std::chrono::steady_clock::now();
            if (total != last_total) {
                last_total = total;
                last_change = now;
                reported = false;
                continue;
            }
            if (total == 0) continue;
            const auto quiet =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - last_change).count();
            if (quiet < quiet_ms) continue;
            // Repeat, on an interval, for as long as the stall lasts. Reporting once cannot answer the
            // question a stall actually poses: is the queue draining slowly or not at all? A scheduler
            // holding many operations in a single snapshot is either the cause of the stall or its
            // consequence -- if progress has stopped being called, work stays held because nothing is
            // asking for it -- and those are opposite conclusions from the same number. Two samples
            // separate them; one cannot.
            const bool first = not reported;
            reported = true;
            if (not first and quiet % kRepeatMs > 500) continue;
            logf("STALL%s: nothing applied for %lld ms after %llu unit(s).%s",
                 first ? "" : " (still)", static_cast<long long>(quiet),
                 static_cast<unsigned long long>(total),
                 first ? " Where every thread is:" : "");
            if (first) g_where.report(&logf);
            for (CollComm* cc : comms) report_windows_once(cc);
        }
    }

    std::mutex mu_;
    std::vector<CollComm*> comms_;
    std::thread thread_;
    std::atomic<bool> running_{false};
};

StallMonitor g_stall_monitor;

ncclResult_t srdsim_connect(void* /*ctx*/, void* handles[], int nranks, int rank,
                            void* listenComm, void** collComm) {
    auto* cc = new CollComm();
    cc->nranks = nranks;
    cc->rank = rank;

    srdsim::ReceiveAgent::Hooks hooks{};
#ifdef SRDSIM_WITH_CUDA
    hooks.device_copy = device_copy;
    hooks.device_add = device_add;
    if (delivery_async()) hooks.idle = delivery_idle;
#endif
    hooks.get_request = srdsim_handle_get_request;
    hooks.get_response_landed = srdsim_handle_get_response_landed;
    // `device_copy`/`device_add` ignore `user`; the get hooks need it, to find which communicator (and
    // so which of its contexts) a request arrived on.
    hooks.user = cc;
    hooks.log = [](const char* m) { logf("agent: %s", m); };
    cc->agent = std::make_unique<srdsim::ReceiveAgent>(&cc->windows, hooks);
    if (log_ops()) {
        g_stall_monitor.watch(cc);
        // The transport reports where it is through the same breadcrumb map the rest of the plugin
        // uses, so one report covers every thread wherever it happens to be.
        cc->links.set_where_hook([](const char* where) { g_where.set(where); });
    }

    // Accept from every peer and dial every peer. Both halves are needed: a rank both sends and
    // receives. Dialling retries, so the order ranks reach this point does not matter.
    auto* lc = static_cast<ListenComm*>(listenComm);
    std::thread acceptor;
    const int expected_inbound = nranks - 1;
    if (lc != nullptr and expected_inbound > 0) {
        acceptor = std::thread([lc, expected_inbound, cc] {
            for (int i = 0; i < expected_inbound; ++ i) {
                const int fd = ::accept(lc->fd, nullptr, nullptr);
                if (fd < 0) break;
                srdsim::set_nodelay(fd);
                if (not cc->agent->serve(fd)) {
                    // Only happens if the comm is being torn down between the accept and here. The
                    // descriptor is ours to close in that case, and dropping it silently would leak
                    // one per teardown race.
                    logf("the receiving agent is shutting down; closing an accepted connection");
                    ::close(fd);
                    continue;
                }
                ++ cc->accepted_connections;
            }
        });
    }

    // A link to itself, over a socket pair. DeepEP puts to its own rank on the rail team -- the local
    // bypass in the dispatch kernel does exactly that -- and on real hardware a NIC loops such a write
    // back and delivers it. Without this link `send_framed` refuses every self-directed operation,
    // which is a reliable transport losing data: the thing this emulator's calibration rule forbids
    // outright. It was refused silently, and the visible symptom was a barrier signal that appeared to
    // vanish while 187 later signals to the same peer arrived.
    //
    // A socket pair rather than a special case in the scheduler, so self traffic takes the same path as
    // everything else: same framing, same release order, same device application. A real fabric reorders
    // loopback traffic too, and a special case would have quietly exempted it from that.
    {
        int sv[2] = {-1, -1};
        if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0) {
            if (cc->agent->serve(sv[1])) {
                cc->links.set(rank, sv[0]);
                logf("rank %d has a loopback link to itself, as the rail team requires", rank);
            } else {
                ::close(sv[0]);
                ::close(sv[1]);
                logf("could not serve the loopback link; self-directed operations will be refused");
            }
        } else {
            logf("could not create the loopback socket pair (%d); self-directed operations will be "
                 "refused", errno);
        }
    }

    for (int peer = 0; peer < nranks; ++ peer) {
        if (peer == rank) continue;
        ConnHandle h{};
        std::memcpy(&h, handles[peer], sizeof(h));
        const int fd = srdsim::dial(h.ip, h.port);
        if (fd < 0) {
            logf("could not reach rank %d at %s:%u", peer, h.ip, static_cast<unsigned>(h.port));
            if (acceptor.joinable()) acceptor.join();
            // The monitor was given this comm a few lines above, so it has to be told before the delete.
            g_stall_monitor.unwatch(cc);
            delete cc;
            return ncclSystemError;
        }
        cc->links.set(peer, fd);
    }
    if (acceptor.joinable()) acceptor.join();

    *collComm = cc;
    logf("connected: rank %d of %d, mesh established", rank, nranks);
    return ncclSuccess;
}

ncclResult_t srdsim_create_context(void* collComm, ncclRmaConfig_v14_t* config, void** rmaCtx) {
    auto* cc = static_cast<CollComm*>(collComm);
    auto* ctx = new RmaCtx();
    ctx->coll = cc;
    ctx->contexts = config ? config->nContexts : 1;
    ctx->sched = std::make_unique<srdsim::Scheduler>(scheduler_options());
    {
        std::lock_guard<std::mutex> g(cc->mu);
        cc->rma_contexts.push_back(ctx);
    }
    *rmaCtx = ctx;
    logf("context created: %d contexts requested", ctx->contexts);
    return ncclSuccess;
}

ncclResult_t srdsim_reg_mr_sym(void* collComm, void* data, size_t size, int type,
                               uint64_t mrFlags, void** mhandle) {
    auto* cc = static_cast<CollComm*>(collComm);
    auto mr = std::make_unique<MrHandle>();
    mr->local_base = data;
    mr->bytes = size;
    mr->ptr_type = type;
    // Registration is collective, so the id assigned here is the same on every rank without any
    // exchange. That id is what travels on the wire; the peer resolves it to its own base.
    srdsim::Window w{};
    w.base = data;
    w.bytes = size;
    w.is_device = (type == NCCL_PTR_CUDA);
    // NCCL asks for properties on a region through `mrFlags`, and its own source says strong ordering
    // on the signals region is "vital to ensure ordering between puts and signals". This plugin does
    // not implement per-region ordering: the scheduler decides ordering globally, and in `srd` mode it
    // keeps exactly the one guarantee EFA gives. Recording the request and saying it is unimplemented
    // is the honest form; discarding the argument silently would leave a contract gap that reads as a
    // working plugin.
    if (mrFlags != 0)
        logf("regMrSym asked for mrFlags 0x%llx on a %zu byte region; this plugin does not implement "
             "per-region ordering, the scheduler owns ordering globally",
             static_cast<unsigned long long>(mrFlags), size);
    mr->mr_flags = mrFlags;
    mr->window_id = cc->windows.add(w);
    {
        std::lock_guard<std::mutex> g(cc->mu);
        if (cc->window_shapes.size() <= mr->window_id) cc->window_shapes.resize(mr->window_id + 1);
        cc->window_shapes[mr->window_id] = {size, w.is_device};
    }
    logf("window %u registered: %zu bytes, %s", mr->window_id, size,
         w.is_device ? "device" : "host");
    lifecycle("window %u registered at %p, %zu bytes, %s", mr->window_id, data, size, w.is_device ? "device" : "host");
#ifdef SRDSIM_WITH_CUDA
    // Build the signal kernel here rather than on the first signal. Two reasons, and the second is why
    // this line exists at all.
    //
    // The first is ordinary: compiling on the data path pays for a compile inside the first signal, and
    // that signal is usually the one a barrier is waiting for.
    //
    // The second is diagnostic. When the load failed later it returned CUDA_ERROR_LAUNCH_FAILED, which
    // is what the driver says about a context that is *already* unusable -- not about the image. That
    // reading cannot be confirmed from the data path, because by then anything could have poisoned the
    // context. Registration happens before any kernel of DeepEP's has run, so a load that succeeds here
    // and fails later says plainly that something in between broke the context, and a load that fails
    // here says the problem is mine. The context in question is the plugin's delivery context, where the
    // kernel is built and launched, not the application's.
    //
    // Only for device windows: a host-only run should not pay for a CUDA context it never uses. A window
    // delivery cannot write is refused here, at the boundary, rather than accepted and failed later on
    // the data path where the only symptom is a peer waiting.
    if (w.is_device and not check_delivery_window(data)) {
        cc->windows.retire(mr->window_id);
        return ncclInvalidUsage;
    }
    if (w.is_device) {
    DeliveryContext in_delivery_context;
    if (not in_delivery_context.ok()) {
        cc->windows.retire(mr->window_id);
        return ncclSystemError;
    }
    {
        if (mrFlags & NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET) g_never_reset.add(data, size);
        prebuild_signal_kernel();
        // The pinned staging pool, for the same reason and one step stronger. Page-locked allocation
        // implicitly synchronizes the context, so it has to happen while the context is still quiet.
        // After DeepEP launches, its forwarding kernels are resident and spinning for the bytes these
        // buffers carry, and an allocation that waits for them to finish waits for something that is
        // waiting for it. Registration is collective and precedes every launch, so this is the last
        // moment at which reserving it is safe.
        g_pinned.reserve();
        g_own_probe.reserve();
#ifdef SRDSIM_WITH_GDRCOPY
        // Mapped here for the same reason everything else is reserved here: `gdr_pin_buffer` commits BAR
        // aperture and talks to the kernel module, and doing that on the data path would put a driver
        // operation back into the hot path this whole change exists to empty.
        g_gdr.add(data, size);
#endif
    }
    }
#endif
    std::lock_guard<std::mutex> g(cc->mu);
    *mhandle = mr.get();
    cc->mrs.push_back(std::move(mr));
    return ncclSuccess;
}

ncclResult_t srdsim_reg_mr_sym_dmabuf(void* collComm, void* data, size_t size, int type,
                                      uint64_t /*offset*/, int /*fd*/, uint64_t mrFlags,
                                      void** mhandle) {
    return srdsim_reg_mr_sym(collComm, data, size, type, mrFlags, mhandle);
}

ncclResult_t srdsim_dereg_mr_sym(void* collComm, void* mhandle) {
    // The window itself, first and regardless of build flags. Leaving it in the table is not a leak: a
    // frame that arrives after deregistration is applied to the freed base, and for a host window that
    // base is ordinary heap, so the write corrupts whatever the application allocated there next, and the
    // crash happens a long way from the write.
    if (auto* cc = static_cast<CollComm*>(collComm)) {
        if (auto* mr = static_cast<MrHandle*>(mhandle)) {
            cc->windows.retire(mr->window_id);   // returns once no write to it is under way
            lifecycle("window %u deregistered", mr->window_id);
        }
    }
#ifdef SRDSIM_WITH_CUDA
    // With asynchronous delivery a copy into (or a staging copy out of) this window may still be on the
    // delivery stream after its write returned. NCCL frees the memory once this returns, so it waits.
    // A queued send may point into it too. The result is reported after the cleanup below, which has to
    // happen either way: NCCL frees the memory regardless of what this returns.
    bool drained = true;
    if (delivery_async()) {
        auto* mr = static_cast<MrHandle*>(mhandle);
        if (mr != nullptr) {
            drained = finish_sends(nullptr, /*send=*/true);
            DeliveryContext in_delivery_context;
            if (in_delivery_context.ok()) {
                const CUresult rc = cuStreamSynchronize(in_delivery_context.stream());
                if (rc != CUDA_SUCCESS) {
                    poison_delivery("waiting for copies before a deregistration", rc);
                    drained = false;
                }
            } else if (in_delivery_context.exists()) {
                // The context exists but this thread cannot enter it: the copies cannot be confirmed. That is not
                // a fault of the stream, so delivery is not stopped for it.
                logf("window %u: cannot enter the delivery context to wait for its copies (%s)", mr->window_id,
                     g_exiting.load() ? "the process is exiting" : "cuCtxPushCurrent failed, logged above");
                drained = false;
            }
            // Without a context there is no delivery stream, so nothing to wait for.
            if (not drained)
                logf("window %u was deregistered with copies or sends that could not be confirmed", mr->window_id);
        }
    }
#endif
    // The signal totals cannot outlive the memory they describe. NCCL frees a context's signals and a
    // later allocation can return the same address, which `createContext` zeroes -- while a surviving
    // total would make the first signal on the new context store a large stale value and satisfy a
    // barrier before anything had arrived.
#ifdef SRDSIM_WITH_CUDA
    if (auto* mr = static_cast<MrHandle*>(mhandle)) {
        forget_signal_totals(mr->local_base, mr->bytes);
        g_never_reset.forget(mr->local_base);
#ifdef SRDSIM_WITH_GDRCOPY
        // The mapping cannot outlive the window for the same reason the signal totals cannot: the address
        // can come back attached to something else.
        g_gdr.forget(mr->local_base);
#endif
    }
#else
    (void)mhandle;
#endif
#ifdef SRDSIM_WITH_CUDA
    if (not drained) return ncclSystemError;
#endif
    return ncclSuccess;
}

ncclResult_t srdsim_destroy_context(void* rmaCtx) {
    auto* ctx = static_cast<RmaCtx*>(rmaCtx);
    if (ctx != nullptr and ctx->coll != nullptr) {
        // Remove it before deleting, or the stall monitor -- which runs on its own thread -- can read a
        // pointer this call is about to free.
        std::lock_guard<std::mutex> g(ctx->coll->mu);
        auto& v = ctx->coll->rma_contexts;
        v.erase(std::remove(v.begin(), v.end(), ctx), v.end());
    }
#ifdef SRDSIM_WITH_CUDA
    // Its queued units point at it. Send what can go, drop the rest, before the pointer dies.
    if (delivery_async()) finish_sends(ctx, /*send=*/false);
#endif
    {   // copies of sources this context will now never send
        std::lock_guard<std::mutex> g(g_snapshots.mu);
        for (auto it = g_snapshots.snaps.begin(); it != g_snapshots.snaps.end();)
            it = it->first.first == ctx ? g_snapshots.snaps.erase(it) : std::next(it);
    }
    {
        // Its gets are marked aborted, not erased: `test` on one must not report it as landed. After this no
        // landed hook can reach the context, so it is deleted outside the lock.
        size_t aborted = 0;
        {
            std::lock_guard<std::mutex> g(g_gets.mu);
            for (auto& entry : g_gets.gets)
                if (entry.second.ctx == ctx) { entry.second.ctx = nullptr; ++ aborted; }
        }
        if (aborted != 0) logf("context %p destroyed with %zu get(s) whose reply had not landed", rmaCtx, aborted);
        delete ctx;
    }
    return ncclSuccess;
}

ncclResult_t srdsim_close_coll(void* collComm) {
    auto* cc = static_cast<CollComm*>(collComm);
    lifecycle("communicator %p closing", collComm);
    if (cc) {
        g_stall_monitor.unwatch(cc);   // before the delete, or the monitor reads freed memory
#ifdef SRDSIM_WITH_CUDA
        if (delivery_async()) drop_sends_on(cc);   // nothing may be sent on it after this
#endif
        cc->responder.stop();          // before the links it sends on go
        if (const size_t r = cc->links.reserved_total())
            logf("communicator %p closing with %zu byte(s) of outbox room still reserved", collComm, r);
        cc->links.shutdown_all();      // tells every peer's agent to finish
        if (cc->agent) cc->agent->stop();
        delete cc;
    }
    return ncclSuccess;
}

ncclResult_t srdsim_close_listen(void* listenComm) {
    auto* lc = static_cast<ListenComm*>(listenComm);
    if (lc) {
        if (lc->fd >= 0) ::close(lc->fd);
        delete lc;
    }
    return ncclSuccess;
}

// A context index outside what `createContext` was asked for is a caller error. It is reported through
// `queryLastError` rather than returned, since an error from a submit entry point stops NCCL's GIN progress
// thread, and the operation is still carried out in an ordering domain of its own.
void check_context(RmaCtx* ctx, int context) {
    if (context >= 0 and context < ctx->contexts) return;
    char detail[160];
    std::snprintf(detail, sizeof(detail), "an operation names context %d, but the RMA context has %d", context,
                  ctx->contexts);
    {
        // The first error is kept and logged; a caller repeating the mistake does not flood the log.
        std::lock_guard<std::mutex> g(ctx->mu);
        if (ctx->has_error) return;
        ctx->has_error = true;
        ctx->error = detail;
    }
    logf("%s", detail);
    std::lock_guard<std::mutex> g(ctx->coll->mu);
    if (ctx->coll->last_error.empty()) ctx->coll->last_error = detail;
}

srdsim::Op build_op(MrHandle* src, uint64_t srcOff, MrHandle* dst, uint64_t dstOff, size_t size,
                    uint32_t rank) {
    srdsim::Op op;
    op.peer = static_cast<int>(rank);
    // The source is local, so a pointer is right here. The destination is named, not pointed at.
    op.src_base = src ? src->local_base : nullptr;
    op.src_off = srcOff;
    op.src_is_device = src != nullptr and src->ptr_type == NCCL_PTR_CUDA;
    op.dst_window = dst ? dst->window_id : 0;
    op.dst_off = dstOff;
    op.bytes = size;
    return op;
}

ncclResult_t srdsim_iput(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle,
                         size_t size, uint64_t dstOff, void* dstMhandle, uint32_t rank,
                         void** request) {
    Mark mark("iput");
    auto* ctx = static_cast<RmaCtx*>(rmaCtx);
    auto op = build_op(static_cast<MrHandle*>(srcMhandle), srcOff,
                       static_cast<MrHandle*>(dstMhandle), dstOff, size, rank);
    op.context = context;
    check_context(ctx, context);
    return submit(ctx, op, request);
}

ncclResult_t srdsim_iput_signal(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle,
                                size_t size, uint64_t dstOff, void* dstMhandle, uint32_t rank,
                                uint64_t signalOff, void* signalMhandle, uint64_t signalValue,
                                uint32_t signalOp, bool isStrongSignal, void** request) {
    Mark mark("iputSignal");
    auto* ctx = static_cast<RmaCtx*>(rmaCtx);
    auto op = build_op(static_cast<MrHandle*>(srcMhandle), srcOff,
                       static_cast<MrHandle*>(dstMhandle), dstOff, size, rank);
    op.context = context;
    check_context(ctx, context);
    auto* sig = static_cast<MrHandle*>(signalMhandle);
    op.has_signal = true;
    op.signal_window = sig ? sig->window_id : 0;
    op.signal_off = signalOff;
    op.signal_val = signalValue;
    op.signal_is_add = (signalOp & NCCL_GIN_SIGNAL_OP_ADD) != 0;
    op.is_strong = isStrongSignal;
    op.signal_is_strong = isStrongSignal;
    // Accepted, and enforced by the scheduler, which holds a strong signal until every earlier operation
    // to that peer has been released. Refusing it was the wrong shape of answer: a non-success return from
    // a submit entry point makes NCCL's GIN progress thread store the code and exit permanently
    // (`src/gin/gin_host.cc`), and the descriptor it had already consumed is never retried -- so one
    // strong signal anywhere ends the run with no message the operator will find. NCCL also reports
    // `supportsStrongSignals = true` on the plugin's behalf regardless of what the plugin thinks, and the
    // default device requirements ask for strong signals, so this is a path the caller can reach without
    // doing anything unusual.
    if (isStrongSignal and log_ops())
        logf("op carries a strong signal; the scheduler will hold it until every earlier operation to "
             "peer %u has been released", rank);
    return submit(ctx, op, request);
}

// A get travels as a request (below, answered on the peer's own context by `srdsim_handle_get_request`)
// and comes back as an ordinary placement tagged GetResponse. The get completes when every byte of the reply
// has landed in the local window (`g_gets`), as an RDMA read completes once its data is in local memory; a
// flush also waits for every outstanding get of the context (`RmaCtx::pending_get_bytes`).
//
// EFA-GDA itself has no Get: the capability this reproduces is aws-ofi-nccl's host-proxy GIN backend
// (`nccl_ofi_gin.h`'s `iget`, over `fi_read`), which DeepEP's engram fetch is written against. Enabling
// it widens this simulator's surface past EFA-GDA on purpose, so that engram can be exercised at all; it
// is not claimed to reproduce EFA-GDA's SRD reordering of a get the way the rest of this file does for
// puts, because EFA-GDA does not have one to reproduce.
ncclResult_t srdsim_iget(void* rmaCtx, int context, uint64_t remoteOff, void* remoteMhandle,
                         size_t size, uint64_t localOff, void* localMhandle, uint32_t rank,
                         void** request) {
    Mark mark("iget");
    auto* ctx = static_cast<RmaCtx*>(rmaCtx);
    auto* remote = static_cast<MrHandle*>(remoteMhandle);
    auto* local = static_cast<MrHandle*>(localMhandle);
    srdsim::Op op;
    op.context = context;
    check_context(ctx, context);
    op.peer = static_cast<int>(rank);
    op.is_get_request = true;
    op.get_src_window = remote ? remote->window_id : 0;
    op.get_src_off = remoteOff;
    op.get_dst_window = local ? local->window_id : 0;
    op.get_dst_off = localOff;
    op.bytes = size;
    op.requester_rank = ctx->coll ? ctx->coll->rank : 0;
    // A get completes when its bytes are in the local window. With nothing to fetch it is complete now.
    if (size == 0) {
        Request* r = g_requests.acquire();
        r->done_at_issue = true;
        r->ctx = ctx;
        *request = r;
        return ncclSuccess;
    }
    if (dry_run()) return submit(ctx, op, request);   // completes at once and fetches nothing
    // The token the peer echoes back on every frame of the reply: this get's entry in `g_gets`.
    uint64_t id;
    {
        std::lock_guard<std::mutex> g(g_gets.mu);
        id = g_gets.next ++;
        g_gets.gets.emplace(id, GetRecord{ctx, static_cast<int64_t>(size)});
    }
    op.get_response_token = id;
    const ncclResult_t rc = submit(ctx, op, request);
    auto* r = static_cast<Request*>(*request);
    if (rc != ncclSuccess or r == nullptr or r->done_at_issue) {   // refused, or a dry run: no reply will come
        std::lock_guard<std::mutex> g(g_gets.mu);
        g_gets.gets.erase(id);
    } else {
        r->get_id = id;
    }
    return rc;
}

// Answers a GetRequest this rank received. A get is one-sided: in RDMA the target's NIC serves the read and its
// CPU takes no part, so nothing here waits for this rank's NCCL progress thread or any of its contexts. The
// receive thread copies the requested bytes (staging a device source through the plugin's own context) and
// hands them to the communicator's responder, which sends them; see `GetResponder`.
bool srdsim_handle_get_request(const void* src, bool src_is_device, size_t length,
                               uint32_t dest_window, uint64_t dest_off, uint32_t requester_rank,
                               uint64_t owner_token, void* user) {
    auto* cc = static_cast<CollComm*>(user);
    if (cc == nullptr) return false;
    if (requester_rank >= static_cast<uint32_t>(cc->nranks)) {
        logf("get %llu: the request names rank %u, outside the communicator's %d; dropping",
             static_cast<unsigned long long>(owner_token), requester_rank, cc->nranks);
        return false;
    }
    // Frames no larger than the requester accepts (the transport's default frame limit), and never empty.
    const size_t chunk = std::min<size_t>(std::max<size_t>(scheduler_options().chunk_bytes, 1), size_t{1} << 22);
    std::vector<char> bytes;
#ifdef SRDSIM_WITH_CUDA
    if (src_is_device) {
        bytes.resize(length);
        if (length != 0 and not read_device(bytes.data(), src, length)) {
            logf("get %llu: could not read %zu byte(s) of the device window for the reply; it is lost",
                 static_cast<unsigned long long>(owner_token), length);
            return false;
        }
    } else
#else
    (void)src_is_device;
#endif
    bytes.assign(static_cast<const char*>(src), static_cast<const char*>(src) + length);
    if (not cc->responder.enqueue(static_cast<int>(requester_rank), dest_window, dest_off, owner_token,
                                  std::move(bytes), chunk)) {
        logf("get %llu: the communicator is closing; the reply is not sent", static_cast<unsigned long long>(owner_token));
        return false;
    }
    return true;
}

// A frame of a get's reply has landed in the local window: its bytes come off the get's entry, and off the
// context's total that a flush waits on.
void srdsim_handle_get_response_landed(uint64_t owner_token, uint64_t bytes, void* /*user*/) {
    std::lock_guard<std::mutex> g(g_gets.mu);
    auto it = g_gets.gets.find(owner_token);
    if (it == g_gets.gets.end()) {
        logf("a get response named get %llu, which is not outstanding; dropping",
             static_cast<unsigned long long>(owner_token));
        return;
    }
    GetRecord& rec = it->second;
    if (rec.ctx == nullptr) {
        logf("a get response for get %llu arrived after its context was destroyed; dropping",
             static_cast<unsigned long long>(owner_token));
        return;
    }
    if (static_cast<int64_t>(bytes) > rec.remaining) {
        logf("get %llu received %llu byte(s) with only %lld outstanding; counting what was outstanding",
             static_cast<unsigned long long>(owner_token), static_cast<unsigned long long>(bytes),
             static_cast<long long>(rec.remaining));
        bytes = static_cast<uint64_t>(rec.remaining);
    }
    rec.remaining -= static_cast<int64_t>(bytes);
    rec.ctx->pending_get_bytes.fetch_sub(static_cast<int64_t>(bytes), std::memory_order_relaxed);
}

ncclResult_t srdsim_iflush(void* rmaCtx, int /*context*/, void* /*mhandle*/, uint32_t /*rank*/,
                           void** request) {
    Mark mark("iflush");
    auto* ctx = static_cast<RmaCtx*>(rmaCtx);
    // A flush completes once everything submitted so far has been released. That is `idle()`, not a
    // question about one operation, so it is marked as such rather than aimed at the newest id.
    Request* r = g_requests.acquire();
    std::lock_guard<std::mutex> g(ctx->mu);
    r->op_id = 0;
    r->is_flush = true;
    r->ctx = ctx;
    *request = r;
    return ncclSuccess;
}


// How the completion path is being driven. NCCL's proxy advances the GPU's consumption index from the
// `done` this function reports, so "was anything ever reported complete" is a different question from
// "did any byte arrive", and a stall can be either. The counters are printed by the stall watchdog
// rather than on a call-count interval: an interval that is right on one machine prints nothing on
// another, and this one printed nothing on the machine it was needed on.
struct TestStats {
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> done{0};
    std::atomic<uint64_t> refused{0};
    std::atomic<uint64_t> flushes{0};
    std::atomic<uint64_t> flushes_done{0};
    // When NCCL last asked about a completion. A call count that has stopped moving is the same
    // observation as a call count that was never large, and only a date separates them.
    std::atomic<uint64_t> last_call_at{0};
};
TestStats g_test_stats;

ncclResult_t srdsim_test(void* /*collComm*/, void* request, int* done) {
    Mark mark("test");
    auto* r = static_cast<Request*>(request);
    ++ g_test_stats.calls;
    g_test_stats.last_call_at.store(srdsim::mono_ms(), std::memory_order_relaxed);
    if (not g_requests.live(r)) {
        ++ g_test_stats.refused;
        // Either NCCL handed back something this plugin never issued, or a request was tested after
        // it was already reported done. Both are contract violations worth naming: silently
        // following the pointer is how this became a SIGSEGV the first time.
        logf("test() called with a request this plugin does not recognise (%p); refusing",
             static_cast<const void*>(r));
        return ncclInternalError;
    }
    if (r->op_id == 0 and not r->is_flush and not r->done_at_issue) {
        // Every other request names an operation, and id 0 is never one: answering from the scheduler would
        // report "not done" forever, which is how this stalled before.
        logf("test() called with a request that names no operation (%p); refusing", static_cast<const void*>(r));
        return ncclInternalError;
    }
    bool complete = r->done_at_issue;
    if (not complete and r->get_id != 0) {
        // A get is complete when its reply has landed, which also means its request was sent.
        std::lock_guard<std::mutex> g(g_gets.mu);
        auto it = g_gets.gets.find(r->get_id);
        if (it == g_gets.gets.end()) {
            logf("test() called on get %llu, which has no entry; refusing", static_cast<unsigned long long>(r->get_id));
            *done = 0;
            return ncclInternalError;
        }
        if (it->second.ctx == nullptr) {
            // Its context is gone, so the reply will never be counted. Reported, and finished, so the request
            // does not stay live forever; its data did not land.
            logf("get %llu completed without its data: its context was destroyed first",
                 static_cast<unsigned long long>(r->get_id));
            complete = true;
        } else {
            complete = it->second.remaining <= 0;
        }
        if (complete) g_gets.gets.erase(it);
    } else if (not complete) {
        std::lock_guard<std::mutex> g(r->ctx->mu);
        // A flush fences both directions on this context: the scheduler being idle is the put side
        // (everything this rank sent has been released), `pending_get_bytes` is the get side (every reply
        // this rank is waiting on has actually landed, not merely been sent by the peer). The same
        // entry point serves both, and `iflush`'s own `rank` argument does not narrow it to one peer.
        complete = r->is_flush
                 ? r->ctx->sched->idle() and r->ctx->pending_get_bytes.load(std::memory_order_relaxed) == 0
                 : r->ctx->sched->complete(r->op_id);
    }
#ifdef SRDSIM_WITH_CUDA
    // Released is not finished while a unit still waits for its source to be staged: reporting it done
    // would let NCCL reuse a source that has not been read yet.
    if (complete and not r->done_at_issue and r->get_id == 0 and delivery_async()) {
        drain_sends();
        complete = r->is_flush
                 ? not send_holds_any_of(r->ctx) and r->ctx->pending_get_bytes.load(std::memory_order_relaxed) == 0
                 : not send_holds(r->ctx, r->op_id);
    }
#endif
    *done = complete ? 1 : 0;
    if (r->is_flush) {
        ++ g_test_stats.flushes;
        if (complete) ++ g_test_stats.flushes_done;
    }
    if (complete) ++ g_test_stats.done;
    // Reuse only after NCCL has been told the request is finished; it will not test it again.
    if (complete)
        g_requests.release(r);
    return ncclSuccess;
}

// What the receiving half of this rank has done. Reported from progress, which is called constantly,
// because "the sender released two units" and "the peer applied two units" are the two numbers that
// separate a send-side bug from a receive-side one, and a wait that never ends looks identical either
// way from the outside.
void report_receive_side(CollComm* cc, uint64_t progress_calls) {
    if (cc == nullptr or cc->agent == nullptr) return;
    if (not log_ops()) return;
    const uint64_t pl = cc->agent->applied_placements();
    const uint64_t sg = cc->agent->applied_signals();
    // The last-reported values have to live on the comm, not in a static. A process holds more than
    // one communicator -- DeepEP builds one per team -- and a shared static makes two independent
    // agents look like one that keeps changing its mind, which is exactly how the first reading of
    // these counters was misread.
    if (pl != cc->last_reported_placements or sg != cc->last_reported_signals or
        progress_calls % 5000000 == 0) {
        cc->last_reported_placements = pl;
        cc->last_reported_signals = sg;
        // Refusals and failed writes are reported apart because they have different owners: a refusal
        // is the peer sending something this agent declined, a failed write is local. One number for
        // both makes a run where every CUDA copy failed read the same as a run where the peer sent
        // nonsense.
        size_t queued = 0;
        for (size_t i = 0; i < cc->links.peers(); ++ i) queued += cc->links.pending_bytes(int(i));
        logf("receive side (comm %p, %d peer connection(s)): %llu placement(s), %llu signal(s) "
             "applied, %llu refused, %llu failed write(s), %zu byte(s) queued to send",
             static_cast<const void*>(cc), cc->accepted_connections,
             static_cast<unsigned long long>(pl), static_cast<unsigned long long>(sg),
             static_cast<unsigned long long>(cc->agent->refused_frames()),
             static_cast<unsigned long long>(cc->agent->failed_writes()), queued);
    }
}


// A stall is the failure mode a fabric simulator has to explain, and it explains nothing by default:
// the caller waits, times out somewhere else entirely, and the simulator's own numbers are never
// looked at. So the plugin watches itself -- from `StallMonitor`, on its own thread.
//
// A reporter that runs on the thread it is reporting about, such as one inside `rmaProgress`, cannot report
// the one failure that matters, because a blocked progress call never reaches it. What a per-comm function
// remembers belongs on the comm, not in a function-local static, which every communicator shares.
void report_windows_once(CollComm* cc);   // defined below, next to the window bookkeeping

// Records a unit the link refused, where a report and `queryLastError` will find it. NCCL discards what
// `rmaProgress` returns, so this is the only channel that works.
void note_send_failure(RmaCtx* ctx, const srdsim::Unit& u) {
    {
        std::lock_guard<std::mutex> g(ctx->mu);
        ctx->has_error = true;
        ctx->error = "the link to a peer failed while sending a released unit";
    }
    char detail[200];
    std::snprintf(detail, sizeof(detail), "the link to peer %d failed while sending a released unit (op %llu)",
                  u.peer, static_cast<unsigned long long>(u.op_id));
    {
        std::lock_guard<std::mutex> g(ctx->coll->mu);
        if (ctx->coll->last_error.empty()) ctx->coll->last_error = detail;
    }
    logf("%s", detail);
}

// A released unit holds room in its peer's outbox until its frame is appended (see `LinkRoom` in
// `srdsim_progress`). One that will not be sent gives the room back.
void release_reservation(CollComm* cc, const srdsim::Unit& u) {
    if (cc == nullptr) return;
    cc->links.unreserve(u.peer, sizeof(srdsim::WireHeader) + (u.kind == srdsim::Unit::Kind::Placement ? u.bytes : 0));
}

ncclResult_t srdsim_progress(void* rmaCtx) {
    Mark mark("rmaProgress");
    auto* ctx = static_cast<RmaCtx*>(rmaCtx);
    // "Nothing arrived" has two very different causes: the units were never released because NCCL
    // never called progress, or they were released and went somewhere the peer is not watching.
    // Counting both sides is what separates them, and the counts are cheap enough to keep always.
    // These are process-wide on purpose -- they answer "is progress being called at all" -- but the
    // context has to be printed with them: without the context pointer, a scheduler holding operations
    // while the global count shows units released does not tell whether progress was never called for
    // that context or called and refusing to release.
    static std::atomic<uint64_t> calls{0}, units{0};
    const uint64_t n = ++ calls;
    if (log_ops() and (n <= 8 or n % 5000000 == 0))
        logf("progress call %llu on context %p (units released so far: %llu)",
             static_cast<unsigned long long>(n), static_cast<const void*>(ctx),
             static_cast<unsigned long long>(units.load()));
    // Push whatever the sockets refused last time. This is the other half of making the send path
    // non-blocking: a frame the socket would not take is queued, and the progress callback -- the one
    // caller NCCL expects to be cheap and repeated -- is where it gets pushed.
    if (ctx->coll) ctx->coll->links.flush_all();
    report_receive_side(ctx->coll, n);
#ifdef SRDSIM_WITH_CUDA
    // One wait per progress call rather than one per signal. Progress is called constantly, so this is
    // a batch boundary that already exists and costs nothing to reuse.
    g_device_atomic_add.flush();
#endif
    // Which contexts to advance. NCCL calls `rmaProgress` with one context, and a plugin that advances
    // only that one depends on NCCL calling it for every context that has work. A real backend has no
    // such dependency: the NIC drains its queues whether or not anybody asks about a particular one. So
    // this is a knob, and the honest default is the faithful behaviour.
    std::vector<RmaCtx*> to_advance;
    static const bool one_context = env_size("SRDSIM_PROGRESS_ONE_CONTEXT", 0) != 0;   // read once; see `env_size`
    if (one_context or ctx->coll == nullptr) {
        to_advance.push_back(ctx);
    } else {
        std::lock_guard<std::mutex> g(ctx->coll->mu);
        to_advance = ctx->coll->rma_contexts;
        if (to_advance.empty()) to_advance.push_back(ctx);
    }

    // Each unit keeps the context that released it. The batch is filled from every context, and a unit handed on
    // under the calling context instead was looked up under the wrong owner: the source snapshots of
    // SRDSIM_SNAPSHOT_SOURCE missed every unit of another context and sent it unchecked, and the asynchronous
    // path keyed its per-operation failure record by the wrong context.
    std::vector<srdsim::Unit> batch;
    std::vector<RmaCtx*> owners;
    // A unit is released only with room reserved for it in its peer's outbox (`SendLinks::reserve`), which
    // `send_unit` then consumes, so a released unit is never refused for budget, whoever else is sending on
    // the link. A peer that refuses once is full for the rest of this call. What is reserved and not released
    // is returned at the end.
    struct LinkRoom final : srdsim::Scheduler::Room {
        srdsim::SendLinks* links = nullptr;
        std::map<int, size_t> granted;   // reserved and not yet taken
        std::set<int> full;
        bool fits(int peer, size_t bytes) override {
            if (links == nullptr) return true;
            if (full.count(peer)) return false;
            const size_t need = sizeof(srdsim::WireHeader) + bytes;
            size_t& have = granted[peer];
            if (have >= need) return true;
            if (not links->reserve(peer, need - have)) { full.insert(peer); return false; }
            have = need;
            return true;
        }
        void take(int peer, size_t bytes) override {
            size_t& have = granted[peer];
            have -= std::min(have, sizeof(srdsim::WireHeader) + bytes);
        }
        ~LinkRoom() override {
            if (links) for (const auto& g : granted) if (g.second) links->unreserve(g.first, g.second);
        }
    };
    for (RmaCtx* c : to_advance) {
        if (c == nullptr or c->sched == nullptr) continue;
        const size_t before = batch.size();
        {
            std::lock_guard<std::mutex> g(c->mu);
            LinkRoom room;
            room.links = c->coll ? &c->coll->links : nullptr;
            c->sched->progress(/*budget=*/8, &batch, &room);
        }
        owners.resize(batch.size(), c);
        c->progress_calls.fetch_add(1, std::memory_order_relaxed);
        c->units_released.fetch_add(batch.size() - before, std::memory_order_relaxed);
    }
    units += batch.size();
    if (log_ops() and not batch.empty())
        logf("progress released %zu unit(s): first is %s op %llu -> peer %d win %u off %llu",
             batch.size(),
             batch.front().kind == srdsim::Unit::Kind::Placement ? "placement" :
             batch.front().kind == srdsim::Unit::Kind::GetRequest ? "get request" : "signal",
             static_cast<unsigned long long>(batch.front().op_id), batch.front().peer,
             batch.front().window,
             static_cast<unsigned long long>(batch.front().window_offset));
    // Every unit is attempted. Returning on the first failure discarded the rest of the batch -- units
    // from other contexts included, since the batch is filled from all of them -- and those units had
    // already been marked released, so `complete()` answered done for operations whose bytes never left.
    // A reliable transport losing data is what this emulator's calibration rule forbids outright, and the
    // return value that reported it is discarded by NCCL anyway (`gin_host_proxy.cc`), so the only channel
    // that works is `queryLastError`.
    size_t failed = 0;
    for (size_t i = 0; i < batch.size(); ++ i) {
        const auto& u = batch[i];
        if (send_unit(owners[i]->coll, u, owners[i]))
            continue;
        release_reservation(owners[i]->coll, u);
        ++ failed;
        note_send_failure(owners[i], u);
    }
#ifdef SRDSIM_WITH_CUDA
    // Units staged by earlier calls whose copies have landed since. Queried, not waited for.
    if (delivery_async()) drain_sends();
#endif
    // The count goes to the context so the stall report can print it. The return value cannot carry it:
    // NCCL discards what `rmaProgress` returns.
    if (failed != 0)
        ctx->send_failures.fetch_add(failed, std::memory_order_relaxed);
    return ncclSuccess;
}

// What each scheduler on this communicator is still holding, every time it is asked. Printing it once
// produced a snapshot that could not be interpreted: a scheduler holding 57 operations ten seconds in
// either drains them or does not, and those are opposite conclusions.
void report_outstanding(CollComm* cc) {
    if (cc == nullptr) return;
    std::vector<RmaCtx*> contexts;
    {
        std::lock_guard<std::mutex> g(cc->mu);
        contexts = cc->rma_contexts;
    }
    for (RmaCtx* ctx : contexts) {
        if (ctx == nullptr or ctx->sched == nullptr) continue;
        // Under the context's own lock. Without it this reads `pending_` while another thread is inside
        // `progress()`, where `reap()` resizes that vector -- a data race, and one that can report counts
        // no reachable state has. A racing instrument reports impossibilities, and they send the reader
        // looking for a bug that is not there.
        srdsim::Scheduler::Outstanding o;
        {
            std::lock_guard<std::mutex> g(ctx->mu);
            o = ctx->sched->outstanding();
        }
        if (o.ops == 0) continue;
        std::string err;
        {
            std::lock_guard<std::mutex> g(ctx->mu);
            if (ctx->has_error) err = ctx->error;
        }
        if (not err.empty())
            logf("REPORT: context %p recorded an error: %s", static_cast<const void*>(ctx), err.c_str());
#ifdef SRDSIM_WITH_CUDA
        if (delivery_async()) logf("REPORT: %s", send_queue_report().c_str());
#endif
        if (ctx->send_failures.load() != 0)
            logf("REPORT: context %p could not send %llu released unit(s)",
                 static_cast<const void*>(ctx),
                 static_cast<unsigned long long>(ctx->send_failures.load()));
        logf("REPORT: context %p is holding %zu operation(s): %zu placement chunk(s) and %zu signal(s) "
             "unreleased, oldest is op %llu to peer %d "
             "| %llu submitted, %llu released, driven %llu time(s)",
             static_cast<const void*>(ctx), o.ops, o.placements, o.signals,
             static_cast<unsigned long long>(o.oldest_op), o.oldest_peer,
             static_cast<unsigned long long>(ctx->ops_submitted.load()),
             static_cast<unsigned long long>(ctx->units_released.load()),
             static_cast<unsigned long long>(ctx->progress_calls.load()));
    }
}

// Who is driving this plugin, and whether anyone still is.
//
// Every line here is a duration rather than a count, because the question a stall poses is "how long
// ago", and a count cannot answer it. Two samples of "driven 254699 times" taken a hundred seconds
// apart say the number did not change, which is equally consistent with three different bugs: NCCL
// stopped calling, the last call never returned, or this report is itself stale.
//
// The distinction is worth stating precisely, because it decides who looks next. NCCL's GIN proxy runs
// one thread by default (`NCCL_GIN_PROXY_NTHREADS`), and one iteration of that thread calls `test`,
// then the submit entry points, then `rmaProgress`. So a block anywhere inside this plugin freezes all
// of those counters at once. A breadcrumb reading `rmaProgress` that last moved 100000 ms ago is a
// call that has not returned -- this plugin's fault. The same breadcrumb 3 ms old is a plugin being
// called and finding nothing to do -- somebody else's.
void report_driving(CollComm* cc) {
    logf("  completion path: test() %llu call(s), %llu done, %llu refused | flush %llu asked, %llu done",
         static_cast<unsigned long long>(g_test_stats.calls.load()),
         static_cast<unsigned long long>(g_test_stats.done.load()),
         static_cast<unsigned long long>(g_test_stats.refused.load()),
         static_cast<unsigned long long>(g_test_stats.flushes.load()),
         static_cast<unsigned long long>(g_test_stats.flushes_done.load()));
    const uint64_t since_test = srdsim::age_ms(g_test_stats.last_call_at.load());
    if (since_test == ~uint64_t(0))
        logf("  no completion has ever been asked about");
    else
        logf("  last test() was %llu ms ago", static_cast<unsigned long long>(since_test));
    // Printed before the rest, because if the device faulted first then every duration below is a
    // consequence and reading them as causes is the mistake this line exists to prevent.
    //
    // Guarded because the whole device path is: a build without CUDA has no context to ask about, and
    // the conformance test builds that way.
#ifdef SRDSIM_WITH_CUDA
    // Asked here rather than on a timer, so the answer and the durations below come from one instant.
    bool stream_idle = false, stream_busy = false;
    probe_device_health(&stream_idle, &stream_busy);
    // After the cheap answers, because it may not come back.
    g_own_probe.run_once();
    const uint64_t err_at = g_first_device_error.at.load(std::memory_order_relaxed);
    if (err_at != 0) {
        const char* w = g_first_device_error.where.load(std::memory_order_relaxed);
        logf("  the device first reported an error %llu ms ago, code %d, seen by %s",
             static_cast<unsigned long long>(srdsim::age_ms(err_at, 0)),
             g_first_device_error.code.load(std::memory_order_relaxed), w ? w : "an unnamed call");
    } else {
        logf("  the device has reported no error and its stream is %s",
             stream_idle ? "idle, so anything blocked in a submit never reached it"
                         : (stream_busy ? "still working" : "in an unreported state"));
    }
#endif
    const auto w = srdsim::write_block_state();
    if (w.threads_inside != 0)
        logf("  %llu thread(s) inside a socket write, the oldest for %llu ms -- a write that does not "
             "drain parks the one thread NCCL drives this plugin with",
             static_cast<unsigned long long>(w.threads_inside),
             static_cast<unsigned long long>(srdsim::age_ms(w.oldest_entered_ms, 0)));
    if (cc != nullptr and cc->agent != nullptr) {
        const auto l = cc->agent->liveness();
        if (l.since_last_header_ms == ~uint64_t(0))
            logf("  the receiving agent has never seen a frame");
        else
            logf("  the receiving agent last read a frame header %llu ms ago",
                 static_cast<unsigned long long>(l.since_last_header_ms));
        // Entered and left are printed as a pair on purpose. Equal counts mean the agent is between
        // applications; a difference of one names an application that started and did not finish. The
        // device is occupied by DeepEP's resident spinning kernels at exactly this moment, so an apply
        // that never returns is the failure this design is most exposed to, and it is invisible in the
        // applied counters -- which simply stop, like everything else.
        logf("  the receiving agent entered %llu device operation(s) and left %llu",
             static_cast<unsigned long long>(l.device_ops_entered),
             static_cast<unsigned long long>(l.device_ops_left));
        if (l.inside_device_op_ms != 0)
            logf("  the receiving agent has been inside a device operation of %llu byte(s) for %llu ms",
                 static_cast<unsigned long long>(l.device_op_bytes),
                 static_cast<unsigned long long>(l.inside_device_op_ms));
    }
    g_where.report(&logf);
}

// Prints every registered window beside what it actually received. Called when the caller asks about
// errors after a stall, which is the moment the difference between "no traffic" and "traffic to the
// wrong place" decides where to look next.
//
// The window table is printed once, because it does not change. The outstanding work is printed every
// time, because it does -- and printing it once produced a snapshot that could not be interpreted: a
// scheduler holding 57 operations ten seconds in either drains them or does not, and those are opposite
// conclusions. A number that is only ever read once cannot distinguish a queue from a leak.
void report_windows_once(CollComm* cc) {
    if (cc == nullptr or cc->agent == nullptr) return;
    report_outstanding(cc);
    // The completion path, the error and what teardown discarded all repeat. Only the window table is
    // printed once, because it does not change. They come before the early return, so that "nothing was
    // refused" is a statement about the whole run and not about its first report.
    report_driving(cc);
    {
        const auto lost = cc->links.discarded();
        if (lost.bytes != 0)
            logf("  teardown discarded %llu byte(s) on %llu link(s)",
                 static_cast<unsigned long long>(lost.bytes),
                 static_cast<unsigned long long>(lost.links));
    }
    {
        std::lock_guard<std::mutex> g(cc->mu);
        if (not cc->last_error.empty())
            logf("  this communicator recorded an error: %s", cc->last_error.c_str());
    }
    if (cc->window_report_done) return;
    cc->window_report_done = true;
    logf("nothing has been applied for a while; this is what every window has received so far");
    // The completion path belongs in the same report. "Did anything arrive" and "was anything ever
    // reported complete" are different questions and a stall can be either, so printing one without
    // the other sends the reader down the wrong half. Reported here rather than on a call-count
    // interval, because an interval that is right on one machine prints nothing on another.
    report_driving(cc);
    std::vector<CollComm::WindowShape> shapes;
    {
        std::lock_guard<std::mutex> g(cc->mu);
        shapes = cc->window_shapes;
    }
    // Per link, before the windows. Which direction stopped is the first question and the totals cannot
    // answer it: three megabytes queued reads the same whether one peer refuses everything or every
    // peer is slightly behind.
    logf("  links out of this rank:");
    for (size_t i = 0; i < cc->links.peers(); ++ i) {
        const auto st = cc->links.send_state(static_cast<int>(i));
        if (st.fd < 0 and st.frames_queued == 0) continue;
        logf("    peer %zu: fd %d, %zu byte(s) waiting, %llu frame(s) queued, %llu byte(s) accepted by "
             "the socket, refused %llu time(s)",
             i, st.fd, st.queued,
             static_cast<unsigned long long>(st.frames_queued),
             static_cast<unsigned long long>(st.bytes_sent),
             static_cast<unsigned long long>(st.would_block));
    }
    logf("  links into this rank:");
    for (const auto& c : cc->agent->connection_states())
        logf("    fd %d: %llu placement(s), %llu signal(s), %llu byte(s)%s",
             c.fd, static_cast<unsigned long long>(c.placements),
             static_cast<unsigned long long>(c.signals),
             static_cast<unsigned long long>(c.bytes),
             c.finished ? " (the peer closed)" : "");
    // Any error this comm has recorded goes at the top of the report. The self-link failure that cost
    // this project a day was logged the moment it happened -- and then buried under a thousand routine
    // lines, so it was never read. A report that omits the one line that explains everything is a
    // report that has to be read twice.
    {
        std::lock_guard<std::mutex> g(cc->mu);
        if (not cc->last_error.empty())
            logf("REPORT: this communicator recorded an error: %s", cc->last_error.c_str());
    }
    logf("window report for comm %p (%zu window(s) registered)", static_cast<const void*>(cc),
         shapes.size());
    for (size_t i = 0; i < shapes.size(); ++ i) {
        const auto a = cc->agent->activity(static_cast<uint32_t>(i));
        const bool touched = a.placements != 0 or a.signals != 0;
        logf("  window %2zu: %9zu bytes %-6s | %6llu placement(s) %6llu signal(s) %10llu byte(s)%s",
             i, shapes[i].bytes, shapes[i].is_device ? "device" : "host",
             static_cast<unsigned long long>(a.placements),
             static_cast<unsigned long long>(a.signals),
             static_cast<unsigned long long>(a.bytes),
             touched ? "" : "   <- nothing arrived here");
        if (touched)
            logf("             offsets %llu..%llu",
                 static_cast<unsigned long long>(a.min_offset),
                 static_cast<unsigned long long>(a.max_end));
        // Every counter that was touched, by slot rather than by total. NCCL's layout is
        // `(signalId + contextId * nSignalsPerContext) * 8`, so the slot number decomposes into a
        // context and a signal id once the per-context count is known -- and the count is the window
        // size divided by 8 divided by the number of contexts. A stalled channel is watching exactly
        // one of these, and a total of 192 signals cannot say whether its one is among them.
        if (not a.added_at.empty()) {
            const size_t slots = shapes[i].bytes / sizeof(uint64_t);
            std::string line;
            char cell[48];
            for (const auto& kv : a.added_at) {
                std::snprintf(cell, sizeof(cell), "%llu:+%llu ",
                              static_cast<unsigned long long>(kv.first / sizeof(uint64_t)),
                              static_cast<unsigned long long>(kv.second));
                line += cell;
                if (line.size() > 900) { logf("             slots %s", line.c_str()); line.clear(); }
            }
            if (not line.empty()) logf("             slots %s", line.c_str());
            logf("             %zu of %zu slot(s) received anything", a.added_at.size(), slots);
        }
    }
}

ncclResult_t srdsim_query_last_error(void* rmaCtx, bool* hasError) {
    auto* ctx = static_cast<RmaCtx*>(rmaCtx);
    std::lock_guard<std::mutex> g(ctx->mu);
    *hasError = ctx->has_error;
    return ncclSuccess;
}

ncclResult_t srdsim_finalize(void* /*context*/) {
    g_stall_monitor.stop();
    return ncclSuccess;
}

// Assigned by name rather than positionally. Several slots in the op-table share a signature
// (`destroyContext`, `closeColl` and `closeListen` are all `ncclResult_t(*)(void*)`), so a
// positional initializer would let two of them be swapped without a type error.
ncclRma_v14_t build_op_table() {
    ncclRma_v14_t t{};
    t.name            = "srdsim";
    t.init            = srdsim_init;
    t.devices         = srdsim_devices;
    t.getProperties   = srdsim_get_properties;
    t.listen          = srdsim_listen;
    t.connect         = srdsim_connect;
    t.createContext   = srdsim_create_context;
    t.regMrSym        = srdsim_reg_mr_sym;
    t.regMrSymDmaBuf  = srdsim_reg_mr_sym_dmabuf;
    t.deregMrSym      = srdsim_dereg_mr_sym;
    t.destroyContext  = srdsim_destroy_context;
    t.closeColl       = srdsim_close_coll;
    t.closeListen     = srdsim_close_listen;
    t.iput            = srdsim_iput;
    t.iputSignal      = srdsim_iput_signal;
    // Was left null: a stub that errors kills NCCL's GIN progress thread permanently (the consumed
    // descriptor is never retried), so `nullptr` -- which NCCL checks for and reports around, from the
    // layer above -- was the safer unimplemented answer than a stub reached by `iget()` calls that
    // then had nowhere good to go. `srdsim_iget` is a real implementation now (see its own comment for
    // what it does and does not reproduce), so that hazard no longer applies.
    t.iget            = srdsim_iget;
    t.iflush          = srdsim_iflush;
    t.test            = srdsim_test;
    t.rmaProgress     = srdsim_progress;
    t.queryLastError  = srdsim_query_last_error;
    t.finalize        = srdsim_finalize;
    return t;
}

// The newer op tables, v15 (NCCL 2.31 knows it) and v16 (2.32), built from the same entry points. NCCL loads
// the newest table it knows, so exporting all three is what keeps 2.31 and 2.32 each on its own path rather
// than both on v14 through a compatibility shim.
//
// What they add, and what this plugin answers:
//   optFlags (v15)       ncclRmaOptFlagsAggregateRequests says more operations for the same peer follow at once,
//                        so a backend may defer its doorbell until the last. The scheduler here never waits for
//                        a doorbell -- it releases what it holds on its own schedule -- so the hint needs no
//                        action; it is counted, so a run can show NCCL used it.
//   getRmaProperties     flushesAllPutsOnAnySignal: whether a received signal, from any peer, implies every put
//   (v16)                received earlier has landed. False. Order is kept per peer and, in `srd` mode, not even
//                        that, so a signal from one peer says nothing about another's puts; true would let NCCL's
//                        barrier skip a flush this fabric needs.
#if defined(RMA_V15_H_)
std::atomic<uint64_t> g_aggregated_ops{0};

void note_opt_flags(uint32_t flags) {
    if ((flags & ncclRmaOptFlagsAggregateRequests) == 0) return;
    const uint64_t k = ++ g_aggregated_ops;
    if ((k & (k - 1)) == 0 and (__builtin_ctzll(k) % 2) == 0)
        lifecycle("operations submitted with AggregateRequests: %llu", static_cast<unsigned long long>(k));
}

ncclResult_t srdsim_iput_v15(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                             uint64_t dstOff, void* dstMhandle, uint32_t rank, uint32_t optFlags, void** request) {
    note_opt_flags(optFlags);
    return srdsim_iput(rmaCtx, context, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, request);
}

ncclResult_t srdsim_iput_signal_v15(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                    uint64_t dstOff, void* dstMhandle, uint32_t rank, uint64_t signalOff,
                                    void* signalMhandle, uint64_t signalValue, uint32_t signalOp,
                                    bool isStrongSignal, uint32_t optFlags, void** request) {
    note_opt_flags(optFlags);
    return srdsim_iput_signal(rmaCtx, context, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, signalOff,
                              signalMhandle, signalValue, signalOp, isStrongSignal, request);
}

// `engram_fetch_impl` sets AggregateRequests on every get but its last to a given peer, the same hint
// `srdsim_iput_v15` already counts, so this counts it too rather than adding a second meaning for it.
ncclResult_t srdsim_iget_v15(void* rmaCtx, int context, uint64_t remoteOff, void* remoteMhandle, size_t size,
                             uint64_t localOff, void* localMhandle, uint32_t rank, uint32_t optFlags,
                             void** request) {
    note_opt_flags(optFlags);
    return srdsim_iget(rmaCtx, context, remoteOff, remoteMhandle, size, localOff, localMhandle, rank, request);
}

// Every slot the v14 table has, by name; the two that changed signature, the new ones by version.
template <typename Table>
Table build_newer_op_table() {
    const ncclRma_v14_t base = build_op_table();
    Table t{};
    t.name            = base.name;
    t.init            = base.init;
    t.devices         = base.devices;
    t.getProperties   = base.getProperties;
    t.listen          = base.listen;
    t.connect         = base.connect;
    t.createContext   = base.createContext;
    t.regMrSym        = base.regMrSym;
    t.regMrSymDmaBuf  = base.regMrSymDmaBuf;
    t.deregMrSym      = base.deregMrSym;
    t.destroyContext  = base.destroyContext;
    t.closeColl       = base.closeColl;
    t.closeListen     = base.closeListen;
    t.iput            = srdsim_iput_v15;
    t.iputSignal      = srdsim_iput_signal_v15;
    t.iget            = srdsim_iget_v15;
    t.iflush          = base.iflush;
    t.test            = base.test;
    t.rmaProgress     = base.rmaProgress;
    t.queryLastError  = base.queryLastError;
    t.finalize        = base.finalize;
    return t;
}
#endif

#if defined(RMA_V16_H_)
ncclResult_t srdsim_get_rma_properties(void* /*collComm*/, ncclRmaProperties_v16_t* props) {
    props->flushesAllPutsOnAnySignal = false;
    return ncclSuccess;
}

ncclRma_v16_t build_v16_op_table() {
    ncclRma_v16_t t = build_newer_op_table<ncclRma_v16_t>();
    t.getRmaProperties = srdsim_get_rma_properties;
    return t;
}
#endif

}   // namespace

extern "C" {

ncclRma_v14_t ncclRmaPlugin_v14 = build_op_table();
#if defined(RMA_V15_H_)
ncclRma_v15_t ncclRmaPlugin_v15 = build_newer_op_table<ncclRma_v15_t>();
#endif
#if defined(RMA_V16_H_)
ncclRma_v16_t ncclRmaPlugin_v16 = build_v16_op_table();
#endif

}   // extern "C"
