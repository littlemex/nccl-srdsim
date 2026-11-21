// SPDX-License-Identifier: Apache-2.0
// One rank of a harness case: the plugin's objects, two registered windows, and a proxy thread that
// calls the plugin the way NCCL's GIN host proxy does (src/gin/gin_host_proxy.cc, ncclGinProxyProgress):
// per context, test the outstanding requests in issue order and retire them at the per-peer frontier,
// then submit up to kPollBatch queued operations per target with the aggregate hint on all but the last,
// and after every context call rmaProgress once, ignoring its result. Registration and teardown run on
// the rank's main thread, as they do on NCCL's.
#pragma once

#include "rma_table.h"

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>
#include <signal.h>
#include <unistd.h>

#include <cerrno>

#if defined(__SANITIZE_ADDRESS__)
#define HARNESS_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define HARNESS_ASAN 1
#endif
#endif
#if defined(HARNESS_ASAN)
#include <sanitizer/lsan_interface.h>
#endif

namespace harness {

using Clock = std::chrono::steady_clock;
constexpr int kMaxRanks = 4;
constexpr int kContexts = 2;
constexpr int kPollBatch = 32;        // NCCL_GIN_PROXY_POLL_BATCH's default
constexpr int kStallMs = 3000;        // nothing observable changed for this long while waiting: a stall
constexpr int kRankWallMs = 10000;    // a rank gives up any single wait after this long
constexpr size_t kDataBytes = 8u << 20;   // room for one put larger than the transport's default frame limit
constexpr size_t kSignalWords = 64;

// Shared between the ranks of one case and the driver, in an anonymous shared mapping.
struct Shared {
    std::atomic<int> arrived{0};
    std::atomic<int> generation{0};
    char handles[kMaxRanks][NCCL_NET_HANDLE_MAXSIZE];
    char status[kMaxRanks][256];
    char where[kMaxRanks][64];        // the wait a rank is in, for a driver that has to kill it
    pid_t pid[kMaxRanks];             // so a rank can stop and continue a peer
    std::atomic<int> flag[16]{};
    std::atomic<int64_t> t0_ns{0};
    std::atomic<int> expected_dead[kMaxRanks]{};   // a case kills this rank on purpose; its SIGKILL is not a failure    // a time one rank publishes for another (steady_clock, shared by processes)
};

inline Shared* g_shared = nullptr;
inline int g_rank = -1, g_nranks = 0;
inline bool g_verbose = false;        // HARNESS_VERBOSE, read once in the driver before any fork

[[noreturn]] inline void pause_forever() { for (;;) std::this_thread::sleep_for(std::chrono::seconds(1)); }

// The first caller on either of the rank's threads writes the status and exits; a second waits for that.
[[noreturn]] inline void finish(const char* status) {
    static std::atomic<bool> finishing{false};
    if (finishing.exchange(true)) pause_forever();
    std::snprintf(g_shared->status[g_rank], sizeof(g_shared->status[g_rank]), "%s", status);
    std::fflush(stdout);
    std::fflush(stderr);
    _exit(std::strcmp(status, "ok") == 0 ? 0 : 1);
}

__attribute__((noreturn, format(printf, 3, 4))) inline void fail(const char* kind, const char* id,
                                                                 const char* fmt, ...) {
    char detail[160];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(detail, sizeof(detail), fmt, ap);
    va_end(ap);
    char s[256];
    std::snprintf(s, sizeof(s), "%s:%s %s", kind, id, detail);
    finish(s);
}

inline void set_where(const char* what) {
    std::snprintf(g_shared->where[g_rank], sizeof(g_shared->where[g_rank]), "%s", what);
}

inline void barrier(const char* what) {
    set_where(what);
    const int gen = g_shared->generation.load(std::memory_order_acquire);
    if (g_shared->arrived.fetch_add(1, std::memory_order_acq_rel) + 1 == g_nranks) {
        g_shared->arrived.store(0, std::memory_order_relaxed);
        g_shared->generation.fetch_add(1, std::memory_order_acq_rel);
        return;
    }
    const auto start = Clock::now();
    while (g_shared->generation.load(std::memory_order_acquire) == gen) {
        if (Clock::now() - start > std::chrono::milliseconds(kRankWallMs)) fail("timeout", "barrier", "%s", what);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

// Waits for a peer to set flag[i]. Bounded by kRankWallMs only: the peer doing the work reports its
// own stall first.
inline void wait_flag(int i, const char* what) {
    set_where(what);
    const auto start = Clock::now();
    while (g_shared->flag[i].load(std::memory_order_acquire) == 0) {
        if (Clock::now() - start > std::chrono::milliseconds(kRankWallMs)) fail("timeout", what, "flag %d", i);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}
// Stops and continues a whole peer process, its plugin threads included.
// stop_peer returns only once every thread of the peer is stopped: the signal alone is only queued.
inline void stop_peer(int r) {
    const pid_t pid = g_shared->pid[r];
    if (kill(pid, SIGSTOP) != 0) fail("harness", "stop", "kill errno %d", errno);
#if defined(__linux__)
    const auto start = Clock::now();
    for (;;) {
        bool all = true;
        const std::string dir = "/proc/" + std::to_string(pid) + "/task";
        if (DIR* d = opendir(dir.c_str())) {
            while (dirent* e = readdir(d)) {
                if (e->d_name[0] == '.') continue;
                char state = 0;
                if (FILE* f = std::fopen((dir + "/" + e->d_name + "/stat").c_str(), "r")) {
                    // The state follows the command name, which is in parentheses and may contain spaces.
                    char buf[512] = {0};
                    const size_t len = std::fread(buf, 1, sizeof(buf) - 1, f);
                    std::fclose(f);
                    if (const char* close = static_cast<const char*>(memrchr(buf, ')', len))) state = close[2];
                }
                if (state != 'T' and state != 't' and state != 0) all = false;
            }
            closedir(d);
        }
        if (all) return;
        if (Clock::now() - start > std::chrono::seconds(1)) fail("harness", "stop", "%s", "peer not stopped after 1 s");
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
#endif
}
inline void continue_peer(int r) { kill(g_shared->pid[r], SIGCONT); }
inline void set_flag(int i) { g_shared->flag[i].store(1, std::memory_order_release); }

// The plugin's log. Warnings are always printed; everything when HARNESS_VERBOSE is set. The line the
// plugin writes at init naming its mode and seed is kept, so a case can check that its settings took.
inline std::mutex g_log_mu;
inline std::string g_init_line;
inline std::string g_poison_line;     // the plugin reported a CUDA error or stopped asynchronous delivery
inline void plugin_log(ncclDebugLogLevel level, unsigned long, const char*, int, const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::lock_guard<std::mutex> g(g_log_mu);
    if (std::strncmp(buf, "SRD simulator: ", 15) == 0) g_init_line = buf;
    if (g_poison_line.empty() and (std::strstr(buf, "FIRST CUDA ERROR") or std::strstr(buf, "asynchronous delivery stopped") or
                                   std::strstr(buf, "outbox room still reserved")))
        g_poison_line = buf;
    if (g_verbose or level == NCCL_LOG_WARN) std::fprintf(stderr, "[rank %d] %s\n", g_rank, buf);
}

// --- Window contents ---------------------------------------------------------------------------------

// A byte that depends on its position, the sender and a salt, with no short period: a chunk landed at
// the wrong offset, swapped or duplicated does not match.
inline unsigned char pattern(size_t i, int sender, int salt) {
    uint64_t x = (uint64_t(i) + 1) * 0x9E3779B97F4A7C15ull ^ (uint64_t(unsigned(sender)) << 32 | unsigned(salt));
    x ^= x >> 29;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 32;
    return static_cast<unsigned char>(x);
}
// Not instrumented: a window filled here is read by the plugin's threads only after a barrier
// between processes, which ThreadSanitizer cannot see.
__attribute__((no_sanitize("thread"))) inline void fill(unsigned char* d, size_t n, int sender, int salt) {
    for (size_t i = 0; i < n; ++ i) d[i] = pattern(i, sender, salt);
}
// Reads a window the plugin's threads may still be writing. The read races with that write by design
// when a case checks what an early signal or completion let through, so it is not instrumented; the
// plugin's own side stays instrumented.
__attribute__((no_sanitize("thread"))) inline size_t mismatches(const unsigned char* d, size_t n, int sender,
                                                                 int salt) {
    size_t bad = 0;
    for (size_t i = 0; i < n; ++ i) bad += d[i] != pattern(i, sender, salt);
    return bad;
}
// The same for bytes [first, first + n) of a put: byte k of the range is compared with pattern(first + k).
__attribute__((no_sanitize("thread"))) inline size_t mismatches_from(const unsigned char* d, size_t first, size_t n,
                                                                      int sender, int salt) {
    size_t bad = 0;
    for (size_t i = 0; i < n; ++ i) bad += d[i] != pattern(first + i, sender, salt);
    return bad;
}
__attribute__((no_sanitize("thread"))) inline size_t nonzero(const unsigned char* d, size_t n) {
    size_t bad = 0;
    for (size_t i = 0; i < n; ++ i) bad += d[i] != 0;
    return bad;
}
constexpr size_t kGuard = 64;          // bytes on each side of a destination that must stay zero
constexpr unsigned char kScribble = 0xA5;

// --- The proxy ---------------------------------------------------------------------------------------

struct Op {
    enum Kind { Put, Signal, PutSignal, Get, Flush } kind = Put;
    int context = 0;
    int peer = 0;
    uint64_t srcOff = 0, dstOff = 0;   // a get reads the peer at srcOff into the local window at dstOff
    size_t size = 0;
    size_t word = 0;                   // signal word
    uint64_t value = 1;
    bool strong = false;
    bool refused = false;              // the case expects the plugin to refuse it; it may name memory outside the window
    bool keep_source = false;          // the source is shared with later operations, so it is not overwritten on retirement
    int  plugin_context = -1;          // the context index passed to the plugin, when not `context` (which picks the queue)
    void* signal_mh = nullptr;         // the signal's window, when it is not the rank's signal window
    void* dst_mh = nullptr;            // the destination window, when it is not the rank's data window
};

// What the rank's main thread sees of one operation: `done` when `test` reported it, `retired` when it
// reached its per-peer frontier, which is what NCCL publishes to the kernel.
struct Ticket {
    Op op;
    std::atomic<int> done{0};
    std::atomic<int> retired{0};
};

struct Rank {
    RmaTable rma;
    void* ctx = nullptr;
    void* coll = nullptr;
    void* rmaCtx = nullptr;
    std::vector<unsigned char> data;
    std::vector<uint64_t> signals;
    void* dataMh = nullptr;
    void* signalMh = nullptr;
    std::atomic<uint64_t> completions{0};   // requests that completed: the stall detector's observable
    std::atomic<uint64_t> steps{0};         // proxy iterations run
    std::atomic<int> step_delay_us{0};      // a pause after each iteration, to keep work in flight longer

    // Submits `op` through the proxy and returns its ticket, which stays valid until teardown.
    Ticket* issue(const Op& op) {
        std::lock_guard<std::mutex> g(mu_);
        tickets_.emplace_back();
        Ticket* t = &tickets_.back();
        t->op = op;
        queue_[op.context][op.peer].push_back(t);
        return t;
    }

    bool last_error() {
        bool has = false;
        check(rma.queryLastError(rmaCtx, &has), "queryLastError");
        return has;
    }

    // Destroys the RMA context and creates a new one in its place, with the proxy paused, as a communicator
    // that rebuilds its contexts would. Operations already issued keep being tested.
    void recreate_context() {
        pause();
        check(rma.destroyContext(rmaCtx), "destroyContext");
        check(rma.createContext(coll, kContexts, &rmaCtx), "createContext");
        resume();
    }

    // Registers one more window, collectively, and deregisters it.
    void* register_window(void* base, size_t bytes, int type, uint64_t flags) {
        void* mh = nullptr;
        check(rma.regMrSym(coll, base, bytes, type, flags, &mh), "regMrSym extra");
        return mh;
    }
    void deregister_window(void* mh) { check(rma.deregMrSym(coll, mh), "deregMrSym extra"); }

    const Ticket* last_ticket() {
        std::lock_guard<std::mutex> g(mu_);
        return tickets_.empty() ? nullptr : &tickets_.back();
    }

    // Registers a window of `size` bytes and deregisters it again, on the main thread while the proxy runs.
    // Collective, as regMrSym is: every rank calls it.
    void register_and_release(size_t size) {
        std::vector<unsigned char> w(size, 0);
        void* mh = nullptr;
        check(rma.regMrSym(coll, w.data(), w.size(), NCCL_PTR_HOST, 0, &mh), "regMrSym extra");
        check(rma.deregMrSym(coll, mh), "deregMrSym extra");
    }

    // The proxy thread runs freely after setup. pause() returns once no iteration is in flight;
    // run_steps(n) runs exactly n iterations and then pauses again.
    void pause() { set_where("pause"); std::lock_guard<std::mutex> g(step_mu_); budget_.store(0); }
    void resume() { std::lock_guard<std::mutex> g(step_mu_); budget_.store(-1); }
    void run_steps(long n) {
        set_where("run_steps");
        { std::lock_guard<std::mutex> g(step_mu_); budget_.store(n); }
        while (budget_.load() != 0) std::this_thread::sleep_for(std::chrono::microseconds(20));
    }

    // Waits until `pred` holds while the proxy runs. Fails as a stall named `what` when neither the
    // completion count nor `observable` changes for kStallMs, and as a timeout past kRankWallMs.
    void wait_until(const std::function<bool()>& pred, const char* what,
                    const std::function<uint64_t()>& observable = nullptr) {
        wait_impl(pred, what, observable, true);
    }
    // Waits for what a peer's work makes visible here, such as a signal word. Bounded by kRankWallMs
    // only, so that a stall is reported by the rank whose own requests stopped moving.
    void wait_peer(const std::function<bool()>& pred, const char* what) { wait_impl(pred, what, nullptr, false); }

    void wait_impl(const std::function<bool()>& pred, const char* what, const std::function<uint64_t()>& observable,
                   bool detect_stall) {
        set_where(what);
        uint64_t last = ~0ull;
        const auto start = Clock::now();
        auto changed_at = start;
        while (not pred()) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            const uint64_t seen = completions.load() + (observable ? observable() : 0);
            const auto now = Clock::now();
            if (seen != last) { last = seen; changed_at = now; }
            else if (detect_stall and now - changed_at > std::chrono::milliseconds(kStallMs)) fail("stall", what, "%s", "");
            if (now - start > std::chrono::milliseconds(kRankWallMs)) fail("timeout", what, "%s", "");
        }
    }
    void wait_retired(const Ticket* t, const char* what) {
        wait_until([&] { return t->retired.load(std::memory_order_acquire) != 0; }, what);
    }
    void wait_done(const Ticket* t, const char* what) {
        wait_until([&] { return t->done.load(std::memory_order_acquire) != 0; }, what);
    }

    uint64_t signal_word(size_t i) const { return __atomic_load_n(&signals[i], __ATOMIC_ACQUIRE); }
    unsigned char* at(uint64_t off) { return data.data() + off; }

    void setup(const std::string& plugin, int table, const char* mode, int seed) {
        std::string why;
        if (not rma.load(plugin, table, &why)) fail("setup", "load", "%s", why.c_str());
        check(rma.init(&ctx, 1, plugin_log), "init");
        {
            // The mode and seed this case asked for are the ones the plugin runs: an unknown mode falls
            // back to srd, which would make every case of that mode a copy of srd.
            char want[96];
            std::snprintf(want, sizeof(want), "mode=%s seed=%d ", mode, seed);
            std::lock_guard<std::mutex> g(g_log_mu);
            if (g_init_line.find(want) == std::string::npos)
                fail("setup", "mode", "asked for '%s', the plugin logged '%s'", want, g_init_line.c_str());
        }
        int ndev = 0;
        check(rma.devices(&ndev), "devices");
        if (ndev < 1) fail("setup", "devices", "%s", "no device");
        ncclNetProperties_v12_t props{};
        check(rma.getProperties(0, &props), "getProperties");
        void* listenComm = nullptr;
        check(rma.listen(ctx, 0, g_shared->handles[g_rank], &listenComm), "listen");
        barrier("handles published");
        void* handles[kMaxRanks];
        for (int r = 0; r < g_nranks; ++ r) handles[r] = g_shared->handles[r];
        check(rma.connect(ctx, handles, g_nranks, g_rank, listenComm, &coll), "connect");
        check(rma.closeListen(listenComm), "closeListen");
#if defined(RMA_V16_H_)
        if (rma.hasRmaProperties()) {
            // Order is kept per peer at most, so a signal from one peer says nothing about another's puts.
            ncclRmaProperties_v16_t rp{};
            rp.flushesAllPutsOnAnySignal = true;
            check(rma.getRmaProperties(coll, &rp), "getRmaProperties");
            if (rp.flushesAllPutsOnAnySignal) fail("assert", "rma-properties", "%s", "flushesAllPutsOnAnySignal");
        }
#endif
        check(rma.createContext(coll, kContexts, &rmaCtx), "createContext");
        // From here on the proxy runs, as NCCL's progress thread does while windows are registered.
        proxy_ = std::thread([this] { loop(); });
        data.assign(kDataBytes, 0);
        signals.assign(kSignalWords, 0);
        // The same windows in the same order on every rank, with the flags NCCL's GIN proxy uses for
        // its signal window.
        check(rma.regMrSym(coll, data.data(), data.size(), NCCL_PTR_HOST, 0, &dataMh), "regMrSym data");
        check(rma.regMrSym(coll, signals.data(), signals.size() * sizeof(uint64_t), NCCL_PTR_HOST,
                           NCCL_NET_MR_FLAG_FORCE_SO | NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET, &signalMh),
              "regMrSym signals");
        barrier("registered");
    }

    // NCCL's order: the progress thread stops, then the context, the windows and the connection go.
    void teardown() {
        // Every operation a case issued must have retired: one that never completes is a failure of the
        // case, not something for teardown to absorb.
        resume();
        wait_until([&] {
            std::lock_guard<std::mutex> g(mu_);
            for (const Ticket& t : tickets_) if (t.retired.load(std::memory_order_acquire) == 0) return false;
            return true;
        }, "unretired");
        barrier("before teardown");
        stop_.store(true);
        proxy_.join();
        check(rma.destroyContext(rmaCtx), "destroyContext");
        check(rma.deregMrSym(coll, signalMh), "deregMrSym signals");
        check(rma.deregMrSym(coll, dataMh), "deregMrSym data");
        check(rma.closeColl(coll), "closeColl");
        check(rma.finalize(ctx), "finalize");
    }

    // The case passed on this rank. Under AddressSanitizer, what teardown left allocated is a finding.
    [[noreturn]] void pass() {
        std::string poison;
        {
            std::lock_guard<std::mutex> g(g_log_mu);
            poison = g_poison_line;
        }
        if (not poison.empty()) fail("assert", "delivery-poisoned", "%s", poison.c_str());
#if defined(HARNESS_ASAN)
        if (__lsan_do_recoverable_leak_check() != 0) fail("leak", "teardown", "%s", "see the report above");
#endif
        finish("ok");
    }

private:
    struct Inflight { Ticket* t; void* req; uint64_t index; };

    std::mutex mu_;                                 // tickets_ and queue_
    std::deque<Ticket> tickets_;
    std::deque<Ticket*> queue_[kContexts][kMaxRanks];
    std::thread proxy_;
    std::atomic<bool> stop_{false};
    std::mutex step_mu_;                            // held for one iteration
    std::atomic<long> budget_{-1};                  // iterations left: -1 without limit
    // Proxy thread only.
    std::vector<Inflight> inflight_[kContexts];     // in issue order
    uint64_t issued_[kContexts][kMaxRanks]{};
    uint64_t frontier_[kContexts][kMaxRanks]{};

    static void check(ncclResult_t r, const char* call) {
        if (r != ncclSuccess) fail("error", call, "returned %d", int(r));
    }

    void loop() {
        while (not stop_.load()) {
            std::unique_lock<std::mutex> g(step_mu_);
            const long b = budget_.load();
            if (b == 0) { g.unlock(); std::this_thread::sleep_for(std::chrono::microseconds(20)); continue; }
            step();
            steps.fetch_add(1);
            if (b > 0) budget_.store(b - 1);
            g.unlock();
            if (const int d = step_delay_us.load()) std::this_thread::sleep_for(std::chrono::microseconds(d));
            else std::this_thread::yield();
        }
    }

    void step() {
        for (int c = 0; c < kContexts; ++ c) {
            poll(c);
            for (int t = 0; t < g_nranks; ++ t) {
                std::vector<Ticket*> batch;
                {
                    std::lock_guard<std::mutex> g(mu_);
                    auto& q = queue_[c][t];
                    while (not q.empty() and int(batch.size()) < kPollBatch) { batch.push_back(q.front()); q.pop_front(); }
                }
                // Aggregate only when this peer's next operation is already queued and this is not the last
                // of the batch, as NCCL decides it. Past the end of a full batch the last-of-batch rule decides,
                // so whether more is queued there does not matter.
                for (size_t p = 0; p < batch.size(); ++ p) submit(c, batch[p], p + 1 < batch.size());
            }
        }
        (void)rma.progress(rmaCtx);
    }

    void submit(int queue, Ticket* t, bool aggregate) {
        const Op& o = t->op;
        const int c = o.plugin_context >= 0 ? o.plugin_context : queue;
        const uint32_t flags = aggregate ? 1u /* ncclRmaOptFlagsAggregateRequests */ : 0u;
        void* req = nullptr;
        switch (o.kind) {
            case Op::Put:
                check(rma.iput(rmaCtx, c, o.srcOff, dataMh, o.size, o.dstOff, dataMh, uint32_t(o.peer), flags, &req), "iput");
                break;
            case Op::Signal:   // NCCL's VA signal: no source and no destination window
                check(rma.iputSignal(rmaCtx, c, 0, nullptr, 0, 0, nullptr, uint32_t(o.peer), o.word * sizeof(uint64_t),
                                     o.signal_mh ? o.signal_mh : signalMh, o.value, NCCL_GIN_SIGNAL_OP_ADD, o.strong,
                                     flags, &req), "iputSignal");
                break;
            case Op::PutSignal:
                check(rma.iputSignal(rmaCtx, c, o.srcOff, dataMh, o.size, o.dstOff, o.dst_mh ? o.dst_mh : dataMh,
                                     uint32_t(o.peer), o.word * sizeof(uint64_t), o.signal_mh ? o.signal_mh : signalMh,
                                     o.value, NCCL_GIN_SIGNAL_OP_ADD, o.strong, flags, &req), "iputSignal");
                break;
            case Op::Get:
                check(rma.iget(rmaCtx, c, o.srcOff, dataMh, o.size, o.dstOff, dataMh, uint32_t(o.peer), flags, &req), "iget");
                break;
            case Op::Flush:
                check(rma.iflush(rmaCtx, c, signalMh, uint32_t(o.peer), &req), "iflush");
                break;
        }
        // NCCL treats a NULL request as done for a flush only (gin_host_proxy.cc, proxyGinProcessGfd). For any
        // other call NCCL would pass the NULL to test(); the harness fails it outright, which is stricter.
        if (req == nullptr and o.kind != Op::Flush) fail("error", "null-request", "kind %d", int(o.kind));
        if (req == nullptr) t->done.store(1, std::memory_order_release);
        inflight_[queue].push_back({t, req, issued_[queue][o.peer] ++});
    }

    // NCCL's proxyGinPollCompletions: test each not-done operation in issue order, and retire one when it
    // is done and its index is the peer's frontier. On retirement the put's source is overwritten, as a
    // kernel reusing its buffer would.
    void poll(int c) {
        auto& v = inflight_[c];
        size_t w = 0;
        for (size_t i = 0; i < v.size(); ++ i) {
            Inflight& f = v[i];
            if (f.t->done.load(std::memory_order_relaxed) == 0) {
                int done = 0;
                check(rma.test(coll, f.req, &done), "test");
                if (done) { f.t->done.store(1, std::memory_order_release); completions.fetch_add(1); }
            }
            const int peer = f.t->op.peer;
            if (f.t->done.load(std::memory_order_relaxed) != 0 and f.index == frontier_[c][peer]) {
                ++ frontier_[c][peer];
                const Op& o = f.t->op;
                if ((o.kind == Op::Put or o.kind == Op::PutSignal) and o.size > 0 and not o.refused and not o.keep_source) {
                    if (o.srcOff > data.size() or o.size > data.size() - o.srcOff)
                        fail("harness", "source-outside-window", "offset %llu size %zu",
                             static_cast<unsigned long long>(o.srcOff), o.size);
                    std::memset(at(o.srcOff), kScribble, o.size);
                }
                f.t->retired.store(1, std::memory_order_release);
                continue;
            }
            v[w ++] = f;
        }
        v.resize(w);
    }
};

}  // namespace harness
