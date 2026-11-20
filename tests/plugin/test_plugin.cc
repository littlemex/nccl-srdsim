// SPDX-License-Identifier: Apache-2.0
// Drives the plugin through its op table as NCCL's GIN proxy does, without NCCL and without a GPU.
//
//   test_plugin PLUGIN_SO TABLE_VERSION [SCENARIO]
//
// Each case (scenario, mode, seed) runs in fresh processes, one per rank, so the plugin's process-wide
// state is per rank as under NCCL. The ranks exchange listen handles and meet at barriers through shared
// memory. Each rank calls the plugin in NCCL's order (init, devices, getProperties, listen, connect,
// createContext, regMrSym, operations, test and rmaProgress, then teardown), polls `test` on every
// outstanding request once per loop and calls `rmaProgress` once per loop, and treats an error from a
// submission or from `test` as fatal, as NCCL does. Scenarios assert only what NCCL or a GPU kernel could
// observe: bytes in a window, a signal word, whether a request completed.
//
// A case fails when a rank reports an assertion, when nothing observable changes for STALL_MS while a
// rank waits, or when it runs past a hard wall clock. Known defects are listed in kExpected with the exact
// failure they produce; such a case is reported XFAIL, and the run fails if a listed case passes (remove it
// from the list with the fix) or fails differently.
#include "rma_table.h"

#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr int kMaxRanks = 4;
constexpr int kStallMs = 3000;        // nothing observable changed for this long while waiting: a stall
constexpr int kWallMs = 60000;        // a case that runs longer is killed
constexpr size_t kDataBytes = 2u << 20;
constexpr size_t kSignalWords = 64;
constexpr int kContexts = 2;

// ---------------------------------------------------------------------------------------------------
// Shared between the ranks of one case
// ---------------------------------------------------------------------------------------------------

struct Shared {
    std::atomic<int> arrived{0};
    std::atomic<int> generation{0};
    char handles[kMaxRanks][NCCL_NET_HANDLE_MAXSIZE];
    char status[kMaxRanks][256];
    std::atomic<int> flag[16]{};
};

Shared* g_shared = nullptr;
int g_rank = -1, g_nranks = 0;

[[noreturn]] void finish(const char* status) {
    std::snprintf(g_shared->status[g_rank], sizeof(g_shared->status[g_rank]), "%s", status);
    std::fflush(stdout);
    std::fflush(stderr);
    _exit(std::strcmp(status, "ok") == 0 ? 0 : 1);
}

[[noreturn]] void fail(const char* kind, const char* id, const char* fmt = "", ...) {
    char detail[160];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(detail, sizeof(detail), fmt, ap);
    va_end(ap);
    char s[256];
    std::snprintf(s, sizeof(s), "%s:%s %s", kind, id, detail);
    finish(s);
}

void barrier(const char* what) {
    const int gen = g_shared->generation.load(std::memory_order_acquire);
    if (g_shared->arrived.fetch_add(1, std::memory_order_acq_rel) + 1 == g_nranks) {
        g_shared->arrived.store(0, std::memory_order_relaxed);
        g_shared->generation.fetch_add(1, std::memory_order_acq_rel);
        return;
    }
    const auto start = Clock::now();
    while (g_shared->generation.load(std::memory_order_acquire) == gen) {
        if (Clock::now() - start > std::chrono::milliseconds(kWallMs)) fail("stall", "barrier", "%s", what);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void quiet_log(ncclDebugLogLevel, unsigned long, const char*, int, const char* fmt, ...) {
    if (std::getenv("HARNESS_VERBOSE") == nullptr) return;
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "[rank %d] ", g_rank);
    std::vfprintf(stderr, fmt, ap);
    std::fputc('\n', stderr);
    va_end(ap);
}

// ---------------------------------------------------------------------------------------------------
// One rank: the plugin's objects, its windows, and a proxy loop shaped like NCCL's
// ---------------------------------------------------------------------------------------------------

struct Rank {
    harness::RmaTable rma;
    void* ctx = nullptr;
    void* coll = nullptr;
    void* rmaCtx = nullptr;
    std::vector<unsigned char> data;
    std::vector<uint64_t> signals;
    void* dataMh = nullptr;
    void* signalMh = nullptr;
    uint64_t completed = 0;   // requests that completed: one observable for the stall detector

    struct Outstanding { void* req; bool done; };
    std::vector<Outstanding> outstanding;

    void check(ncclResult_t r, const char* call) {
        if (r != ncclSuccess) fail("error", call, "returned %d", int(r));
    }

    // One iteration of NCCL's proxy loop: test every outstanding request, then rmaProgress once (whose
    // result NCCL ignores). A request is never tested again once it has completed.
    void step() {
        for (auto& o : outstanding) {
            if (o.done) continue;
            int done = 0;
            check(rma.test(coll, o.req, &done), "test");
            if (done) { o.done = true; ++ completed; }
        }
        (void)rma.progress(rmaCtx);
    }

    Outstanding* track(void* req) {
        outstanding.push_back({req, false});
        return &outstanding.back();
    }

    // Runs the proxy loop (unless `proxy` is false) until `pred` holds. Fails as a stall, named `what`,
    // when neither the number of completed requests nor `observable` changes for kStallMs.
    void wait_until(const std::function<bool()>& pred, const char* what, bool proxy = true,
                    const std::function<uint64_t()>& observable = nullptr) {
        uint64_t last = ~0ull;
        auto changed_at = Clock::now();
        while (not pred()) {
            if (proxy) step(); else std::this_thread::sleep_for(std::chrono::microseconds(50));
            const uint64_t now_seen = completed + (observable ? observable() : 0);
            if (now_seen != last) { last = now_seen; changed_at = Clock::now(); }
            else if (Clock::now() - changed_at > std::chrono::milliseconds(kStallMs)) fail("stall", what);
        }
    }

    // `outstanding` must not reallocate while a pointer into it is held, so it reserves up front.
    void wait_done(size_t index, const char* what) {
        wait_until([&] { step(); return outstanding[index].done; }, what, false);
    }

    uint64_t signal_word(size_t i) const {
        return __atomic_load_n(&signals[i], __ATOMIC_ACQUIRE);
    }

    void setup(const std::string& plugin, int table) {
        std::string why;
        if (not rma.load(plugin, table, &why)) fail("setup", "load", "%s", why.c_str());
        outstanding.reserve(4096);
        check(rma.init(&ctx, 1, quiet_log), "init");
        int ndev = 0;
        check(rma.devices(&ndev), "devices");
        if (ndev < 1) fail("setup", "devices", "no device");
        ncclNetProperties_v12_t props{};
        check(rma.getProperties(0, &props), "getProperties");
        void* listenComm = nullptr;
        check(rma.listen(ctx, 0, g_shared->handles[g_rank], &listenComm), "listen");
        barrier("handles published");
        void* handles[kMaxRanks];
        for (int r = 0; r < g_nranks; ++ r) handles[r] = g_shared->handles[r];
        check(rma.connect(ctx, handles, g_nranks, g_rank, listenComm, &coll), "connect");
        check(rma.closeListen(listenComm), "closeListen");
        check(rma.createContext(coll, kContexts, &rmaCtx), "createContext");
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

    // NCCL's order: the context, then the windows, then the connection.
    void teardown() {
        barrier("before teardown");
        check(rma.destroyContext(rmaCtx), "destroyContext");
        (void)rma.deregMrSym(coll, signalMh);
        (void)rma.deregMrSym(coll, dataMh);
        check(rma.closeColl(coll), "closeColl");
        check(rma.finalize(ctx), "finalize");
    }

    size_t put(int context, uint64_t srcOff, size_t size, uint64_t dstOff, int peer) {
        void* req = nullptr;
        check(rma.iput(rmaCtx, context, srcOff, dataMh, size, dstOff, dataMh, uint32_t(peer), 0, &req), "iput");
        track(req);
        return outstanding.size() - 1;
    }
    size_t put_signal(int context, uint64_t srcOff, size_t size, uint64_t dstOff, int peer, size_t word,
                      uint64_t value, bool strong) {
        void* req = nullptr;
        check(rma.iputSignal(rmaCtx, context, srcOff, dataMh, size, dstOff, dataMh, uint32_t(peer),
                             word * sizeof(uint64_t), signalMh, value, NCCL_GIN_SIGNAL_OP_ADD, strong, 0, &req),
              "iputSignal");
        track(req);
        return outstanding.size() - 1;
    }
    size_t get(int context, uint64_t remoteOff, size_t size, uint64_t localOff, int peer) {
        void* req = nullptr;
        check(rma.iget(rmaCtx, context, remoteOff, dataMh, size, localOff, dataMh, uint32_t(peer), 0, &req), "iget");
        track(req);
        return outstanding.size() - 1;
    }
    size_t flush(int context) {
        void* req = nullptr;
        check(rma.iflush(rmaCtx, context, signalMh, 0, &req), "iflush");
        if (req == nullptr) { outstanding.push_back({nullptr, true}); ++ completed; }   // NULL means done
        else track(req);
        return outstanding.size() - 1;
    }
};

unsigned char pattern(size_t i, int sender, int salt) {
    return static_cast<unsigned char>((i * 131u + unsigned(sender) * 17u + unsigned(salt) * 7u + 1u) & 0xffu);
}
void fill(std::vector<unsigned char>& d, size_t off, size_t n, int sender, int salt) {
    for (size_t i = 0; i < n; ++ i) d[off + i] = pattern(i, sender, salt);
}
size_t mismatches(const std::vector<unsigned char>& d, size_t off, size_t n, int sender, int salt) {
    size_t bad = 0;
    for (size_t i = 0; i < n; ++ i) bad += d[off + i] != pattern(i, sender, salt);
    return bad;
}

// ---------------------------------------------------------------------------------------------------
// Scenarios. Each runs on every rank; most use two.
// ---------------------------------------------------------------------------------------------------

// A put of several chunks, then a strong signal of size 0 on the same context: when the signal word
// reaches its value, every byte of the earlier put has landed.
void put_then_strong_signal(Rank& R) {
    constexpr size_t n = 24 << 10, src = 0, dst = 512 << 10;
    if (g_rank == 0) {
        fill(R.data, src, n, 0, 1);
        const size_t p = R.put(0, src, n, dst, 1);
        const size_t s = R.put_signal(0, 0, 0, 0, 1, 0, 1, /*strong=*/true);
        R.wait_done(p, "put");
        R.wait_done(s, "signal");
    } else if (g_rank == 1) {
        R.wait_until([&] { return R.signal_word(0) >= 1; }, "signal-word");
        if (const size_t bad = mismatches(R.data, dst, n, 0, 1)) fail("assert", "strong-orders-earlier-put", "%zu bytes", bad);
    }
}

// I1 over many rounds: when a weak signal's word reaches its value, the put it was bundled with has landed.
void i1_holds(Rank& R) {
    constexpr size_t n = 64 << 10, src = 0, dst = 1 << 20;
    constexpr int rounds = 12;
    for (int r = 0; r < rounds; ++ r) {
        if (g_rank == 0) {
            fill(R.data, src, n, 0, r);
            const size_t i = R.put_signal(0, src, n, dst, 1, 1, 1, false);
            R.wait_done(i, "put-with-signal");
        } else if (g_rank == 1) {
            R.wait_until([&] { return R.signal_word(1) >= uint64_t(r + 1); }, "signal-word");
            if (const size_t bad = mismatches(R.data, dst, n, 0, r)) fail("assert", "i1", "round %d: %zu bytes", r, bad);
        }
        barrier("round");
    }
}

// The negative control: under violate-i1 the sender releases one batch (the signal, then the first
// chunks) and holds the rest until the target has looked, so the target must see the signal while most
// of its put is missing. A run in which it does not would mean the check does not read what the signal
// guards. Reading the window while the receiving thread may still write it is a race by design.
void violate_control(Rank& R) {
    constexpr size_t n = 1 << 20, src = 0, dst = 1 << 20;
    if (g_rank == 0) {
        fill(R.data, src, n, 0, 3);
        const size_t i = R.put_signal(0, src, n, dst, 1, 2, 1, false);
        R.step();                                             // one rmaProgress: one batch released
        R.wait_until([&] { return g_shared->flag[0].load() != 0; }, "target-looked", false,
                     [&] { return uint64_t(g_shared->flag[0].load()); });
        R.wait_done(i, "put-with-signal");
    } else if (g_rank == 1) {
        R.wait_until([&] { return R.signal_word(2) >= 1; }, "signal-word");
        const size_t bad = mismatches(R.data, dst, n, 0, 3);
        g_shared->flag[0].store(1);
        if (bad == 0) fail("assert", "control-missed", "the signal arrived with its whole put");
    }
}

// A get completes only when its bytes are in the local window, and needs nothing from the owner: the
// owner fills its window and then never calls rmaProgress while the requester waits.
void get_owner_idle(Rank& R) {
    constexpr size_t n = 4 << 10, off = 256 << 10;
    if (g_rank == 1) fill(R.data, off, n, 1, 5);
    barrier("owner filled");
    if (g_rank == 0) {
        const size_t g = R.get(0, off, n, off, 1);
        R.wait_done(g, "get");
        if (const size_t bad = mismatches(R.data, off, n, 1, 5)) fail("assert", "data-after-complete", "%zu bytes", bad);
        g_shared->flag[0].store(1);
    } else if (g_rank == 1) {
        R.wait_until([&] { return g_shared->flag[0].load() != 0; }, "requester-done", false,
                     [&] { return uint64_t(g_shared->flag[0].load()); });
    }
}

// A get of several chunks completes, and a flush issued after its data has landed completes too.
void get_multichunk_flush(Rank& R) {
    constexpr size_t n = 64 << 10, off = 768 << 10;
    if (g_rank == 1) fill(R.data, off, n, 1, 6);
    barrier("owner filled");
    if (g_rank == 0) {
        const size_t g = R.get(0, off, n, off, 1);
        R.wait_done(g, "get");
        R.wait_until([&] { return mismatches(R.data, off, n, 1, 6) == 0; }, "get-data",
                     true, [&] { return uint64_t(n - mismatches(R.data, off, n, 1, 6)); });
        const size_t f = R.flush(0);
        R.wait_done(f, "flush");
        g_shared->flag[0].store(1);
    } else if (g_rank == 1) {
        R.wait_until([&] { return g_shared->flag[0].load() != 0; }, "requester-done", true,
                     [&] { return uint64_t(g_shared->flag[0].load()); });
    }
}

// Under SRDSIM_DRY_RUN=1 every operation completes at once without moving data.
void dry_run(Rank& R) {
    if (g_rank == 0) {
        const size_t p = R.put(0, 0, 4096, 4096, 1);
        R.wait_done(p, "put");
    }
}

struct Scenario {
    const char* name;
    void (*run)(Rank&);
    std::vector<const char*> modes;
    std::vector<int> seeds;
    const char* extra_env;   // NAME=VALUE set in every rank, or nullptr
    bool races_by_design;    // skipped under ThreadSanitizer
};

const std::vector<const char*> kKeepI1 = {"srd", "inorder", "peer-seq", "reverse-ops", "signal-last", "defer-one"};

const std::vector<Scenario>& scenarios() {
    static const std::vector<Scenario> s = {
        {"put_then_strong_signal", put_then_strong_signal, kKeepI1, {1, 2, 3}, nullptr, false},
        {"i1_holds", i1_holds, kKeepI1, {1, 2, 3}, nullptr, false},
        {"violate_control", violate_control, {"violate-i1"}, {1, 2, 3}, nullptr, true},
        {"get_owner_idle", get_owner_idle, {"srd"}, {1}, nullptr, true},
        {"get_multichunk_flush", get_multichunk_flush, {"srd"}, {1}, nullptr, true},
        {"dry_run", dry_run, {"srd"}, {1}, "SRDSIM_DRY_RUN=1", false},
    };
    return s;
}

// Known defects (#26): the scenario fails today with exactly this rank status prefix. A fix removes its
// entry; the run fails if a listed scenario passes or fails differently.
// status_prefix is the kind and the id, "kind:id"; ids contain no spaces.
struct Expected { const char* scenario; const char* status_prefix; const char* issue; };
const Expected kExpected[] = {
    {"get_owner_idle", "assert:data-after-complete", "#26: a get completes before its data arrives"},
    {"get_multichunk_flush", "stall:flush", "#26: pending_gets is miscounted for a get of several chunks"},
    {"dry_run", "stall:put", "#26: a request numbered 0 never completes"},
};

const Expected* expected_for(const char* scenario) {
    for (const auto& e : kExpected) if (std::strcmp(e.scenario, scenario) == 0) return &e;
    return nullptr;
}

bool under_tsan() {
#if defined(__SANITIZE_THREAD__)
    return true;
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
    return true;
#endif
#endif
    return false;
}

// Runs one case in fresh processes and returns the first rank's non-ok status, or "ok".
std::string run_case(const std::string& plugin, int table, const Scenario& sc, const char* mode, int seed) {
    auto* shared = static_cast<Shared*>(mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE,
                                              MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    new (shared) Shared();
    const int nranks = 2;
    std::vector<pid_t> pids;
    for (int r = 0; r < nranks; ++ r) {
        const pid_t pid = fork();
        if (pid == 0) {
            setpgid(0, 0);
            g_shared = shared; g_rank = r; g_nranks = nranks;
            setenv("SRDSIM_MODE", mode, 1);
            setenv("SRDSIM_SEED", std::to_string(seed).c_str(), 1);
            if (sc.extra_env) putenv(const_cast<char*>(sc.extra_env));
            Rank R;
            R.setup(plugin, table);
            sc.run(R);
            R.teardown();
            finish("ok");
        }
        pids.push_back(pid);
    }
    std::string result = "ok";
    std::vector<bool> exited(nranks, false);
    const auto start = Clock::now();
    int left = nranks;
    while (left > 0) {
        for (int r = 0; r < nranks; ++ r) {
            if (exited[r]) continue;
            int st = 0;
            if (waitpid(pids[r], &st, WNOHANG) != pids[r]) continue;
            exited[r] = true; -- left;
            std::string s = shared->status[r];
            if (s.empty()) s = WIFSIGNALED(st) ? "crash:signal-" + std::to_string(WTERMSIG(st)) : "crash:no-status";
            if (s != "ok" and result == "ok") {
                result = "rank " + std::to_string(r) + ": " + s;
                for (int q = 0; q < nranks; ++ q) if (not exited[q]) kill(-pids[q], SIGKILL);   // peers would wait forever
            }
        }
        if (Clock::now() - start > std::chrono::milliseconds(kWallMs)) {
            for (int q = 0; q < nranks; ++ q) if (not exited[q]) kill(-pids[q], SIGKILL);
            if (result == "ok") result = "killed:wall-clock";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    munmap(shared, sizeof(Shared));
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s PLUGIN_SO TABLE_VERSION [SCENARIO]\n", argv[0]); return 2; }
    const std::string plugin = argv[1];
    const int table = std::atoi(argv[2]);
    const char* only = argc > 3 ? argv[3] : nullptr;
    signal(SIGPIPE, SIG_IGN);
    int passed = 0, xfailed = 0, failed = 0, skipped = 0;
    for (const auto& sc : scenarios()) {
        if (only and std::strcmp(only, sc.name) != 0) continue;
        if (sc.races_by_design and under_tsan()) { ++ skipped; std::printf("SKIP   %s (races by design)\n", sc.name); continue; }
        const Expected* e = expected_for(sc.name);
        for (const char* mode : sc.modes) {
            for (int seed : sc.seeds) {
                const std::string r = run_case(plugin, table, sc, mode, seed);
                const std::string where = std::string(sc.name) + " mode=" + mode + " seed=" + std::to_string(seed) +
                                          " table=v" + std::to_string(table);
                // The status after "rank N: " is what the expectation names.
                const std::string status = r == "ok" ? r : r.substr(r.find(": ") + 2);
                if (e == nullptr) {
                    if (r == "ok") { ++ passed; std::printf("PASS   %s\n", where.c_str()); }
                    else { ++ failed; std::printf("FAIL   %s: %s\n", where.c_str(), r.c_str()); }
                } else if (status == e->status_prefix or status.rfind(std::string(e->status_prefix) + " ", 0) == 0) {
                    ++ xfailed; std::printf("XFAIL  %s: %s (%s)\n", where.c_str(), r.c_str(), e->issue);
                } else if (r == "ok") {
                    ++ failed; std::printf("XPASS  %s: passes; remove it from kExpected (%s)\n", where.c_str(), e->issue);
                } else {
                    ++ failed; std::printf("FAIL   %s: %s, expected %s\n", where.c_str(), r.c_str(), e->status_prefix);
                }
                std::fflush(stdout);
            }
        }
    }
    std::printf("%d passed, %d expected failures, %d skipped, %d failed\n", passed, xfailed, skipped, failed);
    return failed == 0 ? 0 : 1;
}
