// SPDX-License-Identifier: Apache-2.0
// Measures the plugin through its op table, the way the harness drives it (harness.h), with two ranks in two
// processes on one host and host windows.
//
//   bench_plugin PLUGIN_SO TABLE_VERSION [--size BYTES]... [MODE ...]
//
// For each mode (default: srd inorder) and each size (default: 64 B, 8 KiB, 64 KiB, 1 MiB; at most 2 MiB) it prints
// one line:
//
//   BENCH mode=srd size=8192 ops=4096 ops/s=... MB/s=... burst_ms=... p50_us=... p99_us=...
//
// Throughput: rank 0 issues `ops` puts back to back, each with a signal on the same word of rank 1, and the burst
// lasts until rank 1 sees the last signal. The burst runs kBursts times; ops/s, MB/s and burst_ms are from the median
// one. Rank 1 then checks that every destination holds the bytes put.
// Latency: one put with a signal at a time, from rank 0 issuing it to rank 1 seeing the signal, over kRounds rounds
// after kWarmup discarded ones; p50_us and p99_us are nearest-rank percentiles.
//
// Nothing judges the numbers, which depend on the host. A case fails, and the program exits 1, when its data is wrong,
// when a rank fails, or when a case does not finish within kCaseWallMs.
#include "harness.h"

#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <new>

extern char** environ;

namespace {

using namespace harness;

constexpr size_t kSrc = 4096;                 // the source, in rank 0's window
constexpr size_t kDst = 4u << 20;             // destinations cycle through [kDst, kDst + kMaxBytes) of rank 1's window
constexpr size_t kMaxBytes = 2u << 20;
constexpr int kBursts = 5;
constexpr int kWarmup = 20, kRounds = 1000;
constexpr int kWaitMs = 60000;                // any single wait inside a case
constexpr int kCaseWallMs = 180000;           // the driver kills a case that runs longer

struct Result { double mbps = 0, ops_per_s = 0, burst_ms = 0; double p50 = 0, p99 = 0; size_t ops = 0; };

// A put of `size` bytes from the source to destination slot i (cycling), with a signal on `word` of rank 1.
Op put_signal_op(size_t size, size_t i, size_t word) {
    Op o;
    o.kind = Op::PutSignal;
    o.peer = 1;
    o.srcOff = kSrc;
    o.size = size;
    o.dstOff = kDst + (i % std::max<size_t>(1, kMaxBytes / size)) * size;
    o.word = word;
    o.keep_source = true;   // the same source is read by every put
    return o;
}

// Waits for `pred` without sleeping, so a time is not rounded up to the harness's polling interval.
template <class Pred> void spin_until(Pred pred, const char* what) {
    set_where(what);
    const auto start = Clock::now();
    while (not pred()) {
        if (Clock::now() - start > std::chrono::milliseconds(kWaitMs)) fail("timeout", what, "%s", "");
        std::this_thread::yield();
    }
}
// The same with a short sleep, for a wait that is not timed, so it does not compete with the threads that are.
template <class Pred> void sleep_until(Pred pred, const char* what) {
    set_where(what);
    const auto start = Clock::now();
    while (not pred()) {
        if (Clock::now() - start > std::chrono::milliseconds(kWaitMs)) fail("timeout", what, "%s", "");
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
}

int64_t now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); }

double nearest_rank(const std::vector<double>& sorted, double q) {
    const size_t i = static_cast<size_t>(std::ceil(q * double(sorted.size())));
    return sorted[std::min(sorted.size() - 1, i == 0 ? 0 : i - 1)];
}

// Rank 0 puts, rank 1 watches its signal words: word 0 counts the bursts' puts, word 1 the latency rounds.
void run(Rank& R, size_t size, Result* out) {
    const size_t ops = std::max<size_t>(64, std::min<size_t>(4096, (64u << 20) / size));
    if (g_rank == 0) fill(R.at(kSrc), size, 0, 1);
    barrier("bench ready");
    std::vector<double> burst_s;
    const Ticket* last = nullptr;
    for (int b = 0; b < kBursts; ++ b) {
        if (g_rank == 0) {
            g_shared->t0_ns.store(now_ns());
            for (size_t i = 0; i < ops; ++ i) last = R.issue(put_signal_op(size, i, 0));
            sleep_until([&] { return last->retired.load(std::memory_order_acquire) != 0; }, "burst-retired");
        } else {
            const uint64_t want = uint64_t(b + 1) * ops;
            spin_until([&] { return R.signal_word(0) == want; }, "burst");
            burst_s.push_back(double(now_ns() - g_shared->t0_ns.load()) / 1e9);
        }
        barrier("burst done");
    }
    if (g_rank == 1) {
        const size_t slots = std::min(ops, std::max<size_t>(1, kMaxBytes / size));
        for (size_t k = 0; k < slots; ++ k)
            if (size_t bad = mismatches(R.at(kDst + k * size), size, 0, 1))
                fail("assert", "data", "slot %zu: %zu of %zu bytes differ", k, bad, size);
        std::sort(burst_s.begin(), burst_s.end());
        const double s = burst_s[burst_s.size() / 2];
        out->ops = ops;
        out->burst_ms = s * 1e3;
        out->ops_per_s = double(ops) / s;
        out->mbps = double(size) * double(ops) / s / 1e6;
    }
    // Latency: one put with a signal at a time, on word 1; rank 1 acknowledges each through flag 1.
    std::vector<double> us;
    for (int r = 0; r < kWarmup + kRounds; ++ r) {
        if (g_rank == 0) {
            g_shared->t0_ns.store(now_ns());
            R.issue(put_signal_op(size, size_t(r), 1));
            sleep_until([&] { return g_shared->flag[1].load() == r + 1; }, "latency-ack");
        } else {
            spin_until([&] { return R.signal_word(1) == uint64_t(r + 1); }, "latency");
            const double t = double(now_ns() - g_shared->t0_ns.load()) / 1e3;
            if (r >= kWarmup) us.push_back(t);
            g_shared->flag[1].store(r + 1);
        }
    }
    if (g_rank == 1) {
        std::sort(us.begin(), us.end());
        out->p50 = nearest_rank(us, 0.50);
        out->p99 = nearest_rank(us, 0.99);
    }
    barrier("latency done");
}

bool parse_size(const char* s, size_t* out) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(s, &end, 10);
    if (errno != 0 or end == s or *end != '\0' or v == 0 or v > kMaxBytes) return false;
    *out = size_t(v);
    return true;
}

std::string describe(int st, const char* status) {
    if (WIFSIGNALED(st)) return "killed by signal " + std::to_string(WTERMSIG(st));
    if (status[0] != '\0') return status;
    return "exit " + std::to_string(WEXITSTATUS(st));
}

}  // namespace

int main(int argc, char** argv) {
    const char* usage = "usage: %s PLUGIN_SO TABLE_VERSION [--size BYTES]... [MODE ...]\n"
                        "  BYTES is 1 to 2097152; the default sizes are 64 8192 65536 1048576; the default modes srd inorder\n";
    if (argc < 3) { std::fprintf(stderr, usage, argv[0]); return 2; }
    const std::string plugin = argv[1];
    char* end = nullptr;
    const long table = std::strtol(argv[2], &end, 10);
    if (end == argv[2] or *end != '\0' or table <= 0) {
        std::fprintf(stderr, "TABLE_VERSION '%s' is not a positive number\n", argv[2]);
        return 2;
    }
    std::vector<std::string> modes;
    std::vector<size_t> sizes;
    for (int i = 3; i < argc; ++ i) {
        if (std::strcmp(argv[i], "--size") == 0) {
            size_t v = 0;
            if (i + 1 >= argc or not parse_size(argv[i + 1], &v)) { std::fprintf(stderr, usage, argv[0]); return 2; }
            sizes.push_back(v);
            ++ i;
        } else if (argv[i][0] == '-') {
            std::fprintf(stderr, usage, argv[0]);
            return 2;
        } else {
            modes.push_back(argv[i]);
        }
    }
    if (modes.empty()) modes = {"srd", "inorder"};
    if (sizes.empty()) sizes = {64, 8192, 64u << 10, 1u << 20};
    // The plugin runs with its defaults and the mode asked for, not with whatever the caller's environment holds.
    std::vector<std::string> names;
    for (char** e = environ; *e; ++ e)
        if (std::strncmp(*e, "SRDSIM_", 7) == 0) names.emplace_back(*e, std::strcspn(*e, "="));
    for (const auto& n : names) unsetenv(n.c_str());
    int failures = 0;
    for (const auto& mode : modes) {
        for (size_t size : sizes) {
            void* mem = mmap(nullptr, sizeof(Shared) + sizeof(Result), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
            if (mem == MAP_FAILED) return 1;
            auto* shared = new (mem) Shared();
            auto* result = new (static_cast<char*>(mem) + sizeof(Shared)) Result();
            std::fflush(nullptr);
            pid_t pids[2] = {-1, -1};
            for (int r = 0; r < 2; ++ r) {
                pids[r] = fork();
                // Rank 0, if already forked, dies with the driver (PR_SET_PDEATHSIG below).
                if (pids[r] < 0) return 1;
                if (pids[r] == 0) {
#if defined(__linux__)
                    prctl(PR_SET_PDEATHSIG, SIGKILL);   // a rank never outlives the driver
#endif
                    g_shared = shared; g_rank = r; g_nranks = 2;
                    shared->pid[r] = getpid();
                    setenv("SRDSIM_MODE", mode.c_str(), 1);
                    setenv("SRDSIM_SEED", "1", 1);
                    Rank R;
                    R.setup(plugin, int(table), mode.c_str(), 1);
                    run(R, size, result);
                    R.teardown();
                    R.pass();
                }
            }
            // Both ranks, with a deadline: a rank blocked inside a plugin call does not end the program's wait.
            int st[2] = {0, 0};
            bool exited[2] = {false, false}, killed = false;
            const auto start = Clock::now();
            while (not (exited[0] and exited[1])) {
                for (int r = 0; r < 2; ++ r)
                    if (not exited[r] and waitpid(pids[r], &st[r], WNOHANG) == pids[r]) exited[r] = true;
                if (not killed and Clock::now() - start > std::chrono::milliseconds(kCaseWallMs)) {
                    std::printf("BENCH mode=%s size=%zu failed: killed after %d s, rank 0 in %s, rank 1 in %s\n",
                                mode.c_str(), size, kCaseWallMs / 1000, shared->where[0], shared->where[1]);
                    for (int r = 0; r < 2; ++ r) if (not exited[r]) kill(pids[r], SIGKILL);
                    killed = true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            bool ok = not killed;
            for (int r = 0; r < 2; ++ r) ok = ok and WIFEXITED(st[r]) and WEXITSTATUS(st[r]) == 0;
            if (ok)
                std::printf("BENCH mode=%s size=%zu ops=%zu ops/s=%.0f MB/s=%.2f burst_ms=%.2f p50_us=%.1f p99_us=%.1f\n",
                            mode.c_str(), size, result->ops, result->ops_per_s, result->mbps, result->burst_ms,
                            result->p50, result->p99);
            else if (not killed)
                std::printf("BENCH mode=%s size=%zu failed: rank 0: %s; rank 1: %s\n", mode.c_str(), size,
                            describe(st[0], shared->status[0]).c_str(), describe(st[1], shared->status[1]).c_str());
            if (not ok) ++ failures;
            std::fflush(stdout);
            munmap(mem, sizeof(Shared) + sizeof(Result));
        }
    }
    return failures == 0 ? 0 : 1;
}
