// SPDX-License-Identifier: Apache-2.0
// Measures the plugin through its op table, the way the harness drives it (harness.h), with two ranks in two
// processes on one host and host windows.
//
//   bench_plugin PLUGIN_SO TABLE_VERSION [--size BYTES]... [MODE ...]
//
// For each mode (default: srd inorder) and each size (default: 64 B, 8 KiB, 64 KiB, 1 MiB; at most 2 MiB) it prints
// one line:
//
//   BENCH mode=srd size=8192 ops=4096 ops/s=... MB/s=... p50_us=... p99_us=...
//
// ops/s and MB/s are puts and bytes put divided by the time from the first issue until the target sees the last signal, with the
// puts issued back to back. p50_us and p99_us are the time from issuing one put with a signal until the target
// sees its signal, one at a time. Numbers depend on the host; nothing here asserts on them.
#include "harness.h"

#include <signal.h>
#include <sys/mman.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif
#include <sys/wait.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <new>

namespace {

using namespace harness;

constexpr size_t kSrc = 4096;                 // sources start here and destinations at kDst, in every rank's window
constexpr size_t kDst = 4u << 20;
constexpr size_t kMaxBytes = 2u << 20;        // destinations cycle through [kDst, kDst + kMaxBytes)

struct Result { double mbps = 0, ops_per_s = 0; double p50 = 0, p99 = 0; size_t ops = 0; };

// A put of `size` bytes with a signal on word 0 of rank 1, to a destination that cycles through the second half of the window.
Op put_signal_op(size_t size, size_t i) {
    Op o;
    o.kind = Op::PutSignal;
    o.peer = 1;
    o.srcOff = kSrc;
    o.size = size;
    o.dstOff = kDst + (i % std::max<size_t>(1, kMaxBytes / size)) * size;
    o.word = 0;
    o.keep_source = true;   // the same source is read by every put
    return o;
}

int64_t now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); }

// Rank 0 puts, rank 1 watches its signal words. Words 0 and 1 count the throughput and latency phases.
void run(Rank& R, size_t size, Result* out) {
    const size_t ops = std::max<size_t>(64, std::min<size_t>(4096, (64u << 20) / size));
    if (g_rank == 0) fill(R.at(kSrc), size, 0, 1);
    barrier("bench ready");
    // Throughput: every put issued at once, each with a signal on word 0.
    if (g_rank == 0) {
        g_shared->t0_ns.store(now_ns());
        for (size_t i = 0; i < ops; ++ i) {
            Op o = put_signal_op(size, i);
            R.issue(o);
        }
        wait_flag(0, "throughput-seen");
    } else {
        R.wait_peer([&] { return R.signal_word(0) == ops; }, "throughput");
        const double s = double(now_ns() - g_shared->t0_ns.load()) / 1e9;
        out->mbps = double(size) * double(ops) / s / 1e6;
        out->ops_per_s = double(ops) / s;
        out->ops = ops;
        set_flag(0);
    }
    barrier("throughput done");
    // Latency: one put with a signal at a time, on word 1.
    constexpr int kRounds = 100;
    std::vector<double> us;
    for (int r = 0; r < kRounds; ++ r) {
        if (g_rank == 0) {
            g_shared->t0_ns.store(now_ns());
            Op o = put_signal_op(size, size_t(r));
            o.word = 1;
            R.issue(o);
            R.wait_peer([&] { return g_shared->flag[1].load() == r + 1; }, "latency-ack");
        } else {
            R.wait_peer([&] { return R.signal_word(1) == uint64_t(r + 1); }, "latency");
            us.push_back(double(now_ns() - g_shared->t0_ns.load()) / 1e3);
            g_shared->flag[1].store(r + 1);
        }
    }
    if (g_rank == 1) {
        std::sort(us.begin(), us.end());
        out->p50 = us[us.size() / 2];
        out->p99 = us[std::min(us.size() - 1, us.size() * 99 / 100)];
    }
    barrier("latency done");
}

}  // namespace


int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s PLUGIN_SO TABLE_VERSION [MODE ...]\n", argv[0]); return 2; }
    const std::string plugin = argv[1];
    const int table = std::atoi(argv[2]);
    std::vector<std::string> modes;
    std::vector<size_t> sizes;
    for (int i = 3; i < argc; ++ i) {
        if (std::strcmp(argv[i], "--size") == 0 and i + 1 < argc) {
            const unsigned long long v = std::strtoull(argv[++ i], nullptr, 10);
            if (v == 0 or v > kMaxBytes) { std::fprintf(stderr, "--size must be 1 to %zu bytes\n", kMaxBytes); return 2; }
            sizes.push_back(size_t(v));
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
            pid_t pids[2];
            for (int r = 0; r < 2; ++ r) {
                pids[r] = fork();
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
                    R.setup(plugin, table, mode.c_str(), 1);
                    run(R, size, result);
                    R.teardown();
                    R.pass();
                }
            }
            bool ok = true;
            for (pid_t p : pids) { int st = 0; waitpid(p, &st, 0); ok = ok and WIFEXITED(st) and WEXITSTATUS(st) == 0; }
            if (ok)
                std::printf("BENCH mode=%s size=%zu ops=%zu ops/s=%.0f MB/s=%.2f p50_us=%.1f p99_us=%.1f\n", mode.c_str(),
                            size, result->ops, result->ops_per_s, result->mbps, result->p50, result->p99);
            else {
                std::printf("BENCH mode=%s size=%zu failed: %s / %s\n", mode.c_str(), size, shared->status[0], shared->status[1]);
                ++ failures;
            }
            std::fflush(stdout);
            munmap(mem, sizeof(Shared) + sizeof(Result));
        }
    }
    return failures == 0 ? 0 : 1;
}
