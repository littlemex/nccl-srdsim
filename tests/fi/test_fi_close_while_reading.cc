// SPDX-License-Identifier: Apache-2.0
// Endpoints are opened, used and closed again and again while another thread keeps reading the
// completion queue they are bound to: nothing may crash, every completion that comes out must belong
// to an operation that was posted, and every endpoint must close.
//
//   test_fi_close_while_reading PROVIDER_SO
//
// Why this is a test. Reading a completion queue makes progress on every endpoint bound to it, by
// walking the queue's endpoint list and calling into each endpoint (ofi_cq_progress). Closing an
// endpoint takes it off that list and then tears it down. If the two can overlap, a reader calls into
// an endpoint that is half freed. libfabric's utility layer is meant to make this safe -- the reader
// holds the list's lock for the walk, and close takes the same lock to unlink -- and this provider
// relies on that and on unlinking before anything else in srdsim_ep_close. This checks the reliance
// under load, with messages still in flight when the close happens, which is the case that matters:
// the provider's agent may be delivering to the very endpoint being closed.
//
// Run it under ThreadSanitizer and AddressSanitizer too. Plain, it catches a crash or a stray
// completion; the sanitizers catch the use-after-free or the race that has not crashed yet.

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kRounds = 60;
constexpr int kMessages = 40;
constexpr size_t kLen = 8192;

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    std::fflush(stdout);
    if (not ok) ++g_failures;
}

[[noreturn]] void die(const char* what, int rc) {
    std::printf("FAIL %s: %d (%s)\n", what, rc, fi_strerror(-rc));
    std::fflush(nullptr);
    std::_Exit(1);
}

// The contexts posted so far, so a completion can be checked against them. A completion for a
// context nobody posted is a stray: the queue handed out memory it should not have.
std::mutex g_posted_mu;
std::set<const void*> g_posted;

void posted(const void* ctx) {
    std::lock_guard<std::mutex> g(g_posted_mu);
    g_posted.insert(ctx);
}

bool known(const void* ctx) {
    std::lock_guard<std::mutex> g(g_posted_mu);
    return g_posted.count(ctx) != 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s PROVIDER_SO\n", argv[0]); return 2; }
    setenv("FI_PROVIDER_PATH", (std::string("+") + argv[1]).c_str(), 1);

    fi_info* hints = fi_allocinfo();
    hints->caps = FI_MSG | FI_LOCAL_COMM | FI_REMOTE_COMM;
    hints->ep_attr->type = FI_EP_RDM;
    hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
    // The model NIXL asks for, in which the queue's own lock depends on this provider's guard.
    hints->domain_attr->threading = FI_THREAD_COMPLETION;
    hints->fabric_attr->prov_name = strdup("efa");
    fi_info* info = nullptr;
    int rc = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0, hints, &info);
    fi_freeinfo(hints);
    if (rc) die("fi_getinfo", rc);

    fid_fabric* fabric = nullptr;
    fid_domain* domain = nullptr;
    fid_av* av = nullptr;
    fid_cq* cq = nullptr;
    if ((rc = fi_fabric(info->fabric_attr, &fabric, nullptr))) die("fi_fabric", rc);
    if ((rc = fi_domain(fabric, info, &domain, nullptr))) die("fi_domain", rc);
    fi_av_attr av_attr = {};
    av_attr.type = FI_AV_TABLE;
    if ((rc = fi_av_open(domain, &av_attr, &av, nullptr))) die("fi_av_open", rc);
    // One queue for every endpoint of every round, read the whole time by the reader below.
    fi_cq_attr cq_attr = {};
    cq_attr.format = FI_CQ_FORMAT_CONTEXT;
    if ((rc = fi_cq_open(domain, &cq_attr, &cq, nullptr))) die("fi_cq_open", rc);

    std::atomic<bool> stop{false};
    std::atomic<long> read_total{0}, strays{0}, errors{0};
    std::thread reader([&] {
        fi_cq_entry e[16];
        while (not stop.load(std::memory_order_relaxed)) {
            const ssize_t n = fi_cq_read(cq, e, 16);
            if (n == -FI_EAGAIN) continue;
            if (n == -FI_EAVAIL) {
                // An endpoint closed with operations outstanding may report them as failed; that is
                // allowed. What is not allowed is an error for a context that was never posted.
                fi_cq_err_entry err = {};
                if (fi_cq_readerr(cq, &err, 0) == 1) {
                    errors.fetch_add(1);
                    if (err.op_context and not known(err.op_context)) strays.fetch_add(1);
                }
                continue;
            }
            if (n < 0) { std::printf("FAIL fi_cq_read: %zd\n", n); std::fflush(stdout); return; }
            for (ssize_t k = 0; k < n; ++k)
                if (not known(e[k].op_context)) strays.fetch_add(1);
            read_total.fetch_add(n);
        }
    });

    // Buffers live for the whole run: a completion can arrive for an endpoint after it has closed,
    // and must not point into freed memory.
    std::vector<std::vector<uint8_t>> bufs(size_t(kRounds) * kMessages * 2, std::vector<uint8_t>(kLen, 0x5a));
    size_t next_buf = 0;
    int closed_ok = 0, close_calls = 0;

    for (int round = 0; round < kRounds; ++round) {
        fid_ep* ep[2] = {};
        fi_addr_t addr[2] = {};
        for (int s = 0; s < 2; ++s) {
            if ((rc = fi_endpoint(domain, info, &ep[s], nullptr))) die("fi_endpoint", rc);
            if ((rc = fi_ep_bind(ep[s], &av->fid, 0))) die("bind av", rc);
            if ((rc = fi_ep_bind(ep[s], &cq->fid, FI_TRANSMIT | FI_RECV))) die("bind cq", rc);
            if ((rc = fi_enable(ep[s]))) die("fi_enable", rc);
            char name[64];
            size_t len = sizeof name;
            if ((rc = fi_getname(&ep[s]->fid, name, &len))) die("fi_getname", rc);
            if (fi_av_insert(av, name, 1, &addr[s], 0, nullptr) != 1) die("fi_av_insert", -FI_EINVAL);
        }
        // Receives on ep[1], sends from ep[0]; post everything and close without waiting for any of
        // it, so the close lands while the agent may still be delivering into ep[1].
        for (int i = 0; i < kMessages; ++i) {
            void* b = bufs[next_buf++].data();
            posted(b);
            while (fi_recv(ep[1], b, kLen, nullptr, FI_ADDR_UNSPEC, b) == -FI_EAGAIN)
                std::this_thread::yield();
        }
        for (int i = 0; i < kMessages; ++i) {
            void* b = bufs[next_buf++].data();
            posted(b);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            ssize_t s;
            while ((s = fi_send(ep[0], b, kLen, nullptr, addr[1], b)) == -FI_EAGAIN and
                   std::chrono::steady_clock::now() < deadline)
                std::this_thread::yield();
            if (s and s != -FI_EAGAIN) die("fi_send", int(s));
        }
        // Half the rounds close the receiver first, half the sender, so both orders are covered.
        const int first = round % 2;
        for (int s : {first, 1 - first}) {
            ++close_calls;
            if (fi_close(&ep[s]->fid) == 0) ++closed_ok;
        }
    }

    // Let the reader drain whatever the last rounds left, then stop it.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    stop.store(true);
    reader.join();

    std::printf("%d rounds, %d of %d closes returned 0, %ld completions read, %ld error completions, %ld strays\n",
                kRounds, closed_ok, close_calls, read_total.load(), errors.load(), strays.load());
    check(closed_ok == close_calls, "every endpoint closed while the queue was being read");
    check(strays.load() == 0, "no completion came out for an operation that was never posted");
    check(read_total.load() > 0, "the reader was actually taking completions while endpoints closed");

    rc = fi_close(&cq->fid);
    check(rc == 0, "the completion queue closed once its endpoints were gone");
    fi_close(&av->fid);
    fi_close(&domain->fid);
    fi_close(&fabric->fid);
    fi_freeinfo(info);
    std::printf("close_while_reading: %s\n", g_failures == 0 ? "passed" : "failed");
    std::fflush(nullptr);
    return g_failures == 0 ? 0 : 1;
}
