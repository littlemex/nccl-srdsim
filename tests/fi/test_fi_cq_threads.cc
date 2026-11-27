// SPDX-License-Identifier: Apache-2.0
// Two threads read one completion queue of a domain opened with FI_THREAD_COMPLETION: every completion must be
// returned exactly once, to exactly one of them.
//
//   test_fi_cq_threads PROVIDER_SO [completion|domain]
//
// Why this is a test and not a consumer bug. libfabric's utility completion queue picks the lock that guards its
// ring from the domain's threading model, and for FI_THREAD_COMPLETION and FI_THREAD_DOMAIN it picks one that does
// nothing (`ofi_cq_init`). Both the writer and the reader take that lock, so in those modes the ring is unguarded
// and two readers can return one entry twice, or lose one. The stock providers behave the same way, and a consumer
// that asks for FI_THREAD_COMPLETION does owe libfabric its own serialisation per queue -- so this is not a defect
// in libfabric.
//
// It is still a defect HERE. This provider exists so that someone can find a bug in the code above it; a consumer
// hunting its own threading mistake must not first have to rule out a corrupted ring inside the tool it is using to
// look. So this provider guards its queues whatever the consumer asked for, and this is the test of that: it breaks
// the consumer's side of the contract on purpose and the completions still come out right.
//
// Both models libfabric disarms the lock for are run: FI_THREAD_COMPLETION, which is what NIXL asks for, and
// FI_THREAD_DOMAIN. They reach ofi_cq_init by the same branch, so this is coverage of the claim rather than of a
// second mechanism.
//
// Run it under ThreadSanitizer too. The counts below catch a duplicated or lost entry; the sanitizer catches the
// race that produces one, in ofi_cq_readfrom, before it has had the luck to matter.

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
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kMessages = 400;
constexpr int kReaders = 2;
constexpr size_t kLen = 64;

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    std::fflush(stdout);
    if (not ok) ++g_failures;
}

[[noreturn]] void die(const char* what, int rc) {
    std::printf("FAIL %s: %d (%s)\n", what, rc, fi_strerror(-rc));
    std::fflush(stdout);
    std::_Exit(1);
}

enum fi_threading g_threading = FI_THREAD_COMPLETION;
const char* g_threading_name = "FI_THREAD_COMPLETION";

struct Node {
    fi_info* info = nullptr;
    fid_fabric* fabric = nullptr;
    fid_domain* domain = nullptr;
    fid_av* av = nullptr;
    fid_cq* cq = nullptr;
    fid_ep* ep = nullptr;
    char name[64] = {};
    fi_addr_t peer = 0;
};

void open_node(Node* n) {
    fi_info* hints = fi_allocinfo();
    hints->caps = FI_MSG | FI_SOURCE | FI_LOCAL_COMM | FI_REMOTE_COMM;
    hints->ep_attr->type = FI_EP_RDM;
    hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
    // The whole point of the test: these are the modes in which libfabric's utility queue stops locking.
    hints->domain_attr->threading = g_threading;
    hints->fabric_attr->prov_name = strdup("efa");
    int rc = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0, hints, &n->info);
    fi_freeinfo(hints);
    if (rc) die("fi_getinfo", rc);
    if (n->info->domain_attr->threading != g_threading)
        die("the domain did not keep the threading model asked for, so this test proves nothing", -FI_EINVAL);
    if ((rc = fi_fabric(n->info->fabric_attr, &n->fabric, nullptr))) die("fi_fabric", rc);
    if ((rc = fi_domain(n->fabric, n->info, &n->domain, nullptr))) die("fi_domain", rc);
    fi_av_attr av_attr = {};
    av_attr.type = FI_AV_TABLE;
    if ((rc = fi_av_open(n->domain, &av_attr, &n->av, nullptr))) die("fi_av_open", rc);
    fi_cq_attr cq_attr = {};
    cq_attr.format = FI_CQ_FORMAT_TAGGED;
    // Deliberately smaller than the number of messages, so the ring wraps and its overflow path is used while
    // both readers are inside it.
    cq_attr.size = 16;
    if ((rc = fi_cq_open(n->domain, &cq_attr, &n->cq, nullptr))) die("fi_cq_open", rc);
    if ((rc = fi_endpoint(n->domain, n->info, &n->ep, nullptr))) die("fi_endpoint", rc);
    if ((rc = fi_ep_bind(n->ep, &n->av->fid, 0))) die("bind av", rc);
    if ((rc = fi_ep_bind(n->ep, &n->cq->fid, FI_TRANSMIT | FI_RECV))) die("bind cq", rc);
    if ((rc = fi_enable(n->ep))) die("fi_enable", rc);
    size_t len = sizeof n->name;
    if ((rc = fi_getname(&n->ep->fid, n->name, &len))) die("fi_getname", rc);
}

// Every close is checked: a -FI_EBUSY here means an object was still referenced, which with a discarded return
// value would leave the rest of this test reading a queue that was supposed to be gone.
void close_node(Node* n, const char* what) {
    bool ok = true;
    if (n->ep) ok = fi_close(&n->ep->fid) == 0 and ok;
    if (n->cq) ok = fi_close(&n->cq->fid) == 0 and ok;
    if (n->av) ok = fi_close(&n->av->fid) == 0 and ok;
    if (n->domain) ok = fi_close(&n->domain->fid) == 0 and ok;
    if (n->fabric) ok = fi_close(&n->fabric->fid) == 0 and ok;
    if (n->info) fi_freeinfo(n->info);
    check(ok, what);
}

uint8_t byte(int i, size_t k) { return uint8_t((i * 31 + k * 7) & 0xff); }

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s PROVIDER_SO [completion|domain]\n", argv[0]); return 2; }
    setenv("FI_PROVIDER_PATH", (std::string("+") + argv[1]).c_str(), 1);
    if (argc > 2 and std::strcmp(argv[2], "domain") == 0) {
        g_threading = FI_THREAD_DOMAIN;
        g_threading_name = "FI_THREAD_DOMAIN";
    } else if (argc > 2 and std::strcmp(argv[2], "completion") != 0) {
        std::fprintf(stderr, "the threading model must be completion or domain, not %s\n", argv[2]);
        return 2;
    }
    std::printf("threading %s\n", g_threading_name);

    static Node tx, rx;
    open_node(&tx);
    open_node(&rx);
    if (fi_av_insert(tx.av, rx.name, 1, &tx.peer, 0, nullptr) != 1) die("fi_av_insert", -FI_EINVAL);
    if (fi_av_insert(rx.av, tx.name, 1, &rx.peer, 0, nullptr) != 1) die("fi_av_insert", -FI_EINVAL);

    // One buffer per message, so a completion identifies its message by its context and nothing is reused while a
    // reader may still be holding it.
    std::vector<std::vector<uint8_t>> inbox(kMessages);
    for (int i = 0; i < kMessages; ++i) {
        inbox[i].assign(kLen, 0);
        const int rc = int(fi_recv(rx.ep, inbox[i].data(), kLen, nullptr, FI_ADDR_UNSPEC, &inbox[i]));
        if (rc) die("fi_recv", rc);
    }

    // How many times each message was returned to a reader. More than once is the ring handing one entry out
    // twice; never is one lost.
    std::vector<std::atomic<int>> seen(kMessages);
    for (auto& s : seen) s.store(0, std::memory_order_relaxed);
    std::atomic<int> total{0};
    std::atomic<bool> stop{false};
    std::atomic<int> unknown_context{0};
    // A reader that leaves early has to reach the verdict: counted here rather than printed and forgotten, or a
    // run where one reader died and the other read all 400 would pass as though both had been inside the queue.
    std::atomic<int> reader_errors{0};
    std::vector<std::atomic<int>> got(kReaders);
    for (auto& g : got) g.store(0, std::memory_order_relaxed);
    // The readers wait here until every send has been posted, so that the first one to look finds a few hundred
    // completions already queued and moves them through a ring of 16 -- which is the overflow path, rather than
    // the one or two at a time a reader spinning from the start would see. Released after posting and not after
    // the sends complete, because a send's completion can wait on the target making progress.
    std::atomic<bool> go{false};

    std::vector<std::thread> readers;
    for (int r = 0; r < kReaders; ++r) {
        readers.emplace_back([&, r] {
            fi_cq_tagged_entry e[4];
            while (not go.load(std::memory_order_acquire)) std::this_thread::yield();
            while (not stop.load(std::memory_order_relaxed)) {
                const ssize_t n = fi_cq_read(rx.cq, e, 4);
                if (n == -FI_EAGAIN) continue;
                if (n == -FI_EAVAIL) {
                    fi_cq_err_entry err = {};
                    fi_cq_readerr(rx.cq, &err, 0);
                    unknown_context.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                if (n < 0) {
                    std::printf("FAIL fi_cq_read: %zd (%s)\n", n, fi_strerror(int(-n)));
                    std::fflush(stdout);
                    reader_errors.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
                for (ssize_t k = 0; k < n; ++k) {
                    int idx = -1;
                    for (int i = 0; i < kMessages; ++i)
                        if (e[k].op_context == &inbox[i]) { idx = i; break; }
                    if (idx < 0) { unknown_context.fetch_add(1, std::memory_order_relaxed); continue; }
                    seen[idx].fetch_add(1, std::memory_order_relaxed);
                    got[r].fetch_add(1, std::memory_order_relaxed);
                    total.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    std::vector<std::vector<uint8_t>> out(kMessages);
    fi_cq_tagged_entry sent[8];
    int acked = 0;
    for (int i = 0; i < kMessages; ++i) {
        out[i].assign(kLen, 0);
        for (size_t k = 0; k < kLen; ++k) out[i][k] = byte(i, k);
        ssize_t rc;
        while ((rc = fi_send(tx.ep, out[i].data(), kLen, nullptr, tx.peer, &out[i])) == -FI_EAGAIN) {
            const ssize_t n = fi_cq_read(tx.cq, sent, 8);
            if (n > 0) acked += int(n);
        }
        if (rc) die("fi_send", int(rc));
    }
    go.store(true, std::memory_order_release);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (acked < kMessages and std::chrono::steady_clock::now() < deadline) {
        const ssize_t n = fi_cq_read(tx.cq, sent, 8);
        if (n > 0) acked += int(n);
    }
    check(acked == kMessages, "every send completed");

    while (total.load() < kMessages and std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    // Let the readers keep going a moment after the count is reached: a duplicate would push the total past the
    // number of messages, and stopping the instant it is first reached would hide that.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    stop.store(true);
    for (auto& t : readers) t.join();

    int once = 0, more = 0, never = 0;
    for (int i = 0; i < kMessages; ++i) {
        const int c = seen[i].load();
        if (c == 1) ++once;
        else if (c > 1) ++more;
        else ++never;
    }
    std::printf("%d of %d messages returned once, %d returned more than once, %d never, %d entries unrecognised\n",
                once, kMessages, more, never, unknown_context.load());
    check(more == 0, "no completion was returned to both readers");
    check(never == 0, "no completion was lost between the readers");
    check(unknown_context.load() == 0, "every entry carried a context this test had posted");
    check(reader_errors.load() == 0, "no reader left the queue on an error");
    int reading = 0;
    for (int r = 0; r < kReaders; ++r) {
        std::printf("reader %d returned %d of them\n", r, got[r].load());
        if (got[r].load() > 0) ++reading;
    }
    check(reading == kReaders, "both readers were inside the queue, so the guarantee was actually exercised");

    bool bodies = true;
    for (int i = 0; i < kMessages and bodies; ++i)
        for (size_t k = 0; k < kLen and bodies; ++k) bodies = inbox[i][k] == byte(i, k);
    check(bodies, "every message holds what its sender wrote");

    close_node(&tx, "the sending endpoint and its objects closed");
    close_node(&rx, "the receiving endpoint and its objects closed");
    std::printf("cq_threads (%s): %s\n", g_threading_name, g_failures == 0 ? "passed" : "failed");
    std::fflush(nullptr);
    return g_failures == 0 ? 0 : 1;
}
