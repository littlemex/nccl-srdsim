// SPDX-License-Identifier: Apache-2.0
// Carries data between two processes through the libfabric provider, as a consumer would: rank 0 and rank 1
// are a parent and a forked child, each with its own endpoint, exchanging their addresses and memory keys over
// pipes.
//
//   test_fi_transfer PROVIDER_SO SCENARIO
//
// SCENARIO is one of sendrecv, tagged, closeinject, closewrite, directsend, inject, unreachable, truncate, read, badkey, writedata, reorder. The program exits 0 when
// the scenario's checks pass and 1 when one fails; each check prints one line.
//
// writedata checks EFA's guarantee that a write's remote completion with CQ data comes only after every byte
// of that write has landed; run with SRDSIM_FI_CQ_DATA_EARLY=1 it must fail. reorder checks that a write can
// land after a later write's remote completion under SRDSIM_MODE=srd (the default), and never under
// SRDSIM_MODE=inorder: the reordering a consumer has to be written for.

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>

#include <arpa/inet.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_rank = 0;
int g_failures = 0;
bool g_direct = false;   // the efa-direct fabric, whose sends are not ordered
int g_to_peer = -1, g_from_peer = -1;

void check(bool ok, const char* what) {
    std::printf("rank %d: %s %s\n", g_rank, ok ? "PASS" : "FAIL", what);
    std::fflush(stdout);
    if (not ok) ++g_failures;
}

[[noreturn]] void die(const char* what, int rc) {
    std::printf("rank %d: FAIL %s: %d (%s)\n", g_rank, what, rc, fi_strerror(-rc));
    std::fflush(stdout);
    std::_Exit(1);
}

void put(const void* p, size_t n) {
    if (write(g_to_peer, p, n) != ssize_t(n)) die("pipe write", -FI_EIO);
}
void get(void* p, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(g_from_peer, static_cast<char*>(p) + got, n - got);
        if (r <= 0) die("pipe read", -FI_EIO);
        got += size_t(r);
    }
}

struct Endpoint {
    fi_info* info = nullptr;
    fid_fabric* fabric = nullptr;
    fid_domain* domain = nullptr;
    fid_av* av = nullptr;
    fid_cq* cq = nullptr;
    fid_ep* ep = nullptr;
    fi_addr_t peer = FI_ADDR_NOTAVAIL;
    bool closed = false;
};

void open_endpoint(Endpoint* e) {
    fi_info* hints = fi_allocinfo();
    hints->caps = FI_MSG | FI_TAGGED | FI_RMA | FI_LOCAL_COMM | FI_REMOTE_COMM;
    hints->ep_attr->type = FI_EP_RDM;
    hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
    hints->domain_attr->threading = FI_THREAD_COMPLETION;
    hints->fabric_attr->prov_name = strdup("efa");
    if (g_direct) {
        free(hints->fabric_attr->name);
        hints->fabric_attr->name = strdup("efa-direct");
        hints->caps &= ~uint64_t(FI_TAGGED);   // efa-direct has no tagged messages
        hints->domain_attr->mr_mode |= FI_MR_LOCAL;
    } else {
        hints->tx_attr->msg_order = FI_ORDER_SAS;
        hints->rx_attr->msg_order = FI_ORDER_SAS;
    }
    int rc = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0, hints, &e->info);
    fi_freeinfo(hints);
    if (rc) die("fi_getinfo", rc);
    if (std::strncmp(e->info->domain_attr->name, "srdsim0", 7) != 0) die("not the srdsim provider", -FI_ENODATA);
    if ((rc = fi_fabric(e->info->fabric_attr, &e->fabric, nullptr))) die("fi_fabric", rc);
    if ((rc = fi_domain(e->fabric, e->info, &e->domain, nullptr))) die("fi_domain", rc);
    fi_av_attr av_attr = {};
    av_attr.type = FI_AV_TABLE;
    if ((rc = fi_av_open(e->domain, &av_attr, &e->av, nullptr))) die("fi_av_open", rc);
    fi_cq_attr cq_attr = {};
    cq_attr.format = FI_CQ_FORMAT_TAGGED;
    if ((rc = fi_cq_open(e->domain, &cq_attr, &e->cq, nullptr))) die("fi_cq_open", rc);
    if ((rc = fi_endpoint(e->domain, e->info, &e->ep, nullptr))) die("fi_endpoint", rc);
    if ((rc = fi_ep_bind(e->ep, &e->av->fid, 0))) die("bind av", rc);
    if ((rc = fi_ep_bind(e->ep, &e->cq->fid, FI_TRANSMIT | FI_RECV))) die("bind cq", rc);
    if ((rc = fi_enable(e->ep))) die("fi_enable", rc);
    char mine[64], theirs[64];
    size_t len = sizeof mine;
    if ((rc = fi_getname(&e->ep->fid, mine, &len))) die("fi_getname", rc);
    put(mine, 32);
    get(theirs, 32);
    if (fi_av_insert(e->av, theirs, 1, &e->peer, 0, nullptr) != 1) die("fi_av_insert", -FI_EINVAL);
}

void close_endpoint(Endpoint* e) {
    if (e->closed) return;
    e->closed = true;
    fi_close(&e->ep->fid);
    fi_close(&e->cq->fid);
    fi_close(&e->av->fid);
    fi_close(&e->domain->fid);
    fi_close(&e->fabric->fid);
    fi_freeinfo(e->info);
}

// Waits for one completion; returns 0, or the error's positive errno with *err filled in.
int wait_cq(Endpoint* e, fi_cq_tagged_entry* out, fi_cq_err_entry* err = nullptr, int timeout_s = 30) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    while (std::chrono::steady_clock::now() < deadline) {
        ssize_t n = fi_cq_read(e->cq, out, 1);
        if (n == 1) return 0;
        if (n == -FI_EAVAIL) {
            fi_cq_err_entry tmp = {};
            fi_cq_readerr(e->cq, &tmp, 0);
            if (err) *err = tmp;
            return tmp.err;
        }
        if (n != -FI_EAGAIN) die("fi_cq_read", int(n));
    }
    die("no completion within the timeout", -FI_ETIMEDOUT);
}

uint8_t pattern(size_t round, size_t i) { return uint8_t((round * 131 + i * 7 + (i >> 9)) & 0xff); }

struct Region {
    std::vector<uint8_t> mem;
    fid_mr* mr = nullptr;
    uint64_t key = 0;
};
void reg(Endpoint* e, Region* r, size_t len, uint64_t access) {
    r->mem.assign(len, 0);
    int rc = fi_mr_reg(e->domain, r->mem.data(), len, access, 0, 0, 0, &r->mr, nullptr);
    if (rc) die("fi_mr_reg", rc);
    r->key = fi_mr_key(r->mr);
}

// An RMA write raises no completion at the target, so two scenarios below read the target's region while the
// provider may still be delivering into it: in `closewrite` that polling IS how the test learns the write landed,
// and in `reorder` catching a half-landed region is the measurement itself. Both are races by the language's rules
// and ThreadSanitizer is right to see them.
//
// What is silenced is the READ, for the few lines that do the polling -- not the region. Marking the region benign
// would have gone further than the truth: it would also hide the provider delivering the same chunk twice, or
// delivering into a region whose registration is gone, which is exactly what this suite is here to catch. The
// symbols are weak, so a build without the sanitizer links unchanged and this costs nothing.
extern "C" {
void AnnotateIgnoreReadsBegin(const char*, int) __attribute__((weak));
void AnnotateIgnoreReadsEnd(const char*, int) __attribute__((weak));
}

// Reads of the target region inside this object's lifetime are the test's own doing and are not reported.
struct IntendedRaceOnReads {
    const bool on;
    explicit IntendedRaceOnReads(bool enable = true) : on(enable) {
        if (on and AnnotateIgnoreReadsBegin) AnnotateIgnoreReadsBegin(__FILE__, __LINE__);
    }
    ~IntendedRaceOnReads() { if (on and AnnotateIgnoreReadsEnd) AnnotateIgnoreReadsEnd(__FILE__, __LINE__); }
    IntendedRaceOnReads(const IntendedRaceOnReads&) = delete;
    IntendedRaceOnReads& operator=(const IntendedRaceOnReads&) = delete;
};

bool early_cq_data() {
    const char* v = std::getenv("SRDSIM_FI_CQ_DATA_EARLY");
    return v and std::strcmp(v, "0") != 0 and *v;
}

void barrier() {
    char b = 1;
    put(&b, 1);
    get(&b, 1);
}

// Sends in order, some received only after they arrive, with CQ data on every fifth.
void sendrecv(Endpoint* e) {
    constexpr int kCount = 200;
    auto size_of = [](int i) { return size_t((i * 3779) % 40000 + 1); };
    if (g_rank == 0) {
        std::vector<std::vector<uint8_t>> bufs(kCount);
        for (int i = 0; i < kCount; ++i) {
            bufs[i].resize(size_of(i));
            for (size_t k = 0; k < bufs[i].size(); ++k) bufs[i][k] = pattern(i, k);
            int rc = i % 5 == 0 ? int(fi_senddata(e->ep, bufs[i].data(), bufs[i].size(), nullptr, 1000 + i, e->peer, &bufs[i]))
                                : int(fi_send(e->ep, bufs[i].data(), bufs[i].size(), nullptr, e->peer, &bufs[i]));
            if (rc) die("fi_send", rc);
        }
        int done = 0;
        fi_cq_tagged_entry c;
        while (done < kCount) {
            if (wait_cq(e, &c)) die("send completion", -FI_EIO);
            ++done;
        }
        check(true, "every send completed");
        barrier();
        return;
    }
    std::vector<std::vector<uint8_t>> bufs(kCount);
    bool ok = true, data_ok = true;
    for (int i = 0; i < kCount; ++i) {
        // Half the receives are posted after their message has surely arrived (unexpected messages).
        if (i == kCount / 2) usleep(200000);
        bufs[i].assign(size_of(i), 0);
        int rc = int(fi_recv(e->ep, bufs[i].data(), bufs[i].size(), nullptr, FI_ADDR_UNSPEC, &bufs[i]));
        if (rc) die("fi_recv", rc);
        fi_cq_tagged_entry c;
        if (wait_cq(e, &c)) die("recv completion", -FI_EIO);
        if (c.op_context != &bufs[i] or c.len != bufs[i].size()) ok = false;
        for (size_t k = 0; k < bufs[i].size() and ok; ++k)
            if (bufs[i][k] != pattern(i, k)) ok = false;
        if (i % 5 == 0 and (not (c.flags & FI_REMOTE_CQ_DATA) or c.data != uint64_t(1000 + i))) data_ok = false;
    }
    check(ok, "200 messages of 1 to 40000 bytes arrive whole and in order, expected or not");
    check(data_ok, "the CQ data of fi_senddata arrives with its message");
    barrier();
}

// On efa-direct, sends to one peer are not ordered, but the pieces of one message are never mixed with another's: several
// messages longer than one chunk, posted back to back, each arrive whole.
void directsend(Endpoint* e) {
    constexpr int kCount = 8;
    constexpr size_t kLen = 40000;   // several chunks
    if (g_rank == 0) {
        std::vector<std::vector<uint8_t>> bufs(kCount, std::vector<uint8_t>(kLen));
        for (int i = 0; i < kCount; ++i) {
            for (size_t k = 0; k < kLen; ++k) bufs[i][k] = pattern(i, k);
            if (int rc = int(fi_send(e->ep, bufs[i].data(), kLen, nullptr, e->peer, &bufs[i]))) die("fi_send", rc);
        }
        fi_cq_tagged_entry c;
        for (int i = 0; i < kCount; ++i)
            if (wait_cq(e, &c)) die("send completion", -FI_EIO);
        check(true, "every long send completed");
        barrier();
        return;
    }
    std::vector<std::vector<uint8_t>> bufs(kCount, std::vector<uint8_t>(kLen, 0));
    for (int i = 0; i < kCount; ++i)
        if (int rc = int(fi_recv(e->ep, bufs[i].data(), kLen, nullptr, FI_ADDR_UNSPEC, &bufs[i]))) die("fi_recv", rc);
    bool whole = true;
    bool seen[kCount] = {};
    for (int i = 0; i < kCount; ++i) {
        fi_cq_tagged_entry c;
        fi_cq_err_entry err = {};
        if (wait_cq(e, &c, &err)) { whole = false; continue; }
        const auto* b = static_cast<std::vector<uint8_t>*>(c.op_context);
        int which = -1;
        for (int m = 0; m < kCount and which < 0; ++m) {
            bool same = true;
            for (size_t k = 0; k < kLen and same; ++k) same = (*b)[k] == pattern(m, k);
            if (same) which = m;
        }
        if (which < 0 or seen[which] or c.len != kLen) whole = false; else seen[which] = true;
    }
    check(whole, "eight messages of 40000 bytes each arrive whole and once, whatever the order between them");
    barrier();
}

// What a sender had accepted before it closed still reaches its peer: an inject nobody read a completion after, and the
// chunks of a write that are still in flight when the connection ends.
void closeafter(Endpoint* e, bool write) {
    constexpr size_t kLen = 200000;   // several chunks
    Region r;
    uint64_t desc[2] = {};
    if (g_rank == 1) {
        reg(e, &r, kLen, FI_REMOTE_WRITE);
        desc[0] = uint64_t(uintptr_t(r.mem.data()));
        desc[1] = r.key;
        put(desc, sizeof desc);
    } else {
        get(desc, sizeof desc);
    }
    char in[16] = {};
    if (g_rank == 1 and not write)
        if (int rc = int(fi_recv(e->ep, in, sizeof in, nullptr, FI_ADDR_UNSPEC, nullptr))) die("fi_recv", rc);
    barrier();
    if (g_rank == 0) {
        std::vector<uint8_t> b(kLen);
        for (size_t k = 0; k < kLen; ++k) b[k] = pattern(7, k);
        if (write) {
            if (int rc = int(fi_write(e->ep, b.data(), kLen, nullptr, e->peer, desc[0], desc[1], nullptr))) die("fi_write", rc);
        } else {
            char m[16] = "inject, close";
            if (int rc = int(fi_inject(e->ep, m, sizeof m, e->peer))) die("fi_inject", rc);
        }
        close_endpoint(e);   // with no completion read in between
    } else if (write) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        bool landed = false;
        while (not landed and std::chrono::steady_clock::now() < deadline) {
            landed = true;
            {
                const IntendedRaceOnReads polling;   // the write is still landing; that is what is being waited for
                for (size_t k = 0; k < kLen and landed; ++k) landed = r.mem[k] == pattern(7, k);
            }
            if (not landed) usleep(2000);
        }
        check(landed, "a write whose sender closed right after it was posted landed whole");
    } else {
        fi_cq_tagged_entry c;
        const int rc = wait_cq(e, &c, nullptr, 10);
        check(rc == 0 and std::strncmp(in, "inject, close", 14) == 0, "an inject whose sender closed right after it reaches its peer");
    }
    barrier();
    // Registered memory cannot be freed while an endpoint that may still be delivering into it is open: the rule
    // is the same on real hardware, and ThreadSanitizer sees the region's free racing the provider's last chunk.
    // Closing the endpoint here joins the thread doing the delivering, and that is the edge that makes the free
    // safe. close_endpoint is idempotent, so main closing it again is harmless.
    if (g_rank == 1) close_endpoint(e);
}

// Receives match by tag, not by order.
void tagged(Endpoint* e) {
    const uint64_t tags[3] = {0x30, 0x10, 0x20};
    if (g_rank == 0) {
        char m[3][8] = {};   // the whole 8 bytes are sent, so none may be left indeterminate
        for (int i = 0; i < 3; ++i) {
            std::snprintf(m[i], sizeof m[i], "tag%llx", (unsigned long long) tags[i]);
            if (int rc = int(fi_tsend(e->ep, m[i], sizeof m[i], nullptr, e->peer, tags[i], nullptr))) die("fi_tsend", rc);
        }
        fi_cq_tagged_entry c;
        for (int i = 0; i < 3; ++i)
            if (wait_cq(e, &c)) die("tsend completion", -FI_EIO);
        barrier();
        return;
    }
    usleep(100000);
    bool ok = true;
    for (uint64_t want : {uint64_t(0x20), uint64_t(0x10), uint64_t(0x30)}) {
        char b[8] = {};
        if (int rc = int(fi_trecv(e->ep, b, sizeof b, nullptr, FI_ADDR_UNSPEC, want, 0, nullptr))) die("fi_trecv", rc);
        fi_cq_tagged_entry c;
        if (wait_cq(e, &c)) die("trecv completion", -FI_EIO);
        char expect[8] = {};
        std::snprintf(expect, sizeof expect, "tag%llx", (unsigned long long) want);
        if (c.tag != want or std::memcmp(b, expect, sizeof b) != 0) {
            std::printf("rank 1: wanted tag %llx, completion tag %llx, buffer '%.8s', flags %llx\n", (unsigned long long) want,
                        (unsigned long long) c.tag, b, (unsigned long long) c.flags);
            ok = false;
        }
    }
    check(ok, "three tagged messages are matched by tag, received in another order than sent");
    barrier();
}

// Injected sends complete nothing, unless sent with FI_COMPLETION; iovecs gather and scatter.
void inject(Endpoint* e) {
    if (g_rank == 0) {
        char a[16] = "first", b[16] = "second";
        if (int rc = int(fi_inject(e->ep, a, sizeof a, e->peer))) die("fi_inject", rc);
        std::memset(a, 0, sizeof a);   // reusable on return
        iovec iov[3] = {{b, 4}, {b + 4, 4}, {b + 8, 8}};
        if (int rc = int(fi_sendv(e->ep, iov, nullptr, 3, e->peer, &iov))) die("fi_sendv", rc);
        char c3[16] = "third";
        iovec one = {c3, sizeof c3};
        fi_msg m = {};
        m.msg_iov = &one;
        m.iov_count = 1;
        m.addr = e->peer;
        m.context = c3;
        if (int rc = int(fi_sendmsg(e->ep, &m, FI_INJECT | FI_COMPLETION))) die("fi_sendmsg", rc);
        fi_cq_tagged_entry c;
        void* seen[2] = {};
        for (int i = 0; i < 2; ++i) {
            if (wait_cq(e, &c)) die("send completion", -FI_EIO);
            seen[i] = c.op_context;
        }
        check((seen[0] == &iov or seen[1] == &iov) and (seen[0] == c3 or seen[1] == c3),
              "the sendv and the FI_INJECT | FI_COMPLETION sendmsg complete; the fi_inject does not");
        barrier();
        fi_cq_tagged_entry extra;
        check(fi_cq_read(e->cq, &extra, 1) == -FI_EAGAIN, "no completion for the fi_inject");
        return;
    }
    char r1[16], r2a[6], r2b[10], r3[16];
    if (int rc = int(fi_recv(e->ep, r1, sizeof r1, nullptr, FI_ADDR_UNSPEC, nullptr))) die("fi_recv", rc);
    iovec iov[2] = {{r2a, sizeof r2a}, {r2b, sizeof r2b}};
    if (int rc = int(fi_recvv(e->ep, iov, nullptr, 2, FI_ADDR_UNSPEC, nullptr))) die("fi_recvv", rc);
    if (int rc = int(fi_recv(e->ep, r3, sizeof r3, nullptr, FI_ADDR_UNSPEC, nullptr))) die("fi_recv", rc);
    fi_cq_tagged_entry c;
    for (int i = 0; i < 3; ++i)
        if (wait_cq(e, &c)) die("recv completion", -FI_EIO);
    char joined[16];
    std::memcpy(joined, r2a, 6);
    std::memcpy(joined + 6, r2b, 10);
    check(std::strcmp(r1, "first") == 0 and std::strcmp(joined, "second") == 0 and std::strcmp(r3, "third") == 0,
          "an injected message, a gathered one scattered over two buffers, and an injected one with a completion arrive");
    barrier();
}

// A send to an endpoint that is not listening fails with FI_EHOSTUNREACH on the sender's CQ, and the sender goes on.
void unreachable(Endpoint* e) {
    if (g_rank == 0) {
        char mine[32];
        size_t len = sizeof mine;
        if (int rc = fi_getname(&e->ep->fid, mine, &len)) die("fi_getname", rc);
        char other[32];
        std::memcpy(other, mine, sizeof other);
        const uint16_t closed_port = htons(1);            // nothing listens on port 1
        std::memcpy(other + 12, &closed_port, sizeof closed_port);
        other[16] ^= 0x5a;                                // a different endpoint, by nonce
        fi_addr_t dead = FI_ADDR_NOTAVAIL;
        if (fi_av_insert(e->av, other, 1, &dead, 0, nullptr) != 1) die("fi_av_insert", -FI_EINVAL);
        char m[16] = "to nobody";
        if (int rc = int(fi_send(e->ep, m, sizeof m, nullptr, dead, nullptr))) die("fi_send", rc);
        fi_cq_tagged_entry c;
        fi_cq_err_entry err = {};
        check(wait_cq(e, &c, &err) == FI_EHOSTUNREACH, "a send to an endpoint nobody listens on completes with FI_EHOSTUNREACH");
        if (int rc = int(fi_send(e->ep, m, sizeof m, nullptr, e->peer, nullptr))) die("fi_send", rc);
        check(wait_cq(e, &c) == 0, "and the endpoint still sends to a live peer");
    } else {
        char b[16];
        if (int rc = int(fi_recv(e->ep, b, sizeof b, nullptr, FI_ADDR_UNSPEC, nullptr))) die("fi_recv", rc);
        fi_cq_tagged_entry c;
        if (wait_cq(e, &c)) die("recv completion", -FI_EIO);
    }
    barrier();
}

// A message larger than its receive is truncated and reported as such.
void truncate(Endpoint* e) {
    if (g_rank == 0) {
        std::vector<uint8_t> m(1000, 7);
        if (int rc = int(fi_send(e->ep, m.data(), m.size(), nullptr, e->peer, nullptr))) die("fi_send", rc);
        fi_cq_tagged_entry c;
        wait_cq(e, &c);
        barrier();
        return;
    }
    std::vector<uint8_t> b(100, 0);
    if (int rc = int(fi_recv(e->ep, b.data(), b.size(), nullptr, FI_ADDR_UNSPEC, nullptr))) die("fi_recv", rc);
    fi_cq_tagged_entry c;
    fi_cq_err_entry err = {};
    int rc = wait_cq(e, &c, &err);
    check(rc == FI_ETRUNC and err.olen == 900 and b[99] == 7, "a 1000-byte message into 100 bytes reports FI_ETRUNC, 900 left over");
    barrier();
}

// A read of a registered region returns its bytes.
void read_region(Endpoint* e) {
    constexpr size_t kLen = 300000;
    if (g_rank == 1) {
        Region r;
        reg(e, &r, kLen, FI_REMOTE_READ);
        for (size_t k = 0; k < kLen; ++k) r.mem[k] = pattern(5, k);
        uint64_t desc[2] = {uint64_t(uintptr_t(r.mem.data())), r.key};
        put(desc, sizeof desc);
        barrier();
        fi_close(&r.mr->fid);
        return;
    }
    uint64_t desc[2];
    get(desc, sizeof desc);
    std::vector<uint8_t> b(kLen, 0);
    if (int rc = int(fi_read(e->ep, b.data(), kLen, nullptr, e->peer, desc[0], desc[1], nullptr))) die("fi_read", rc);
    fi_cq_tagged_entry c;
    bool ok = wait_cq(e, &c) == 0 and (c.flags & FI_READ);
    for (size_t k = 0; k < kLen and ok; ++k) ok = b[k] == pattern(5, k);
    check(ok, "a 300000-byte read returns the remote bytes");
    barrier();
}

// A write under a key the target never handed out fails on the initiator.
void badkey(Endpoint* e) {
    if (g_rank == 1) {
        Region r;
        reg(e, &r, 4096, FI_REMOTE_WRITE);
        uint64_t desc[2] = {uint64_t(uintptr_t(r.mem.data())), r.key + 12345};
        put(desc, sizeof desc);
        barrier();
        check(r.mem[0] == 0, "the target's memory is untouched");
        fi_close(&r.mr->fid);
        return;
    }
    uint64_t desc[2];
    get(desc, sizeof desc);
    std::vector<uint8_t> b(4096, 9);
    if (int rc = int(fi_write(e->ep, b.data(), b.size(), nullptr, e->peer, desc[0], desc[1], nullptr))) die("fi_write", rc);
    fi_cq_tagged_entry c;
    check(wait_cq(e, &c) == FI_EACCES, "a write under an unknown key completes with FI_EACCES");
    barrier();
}

// The remote completion of fi_writedata comes after every byte of its write.
void writedata(Endpoint* e) {
    constexpr size_t kLen = 1 << 20;
    constexpr int kRounds = 10;
    Region r;
    if (g_rank == 1) reg(e, &r, kLen, FI_REMOTE_WRITE);
    uint64_t desc[2] = {uint64_t(uintptr_t(r.mem.data())), r.key};
    if (g_rank == 1) put(desc, sizeof desc); else get(desc, sizeof desc);
    int bad_rounds = 0;
    for (int round = 0; round < kRounds; ++round) {
        if (g_rank == 0) {
            std::vector<uint8_t> b(kLen);
            for (size_t k = 0; k < kLen; ++k) b[k] = pattern(round, k);
            if (int rc = int(fi_writedata(e->ep, b.data(), kLen, nullptr, uint64_t(round), e->peer, desc[0], desc[1], nullptr)))
                die("fi_writedata", rc);
            fi_cq_tagged_entry c;
            if (wait_cq(e, &c)) die("write completion", -FI_EIO);
            barrier();
            continue;
        }
        fi_cq_tagged_entry c;
        if (wait_cq(e, &c)) die("remote completion", -FI_EIO);
        bool whole = (c.flags & FI_REMOTE_CQ_DATA) and c.data == uint64_t(round);
        {
            // Only under SRDSIM_FI_CQ_DATA_EARLY is the write still landing while this reads, and there that is
            // the whole point of the control. Without it the write is finished before the completion, so leave
            // the read tracked: that the two do not overlap is itself worth checking.
            const bool early = early_cq_data();
            const IntendedRaceOnReads checking(early);
            for (size_t k = 0; k < kLen and whole; ++k) whole = r.mem[k] == pattern(round, k);
        }
        if (not whole) ++bad_rounds;
        barrier();
    }
    if (g_rank == 1) {
        std::printf("rank 1: %d of %d remote completions came before their write had landed\n", bad_rounds, kRounds);
        check(bad_rounds == 0, "every remote completion with CQ data found its whole 1 MiB write landed");
        fi_close(&r.mr->fid);
    }
    // Registered memory cannot be freed while an endpoint that may still be delivering into it is open: the rule
    // is the same on real hardware, and ThreadSanitizer sees the region's free racing the provider's last chunk.
    // Closing the endpoint here joins the thread doing the delivering, and that is the edge that makes the free
    // safe. close_endpoint is idempotent, so main closing it again is harmless.
    if (g_rank == 1) close_endpoint(e);
}

// A write may land after a later write's remote completion, under srd; never under inorder.
void reorder(Endpoint* e) {
    constexpr size_t kLen = 256 << 10;
    constexpr int kRounds = 20;
    Region a, flag;
    if (g_rank == 1) {
        reg(e, &a, kLen, FI_REMOTE_WRITE);
        reg(e, &flag, 8, FI_REMOTE_WRITE);
    }
    uint64_t desc[4] = {uint64_t(uintptr_t(a.mem.data())), a.key, uint64_t(uintptr_t(flag.mem.data())), flag.key};
    if (g_rank == 1) put(desc, sizeof desc); else get(desc, sizeof desc);
    int overtaken = 0;
    for (int round = 0; round < kRounds; ++round) {
        if (g_rank == 0) {
            std::vector<uint8_t> b(kLen, uint8_t(round + 1));
            uint64_t f = round;
            if (int rc = int(fi_write(e->ep, b.data(), kLen, nullptr, e->peer, desc[0], desc[1], nullptr))) die("fi_write", rc);
            if (int rc = int(fi_writedata(e->ep, &f, sizeof f, nullptr, uint64_t(round), e->peer, desc[2], desc[3], nullptr)))
                die("fi_writedata", rc);
            fi_cq_tagged_entry c;
            wait_cq(e, &c);
            wait_cq(e, &c);
            barrier();
            continue;
        }
        fi_cq_tagged_entry c;
        if (wait_cq(e, &c)) die("remote completion", -FI_EIO);
        bool landed = true;
        {
            const IntendedRaceOnReads measuring;   // catching the earlier write half-landed IS the measurement
            for (size_t k = 0; k < kLen and landed; ++k) landed = a.mem[k] == uint8_t(round + 1);
        }
        if (not landed) ++overtaken;
        barrier();
    }
    if (g_rank == 1) {
        const char* mode = std::getenv("SRDSIM_MODE");
        const bool inorder = mode and std::strcmp(mode, "inorder") == 0;
        const char* jitter = std::getenv("SRDSIM_FI_JITTER_US");
        std::printf("rank 1: in %d of %d rounds the earlier write had not fully landed at the later one's remote "
                    "completion (jitter %s us, mode %s)\n",
                    overtaken, kRounds, jitter ? jitter : "default", mode ? mode : "srd");
        if (inorder) check(overtaken == 0, "under inorder, the earlier write always landed first");
        else check(overtaken > 0, "under srd, a later write's completion overtook an earlier write");
        fi_close(&a.mr->fid);
        fi_close(&flag.mr->fid);
    }
    // Registered memory cannot be freed while an endpoint that may still be delivering into it is open: the rule
    // is the same on real hardware, and ThreadSanitizer sees the region's free racing the provider's last chunk.
    // Closing the endpoint here joins the thread doing the delivering, and that is the edge that makes the free
    // safe. close_endpoint is idempotent, so main closing it again is harmless.
    if (g_rank == 1) close_endpoint(e);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s PROVIDER_SO SCENARIO\n", argv[0]); return 2; }
    setenv("FI_PROVIDER_PATH", (std::string("+") + argv[1]).c_str(), 1);
    const std::string scenario = argv[2];
    g_direct = scenario == "directsend";
    // `reorder` asserts that a later write's completion can overtake an earlier write. That has to come from the
    // simulated transport, not from how slowly this machine copies: with the default 200us it passed only because
    // an instrumented memcpy made the 256 KiB write slow, and it failed outright in all 20 rounds once that copy
    // was fast again.
    //
    // The provider gives every chunk a due time drawn uniformly from [0, jitter]. The earlier write is 32 chunks
    // (256 KiB in 8 KiB chunks) and the later one is a single 8-byte chunk, so the later one is last to come due
    // with probability 1/33, and a round shows the overtaking otherwise -- over 20 rounds the chance of seeing
    // none is (1/33)^20. That argument holds for any jitter wide enough to dominate the copying, which is the only
    // thing 3000us is chosen for: a hundred times the default, and still 60ms for the whole scenario. Not forced,
    // so a caller can widen it further.
    if (scenario == "reorder") setenv("SRDSIM_FI_JITTER_US", "3000", 0);
    int p01[2], p10[2];
    if (pipe(p01) or pipe(p10)) return 2;
    std::fflush(nullptr);
    pid_t child = fork();
    if (child < 0) return 2;
    g_rank = child == 0 ? 1 : 0;
    g_to_peer = g_rank == 0 ? p01[1] : p10[1];
    g_from_peer = g_rank == 0 ? p10[0] : p01[0];
    alarm(120);

    Endpoint e;
    open_endpoint(&e);
    if (scenario == "sendrecv") sendrecv(&e);
    else if (scenario == "tagged") tagged(&e);
    else if (scenario == "closeinject") closeafter(&e, false);
    else if (scenario == "closewrite") closeafter(&e, true);
    else if (scenario == "directsend") directsend(&e);
    else if (scenario == "truncate") truncate(&e);
    else if (scenario == "inject") inject(&e);
    else if (scenario == "unreachable") unreachable(&e);
    else if (scenario == "read") read_region(&e);
    else if (scenario == "badkey") badkey(&e);
    else if (scenario == "writedata") writedata(&e);
    else if (scenario == "reorder") reorder(&e);
    else die("unknown scenario", -FI_EINVAL);
    barrier();
    close_endpoint(&e);

    /* _Exit does not flush stdio: with stdout redirected to a file, rank 1's PASS and FAIL lines
       sat in the buffer and were thrown away. The `die` path already flushed; this one did not. */
    if (g_rank == 1) { std::fflush(nullptr); std::_Exit(g_failures ? 1 : 0); }
    int st = 0;
    waitpid(child, &st, 0);
    const bool child_ok = WIFEXITED(st) and WEXITSTATUS(st) == 0;
    std::printf("%s: %s\n", scenario.c_str(), g_failures == 0 and child_ok ? "passed" : "failed");
    return g_failures == 0 and child_ok ? 0 : 1;
}
