// SPDX-License-Identifier: Apache-2.0
// Three endpoints in one process, each holding addresses of the two others: a write or a send must reach the peer
// it names and no other, and a receive from any source must report which peer sent.
//
//   test_fi_multipeer PROVIDER_SO
//
// The two-process test (test_fi_transfer) has one peer per endpoint, so it cannot tell a transfer that went to the
// right peer from one that went to the only peer there is. Here every endpoint writes a slice that only its
// sender can have produced into the region of each of the others, then sends a message to each of them.

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kN = 3;
constexpr size_t kSlice = 4096;

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

struct Node {
    fi_info* info = nullptr;
    fid_fabric* fabric = nullptr;
    fid_domain* domain = nullptr;
    fid_av* av = nullptr;
    fid_cq* cq = nullptr;
    fid_ep* ep = nullptr;
    char name[64] = {};
    fi_addr_t peer[kN] = {};          // peer[j]: the address of node j in this node's table
    std::vector<uint8_t> region;      // kN slices; slice i is written by node i
    fid_mr* mr = nullptr;
    uint64_t key = 0;
};

void open_node(Node* n) {
    fi_info* hints = fi_allocinfo();
    hints->caps = FI_MSG | FI_RMA | FI_SOURCE | FI_LOCAL_COMM | FI_REMOTE_COMM;
    hints->ep_attr->type = FI_EP_RDM;
    hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
    hints->domain_attr->threading = FI_THREAD_COMPLETION;
    hints->fabric_attr->prov_name = strdup("efa");
    hints->tx_attr->msg_order = FI_ORDER_SAS;
    hints->rx_attr->msg_order = FI_ORDER_SAS;
    int rc = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0, hints, &n->info);
    fi_freeinfo(hints);
    if (rc) die("fi_getinfo", rc);
    if ((rc = fi_fabric(n->info->fabric_attr, &n->fabric, nullptr))) die("fi_fabric", rc);
    if ((rc = fi_domain(n->fabric, n->info, &n->domain, nullptr))) die("fi_domain", rc);
    fi_av_attr av_attr = {};
    av_attr.type = FI_AV_TABLE;
    if ((rc = fi_av_open(n->domain, &av_attr, &n->av, nullptr))) die("fi_av_open", rc);
    fi_cq_attr cq_attr = {};
    cq_attr.format = FI_CQ_FORMAT_TAGGED;
    if ((rc = fi_cq_open(n->domain, &cq_attr, &n->cq, nullptr))) die("fi_cq_open", rc);
    if ((rc = fi_endpoint(n->domain, n->info, &n->ep, nullptr))) die("fi_endpoint", rc);
    if ((rc = fi_ep_bind(n->ep, &n->av->fid, 0))) die("bind av", rc);
    if ((rc = fi_ep_bind(n->ep, &n->cq->fid, FI_TRANSMIT | FI_RECV))) die("bind cq", rc);
    if ((rc = fi_enable(n->ep))) die("fi_enable", rc);
    size_t len = sizeof n->name;
    if ((rc = fi_getname(&n->ep->fid, n->name, &len))) die("fi_getname", rc);
    n->region.assign(kN * kSlice, 0);
    rc = fi_mr_reg(n->domain, n->region.data(), n->region.size(), FI_REMOTE_WRITE | FI_REMOTE_READ, 0, 0, 0, &n->mr, nullptr);
    if (rc) die("fi_mr_reg", rc);
    n->key = fi_mr_key(n->mr);
}

// Waits for one completion on `n`; returns the source in *src when the entry carries one.
int wait_one(Node* n, fi_cq_tagged_entry* out, fi_addr_t* src, int timeout_s = 30) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    while (std::chrono::steady_clock::now() < deadline) {
        fi_addr_t from = FI_ADDR_NOTAVAIL;
        ssize_t r = fi_cq_readfrom(n->cq, out, 1, &from);
        if (r == 1) {
            if (src) *src = from;
            return 0;
        }
        if (r == -FI_EAVAIL) {
            fi_cq_err_entry e = {};
            fi_cq_readerr(n->cq, &e, 0);
            return e.err;
        }
        if (r != -FI_EAGAIN) die("fi_cq_readfrom", int(r));
    }
    die("no completion within the timeout", -FI_ETIMEDOUT);
}

uint8_t byte(int from, int to, size_t i) { return uint8_t((from * 61 + to * 17 + i * 7 + (i >> 8)) & 0xff); }

// Every node writes its own slice into every other node's region: a slice that lands in the wrong region, or in the
// wrong slice, fails the comparison, and so does a slice that did not land.
void writes(Node* nodes) {
    for (int i = 0; i < kN; ++i) {
        for (int j = 0; j < kN; ++j) {
            if (i == j) continue;
            std::vector<uint8_t> src(kSlice);
            for (size_t k = 0; k < kSlice; ++k) src[k] = byte(i, j, k);
            const uint64_t addr = uint64_t(uintptr_t(nodes[j].region.data())) + uint64_t(i) * kSlice;
            int rc = int(fi_write(nodes[i].ep, src.data(), kSlice, nullptr, nodes[i].peer[j], addr, nodes[j].key, nullptr));
            if (rc) die("fi_write", rc);
            fi_cq_tagged_entry c;
            if (wait_one(&nodes[i], &c, nullptr)) die("write completion", -FI_EIO);
        }
    }
    bool ok = true;
    for (int j = 0; j < kN; ++j)
        for (int i = 0; i < kN; ++i)
            for (size_t k = 0; k < kSlice; ++k) {
                const uint8_t want = i == j ? 0 : byte(i, j, k);
                if (nodes[j].region[size_t(i) * kSlice + k] != want) ok = false;
            }
    check(ok, "each node's write landed in the slice of its sender in the region of the node it named, and nowhere else");
}

// Every node sends one message to each other node; a receive from any source reports the sender.
void sends(Node* nodes) {
    std::vector<std::vector<uint8_t>> rx(kN * kN);
    for (int j = 0; j < kN; ++j)
        for (int k = 0; k < kN - 1; ++k) {
            auto& b = rx[size_t(j) * kN + size_t(k)];
            b.assign(64, 0);
            if (int rc = int(fi_recv(nodes[j].ep, b.data(), b.size(), nullptr, FI_ADDR_UNSPEC, &b))) die("fi_recv", rc);
        }
    std::vector<std::vector<uint8_t>> tx(kN * kN);
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j) {
            if (i == j) continue;
            auto& b = tx[size_t(i) * kN + size_t(j)];
            b.assign(64, uint8_t(0x40 + i * 8 + j));
            if (int rc = int(fi_send(nodes[i].ep, b.data(), b.size(), nullptr, nodes[i].peer[j], &b))) die("fi_send", rc);
        }
    bool sent = true;
    // Every node completes kN-1 sends and kN-1 receives, in any order. A send completes only once its target has
    // taken it in, which the target does when it reads its completion queue, so all the queues are read in turn.
    bool sources_ok = true, data_ok = true;
    int sends_done[kN] = {}, recvs_done[kN] = {};
    bool seen_from[kN][kN] = {};
    auto finished = [&] {
        for (int n = 0; n < kN; ++n)
            if (sends_done[n] < kN - 1 or recvs_done[n] < kN - 1) return false;
        return true;
    };
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (not finished() and std::chrono::steady_clock::now() < deadline) {
        for (int n = 0; n < kN; ++n) {
            fi_cq_tagged_entry c;
            fi_addr_t src = FI_ADDR_NOTAVAIL;
            const ssize_t r = fi_cq_readfrom(nodes[n].cq, &c, 1, &src);
            if (r == -FI_EAGAIN) continue;
            if (r != 1) die("fi_cq_readfrom", int(r));
            if (c.flags & FI_RECV) {
                ++recvs_done[n];
                int from = -1;
                for (int p = 0; p < kN; ++p)
                    if (p != n and nodes[n].peer[p] == src) from = p;
                if (from < 0 or seen_from[n][from]) { sources_ok = false; continue; }
                seen_from[n][from] = true;
                const auto* b = static_cast<std::vector<uint8_t>*>(c.op_context);
                if (c.len != 64) data_ok = false;
                for (size_t k = 0; k < 64; ++k)
                    if ((*b)[k] != uint8_t(0x40 + from * 8 + n)) data_ok = false;
            } else {
                ++sends_done[n];
            }
        }
    }
    sent = finished();
    check(sent, "every send completed");
    check(sources_ok, "a receive from any source reports the peer that sent it, once per peer");
    check(data_ok, "each message holds what its sender wrote for this receiver");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s PROVIDER_SO\n", argv[0]); return 2; }
    setenv("FI_PROVIDER_PATH", (std::string("+") + argv[1]).c_str(), 1);
    static Node nodes[kN];
    for (auto& n : nodes) open_node(&n);
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j) {
            if (i == j) continue;
            if (fi_av_insert(nodes[i].av, nodes[j].name, 1, &nodes[i].peer[j], 0, nullptr) != 1) die("fi_av_insert", -FI_EINVAL);
        }
    writes(nodes);
    sends(nodes);
    for (auto& n : nodes) {
        fi_close(&n.mr->fid);
        fi_close(&n.ep->fid);
        fi_close(&n.cq->fid);
        fi_close(&n.av->fid);
        fi_close(&n.domain->fid);
        fi_close(&n.fabric->fid);
        fi_freeinfo(n.info);
    }
    std::printf("multipeer: %s\n", g_failures == 0 ? "passed" : "failed");
    return g_failures == 0 ? 0 : 1;
}
