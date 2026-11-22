// SPDX-License-Identifier: Apache-2.0
// Checks the libfabric provider through libfabric itself, the way a consumer reaches it: the provider is named
// in FI_PROVIDER_PATH in the preferred form, and every call goes through fi_getinfo, fi_fabric and the objects
// they return.
//
//   test_fi_provider PROVIDER_SO
//
// The hints are those aws-ofi-nccl and NIXL's libfabric backend build, at the commits named in
// docs/libfabric-provider.md. Each check prints one line; the program exits 1 if any fails.

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_ext.h>
#include <rdma/fi_tagged.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>

namespace {

constexpr const char* kDomain = "srdsim0-rdm";
int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (not ok) ++g_failures;
}

using Hints = std::function<void(fi_info*)>;

// The provider's info for these hints, or nullptr. Infos of other providers (the built-in ones) are skipped.
fi_info* ours(const Hints& fill, uint32_t version = FI_VERSION(1, 18)) {
    fi_info* hints = fi_allocinfo();
    fill(hints);
    fi_info* list = nullptr;
    const int rc = fi_getinfo(version, nullptr, nullptr, 0, hints, &list);
    fi_freeinfo(hints);
    if (rc != 0) return nullptr;
    fi_info* found = nullptr;
    for (fi_info* i = list; i; i = i->next)
        if (std::strcmp(i->fabric_attr->prov_name, "efa") == 0 and std::strcmp(i->domain_attr->name, kDomain) == 0) {
            found = fi_dupinfo(i);
            break;
        }
    fi_freeinfo(list);
    return found;
}

// aws-ofi-nccl, RDMA protocol: get_hints() in src/nccl_ofi_rdma.cpp.
void aws_ofi_nccl_rdma(fi_info* h) {
    h->caps = FI_MSG | FI_RMA | FI_HMEM | FI_LOCAL_COMM | FI_REMOTE_COMM;
    h->mode = FI_CONTEXT | FI_CONTEXT2;
    h->ep_attr->type = FI_EP_RDM;
    h->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_HMEM | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY |
                              FI_MR_ENDPOINT;
    h->domain_attr->mr_key_size = 2;
    h->domain_attr->threading = FI_THREAD_COMPLETION;
    h->domain_attr->cq_data_size = 4;
}
// aws-ofi-nccl, send/recv protocol: get_hints() in src/nccl_ofi_sendrecv.cpp, with GPU buffers.
void aws_ofi_nccl_sendrecv(fi_info* h) {
    h->caps = FI_LOCAL_COMM | FI_REMOTE_COMM | FI_TAGGED | FI_MSG | FI_HMEM | FI_RMA | FI_READ;
    h->mode = FI_CONTEXT | FI_CONTEXT2;
    h->ep_attr->type = FI_EP_RDM;
    h->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_ENDPOINT | FI_MR_HMEM | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED |
                              FI_MR_PROV_KEY;
    h->domain_attr->mr_key_size = 2;
    h->domain_attr->threading = FI_THREAD_SAFE;
    h->tx_attr->msg_order = FI_ORDER_SAS;
    h->rx_attr->msg_order = FI_ORDER_SAS;
}
// NIXL's libfabric backend: the hints in src/utils/libfabric/libfabric_rail.cpp for a provider named efa.
void nixl(fi_info* h) {
    h->caps = FI_MSG | FI_RMA | FI_HMEM | FI_LOCAL_COMM | FI_REMOTE_COMM | FI_SOURCE;
    h->mode = FI_CONTEXT;
    h->ep_attr->type = FI_EP_RDM;
    h->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_HMEM | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
    h->domain_attr->mr_key_size = 2;
    h->domain_attr->name = strdup(kDomain);
    h->domain_attr->threading = FI_THREAD_COMPLETION;
}

void getinfo_checks() {
    fi_info* a = ours(aws_ofi_nccl_rdma);
    check(a != nullptr, "aws-ofi-nccl's RDMA hints select the provider");
    if (a) {
        check(a->ep_attr->type == FI_EP_RDM and a->ep_attr->protocol == FI_PROTO_EFA, "an RDM endpoint, protocol EFA");
        check(a->addr_format == FI_ADDR_EFA, "address format FI_ADDR_EFA");
        check(a->domain_attr->cq_data_size == 4, "4 bytes of CQ data, as fi_writedata needs");
        check((a->domain_attr->mr_mode & FI_MR_LOCAL) == 0, "FI_MR_LOCAL is not required");
        check(a->domain_attr->mr_key_size == 2, "the memory key size asked for");
        check(a->domain_attr->data_progress == FI_PROGRESS_MANUAL, "manual progress");
        check((a->caps & FI_HMEM) != 0, "FI_HMEM");
        constexpr uint64_t rma_orders = FI_ORDER_RAR | FI_ORDER_RAW | FI_ORDER_WAR | FI_ORDER_WAW |
                                        FI_ORDER_RMA_RAR | FI_ORDER_RMA_RAW | FI_ORDER_RMA_WAR | FI_ORDER_RMA_WAW;
        check(((a->tx_attr->msg_order | a->rx_attr->msg_order) & rma_orders) == 0, "no RMA ordering is claimed");
        fi_freeinfo(a);
    }
    fi_info* s = ours(aws_ofi_nccl_sendrecv);
    check(s != nullptr, "aws-ofi-nccl's send/recv hints, which need FI_ORDER_SAS, select the provider");
    fi_freeinfo(s);
    fi_info* n = ours(nixl, FI_VERSION(1, 20));
    check(n != nullptr, "NIXL's hints, with the domain named, select the provider");
    fi_freeinfo(n);

    // What SRD does not give is refused, so a consumer never runs on a promise the fabric does not keep.
    fi_info* w = ours([](fi_info* h) { aws_ofi_nccl_rdma(h); h->tx_attr->msg_order = FI_ORDER_RMA_WAW; });
    check(w == nullptr, "write-after-write ordering is refused");
    fi_freeinfo(w);
    fi_info* r = ours([](fi_info* h) { aws_ofi_nccl_rdma(h); h->rx_attr->msg_order = FI_ORDER_RAW; });
    check(r == nullptr, "read-after-write ordering is refused");
    fi_freeinfo(r);
    fi_info* m = ours([](fi_info* h) { aws_ofi_nccl_rdma(h); h->ep_attr->type = FI_EP_MSG; });
    check(m == nullptr, "a connected (FI_EP_MSG) endpoint is refused");
    fi_freeinfo(m);
    fi_info* p = ours([](fi_info* h) { aws_ofi_nccl_rdma(h); h->domain_attr->data_progress = FI_PROGRESS_AUTO; });
    check(p == nullptr, "automatic progress is refused");
    fi_freeinfo(p);
    fi_info* k = ours([](fi_info* h) { aws_ofi_nccl_rdma(h); h->domain_attr->mr_mode = FI_MR_LOCAL; });
    check(k == nullptr, "an application that cannot supply provider keys and virtual addresses is refused");
    fi_freeinfo(k);
    fi_info* d = ours([](fi_info* h) { aws_ofi_nccl_rdma(h); h->domain_attr->cq_data_size = 8; });
    check(d == nullptr, "8 bytes of CQ data are refused");
    fi_freeinfo(d);
    fi_info* any = ours([](fi_info*) {});
    check(any != nullptr and (any->caps & FI_RMA) != 0, "empty hints select the provider");
    fi_freeinfo(any);
}

struct Opened {
    fid_fabric* fabric = nullptr;
    fid_domain* domain = nullptr;
    fid_av* av = nullptr;
    fid_cq* cq = nullptr;
    fid_ep* ep = nullptr;
    ~Opened() {
        if (ep) fi_close(&ep->fid);
        if (cq) fi_close(&cq->fid);
        if (av) fi_close(&av->fid);
        if (domain) fi_close(&domain->fid);
        if (fabric) fi_close(&fabric->fid);
    }
};

int open_all(fi_info* info, Opened* o, bool bind_cq = true) {
    if (int rc = fi_fabric(info->fabric_attr, &o->fabric, nullptr)) return rc;
    if (int rc = fi_domain(o->fabric, info, &o->domain, nullptr)) return rc;
    fi_av_attr av_attr = {};
    av_attr.type = FI_AV_TABLE;
    if (int rc = fi_av_open(o->domain, &av_attr, &o->av, nullptr)) return rc;
    fi_cq_attr cq_attr = {};
    cq_attr.format = FI_CQ_FORMAT_DATA;
    if (int rc = fi_cq_open(o->domain, &cq_attr, &o->cq, nullptr)) return rc;
    if (int rc = fi_endpoint(o->domain, info, &o->ep, nullptr)) return rc;
    if (int rc = fi_ep_bind(o->ep, &o->av->fid, 0)) return rc;
    if (bind_cq)
        if (int rc = fi_ep_bind(o->ep, &o->cq->fid, FI_TRANSMIT | FI_RECV)) return rc;
    return 0;
}

template <class T> int setopt(fid_ep* ep, int name, T v) { return fi_setopt(&ep->fid, FI_OPT_ENDPOINT, name, &v, sizeof v); }

void object_checks() {
    fi_info* info = ours(aws_ofi_nccl_rdma);
    if (info == nullptr) { check(false, "objects: no info to open"); return; }
    {
        Opened o;
        check(open_all(info, &o) == 0, "fabric, domain, address vector, CQ and endpoint open and bind");
        // The options aws-ofi-nccl and NIXL set, before the endpoint is enabled.
        check(setopt(o.ep, FI_OPT_EFA_RNR_RETRY, size_t(7)) == 0, "FI_OPT_EFA_RNR_RETRY before enabling");
        check(setopt(o.ep, FI_OPT_EFA_USE_UNSOLICITED_WRITE_RECV, false) == 0, "FI_OPT_EFA_USE_UNSOLICITED_WRITE_RECV");
        check(setopt(o.ep, FI_OPT_CUDA_API_PERMITTED, false) == 0, "FI_OPT_CUDA_API_PERMITTED");
        check(setopt(o.ep, FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES, true) == -FI_EOPNOTSUPP,
              "FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES is refused, as by the real RDM endpoint");
        check(setopt(o.ep, FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES, true) == 0,
              "FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES is accepted by default");
        setenv("SRDSIM_FI_WRITE_IN_ORDER", "0", 1);
        check(setopt(o.ep, FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES, true) == -FI_EOPNOTSUPP,
              "SRDSIM_FI_WRITE_IN_ORDER=0 refuses it");
        unsetenv("SRDSIM_FI_WRITE_IN_ORDER");
        check(setopt(o.ep, FI_OPT_EFA_RNR_RETRY, 1) == -FI_EINVAL, "an option of the wrong size is refused");
        check(setopt(o.ep, 0x7fff, 1) == -FI_ENOPROTOOPT, "an unknown option is refused");
        bool emulated = true;
        size_t len = sizeof emulated;
        check(fi_getopt(&o.ep->fid, FI_OPT_ENDPOINT, FI_OPT_EFA_EMULATED_WRITE, &emulated, &len) == 0 and
                  len == sizeof(bool) and not emulated,
              "FI_OPT_EFA_EMULATED_WRITE reads false, as aws-ofi-nccl's RDMA protocol requires");

        check(fi_enable(o.ep) == 0, "the endpoint enables");
        check(setopt(o.ep, FI_OPT_EFA_RNR_RETRY, size_t(3)) == -FI_EINVAL, "FI_OPT_EFA_RNR_RETRY after enabling is refused");

        char addr[64];
        size_t addrlen = sizeof addr;
        check(fi_getname(&o.ep->fid, addr, &addrlen) == 0 and addrlen == 32, "fi_getname gives a 32-byte address");
        size_t small = 8;
        check(fi_getname(&o.ep->fid, addr, &small) == -FI_ETOOSMALL and small == 32, "too small a buffer is reported");
        fi_addr_t fa = FI_ADDR_NOTAVAIL;
        check(fi_av_insert(o.av, addr, 1, &fa, 0, nullptr) == 1 and fa == 0, "the address inserts as fi_addr 0");
        char back[32];
        size_t backlen = sizeof back;
        check(fi_av_lookup(o.av, fa, back, &backlen) == 0 and std::memcmp(back, addr, 32) == 0, "and looks up the same");
        char garbage[32] = {};
        fi_addr_t bad = 0;
        check(fi_av_insert(o.av, garbage, 1, &bad, 0, nullptr) == 0 and bad == FI_ADDR_NOTAVAIL,
              "an address that is not the provider's does not insert");
        // Transfers are not carried yet, and say so rather than succeed.
        char buf[8] = {};
        check(fi_send(o.ep, buf, sizeof buf, nullptr, fa, nullptr) == -FI_ENOSYS, "a send reports -FI_ENOSYS");
        fi_cq_data_entry e;
        check(fi_cq_read(o.cq, &e, 1) == -FI_EAGAIN, "an empty CQ reads -FI_EAGAIN");
    }
    {
        Opened o;
        check(open_all(info, &o, false) == 0 and fi_enable(o.ep) == -FI_ENOCQ,
              "an endpoint without completion queues does not enable");
    }
    fi_freeinfo(info);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s PROVIDER_SO\n", argv[0]); return 2; }
    // Read by libfabric when it first initialises, which is the first fi_getinfo below.
    setenv("FI_PROVIDER_PATH", (std::string("+") + argv[1]).c_str(), 1);
    getinfo_checks();
    object_checks();
    std::printf("%d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
