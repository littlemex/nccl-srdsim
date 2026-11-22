// SPDX-License-Identifier: Apache-2.0
// A libfabric provider that registers under the name "efa" and gives consumers that select EFA by that name
// (aws-ofi-nccl, NIXL's libfabric backend) an RDM endpoint with SRD's delivery rules, on a host whose real
// EFA devices are not used or do not exist. It is selected only by naming it in FI_PROVIDER_PATH, in the
// preferred form "+/path/libsrdsim-fi.so"; it is never installed where libfabric looks by default.
//
// This file is the front end: fi_getinfo, the fabric, domain, address vector, completion queue and endpoint
// objects, and the EFA endpoint options. Data transfer calls answer -FI_ENOSYS until the transfer path is
// added (docs/libfabric-provider.md, "Status").

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_ext.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_collective.h>
#include <rdma/providers/fi_log.h>
#include <rdma/providers/fi_prov.h>

#include <strings.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace srdsim::fi {
namespace {

// ---- What the provider reports -------------------------------------------------------------------------

constexpr const char* kName = "efa";            // fabric_attr->prov_name and fabric_attr->name
constexpr const char* kDomain = "srdsim0-rdm";  // the real provider names its RDM domain "<device>-rdm"
constexpr uint32_t kProviderVersion = FI_VERSION(0, 1);
constexpr size_t kCqDataSize = 4;               // as EFA_CQ_DATA_SIZE
constexpr size_t kMrKeySize = 4;                // as the real provider: an ibv lkey
constexpr size_t kIovLimit = 4;
constexpr size_t kInjectSize = 4096;
constexpr size_t kQueueSize = 8192;
// libfabric's FI_TAG_GENERIC (include/ofi.h, not installed): every tag bit available to the application.
constexpr uint64_t kTagGeneric = 0xAAAAAAAAAAAAAAAAull;

// Two-sided operations keep their order (the real provider reorders them back in software); RMA reports no
// ordering, which is what SRD gives.
constexpr uint64_t kMsgOrder = FI_ORDER_SAS;

constexpr uint64_t kPrimaryCaps = FI_MSG | FI_RMA | FI_TAGGED | FI_READ | FI_WRITE | FI_SEND | FI_RECV |
                                  FI_REMOTE_READ | FI_REMOTE_WRITE;
constexpr uint64_t kSecondaryCaps = FI_SOURCE | FI_DIRECTED_RECV | FI_LOCAL_COMM | FI_REMOTE_COMM;
// FI_HMEM is offered, as both consumers require it; device memory is carried once the transfer path exists.
constexpr uint64_t kCaps = kPrimaryCaps | kSecondaryCaps | FI_HMEM;
// The registration modes the provider requires of the application, as the real RDM endpoint does.
constexpr uint64_t kMrModeRequired = FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;

extern fi_provider g_provider;

#define SRDSIM_FI_WARN(...) FI_WARN(&g_provider, FI_LOG_CORE, __VA_ARGS__)
#define SRDSIM_FI_INFO(...) FI_INFO(&g_provider, FI_LOG_CORE, __VA_ARGS__)

// ---- Configuration -------------------------------------------------------------------------------------

// SRDSIM_FI_WRITE_IN_ORDER: 1 (default) accepts FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES as a device that
// supports it does, 0 refuses it with -FI_EOPNOTSUPP as a device that does not, so both paths of a consumer
// can be tested.
bool write_in_order_supported() {
    const char* v = std::getenv("SRDSIM_FI_WRITE_IN_ORDER");
    if (v == nullptr or *v == '\0') return true;
    if (std::strcmp(v, "1") == 0) return true;
    if (std::strcmp(v, "0") == 0) return false;
    SRDSIM_FI_WARN("SRDSIM_FI_WRITE_IN_ORDER=%s is neither 0 nor 1; using 1\n", v);
    return true;
}

// ---- Stubs for calls the provider does not implement ---------------------------------------------------

// Fills a function pointer with a function of exactly its type that reports the call as not supported.
template <class F> struct Unsupported;
template <class R, class... A> struct Unsupported<R (*)(A...)> {
    static R call(A...) {
        if constexpr (std::is_pointer_v<R>) return nullptr;
        else return static_cast<R>(-FI_ENOSYS);
    }
};
template <class F> void unsupported(F& f) { f = &Unsupported<F>::call; }

// ---- Objects -------------------------------------------------------------------------------------------

// The address an endpoint reports from fi_getname and a peer inserts into its address vector. 32 bytes, the
// size of the real provider's raw address, so a consumer's buffers sized for EFA fit it.
struct Address {
    char magic[8];       // "SRDSIMFI"
    uint32_t version;    // of this layout
    uint32_t pid;
    uint64_t nonce;      // tells apart endpoints of one process
    uint64_t reserved;
};
static_assert(sizeof(Address) == 32);
constexpr char kMagic[8] = {'S', 'R', 'D', 'S', 'I', 'M', 'F', 'I'};
constexpr uint32_t kAddressVersion = 1;

struct Fabric { fid_fabric fid{}; };
struct Domain { fid_domain fid{}; Fabric* fabric = nullptr; fi_info* info = nullptr; };
struct Cq { fid_cq fid{}; Domain* domain = nullptr; fi_cq_format format = FI_CQ_FORMAT_UNSPEC; };
struct Av {
    fid_av fid{};
    Domain* domain = nullptr;
    std::mutex mu;
    std::vector<Address> peers;   // fi_addr_t is the index
};
struct Ep {
    fid_ep fid{};
    Domain* domain = nullptr;
    fi_info* info = nullptr;
    Av* av = nullptr;
    Cq* tx_cq = nullptr;
    Cq* rx_cq = nullptr;
    bool enabled = false;
    Address addr{};
    // EFA endpoint options, as set by the consumer.
    size_t rnr_retry = 7;
    size_t min_multi_recv = 0;
    bool use_unsolicited_write_recv = true;
    bool write_in_order_128 = false;
    bool homogeneous_peers = false;
    bool use_device_rdma = true;
    bool cuda_api_permitted = true;
    bool shared_memory_permitted = true;
    int hmem_p2p = FI_HMEM_P2P_ENABLED;
};

template <class T, class F> T* outer(F* f) {
    // Every object starts with its fid, so the object's address is the fid's.
    static_assert(std::is_standard_layout_v<T>);
    return reinterpret_cast<T*>(f);
}

// ---- fi_getinfo ----------------------------------------------------------------------------------------

bool name_is(const char* got, const char* want) { return got == nullptr or strcasecmp(got, want) == 0; }

// Why `hints` cannot be served, or nullptr when they can.
const char* mismatch(const fi_info* hints) {
    if (hints == nullptr) return nullptr;
    if ((hints->caps & ~kCaps) != 0) return "capabilities the provider does not offer";
    if (hints->addr_format != FI_FORMAT_UNSPEC and hints->addr_format != FI_ADDR_EFA) return "address format";
    if (hints->ep_attr) {
        const auto* e = hints->ep_attr;
        if (e->type != FI_EP_UNSPEC and e->type != FI_EP_RDM) return "endpoint type other than FI_EP_RDM";
        if (e->protocol != FI_PROTO_UNSPEC and e->protocol != FI_PROTO_EFA) return "protocol";
    }
    if (hints->fabric_attr) {
        if (not name_is(hints->fabric_attr->prov_name, kName)) return "provider name";
        if (not name_is(hints->fabric_attr->name, kName)) return "fabric name";
    }
    if (hints->domain_attr) {
        const auto* d = hints->domain_attr;
        if (not name_is(d->name, kDomain)) return "domain name";
        if (d->cq_data_size > kCqDataSize) return "more CQ data than 4 bytes";
        if (d->mr_key_size > kMrKeySize) return "a larger memory key";
        if (d->control_progress == FI_PROGRESS_AUTO or d->data_progress == FI_PROGRESS_AUTO)
            return "automatic progress";
        // 0 leaves the modes to the provider, as libfabric's own check (ofi_check_mr_mode) reads it.
        const uint64_t need = kMrModeRequired | ((hints->caps & FI_HMEM) ? FI_MR_HMEM : 0);
        if (d->mr_mode != 0 and (d->mr_mode & need) != need) return "registration modes the provider requires";
    }
    const uint64_t order_needed = (hints->tx_attr ? hints->tx_attr->msg_order : 0) |
                                  (hints->rx_attr ? hints->rx_attr->msg_order : 0);
    if ((order_needed & ~kMsgOrder) != 0) return "an ordering SRD does not give";
    return nullptr;
}

fi_info* make_info(const fi_info* hints) {
    fi_info* i = fi_allocinfo();
    if (i == nullptr) return nullptr;
    i->caps = (hints and hints->caps) ? (hints->caps | FI_LOCAL_COMM | FI_REMOTE_COMM) : kCaps;
    i->mode = 0;
    i->addr_format = FI_ADDR_EFA;

    i->fabric_attr->name = strdup(kName);
    i->fabric_attr->prov_name = strdup(kName);
    i->fabric_attr->prov_version = kProviderVersion;

    auto* d = i->domain_attr;
    d->name = strdup(kDomain);
    d->threading = (hints and hints->domain_attr and hints->domain_attr->threading != FI_THREAD_UNSPEC)
                       ? hints->domain_attr->threading : FI_THREAD_SAFE;
    d->control_progress = FI_PROGRESS_MANUAL;
    d->data_progress = FI_PROGRESS_MANUAL;
    d->resource_mgmt = FI_RM_ENABLED;
    d->av_type = FI_AV_TABLE;
    d->mr_mode = kMrModeRequired | ((i->caps & FI_HMEM) ? FI_MR_HMEM : 0);
    d->mr_key_size = (hints and hints->domain_attr and hints->domain_attr->mr_key_size)
                         ? hints->domain_attr->mr_key_size : kMrKeySize;
    d->cq_data_size = kCqDataSize;
    d->cq_cnt = 1024;
    d->ep_cnt = 1024;
    d->tx_ctx_cnt = 1;
    d->rx_ctx_cnt = 1;
    d->max_ep_tx_ctx = 1;
    d->max_ep_rx_ctx = 1;
    d->mr_iov_limit = 1;
    d->caps = FI_LOCAL_COMM | FI_REMOTE_COMM;

    auto* e = i->ep_attr;
    e->type = FI_EP_RDM;
    e->protocol = FI_PROTO_EFA;
    e->protocol_version = 1;
    e->max_msg_size = UINT64_MAX;
    e->mem_tag_format = kTagGeneric;
    e->tx_ctx_cnt = 1;
    e->rx_ctx_cnt = 1;

    i->tx_attr->caps = i->caps;
    i->tx_attr->msg_order = kMsgOrder;
    i->tx_attr->comp_order = 0;
    i->tx_attr->inject_size = kInjectSize;
    i->tx_attr->size = kQueueSize;
    i->tx_attr->iov_limit = kIovLimit;
    i->tx_attr->rma_iov_limit = 1;
    i->rx_attr->caps = i->caps;
    i->rx_attr->msg_order = kMsgOrder;
    i->rx_attr->comp_order = 0;
    i->rx_attr->size = kQueueSize;
    i->rx_attr->iov_limit = kIovLimit;
    i->rx_attr->op_flags = FI_COMPLETION;
    if (i->fabric_attr->name == nullptr or i->fabric_attr->prov_name == nullptr or d->name == nullptr) {
        fi_freeinfo(i);
        return nullptr;
    }
    return i;
}

int getinfo(uint32_t, const char* node, const char* service, uint64_t flags, const fi_info* hints,
            fi_info** info) {
    *info = nullptr;
    // An address string for a node or a service is not parsed yet: consumers that select EFA pass neither.
    if (node or service or (flags & FI_SOURCE)) {
        SRDSIM_FI_INFO("a node, a service or FI_SOURCE was given; not supported\n");
        return -FI_ENODATA;
    }
    if (const char* why = mismatch(hints)) {
        SRDSIM_FI_INFO("hints not served: %s\n", why);
        return -FI_ENODATA;
    }
    *info = make_info(hints);
    return *info ? 0 : -FI_ENOMEM;
}

// ---- Completion queue ----------------------------------------------------------------------------------

int cq_close(fid* f) { delete outer<Cq>(f); return 0; }
// Nothing completes until the transfer path exists.
ssize_t cq_read(fid_cq*, void*, size_t) { return -FI_EAGAIN; }
ssize_t cq_readfrom(fid_cq*, void*, size_t, fi_addr_t*) { return -FI_EAGAIN; }
ssize_t cq_readerr(fid_cq*, fi_cq_err_entry*, uint64_t) { return -FI_EAGAIN; }
const char* cq_strerror(fid_cq*, int prov_errno, const void*, char* buf, size_t len) {
    const char* s = fi_strerror(prov_errno);
    if (buf and len) { std::snprintf(buf, len, "%s", s); return buf; }
    return s;
}

fi_ops g_cq_fid_ops;
fi_ops_cq g_cq_ops;

int cq_open(fid_domain* dom, fi_cq_attr* attr, fid_cq** out, void* context) {
    if (attr and attr->wait_obj != FI_WAIT_NONE and attr->wait_obj != FI_WAIT_UNSPEC) return -FI_ENOSYS;
    auto* cq = new (std::nothrow) Cq;
    if (cq == nullptr) return -FI_ENOMEM;
    cq->domain = outer<Domain>(dom);
    cq->format = attr ? attr->format : FI_CQ_FORMAT_UNSPEC;
    cq->fid.fid.fclass = FI_CLASS_CQ;
    cq->fid.fid.context = context;
    cq->fid.fid.ops = &g_cq_fid_ops;
    cq->fid.ops = &g_cq_ops;
    *out = &cq->fid;
    return 0;
}

// ---- Address vector ------------------------------------------------------------------------------------

int av_close(fid* f) { delete outer<Av>(f); return 0; }

int av_insert(fid_av* fav, const void* addr, size_t count, fi_addr_t* fi_addr, uint64_t flags, void*) {
    if (flags != 0) return -FI_EBADFLAGS;
    auto* av = outer<Av>(fav);
    const auto* in = static_cast<const Address*>(addr);
    std::lock_guard<std::mutex> lock(av->mu);
    size_t inserted = 0;
    for (size_t k = 0; k < count; ++k) {
        if (std::memcmp(in[k].magic, kMagic, sizeof kMagic) != 0 or in[k].version != kAddressVersion) {
            if (fi_addr) fi_addr[k] = FI_ADDR_NOTAVAIL;
            continue;
        }
        av->peers.push_back(in[k]);
        if (fi_addr) fi_addr[k] = av->peers.size() - 1;
        ++inserted;
    }
    return static_cast<int>(inserted);
}

int av_remove(fid_av* fav, fi_addr_t* fi_addr, size_t count, uint64_t) {
    auto* av = outer<Av>(fav);
    std::lock_guard<std::mutex> lock(av->mu);
    for (size_t k = 0; k < count; ++k)
        if (fi_addr[k] >= av->peers.size()) return -FI_EINVAL;
    for (size_t k = 0; k < count; ++k) std::memset(&av->peers[fi_addr[k]], 0, sizeof(Address));
    return 0;
}

int av_lookup(fid_av* fav, fi_addr_t fi_addr, void* addr, size_t* addrlen) {
    auto* av = outer<Av>(fav);
    std::lock_guard<std::mutex> lock(av->mu);
    if (fi_addr >= av->peers.size()) return -FI_EINVAL;
    std::memcpy(addr, &av->peers[fi_addr], std::min(*addrlen, sizeof(Address)));
    *addrlen = sizeof(Address);
    return 0;
}

const char* av_straddr(fid_av*, const void* addr, char* buf, size_t* len) {
    const auto* a = static_cast<const Address*>(addr);
    const int n = std::snprintf(buf, *len, "srdsim://%u/%016llx", a->pid,
                                static_cast<unsigned long long>(a->nonce));
    *len = static_cast<size_t>(n) + 1;
    return buf;
}

fi_ops g_av_fid_ops;
fi_ops_av g_av_ops;

int av_open(fid_domain* dom, fi_av_attr* attr, fid_av** out, void* context) {
    if (attr and attr->type != FI_AV_UNSPEC and attr->type != FI_AV_TABLE) return -FI_ENOSYS;
    if (attr and attr->name) return -FI_ENOSYS;   // named, shared address vectors
    auto* av = new (std::nothrow) Av;
    if (av == nullptr) return -FI_ENOMEM;
    av->domain = outer<Domain>(dom);
    av->fid.fid.fclass = FI_CLASS_AV;
    av->fid.fid.context = context;
    av->fid.fid.ops = &g_av_fid_ops;
    av->fid.ops = &g_av_ops;
    *out = &av->fid;
    return 0;
}

// ---- Endpoint ------------------------------------------------------------------------------------------

int ep_close(fid* f) {
    auto* ep = outer<Ep>(f);
    fi_freeinfo(ep->info);
    delete ep;
    return 0;
}

int ep_bind(fid* f, fid* bfid, uint64_t flags) {
    auto* ep = outer<Ep>(f);
    if (ep->enabled) return -FI_EOPBADSTATE;
    switch (bfid->fclass) {
    case FI_CLASS_AV:
        if (ep->av) return -FI_EINVAL;
        ep->av = outer<Av>(bfid);
        return 0;
    case FI_CLASS_CQ:
        if ((flags & (FI_TRANSMIT | FI_RECV)) == 0) return -FI_EBADFLAGS;
        if (flags & FI_TRANSMIT) { if (ep->tx_cq) return -FI_EINVAL; ep->tx_cq = outer<Cq>(bfid); }
        if (flags & FI_RECV) { if (ep->rx_cq) return -FI_EINVAL; ep->rx_cq = outer<Cq>(bfid); }
        return 0;
    default:
        return -FI_ENOSYS;
    }
}

int ep_control(fid* f, int command, void*) {
    auto* ep = outer<Ep>(f);
    if (command != FI_ENABLE) return -FI_ENOSYS;
    // As the real RDM endpoint: an address vector and both completion queues before it is enabled.
    if (ep->av == nullptr or ep->tx_cq == nullptr or ep->rx_cq == nullptr) return -FI_ENOCQ;
    ep->enabled = true;
    return 0;
}

int ep_getname(fid_t f, void* addr, size_t* addrlen) {
    auto* ep = outer<Ep>(f);
    const size_t have = *addrlen;
    *addrlen = sizeof(Address);
    if (have < sizeof(Address)) return -FI_ETOOSMALL;
    std::memcpy(addr, &ep->addr, sizeof(Address));
    return 0;
}

template <class T> int set_value(const void* optval, size_t optlen, T* out) {
    if (optlen != sizeof(T)) return -FI_EINVAL;
    std::memcpy(out, optval, sizeof(T));
    return 0;
}
template <class T> int get_value(T value, void* optval, size_t* optlen) {
    if (*optlen < sizeof(T)) { *optlen = sizeof(T); return -FI_ETOOSMALL; }
    std::memcpy(optval, &value, sizeof(T));
    *optlen = sizeof(T);
    return 0;
}

// The options and the answers of the real RDM endpoint in libfabric's prov/efa/src/rdm/efa_rdm_ep_fiops.c.
int ep_setopt(fid_t f, int level, int optname, const void* optval, size_t optlen) {
    auto* ep = outer<Ep>(f);
    if (level != FI_OPT_ENDPOINT) return -FI_ENOPROTOOPT;
    switch (optname) {
    case FI_OPT_MIN_MULTI_RECV: return set_value(optval, optlen, &ep->min_multi_recv);
    case FI_OPT_EFA_RNR_RETRY:
        if (ep->enabled) return -FI_EINVAL;   // to be set before the endpoint is enabled
        return set_value(optval, optlen, &ep->rnr_retry);
    case FI_OPT_FI_HMEM_P2P: return set_value(optval, optlen, &ep->hmem_p2p);
    case FI_OPT_CUDA_API_PERMITTED: return set_value(optval, optlen, &ep->cuda_api_permitted);
    case FI_OPT_SHARED_MEMORY_PERMITTED: return set_value(optval, optlen, &ep->shared_memory_permitted);
    case FI_OPT_EFA_USE_DEVICE_RDMA: return set_value(optval, optlen, &ep->use_device_rdma);
    case FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES:
        if (optlen != sizeof(bool)) return -FI_EINVAL;
        return -FI_EOPNOTSUPP;   // the real RDM endpoint refuses it
    case FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES: {
        bool v = false;
        if (int rc = set_value(optval, optlen, &v)) return rc;
        if (v and not write_in_order_supported()) return -FI_EOPNOTSUPP;
        ep->write_in_order_128 = v;
        return 0;
    }
    case FI_OPT_EFA_HOMOGENEOUS_PEERS: return set_value(optval, optlen, &ep->homogeneous_peers);
    case FI_OPT_EFA_USE_UNSOLICITED_WRITE_RECV: return set_value(optval, optlen, &ep->use_unsolicited_write_recv);
    default: return -FI_ENOPROTOOPT;
    }
}

int ep_getopt(fid_t f, int level, int optname, void* optval, size_t* optlen) {
    auto* ep = outer<Ep>(f);
    if (level != FI_OPT_ENDPOINT) return -FI_ENOPROTOOPT;
    switch (optname) {
    case FI_OPT_MIN_MULTI_RECV: return get_value(ep->min_multi_recv, optval, optlen);
    case FI_OPT_EFA_RNR_RETRY: return get_value(ep->rnr_retry, optval, optlen);
    case FI_OPT_FI_HMEM_P2P: return get_value(ep->hmem_p2p, optval, optlen);
    case FI_OPT_MAX_MSG_SIZE:
    case FI_OPT_MAX_TAGGED_SIZE:
    case FI_OPT_MAX_RMA_SIZE: return get_value(size_t(ep->info->ep_attr->max_msg_size), optval, optlen);
    case FI_OPT_INJECT_MSG_SIZE:
    case FI_OPT_INJECT_TAGGED_SIZE:
    case FI_OPT_INJECT_RMA_SIZE: return get_value(size_t(ep->info->tx_attr->inject_size), optval, optlen);
    // Reads and writes are carried as such, not emulated with sends; atomics are not offered at all.
    case FI_OPT_EFA_EMULATED_READ: return get_value(false, optval, optlen);
    case FI_OPT_EFA_EMULATED_WRITE: return get_value(false, optval, optlen);
    default: return -FI_ENOPROTOOPT;
    }
}

fi_ops g_ep_fid_ops;
fi_ops_ep g_ep_ops;
fi_ops_cm g_ep_cm;
fi_ops_msg g_ep_msg;
fi_ops_rma g_ep_rma;
fi_ops_tagged g_ep_tagged;
fi_ops_atomic g_ep_atomic;
fi_ops_collective g_ep_collective;

std::atomic<uint64_t> g_nonce_counter{0};

int endpoint(fid_domain* dom, fi_info* info, fid_ep** out, void* context) {
    if (info == nullptr or mismatch(info) != nullptr) return -FI_EINVAL;
    auto* ep = new (std::nothrow) Ep;
    if (ep == nullptr) return -FI_ENOMEM;
    ep->info = fi_dupinfo(info);
    if (ep->info == nullptr) { delete ep; return -FI_ENOMEM; }
    ep->domain = outer<Domain>(dom);
    std::memcpy(ep->addr.magic, kMagic, sizeof kMagic);
    ep->addr.version = kAddressVersion;
    ep->addr.pid = static_cast<uint32_t>(getpid());
    static const uint64_t seed = std::random_device{}();
    ep->addr.nonce = seed ^ (++g_nonce_counter * 0x9E3779B97F4A7C15ull);
    ep->fid.fid.fclass = FI_CLASS_EP;
    ep->fid.fid.context = context;
    ep->fid.fid.ops = &g_ep_fid_ops;
    ep->fid.ops = &g_ep_ops;
    ep->fid.cm = &g_ep_cm;
    ep->fid.msg = &g_ep_msg;
    ep->fid.rma = &g_ep_rma;
    ep->fid.tagged = &g_ep_tagged;
    ep->fid.atomic = &g_ep_atomic;
    ep->fid.collective = &g_ep_collective;
    *out = &ep->fid;
    return 0;
}

// ---- Domain and fabric ---------------------------------------------------------------------------------

int domain_close(fid* f) {
    auto* d = outer<Domain>(f);
    fi_freeinfo(d->info);
    delete d;
    return 0;
}

fi_ops g_domain_fid_ops;
fi_ops_domain g_domain_ops;
fi_ops_mr g_domain_mr;

int domain_open(fid_fabric* fab, fi_info* info, fid_domain** out, void* context) {
    if (info == nullptr or mismatch(info) != nullptr) return -FI_EINVAL;
    auto* d = new (std::nothrow) Domain;
    if (d == nullptr) return -FI_ENOMEM;
    d->info = fi_dupinfo(info);
    if (d->info == nullptr) { delete d; return -FI_ENOMEM; }
    d->fabric = outer<Fabric>(fab);
    d->fid.fid.fclass = FI_CLASS_DOMAIN;
    d->fid.fid.context = context;
    d->fid.fid.ops = &g_domain_fid_ops;
    d->fid.ops = &g_domain_ops;
    d->fid.mr = &g_domain_mr;
    *out = &d->fid;
    return 0;
}

int domain2(fid_fabric* fab, fi_info* info, fid_domain** out, uint64_t flags, void* context) {
    if (flags != 0) return -FI_EBADFLAGS;
    return domain_open(fab, info, out, context);
}

int fabric_close(fid* f) { delete outer<Fabric>(f); return 0; }

fi_ops g_fabric_fid_ops;
fi_ops_fabric g_fabric_ops;

int fabric(fi_fabric_attr* attr, fid_fabric** out, void* context) {
    if (attr and not name_is(attr->name, kName)) return -FI_ENODATA;
    auto* f = new (std::nothrow) Fabric;
    if (f == nullptr) return -FI_ENOMEM;
    f->fid.fid.fclass = FI_CLASS_FABRIC;
    f->fid.fid.context = context;
    f->fid.fid.ops = &g_fabric_fid_ops;
    f->fid.ops = &g_fabric_ops;
    *out = &f->fid;
    return 0;
}

// ---- Op tables -----------------------------------------------------------------------------------------

void fid_ops_with(fi_ops& o, int (*close)(fid*)) {
    o.size = sizeof o;
    o.close = close;
    unsupported(o.bind); unsupported(o.control); unsupported(o.ops_open); unsupported(o.tostr);
    unsupported(o.ops_set);
}

void init_tables() {
    fid_ops_with(g_fabric_fid_ops, fabric_close);
    g_fabric_ops.size = sizeof g_fabric_ops;
    g_fabric_ops.domain = domain_open;
    g_fabric_ops.domain2 = domain2;
    unsupported(g_fabric_ops.passive_ep); unsupported(g_fabric_ops.eq_open); unsupported(g_fabric_ops.wait_open);
    unsupported(g_fabric_ops.trywait);

    fid_ops_with(g_domain_fid_ops, domain_close);
    g_domain_ops.size = sizeof g_domain_ops;
    g_domain_ops.av_open = av_open;
    g_domain_ops.cq_open = cq_open;
    g_domain_ops.endpoint = endpoint;
    unsupported(g_domain_ops.scalable_ep); unsupported(g_domain_ops.cntr_open); unsupported(g_domain_ops.poll_open);
    unsupported(g_domain_ops.stx_ctx); unsupported(g_domain_ops.srx_ctx); unsupported(g_domain_ops.query_atomic);
    unsupported(g_domain_ops.query_collective); unsupported(g_domain_ops.endpoint2);
    unsupported(g_domain_ops.xpu_ctx);
    g_domain_mr.size = sizeof g_domain_mr;
    unsupported(g_domain_mr.reg); unsupported(g_domain_mr.regv); unsupported(g_domain_mr.regattr);

    fid_ops_with(g_cq_fid_ops, cq_close);
    g_cq_ops.size = sizeof g_cq_ops;
    g_cq_ops.read = cq_read;
    g_cq_ops.readfrom = cq_readfrom;
    g_cq_ops.readerr = cq_readerr;
    g_cq_ops.strerror = cq_strerror;
    unsupported(g_cq_ops.sread); unsupported(g_cq_ops.sreadfrom); unsupported(g_cq_ops.signal);
    unsupported(g_cq_ops.export_xpu);

    fid_ops_with(g_av_fid_ops, av_close);
    g_av_ops.size = sizeof g_av_ops;
    g_av_ops.insert = av_insert;
    g_av_ops.remove = av_remove;
    g_av_ops.lookup = av_lookup;
    g_av_ops.straddr = av_straddr;
    unsupported(g_av_ops.insertsvc); unsupported(g_av_ops.insertsym); unsupported(g_av_ops.av_set);
    unsupported(g_av_ops.insert_auth_key); unsupported(g_av_ops.lookup_auth_key);
    unsupported(g_av_ops.set_user_id); unsupported(g_av_ops.lookup2);

    fid_ops_with(g_ep_fid_ops, ep_close);
    g_ep_fid_ops.bind = ep_bind;
    g_ep_fid_ops.control = ep_control;
    g_ep_ops.size = sizeof g_ep_ops;
    g_ep_ops.getopt = ep_getopt;
    g_ep_ops.setopt = ep_setopt;
    unsupported(g_ep_ops.cancel); unsupported(g_ep_ops.tx_ctx); unsupported(g_ep_ops.rx_ctx);
    unsupported(g_ep_ops.rx_size_left); unsupported(g_ep_ops.tx_size_left); unsupported(g_ep_ops.export_xpu);
    g_ep_cm.size = sizeof g_ep_cm;
    g_ep_cm.getname = ep_getname;
    unsupported(g_ep_cm.setname); unsupported(g_ep_cm.getpeer); unsupported(g_ep_cm.connect);
    unsupported(g_ep_cm.listen); unsupported(g_ep_cm.accept); unsupported(g_ep_cm.reject);
    unsupported(g_ep_cm.shutdown); unsupported(g_ep_cm.join);
    g_ep_msg.size = sizeof g_ep_msg;
    unsupported(g_ep_msg.recv); unsupported(g_ep_msg.recvv); unsupported(g_ep_msg.recvmsg);
    unsupported(g_ep_msg.send); unsupported(g_ep_msg.sendv); unsupported(g_ep_msg.sendmsg);
    unsupported(g_ep_msg.inject); unsupported(g_ep_msg.senddata); unsupported(g_ep_msg.injectdata);
    g_ep_rma.size = sizeof g_ep_rma;
    unsupported(g_ep_rma.read); unsupported(g_ep_rma.readv); unsupported(g_ep_rma.readmsg);
    unsupported(g_ep_rma.write); unsupported(g_ep_rma.writev); unsupported(g_ep_rma.writemsg);
    unsupported(g_ep_rma.inject); unsupported(g_ep_rma.writedata); unsupported(g_ep_rma.injectdata);
    g_ep_tagged.size = sizeof g_ep_tagged;
    unsupported(g_ep_tagged.recv); unsupported(g_ep_tagged.recvv); unsupported(g_ep_tagged.recvmsg);
    unsupported(g_ep_tagged.send); unsupported(g_ep_tagged.sendv); unsupported(g_ep_tagged.sendmsg);
    unsupported(g_ep_tagged.inject); unsupported(g_ep_tagged.senddata); unsupported(g_ep_tagged.injectdata);
    g_ep_atomic.size = sizeof g_ep_atomic;
    unsupported(g_ep_atomic.write); unsupported(g_ep_atomic.writev); unsupported(g_ep_atomic.writemsg);
    unsupported(g_ep_atomic.inject); unsupported(g_ep_atomic.readwrite); unsupported(g_ep_atomic.readwritev);
    unsupported(g_ep_atomic.readwritemsg); unsupported(g_ep_atomic.compwrite); unsupported(g_ep_atomic.compwritev);
    unsupported(g_ep_atomic.compwritemsg); unsupported(g_ep_atomic.writevalid);
    unsupported(g_ep_atomic.readwritevalid); unsupported(g_ep_atomic.compwritevalid);
    g_ep_collective.size = sizeof g_ep_collective;
    unsupported(g_ep_collective.barrier); unsupported(g_ep_collective.broadcast);
    unsupported(g_ep_collective.alltoall); unsupported(g_ep_collective.allreduce);
    unsupported(g_ep_collective.allgather); unsupported(g_ep_collective.reduce_scatter);
    unsupported(g_ep_collective.reduce); unsupported(g_ep_collective.scatter); unsupported(g_ep_collective.gather);
    unsupported(g_ep_collective.msg); unsupported(g_ep_collective.barrier2);
}

void cleanup() {}

fi_provider g_provider = {
    kProviderVersion,
    FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
    {},
    kName,
    getinfo,
    fabric,
    cleanup,
};

}  // namespace
}  // namespace srdsim::fi

extern "C" __attribute__((visibility("default"))) struct fi_provider* fi_prov_ini(void) {
    static std::once_flag once;
    std::call_once(once, srdsim::fi::init_tables);
    return &srdsim::fi::g_provider;
}
