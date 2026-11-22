/* SPDX-License-Identifier: Apache-2.0 */
/*
 * The RDM endpoint. Its options are answered as the real RDM endpoint answers them, in libfabric's
 * prov/efa/src/rdm/efa_rdm_ep_fiops.c. Transfers are not carried yet: every data call returns -FI_ENOSYS.
 */
#include <sys/random.h>

#include "srdsim.h"

/*
 * SRDSIM_FI_WRITE_IN_ORDER: 1 (the default) accepts FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES, as an EFA
 * device that supports it does; 0 refuses it with -FI_EOPNOTSUPP, as one that does not. Read each time the
 * option is set.
 */
static bool srdsim_write_in_order_supported(void)
{
	const char *v = getenv("SRDSIM_FI_WRITE_IN_ORDER");

	if (!v || !*v || !strcmp(v, "1"))
		return true;
	if (!strcmp(v, "0"))
		return false;
	SRDSIM_WARN(FI_LOG_EP_CTRL, "SRDSIM_FI_WRITE_IN_ORDER=%s is neither 0 nor 1; using 1\n", v);
	return true;
}

static void srdsim_ep_progress(struct util_ep *util_ep)
{
	OFI_UNUSED(util_ep);
}

static int srdsim_ep_close(struct fid *fid)
{
	struct srdsim_ep *ep = container_of(fid, struct srdsim_ep, util_ep.ep_fid.fid);
	int ret = ofi_endpoint_close(&ep->util_ep);

	if (ret)
		return ret;
	free(ep);
	return 0;
}

static int srdsim_ep_control(struct fid *fid, int command, void *arg)
{
	struct srdsim_ep *ep = container_of(fid, struct srdsim_ep, util_ep.ep_fid.fid);

	OFI_UNUSED(arg);
	if (command != FI_ENABLE)
		return -FI_ENOSYS;
	/* As the real RDM endpoint: an address vector and both completion queues before it is enabled. */
	if (!ep->util_ep.av)
		return -FI_ENOAV;
	if (!ep->util_ep.tx_cq || !ep->util_ep.rx_cq)
		return -FI_ENOCQ;
	ep->enabled = true;
	return 0;
}

static int srdsim_ep_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct srdsim_ep *ep = container_of(fid, struct srdsim_ep, util_ep.ep_fid.fid);

	if (ep->enabled)
		return -FI_EOPBADSTATE;
	return ofi_ep_bind(&ep->util_ep, bfid, flags);
}

#define SRDSIM_SET(field)                                                                                          \
	do {                                                                                                       \
		if (optlen != sizeof(ep->field))                                                                    \
			return -FI_EINVAL;                                                                         \
		memcpy(&ep->field, optval, sizeof(ep->field));                                                     \
		return 0;                                                                                          \
	} while (0)

static int srdsim_ep_setopt(fid_t fid, int level, int optname, const void *optval, size_t optlen)
{
	struct srdsim_ep *ep = container_of(fid, struct srdsim_ep, util_ep.ep_fid.fid);
	bool v;

	if (level != FI_OPT_ENDPOINT)
		return -FI_ENOPROTOOPT;

	switch (optname) {
	case FI_OPT_MIN_MULTI_RECV:
		SRDSIM_SET(min_multi_recv);
	case FI_OPT_EFA_RNR_RETRY:
		if (optlen != sizeof(size_t))
			return -FI_EINVAL;
		if (ep->enabled)
			return -FI_EINVAL;	/* to be set before the endpoint is enabled */
		memcpy(&ep->rnr_retry, optval, sizeof(size_t));
		return 0;
	case FI_OPT_FI_HMEM_P2P:
		SRDSIM_SET(hmem_p2p);
	case FI_OPT_CUDA_API_PERMITTED:
		SRDSIM_SET(cuda_api_permitted);
	case FI_OPT_SHARED_MEMORY_PERMITTED:
		SRDSIM_SET(shared_memory_permitted);
	case FI_OPT_EFA_USE_DEVICE_RDMA:
		SRDSIM_SET(use_device_rdma);
	case FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES:
		if (optlen != sizeof(bool))
			return -FI_EINVAL;
		return -FI_EOPNOTSUPP;	/* the real RDM endpoint refuses it, whatever the value */
	case FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES:
		if (optlen != sizeof(bool))
			return -FI_EINVAL;
		memcpy(&v, optval, sizeof(v));
		if (v && !srdsim_write_in_order_supported())
			return -FI_EOPNOTSUPP;
		ep->write_in_order_128 = v;
		return 0;
	case FI_OPT_EFA_HOMOGENEOUS_PEERS:
		SRDSIM_SET(homogeneous_peers);
	case FI_OPT_EFA_USE_UNSOLICITED_WRITE_RECV:
		SRDSIM_SET(use_unsolicited_write_recv);
	default:
		return -FI_ENOPROTOOPT;
	}
}

static int srdsim_get(void *optval, size_t *optlen, const void *value, size_t size)
{
	if (*optlen < size) {
		*optlen = size;
		return -FI_ETOOSMALL;
	}
	memcpy(optval, value, size);
	*optlen = size;
	return 0;
}

static int srdsim_ep_getopt(fid_t fid, int level, int optname, void *optval, size_t *optlen)
{
	struct srdsim_ep *ep = container_of(fid, struct srdsim_ep, util_ep.ep_fid.fid);
	size_t size;
	bool emulated, yes = true;

	if (level != FI_OPT_ENDPOINT)
		return -FI_ENOPROTOOPT;

	switch (optname) {
	case FI_OPT_MIN_MULTI_RECV:
		return srdsim_get(optval, optlen, &ep->min_multi_recv, sizeof(size_t));
	case FI_OPT_EFA_RNR_RETRY:
		return srdsim_get(optval, optlen, &ep->rnr_retry, sizeof(size_t));
	case FI_OPT_FI_HMEM_P2P:
		return srdsim_get(optval, optlen, &ep->hmem_p2p, sizeof(int));
	case FI_OPT_MAX_MSG_SIZE:
	case FI_OPT_MAX_TAGGED_SIZE:
	case FI_OPT_MAX_RMA_SIZE:
		size = ep->max_msg_size;
		return srdsim_get(optval, optlen, &size, sizeof(size));
	case FI_OPT_INJECT_MSG_SIZE:
	case FI_OPT_INJECT_TAGGED_SIZE:
	case FI_OPT_INJECT_RMA_SIZE:
		size = ep->inject_size;
		return srdsim_get(optval, optlen, &size, sizeof(size));
	/*
	 * Reads and writes are carried as such unless the consumer turned device RDMA off; atomics, which are not
	 * offered, read as emulated, as on the real endpoint.
	 */
	case FI_OPT_EFA_EMULATED_READ:
	case FI_OPT_EFA_EMULATED_WRITE:
		emulated = !ep->use_device_rdma;
		return srdsim_get(optval, optlen, &emulated, sizeof(emulated));
	case FI_OPT_EFA_EMULATED_ATOMICS:
		return srdsim_get(optval, optlen, &yes, sizeof(yes));
	default:
		return -FI_ENOPROTOOPT;
	}
}

static int srdsim_ep_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct srdsim_ep *ep = container_of(fid, struct srdsim_ep, util_ep.ep_fid.fid);

	return srdsim_get(addr, addrlen, &ep->addr, sizeof(ep->addr));
}

static struct fi_ops srdsim_ep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = srdsim_ep_close,
	.bind = srdsim_ep_bind,
	.control = srdsim_ep_control,
	.ops_open = fi_no_ops_open,
	.tostr = fi_no_tostr,
	.ops_set = fi_no_ops_set,
};

static struct fi_ops_ep srdsim_ep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = fi_no_cancel,
	.getopt = srdsim_ep_getopt,
	.setopt = srdsim_ep_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static struct fi_ops_cm srdsim_ep_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = fi_no_setname,
	.getname = srdsim_ep_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_no_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
	.join = fi_no_join,
};

static struct fi_ops_msg srdsim_ep_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = fi_no_msg_recv,
	.recvv = fi_no_msg_recvv,
	.recvmsg = fi_no_msg_recvmsg,
	.send = fi_no_msg_send,
	.sendv = fi_no_msg_sendv,
	.sendmsg = fi_no_msg_sendmsg,
	.inject = fi_no_msg_inject,
	.senddata = fi_no_msg_senddata,
	.injectdata = fi_no_msg_injectdata,
};

static struct fi_ops_rma srdsim_ep_rma_ops = {
	.size = sizeof(struct fi_ops_rma),
	.read = fi_no_rma_read,
	.readv = fi_no_rma_readv,
	.readmsg = fi_no_rma_readmsg,
	.write = fi_no_rma_write,
	.writev = fi_no_rma_writev,
	.writemsg = fi_no_rma_writemsg,
	.inject = fi_no_rma_inject,
	.writedata = fi_no_rma_writedata,
	.injectdata = fi_no_rma_injectdata,
};

static struct fi_ops_tagged srdsim_ep_tagged_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = fi_no_tagged_recv,
	.recvv = fi_no_tagged_recvv,
	.recvmsg = fi_no_tagged_recvmsg,
	.send = fi_no_tagged_send,
	.sendv = fi_no_tagged_sendv,
	.sendmsg = fi_no_tagged_sendmsg,
	.inject = fi_no_tagged_inject,
	.senddata = fi_no_tagged_senddata,
	.injectdata = fi_no_tagged_injectdata,
};

static struct fi_ops_atomic srdsim_ep_atomic_ops = {
	.size = sizeof(struct fi_ops_atomic),
	.write = fi_no_atomic_write,
	.writev = fi_no_atomic_writev,
	.writemsg = fi_no_atomic_writemsg,
	.inject = fi_no_atomic_inject,
	.readwrite = fi_no_atomic_readwrite,
	.readwritev = fi_no_atomic_readwritev,
	.readwritemsg = fi_no_atomic_readwritemsg,
	.compwrite = fi_no_atomic_compwrite,
	.compwritev = fi_no_atomic_compwritev,
	.compwritemsg = fi_no_atomic_compwritemsg,
	.writevalid = fi_no_atomic_writevalid,
	.readwritevalid = fi_no_atomic_readwritevalid,
	.compwritevalid = fi_no_atomic_compwritevalid,
};

int srdsim_endpoint(struct fid_domain *domain, struct fi_info *info, struct fid_ep **ep_fid, void *context)
{
	struct util_domain *util_domain = container_of(domain, struct util_domain, domain_fid);
	struct srdsim_ep *ep;
	int ret;

	ret = ofi_prov_check_info(&srdsim_util_prov, util_domain->fabric->fabric_fid.api_version, info);
	if (ret)
		return ret;

	ep = calloc(1, sizeof(*ep));
	if (!ep)
		return -FI_ENOMEM;

	ret = ofi_endpoint_init(domain, &srdsim_util_prov, info, &ep->util_ep, context, srdsim_ep_progress);
	if (ret) {
		free(ep);
		return ret;
	}

	memcpy(ep->addr.magic, SRDSIM_ADDR_MAGIC, sizeof(ep->addr.magic));
	ep->addr.version = SRDSIM_ADDR_VERSION;
	if (getrandom(&ep->addr.nonce, sizeof(ep->addr.nonce), 0) != sizeof(ep->addr.nonce)) {
		ofi_endpoint_close(&ep->util_ep);
		free(ep);
		return -FI_EOTHER;
	}
	ep->max_msg_size = info->ep_attr->max_msg_size;
	ep->inject_size = info->tx_attr->inject_size;
	/* The real endpoint's defaults. */
	ep->rnr_retry = 3;	/* EFA_RDM_DEFAULT_RNR_RETRY */
	ep->hmem_p2p = FI_HMEM_P2P_ENABLED;
	ep->cuda_api_permitted = true;
	ep->shared_memory_permitted = true;
	ep->use_device_rdma = true;
	ep->use_unsolicited_write_recv = true;

	*ep_fid = &ep->util_ep.ep_fid;
	(*ep_fid)->fid.ops = &srdsim_ep_fi_ops;
	(*ep_fid)->ops = &srdsim_ep_ops;
	(*ep_fid)->cm = &srdsim_ep_cm_ops;
	(*ep_fid)->msg = &srdsim_ep_msg_ops;
	(*ep_fid)->rma = &srdsim_ep_rma_ops;
	(*ep_fid)->tagged = &srdsim_ep_tagged_ops;
	(*ep_fid)->atomic = &srdsim_ep_atomic_ops;
	return 0;
}
