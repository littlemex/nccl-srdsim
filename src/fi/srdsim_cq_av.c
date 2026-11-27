/* SPDX-License-Identifier: Apache-2.0 */
/* The completion queue and the address vector: libfabric's utility objects. */
#include "srdsim.h"

static int srdsim_cq_close(struct fid *fid)
{
	struct util_cq *cq = container_of(fid, struct util_cq, cq_fid.fid);
	int ret = ofi_cq_cleanup(cq);

	if (ret)
		return ret;
	free(cq);
	return 0;
}

static struct fi_ops srdsim_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = srdsim_cq_close,
	.bind = fi_no_bind,
	.control = ofi_cq_control,
	.ops_open = fi_no_ops_open,
	.tostr = fi_no_tostr,
	.ops_set = fi_no_ops_set,
};

int srdsim_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr, struct fid_cq **cq_fid, void *context)
{
	struct util_cq *cq;
	int ret;

	cq = calloc(1, sizeof(*cq));
	if (!cq)
		return -FI_ENOMEM;

	/* Reading the CQ progresses every endpoint bound to it, as the manual progress model requires. */
	ret = ofi_cq_init(&srdsim_prov, domain, attr, cq, &ofi_cq_progress, context);
	if (ret) {
		free(cq);
		return ret;
	}

	*cq_fid = &cq->cq_fid;
	(*cq_fid)->fid.ops = &srdsim_cq_fi_ops;
	return 0;
}

static bool srdsim_addr_valid(const struct srdsim_addr *a)
{
	return memcmp(a->magic, SRDSIM_ADDR_MAGIC, sizeof(a->magic)) == 0 && a->version == SRDSIM_ADDR_VERSION;
}

static int srdsim_av_insert(struct fid_av *av_fid, const void *addr, size_t count, fi_addr_t *fi_addr,
			    uint64_t flags, void *context)
{
	struct util_av *av = container_of(av_fid, struct util_av, av_fid);
	struct srdsim_addr a;
	size_t i;
	int inserted = 0, ret;

	OFI_UNUSED(context);
	if (flags)
		return -FI_EBADFLAGS;

	ofi_genlock_lock(&av->lock);
	for (i = 0; i < count; i++) {
		/* The caller's buffer need not be aligned for the struct. */
		memcpy(&a, (const char *) addr + i * sizeof(a), sizeof(a));
		if (!srdsim_addr_valid(&a)) {
			SRDSIM_INFO(FI_LOG_AV, "address %zu is not one of this provider's\n", i);
			if (fi_addr)
				fi_addr[i] = FI_ADDR_NOTAVAIL;
			continue;
		}
		ret = ofi_av_insert_addr(av, &a, fi_addr ? &fi_addr[i] : NULL);
		if (ret == 0)
			inserted++;
		else if (fi_addr)
			fi_addr[i] = FI_ADDR_NOTAVAIL;
	}
	ofi_genlock_unlock(&av->lock);
	return inserted;
}

static int srdsim_av_remove(struct fid_av *av_fid, fi_addr_t *fi_addr, size_t count, uint64_t flags)
{
	struct util_av *av = container_of(av_fid, struct util_av, av_fid);
	size_t i;
	int ret = 0;

	if (flags)
		return -FI_EBADFLAGS;

	ofi_genlock_lock(&av->lock);
	for (i = 0; i < count && ret == 0; i++)
		ret = ofi_bufpool_ibuf_is_valid(av->av_entry_pool, fi_addr[i]) ?
		      ofi_av_remove_addr(av, fi_addr[i]) : -FI_EINVAL;
	ofi_genlock_unlock(&av->lock);
	return ret;
}

static int srdsim_av_lookup(struct fid_av *av_fid, fi_addr_t fi_addr, void *addr, size_t *addrlen)
{
	struct util_av *av = container_of(av_fid, struct util_av, av_fid);
	void *entry;
	int ret = 0;

	ofi_genlock_lock(&av->lock);
	if (!ofi_bufpool_ibuf_is_valid(av->av_entry_pool, fi_addr)) {
		ret = -FI_EINVAL;
	} else {
		entry = ofi_av_get_addr(av, fi_addr);
		memcpy(addr, entry, MIN(*addrlen, av->addrlen));
		*addrlen = av->addrlen;
	}
	ofi_genlock_unlock(&av->lock);
	return ret;
}

static const char *srdsim_av_straddr(struct fid_av *av, const void *addr, char *buf, size_t *len)
{
	struct srdsim_addr a;

	OFI_UNUSED(av);
	memcpy(&a, addr, sizeof(a));
	*len = (size_t) snprintf(buf, *len, "srdsim://%016" PRIx64, a.nonce) + 1;
	return buf;
}

static int srdsim_av_close(struct fid *fid)
{
	struct util_av *av = container_of(fid, struct util_av, av_fid.fid);
	int ret = ofi_av_close(av);

	if (ret)
		return ret;
	free(av);
	return 0;
}

static struct fi_ops srdsim_av_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = srdsim_av_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
	.tostr = fi_no_tostr,
	.ops_set = fi_no_ops_set,
};

static struct fi_ops_av srdsim_av_ops = {
	.size = sizeof(struct fi_ops_av),
	.insert = srdsim_av_insert,
	.insertsvc = fi_no_av_insertsvc,
	.insertsym = fi_no_av_insertsym,
	.remove = srdsim_av_remove,
	.lookup = srdsim_av_lookup,
	.straddr = srdsim_av_straddr,
	.lookup2 = ofi_av_lookup2,
};

int srdsim_av_open(struct fid_domain *domain_fid, struct fi_av_attr *attr, struct fid_av **av_fid,
		   void *context)
{
	struct util_domain *domain = container_of(domain_fid, struct util_domain, domain_fid);
	struct util_av_attr util_attr = {
		.addrlen = sizeof(struct srdsim_addr),
		.context_len = 0,
		.flags = 0,
	};
	struct util_av *av;
	int ret;

	struct fi_av_attr a = *attr;

	if (attr->name)
		return -FI_ENOSYS;	/* named, shared address vectors */
	/* As the efa provider: an unspecified type opens a table. */
	if (a.type == FI_AV_UNSPEC)
		a.type = FI_AV_TABLE;

	av = calloc(1, sizeof(*av));
	if (!av)
		return -FI_ENOMEM;

	ret = ofi_av_init(domain, &a, &util_attr, av, context);
	if (ret) {
		free(av);
		return ret;
	}

	*av_fid = &av->av_fid;
	(*av_fid)->fid.ops = &srdsim_av_fi_ops;
	(*av_fid)->ops = &srdsim_av_ops;
	return 0;
}
