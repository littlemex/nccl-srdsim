/* SPDX-License-Identifier: Apache-2.0 */
/* The fabric and the domain: libfabric's utility objects, with its memory registration. */
#include "srdsim.h"

static int srdsim_domain_close(fid_t fid)
{
	struct util_domain *domain = container_of(fid, struct util_domain, domain_fid.fid);
	int ret = ofi_domain_close(domain);

	if (ret)
		return ret;
	free(domain);
	return 0;
}

static struct fi_ops srdsim_domain_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = srdsim_domain_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
	.tostr = fi_no_tostr,
	.ops_set = fi_no_ops_set,
};

static struct fi_ops_domain srdsim_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = srdsim_av_open,
	.cq_open = srdsim_cq_open,
	.endpoint = srdsim_endpoint,
	.scalable_ep = fi_no_scalable_ep,
	.cntr_open = fi_no_cntr_open,
	.poll_open = fi_no_poll_open,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_no_srx_context,
	.query_atomic = fi_no_query_atomic,
	.query_collective = fi_no_query_collective,
	.endpoint2 = fi_no_endpoint2,
};

static struct fi_ops_mr srdsim_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = ofi_mr_reg,
	.regv = ofi_mr_regv,
	.regattr = ofi_mr_regattr,
};

int srdsim_domain_open(struct fid_fabric *fabric, struct fi_info *info, struct fid_domain **domain_fid,
		       void *context)
{
	struct util_domain *domain;
	int ret;

	ret = ofi_prov_check_info(&srdsim_util_prov, fabric->api_version, info);
	if (ret)
		return ret;

	domain = calloc(1, sizeof(*domain));
	if (!domain)
		return -FI_ENOMEM;

	ret = ofi_domain_init(fabric, info, domain, context, OFI_LOCK_MUTEX);
	if (ret) {
		free(domain);
		return ret;
	}

	*domain_fid = &domain->domain_fid;
	(*domain_fid)->fid.ops = &srdsim_domain_fi_ops;
	(*domain_fid)->ops = &srdsim_domain_ops;
	(*domain_fid)->mr = &srdsim_mr_ops;
	return 0;
}

static int srdsim_fabric_close(fid_t fid)
{
	struct util_fabric *fabric = container_of(fid, struct util_fabric, fabric_fid.fid);
	int ret = ofi_fabric_close(fabric);

	if (ret)
		return ret;
	free(fabric);
	return 0;
}

static struct fi_ops srdsim_fabric_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = srdsim_fabric_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
	.tostr = fi_no_tostr,
	.ops_set = fi_no_ops_set,
};

static struct fi_ops_fabric srdsim_fabric_ops = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = srdsim_domain_open,
	.passive_ep = fi_no_passive_ep,
	.eq_open = ofi_eq_create,
	.wait_open = ofi_wait_fd_open,
	.trywait = ofi_trywait,
	.domain2 = fi_no_domain2,
};

int srdsim_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric_fid, void *context)
{
	struct util_fabric *fabric;
	int ret;

	fabric = calloc(1, sizeof(*fabric));
	if (!fabric)
		return -FI_ENOMEM;

	ret = ofi_fabric_init(&srdsim_prov, srdsim_info.fabric_attr, attr, fabric, context);
	if (ret) {
		free(fabric);
		return ret;
	}

	*fabric_fid = &fabric->fabric_fid;
	(*fabric_fid)->fid.ops = &srdsim_fabric_fi_ops;
	(*fabric_fid)->ops = &srdsim_fabric_ops;
	return 0;
}
