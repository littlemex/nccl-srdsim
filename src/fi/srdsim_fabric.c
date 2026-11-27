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

/*
 * Whether the domain's utility objects lock whatever the consumer asked for. 1, the default, is the only setting
 * for ordinary use; 0 leaves libfabric's choice alone and so leaves the completion ring unguarded under
 * FI_THREAD_COMPLETION and FI_THREAD_DOMAIN. It exists for one reason: a promise with no way to break it is not
 * tested, and tests/fi/test_fi_cq_threads.cc is run both ways so that the passing run means something. Read once
 * per domain.
 */
static bool srdsim_guard_cq(void)
{
	const char *v = getenv("SRDSIM_FI_GUARD_CQ");

	if (!v || !*v || !strcmp(v, "1"))
		return true;
	if (!strcmp(v, "0"))
		return false;
	SRDSIM_WARN(FI_LOG_DOMAIN, "SRDSIM_FI_GUARD_CQ=%s is neither 0 nor 1; using 1\n", v);
	return true;
}

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

	/*
	 * Every CQ of the domain keeps completion sources, as the efa provider's CQs do: a consumer may open the
	 * domain with an info without FI_SOURCE and an endpoint on it with one, as aws-ofi-nccl's GIN does, and
	 * then reads sources with fi_cq_readfrom.
	 */
	domain->info_domain_caps |= FI_SOURCE;

	/*
	 * The objects of this domain lock whatever the consumer asked for. libfabric's utility layer reads this
	 * one field to choose their locks, and asking for FI_THREAD_COMPLETION or FI_THREAD_DOMAIN disarms two of
	 * them in this provider's configuration: the completion ring (ofi_cq_init picks OFI_LOCK_NOOP for both
	 * models) and the endpoint's own lock (ofi_endpoint_init picks it for anything other than FI_THREAD_SAFE).
	 * The endpoint-list and address-vector locks are not affected here -- those turn into no-ops only when
	 * control progress is FI_PROGRESS_CONTROL_UNIFIED, and this provider reports manual control progress.
	 *
	 * In those models a consumer owes libfabric its own serialisation and the stock providers leave it at
	 * that, the real efa provider included: its CQ is a util_cq too, so it gets the same no-op. None of that
	 * is a defect in libfabric.
	 *
	 * It would be one here. This provider exists so that someone can find a bug in the code above it, and a
	 * consumer hunting its own threading mistake must not first have to rule out a corrupted ring inside the
	 * tool it is looking through. tests/fi/test_fi_cq_threads.cc is that promise written as a test.
	 *
	 * The cost is stated in the warning below and in docs/libfabric-provider.md: a consumer that forgets the
	 * serialisation it owes works here and breaks on the real device.
	 *
	 * Nothing the consumer can observe changes: this is util_domain's own copy, used for picking locks and
	 * nothing else (util_domain_init copies every field it wants out of `info` and keeps no pointer into it),
	 * and the fi_info the consumer holds still says what it asked for. The domain's own lock is already a
	 * real mutex -- the argument above.
	 */
	if (info->domain_attr->threading != FI_THREAD_SAFE && srdsim_guard_cq()) {
		SRDSIM_WARN(FI_LOG_DOMAIN,
			    "this domain asked for threading %d; its completion queues and endpoints are locked "
			    "anyway, so a missing serialisation of your own will not show here -- it will on a "
			    "real EFA device. SRDSIM_FI_GUARD_CQ=0 leaves libfabric's choice alone\n",
			    (int) info->domain_attr->threading);
		domain->threading = FI_THREAD_SAFE;
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

	ret = ofi_fabric_init(&srdsim_prov,
			      attr && attr->name && !strcmp(attr->name, SRDSIM_DIRECT_FABRIC_NAME) ?
			      srdsim_direct_info.fabric_attr : srdsim_info.fabric_attr,
			      attr, fabric, context);
	if (ret) {
		free(fabric);
		return ret;
	}

	*fabric_fid = &fabric->fabric_fid;
	(*fabric_fid)->fid.ops = &srdsim_fabric_fi_ops;
	(*fabric_fid)->ops = &srdsim_fabric_ops;
	return 0;
}
