/* SPDX-License-Identifier: Apache-2.0 */
/*
 * The fi_info the provider starts from, and what it changes for a consumer's hints. Values follow the real
 * provider's RDM endpoint in libfabric's prov/efa/src/efa_prov_info.c and efa_user_info.c, except where a
 * comment says otherwise.
 */
#include "srdsim.h"

/* Queue depths and the inject size are not those of any one device; they bound nothing yet. */
#define SRDSIM_QUEUE_SIZE 4096
#define SRDSIM_INJECT_SIZE 4096

/*
 * Atomics are not offered: the real provider emulates them in software, and nothing this provider exists
 * for uses them. FI_MULTI_RECV is not offered either.
 */
#define SRDSIM_TX_CAPS (OFI_TX_MSG_CAPS | FI_TAGGED | OFI_TX_RMA_CAPS | FI_HMEM)
#define SRDSIM_RX_CAPS (OFI_RX_MSG_CAPS | FI_TAGGED | OFI_RX_RMA_CAPS | FI_SOURCE | FI_DIRECTED_RECV | FI_HMEM)
#define SRDSIM_DOMAIN_CAPS (FI_LOCAL_COMM | FI_REMOTE_COMM)

/*
 * Two-sided messages keep their order, which the real provider restores in software. RMA has no ordering,
 * which is what SRD gives; hints that ask for one are refused by the generic check.
 */
#define SRDSIM_MSG_ORDER FI_ORDER_SAS

static struct fi_tx_attr srdsim_tx_attr = {
	.caps = SRDSIM_TX_CAPS,
	.op_flags = FI_INJECT | FI_COMPLETION | FI_TRANSMIT_COMPLETE | FI_DELIVERY_COMPLETE,
	.msg_order = SRDSIM_MSG_ORDER,
	.inject_size = SRDSIM_INJECT_SIZE,
	.size = SRDSIM_QUEUE_SIZE,
	.iov_limit = SRDSIM_IOV_MAX,
	.rma_iov_limit = 1,
};

static struct fi_rx_attr srdsim_rx_attr = {
	.caps = SRDSIM_RX_CAPS,
	.op_flags = FI_COMPLETION,
	.msg_order = SRDSIM_MSG_ORDER,
	.size = SRDSIM_QUEUE_SIZE,
	.iov_limit = SRDSIM_IOV_MAX,
};

static struct fi_ep_attr srdsim_ep_attr = {
	.type = FI_EP_RDM,
	.protocol = FI_PROTO_EFA,
	.protocol_version = 4,		/* EFA_RDM_PROTOCOL_VERSION */
	.max_msg_size = SIZE_MAX,
	.mem_tag_format = FI_TAG_GENERIC,
	.tx_ctx_cnt = 1,
	.rx_ctx_cnt = 1,
};

static struct fi_domain_attr srdsim_domain_attr = {
	.name = SRDSIM_DOMAIN_NAME,
	.caps = SRDSIM_DOMAIN_CAPS,
	.threading = FI_THREAD_SAFE,
	.control_progress = FI_PROGRESS_MANUAL,
	.progress = FI_PROGRESS_MANUAL,
	.resource_mgmt = FI_RM_ENABLED,
	.av_type = FI_AV_UNSPEC,	/* any type is accepted; FI_AV_TABLE is what is opened */
	.mr_mode = FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_HMEM,
	.mr_key_size = 4,		/* an ibv lkey */
	.cq_data_size = 4,		/* EFA_CQ_DATA_SIZE */
	.cq_cnt = 1024,
	.ep_cnt = 1024,
	.tx_ctx_cnt = 1024,
	.rx_ctx_cnt = 1024,
	.max_ep_tx_ctx = 1,
	.max_ep_rx_ctx = 1,
	.mr_iov_limit = 1,		/* EFA_MR_IOV_LIMIT */
};

static struct fi_fabric_attr srdsim_fabric_attr = {
	.name = SRDSIM_FABRIC_NAME,
	.prov_version = OFI_VERSION_DEF_PROV,
};

struct fi_info srdsim_info = {
	.caps = SRDSIM_TX_CAPS | SRDSIM_RX_CAPS | SRDSIM_DOMAIN_CAPS,
	.addr_format = FI_ADDR_EFA,
	.tx_attr = &srdsim_tx_attr,
	.rx_attr = &srdsim_rx_attr,
	.ep_attr = &srdsim_ep_attr,
	.domain_attr = &srdsim_domain_attr,
	.fabric_attr = &srdsim_fabric_attr,
};

struct util_prov srdsim_util_prov = {
	.prov = &srdsim_prov,
	.info = &srdsim_info,
	.flags = 0,
};

/*
 * What the real provider changes after the generic check, in efa_user_info_alter_rdm(): FI_HMEM only when
 * asked for (and then FI_MR_HMEM is required), the message order narrowed to what the hints ask for, and
 * FI_MR_LOCAL echoed when the consumer offers it, so that its registrations are used.
 */
int srdsim_getinfo_alter(uint32_t version, const struct fi_info *hints, struct fi_info *info)
{
	OFI_UNUSED(version);

	if (hints && !(hints->caps & FI_HMEM)) {
		info->caps &= ~FI_HMEM;
		info->tx_attr->caps &= ~FI_HMEM;
		info->rx_attr->caps &= ~FI_HMEM;
	}
	if (info->caps & FI_HMEM) {
		if (hints && hints->domain_attr && !(hints->domain_attr->mr_mode & FI_MR_HMEM)) {
			SRDSIM_INFO(FI_LOG_CORE, "FI_HMEM requires FI_MR_HMEM\n");
			return -FI_ENODATA;
		}
		info->domain_attr->mr_mode |= FI_MR_HMEM;
	} else {
		info->domain_attr->mr_mode &= ~FI_MR_HMEM;
	}

	if (hints) {
		if (hints->tx_attr)
			info->tx_attr->msg_order &= hints->tx_attr->msg_order;
		if (hints->rx_attr)
			info->rx_attr->msg_order &= hints->rx_attr->msg_order;
		if (hints->domain_attr && (hints->domain_attr->mr_mode & FI_MR_LOCAL))
			info->domain_attr->mr_mode |= FI_MR_LOCAL;
	}
	info->domain_attr->av_type = FI_AV_TABLE;
	if (!hints || !hints->domain_attr || hints->domain_attr->resource_mgmt == FI_RM_UNSPEC)
		info->domain_attr->resource_mgmt = FI_RM_ENABLED;
	return 0;
}
