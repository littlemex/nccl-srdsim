/* SPDX-License-Identifier: Apache-2.0 */
/*
 * A libfabric provider that registers under the name "efa" and gives consumers that select EFA by that name
 * (aws-ofi-nccl, NIXL's libfabric backend) an RDM endpoint with SRD's delivery rules, between processes on
 * one host. It is built as a DSO inside a libfabric source tree (scripts/build_fi_provider.sh), so it uses
 * libfabric's own utility layer for hint matching, objects and memory registration, and it is selected only
 * by naming it in FI_PROVIDER_PATH in the preferred form "+/path/libsrdsim-fi.so".
 */
#ifndef SRDSIM_FI_H
#define SRDSIM_FI_H

#include "config.h"

#include <stdbool.h>
#include <stdint.h>

#include <rdma/fabric.h>
#include <rdma/fi_ext.h>

#include "ofi.h"
#include "ofi_enosys.h"
#include "ofi_prov.h"
#include "ofi_util.h"

#define SRDSIM_PROV_NAME "efa"
#define SRDSIM_FABRIC_NAME "efa"
#define SRDSIM_DIRECT_FABRIC_NAME "efa-direct"
/* The real provider names its RDM domain "<device>-rdm". */
#define SRDSIM_DOMAIN_NAME "srdsim0-rdm"
/* As EFA_RDM_IOV_LIMIT. */
#define SRDSIM_IOV_MAX 4

extern struct fi_provider srdsim_prov;
extern struct util_prov srdsim_util_prov;
extern struct fi_info srdsim_info;
extern struct fi_info srdsim_direct_info;
bool srdsim_info_is_direct(const struct fi_info *info);

#define SRDSIM_WARN(subsys, ...) FI_WARN(&srdsim_prov, subsys, __VA_ARGS__)
#define SRDSIM_INFO(subsys, ...) FI_INFO(&srdsim_prov, subsys, __VA_ARGS__)

/*
 * The address an endpoint reports from fi_getname and a peer inserts into its address vector: 32 bytes, the
 * size of the real provider's raw address, so buffers a consumer sizes for EFA hold it. The endpoint listens on
 * the loopback interface at `port`.
 */
struct srdsim_addr {
	char magic[8];		/* SRDSIM_ADDR_MAGIC */
	uint32_t version;	/* SRDSIM_ADDR_VERSION */
	uint16_t port;		/* network byte order */
	uint16_t reserved;
	uint64_t nonce;		/* unique per endpoint */
	uint32_t ipv4;		/* network byte order */
	uint32_t reserved2;
};
#define SRDSIM_ADDR_MAGIC "SRDSIMFI"
#define SRDSIM_ADDR_VERSION 1

struct srdsim_xfer;	/* the transfer engine of one endpoint, srdsim_xfer.c */

struct srdsim_ep {
	struct util_ep util_ep;
	struct srdsim_addr addr;
	struct srdsim_xfer *xfer;
	bool enabled;
	size_t max_msg_size;	/* from the fi_info it was opened with */
	size_t inject_size;
	size_t tx_size;
	/* The EFA endpoint options, as the consumer set them. */
	size_t rnr_retry;
	size_t min_multi_recv;
	int hmem_p2p;
	bool cuda_api_permitted;
	bool shared_memory_permitted;
	bool use_device_rdma;
	bool write_in_order_128;
	bool homogeneous_peers;
	bool use_unsolicited_write_recv;
};

/* srdsim_xfer.c */
int srdsim_xfer_open(struct srdsim_ep *ep);
int srdsim_xfer_start(struct srdsim_ep *ep);
void srdsim_xfer_close(struct srdsim_ep *ep);
void srdsim_xfer_progress(struct util_ep *util_ep);
extern struct fi_ops_msg srdsim_msg_ops;
extern struct fi_ops_tagged srdsim_tagged_ops;
extern struct fi_ops_rma srdsim_rma_ops;

int srdsim_build_infos(void);
void srdsim_free_infos(void);
int srdsim_getinfo_alter(uint32_t version, const struct fi_info *hints, struct fi_info *info);
int srdsim_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric, void *context);
int srdsim_domain_open(struct fid_fabric *fabric, struct fi_info *info, struct fid_domain **domain,
		       void *context);
int srdsim_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr, struct fid_cq **cq, void *context);
int srdsim_av_open(struct fid_domain *domain, struct fi_av_attr *attr, struct fid_av **av, void *context);
int srdsim_endpoint(struct fid_domain *domain, struct fi_info *info, struct fid_ep **ep, void *context);

#endif
