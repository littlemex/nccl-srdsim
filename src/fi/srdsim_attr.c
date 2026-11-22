/* SPDX-License-Identifier: Apache-2.0 */
/*
 * The fi_info the provider starts from, and what it changes for a consumer's hints. Values follow the real
 * provider's RDM endpoint in libfabric's prov/efa/src/efa_prov_info.c and efa_user_info.c, except where a
 * comment says otherwise.
 */
#include <dirent.h>
#include <stdio.h>

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

/*
 * The NICs the provider reports: one per NVIDIA GPU on the host, each named srdsim<N>-rdm and placed at that
 * GPU's PCI address, as an EFA instance has a NIC next to each GPU. Consumers that pair GPUs with NICs by
 * their place in the PCI topology (aws-ofi-nccl does) then pair each GPU with its own emulated NIC. A host
 * without NVIDIA GPUs gets one NIC with no bus address.
 */
#define SRDSIM_MAX_NICS 64

struct srdsim_pci { unsigned domain, bus, dev, func; };

static int srdsim_pci_cmp(const void *a, const void *b)
{
	const struct srdsim_pci *x = a, *y = b;

	if (x->domain != y->domain)
		return x->domain < y->domain ? -1 : 1;
	if (x->bus != y->bus)
		return x->bus < y->bus ? -1 : 1;
	if (x->dev != y->dev)
		return x->dev < y->dev ? -1 : 1;
	return x->func < y->func ? -1 : x->func > y->func;
}

static bool srdsim_read_hex(const char *dir, const char *file, unsigned long *out)
{
	char path[512], buf[32];
	FILE *f;
	bool ok;

	snprintf(path, sizeof(path), "%s/%s", dir, file);
	f = fopen(path, "r");
	if (!f)
		return false;
	ok = fgets(buf, sizeof(buf), f) && sscanf(buf, "%lx", out) == 1;
	fclose(f);
	return ok;
}

static size_t srdsim_find_gpus(struct srdsim_pci *out, size_t max)
{
	const char *root = "/sys/bus/pci/devices";
	struct srdsim_pci p;
	struct dirent *d;
	unsigned long vendor, cls;
	char dir[512];
	size_t n = 0;
	DIR *dp = opendir(root);

	if (!dp)
		return 0;
	while ((d = readdir(dp)) && n < max) {
		if (sscanf(d->d_name, "%x:%x:%x.%x", &p.domain, &p.bus, &p.dev, &p.func) != 4)
			continue;
		snprintf(dir, sizeof(dir), "%s/%s", root, d->d_name);
		/* NVIDIA's vendor id, and the display (0x03xx) device classes GPUs report. */
		if (srdsim_read_hex(dir, "vendor", &vendor) && vendor == 0x10de &&
		    srdsim_read_hex(dir, "class", &cls) && (cls >> 16) == 0x03)
			out[n++] = p;
	}
	closedir(dp);
	qsort(out, n, sizeof(*out), srdsim_pci_cmp);
	return n;
}

static struct fi_info *srdsim_nic_info(size_t index, const struct srdsim_pci *pci)
{
	struct fi_info *info = fi_dupinfo(&srdsim_info);
	struct fid_nic *nic;
	char name[64];

	if (!info)
		return NULL;
	snprintf(name, sizeof(name), "srdsim%zu-rdm", index);
	free(info->domain_attr->name);
	info->domain_attr->name = strdup(name);
	nic = ofi_nic_dup(NULL);
	if (!info->domain_attr->name || !nic) {
		fi_freeinfo(info);
		return NULL;
	}
	/*
	 * No device name: aws-ofi-nccl reads /sys/class/infiniband/<name>/node_guid for any NIC that has one and
	 * fails when the file does not exist, which it cannot for an emulated device. Without a name it numbers
	 * the NIC itself, as it does for EFA devices 0xefa0 to 0xefa2 in any case.
	 */
	snprintf(name, sizeof(name), "srdsim%zu", index);
	nic->device_attr->vendor_id = strdup("0x1d0f");	/* Amazon, as EFA reports */
	nic->device_attr->device_id = strdup("0xefa1");
	nic->device_attr->driver = strdup("srdsim");
	nic->link_attr->address = strdup(name);
	nic->link_attr->network_type = strdup("Ethernet");
	nic->link_attr->mtu = 8192;
	nic->link_attr->speed = 100000000000ull;
	nic->link_attr->state = FI_LINK_UP;
	if (pci) {
		nic->bus_attr->bus_type = FI_BUS_PCI;
		nic->bus_attr->attr.pci.domain_id = (uint16_t) pci->domain;
		nic->bus_attr->attr.pci.bus_id = (uint8_t) pci->bus;
		nic->bus_attr->attr.pci.device_id = (uint8_t) pci->dev;
		nic->bus_attr->attr.pci.function_id = (uint8_t) pci->func;
	}
	info->nic = nic;
	return info;
}

int srdsim_build_infos(void)
{
	struct srdsim_pci gpus[SRDSIM_MAX_NICS];
	struct fi_info *head = NULL, **tail = &head, *info;
	size_t n = srdsim_find_gpus(gpus, SRDSIM_MAX_NICS), i;

	for (i = 0; i < (n ? n : 1); i++) {
		info = srdsim_nic_info(i, n ? &gpus[i] : NULL);
		if (!info) {
			fi_freeinfo(head);
			return -FI_ENOMEM;
		}
		*tail = info;
		tail = &info->next;
	}
	srdsim_util_prov.info = head;
	return 0;
}

void srdsim_free_infos(void)
{
	if (srdsim_util_prov.info != &srdsim_info)
		fi_freeinfo(srdsim_util_prov.info);
	srdsim_util_prov.info = &srdsim_info;
}
