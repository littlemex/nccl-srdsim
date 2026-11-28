/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Device attributes the stand-in reports.
 *
 * Values marked "observed" were read with ibv_devinfo on an EFA device
 * (vendor part 0xefa1). Values marked "assumed" have no hardware record yet
 * and are to be replaced from a recorded hardware profile (#81).
 * SRDSIM_EFA_DEVICE_CAPS overrides SD_EFA_DEVICE_CAPS at run time.
 */
#ifndef SRDSIM_DEVICE_PROFILE_H
#define SRDSIM_DEVICE_PROFILE_H

#define SD_VENDOR_ID		0x1d0f		/* observed */
#define SD_VENDOR_PART_ID	0xefa1		/* observed */
#define SD_HW_VER		0xefa1		/* observed */
#define SD_MAX_MR_SIZE		0x1800000000ULL	/* observed */
#define SD_PAGE_SIZE_CAP	0xfffff000ULL	/* observed */
#define SD_MAX_QP		256		/* observed */
#define SD_MAX_QP_WR		4096		/* observed */
#define SD_MAX_SGE		2		/* observed */
#define SD_MAX_SGE_RD		1		/* observed */
#define SD_MAX_CQ		512		/* observed */
#define SD_MAX_CQE		32768		/* observed */
#define SD_MAX_MR		262144		/* observed */
#define SD_MAX_PD		256		/* observed */
#define SD_MAX_AH		8192		/* observed */
#define SD_NUM_COMP_VECTORS	8		/* observed */
#define SD_VERBS_ABI_VERSION	"6"		/* assumed */
#define SD_PORT_MAX_MSG_SZ	0x22e0		/* observed */

#define SD_EFA_MAX_SQ_WR	4096		/* assumed */
#define SD_EFA_MAX_RQ_WR	32768		/* assumed */
#define SD_EFA_MAX_SQ_SGE	2		/* assumed */
#define SD_EFA_MAX_RQ_SGE	3		/* assumed */
#define SD_EFA_INLINE_BUF_SIZE	32		/* assumed */
#define SD_EFA_MAX_RDMA_SIZE	1073741824u	/* assumed */
/*
 * Capability bits, by value so that older efadv headers that lack the names
 * still build: RDMA_READ (1 << 0), RNR_RETRY (1 << 1), CQ_WITH_SGID (1 << 2),
 * RDMA_WRITE (1 << 3), UNSOLICITED_WRITE_RECV (1 << 4).
 */
#define SD_EFA_DEVICE_CAPS	0x1fu		/* assumed */

#endif
