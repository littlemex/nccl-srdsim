/* SPDX-License-Identifier: Apache-2.0 */
/*
 * A stand-in for the EFA device below the libfabric efa provider.
 *
 * Built twice, as libibverbs.so.1 and libefa.so.1, exporting the symbols the
 * provider links against with the same symbol versions. Objects are laid out
 * with the real rdma-core headers, so the provider's inline verbs calls reach
 * the function pointers filled in here.
 *
 * It keeps state for the objects the provider creates (device, PD, MR, CQ,
 * QP, AH) and moves no data yet: posting work returns ENOSYS and every CQ
 * stays empty. It does not provide the rings of the direct data path, so the
 * provider takes the verbs path; efadv_query_qp_wqs and efadv_query_cq say
 * EOPNOTSUPP. Set SRDSIM_DEVICE_LOG=1 to log each call on stderr.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/eventfd.h>
#include <sys/stat.h>
#include <unistd.h>

#include <infiniband/verbs.h>
#include <infiniband/efadv.h>

#include "profile.h"

#define EXPORT __attribute__((visibility("default")))

static int log_on = -1;

static void dlog(const char *fmt, ...)
{
	va_list ap;

	if (log_on < 0) {
		const char *v = getenv("SRDSIM_DEVICE_LOG");
		log_on = v && *v && strcmp(v, "0");
	}
	if (!log_on)
		return;
	fprintf(stderr, "[srdsim-device %d] ", (int)getpid());
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	fputc('\n', stderr);
}


/* ---- a private sysfs tree ----------------------------------------------
 *
 * The provider reads the driver name, the PCI address and the verbs ABI
 * version from sysfs, below SYSFS_PATH as rdma-core does. The stand-in
 * writes a small tree of its own and, when SYSFS_PATH is not set, points
 * SYSFS_PATH at it.
 */

static int write_file(const char *path, const char *text)
{
	FILE *f = fopen(path, "w");

	if (!f)
		return -1;
	fputs(text, f);
	return fclose(f);
}

static int mkdirs(const char *path)
{
	char buf[PATH_MAX];
	size_t i, n = strlen(path);

	if (n >= sizeof(buf))
		return -1;
	memcpy(buf, path, n + 1);
	for (i = 1; i <= n; i++) {
		if (buf[i] == '/' || buf[i] == '\0') {
			char c = buf[i];

			buf[i] = '\0';
			if (mkdir(buf, 0755) && errno != EEXIST)
				return -1;
			buf[i] = c;
		}
	}
	return 0;
}

/* Short enough that every path below it fits in ibv_device.ibdev_path. */
static char sysfs_root[160];

static const char *sysfs_tree(int ndev)
{
	char p[1024], t[1024];
	const char *tmp = getenv("TMPDIR");
	int i;

	if (!*sysfs_root &&
	    snprintf(sysfs_root, sizeof(sysfs_root), "%s/srdsim-sysfs-%u",
		     tmp && *tmp ? tmp : "/tmp", (unsigned)getuid()) >= (int)sizeof(sysfs_root)) {
		*sysfs_root = '\0';
		return NULL;
	}
	snprintf(p, sizeof(p), "%s/class/infiniband_verbs", sysfs_root);
	snprintf(t, sizeof(t), "%s/bus/pci/drivers/efa", sysfs_root);
	if (mkdirs(p) || mkdirs(t))
		return NULL;
	snprintf(p, sizeof(p), "%s/class/infiniband_verbs/abi_version", sysfs_root);
	if (write_file(p, SD_VERBS_ABI_VERSION "\n"))
		return NULL;
	for (i = 0; i < ndev; i++) {
		char dev[512];

		snprintf(dev, sizeof(dev), "%s/devices/pci0000:00/0000:00:%02x.0", sysfs_root, 0x10 + i);
		if (mkdirs(dev))
			return NULL;
		snprintf(p, sizeof(p), "%s/vendor", dev);
		snprintf(t, sizeof(t), "0x%04x\n", SD_VENDOR_ID);
		write_file(p, t);
		snprintf(p, sizeof(p), "%s/device", dev);
		snprintf(t, sizeof(t), "0x%04x\n", SD_VENDOR_PART_ID);
		write_file(p, t);
		snprintf(p, sizeof(p), "%s/driver", dev);
		snprintf(t, sizeof(t), "%s/bus/pci/drivers/efa", sysfs_root);
		unlink(p);
		if (symlink(t, p))
			return NULL;
		snprintf(p, sizeof(p), "%s/class/infiniband/srdsim%d", sysfs_root, i);
		if (mkdirs(p))
			return NULL;
		snprintf(p, sizeof(p), "%s/class/infiniband/srdsim%d/device", sysfs_root, i);
		unlink(p);
		if (symlink(dev, p))
			return NULL;
	}
	if (!getenv("SYSFS_PATH")) {
		setenv("SYSFS_PATH", sysfs_root, 0);
		dlog("SYSFS_PATH=%s", sysfs_root);
	}
	return sysfs_root;
}

/* ---- device and context ---------------------------------------------- */

struct sd_device {
	struct ibv_device dev;
	int index;
};

struct sd_context {
	struct verbs_context vctx;
	struct sd_device *device;
	pthread_mutex_t lock;
	uint32_t next_qpn;
	uint16_t next_ahn;
	uint32_t next_key;
};

static struct sd_context *ctx_of(struct ibv_context *c)
{
	return (struct sd_context *)((char *)c - offsetof(struct verbs_context, context));
}

static int device_count(void)
{
	const char *v = getenv("SRDSIM_DEVICE_COUNT");
	int n = v ? atoi(v) : 1;

	return n < 0 ? 0 : n > 32 ? 32 : n;
}

EXPORT struct ibv_device **ibv_get_device_list(int *num_devices)
{
	int n = device_count(), i;
	const char *root = sysfs_tree(n);

	if (!root) {
		dlog("cannot write the sysfs tree under %s", sysfs_root);
		errno = EIO;
		return NULL;
	}
	struct ibv_device **list = calloc((size_t)n + 1, sizeof(*list));

	if (!list) {
		errno = ENOMEM;
		return NULL;
	}
	for (i = 0; i < n; i++) {
		struct sd_device *d = calloc(1, sizeof(*d));

		if (!d) {
			while (i--)
				free(list[i]);
			free(list);
			errno = ENOMEM;
			return NULL;
		}
		d->index = i;
		d->dev.node_type = IBV_NODE_UNSPECIFIED;
		d->dev.transport_type = IBV_TRANSPORT_UNSPECIFIED;
		snprintf(d->dev.name, sizeof(d->dev.name), "srdsim%d", i);
		snprintf(d->dev.dev_name, sizeof(d->dev.dev_name), "uverbs%d", i);
		snprintf(d->dev.dev_path, sizeof(d->dev.dev_path), "/nonexistent/srdsim/uverbs%d", i);
		snprintf(d->dev.ibdev_path, sizeof(d->dev.ibdev_path), "%s/class/infiniband/srdsim%d", root, i);
		list[i] = &d->dev;
	}
	if (num_devices)
		*num_devices = n;
	dlog("get_device_list -> %d", n);
	return list;
}

EXPORT void ibv_free_device_list(struct ibv_device **list)
{
	int i;

	if (!list)
		return;
	for (i = 0; list[i]; i++)
		free(list[i]);
	free(list);
}

static int sd_query_port(struct ibv_context *c, uint8_t port, struct ibv_port_attr *a, size_t len);
static struct ibv_cq_ex *sd_create_cq_ex(struct ibv_context *c, struct ibv_cq_init_attr_ex *attr);
static struct ibv_qp *sd_create_qp_ex(struct ibv_context *c, struct ibv_qp_init_attr_ex *attr);
static int sd_req_notify_cq(struct ibv_cq *cq, int solicited);
static int sd_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad);
static int sd_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad);
static int sd_poll_cq(struct ibv_cq *cq, int n, struct ibv_wc *wc);

EXPORT struct ibv_context *ibv_open_device(struct ibv_device *device)
{
	struct sd_context *sc = calloc(1, sizeof(*sc));
	struct ibv_context *c;

	if (!sc) {
		errno = ENOMEM;
		return NULL;
	}
	sc->device = (struct sd_device *)device;
	pthread_mutex_init(&sc->lock, NULL);
	sc->next_qpn = 1;
	sc->next_ahn = 1;
	sc->next_key = 0x100;
	sc->vctx.sz = sizeof(sc->vctx);
	sc->vctx.query_port = sd_query_port;
	sc->vctx.create_cq_ex = sd_create_cq_ex;
	sc->vctx.create_qp_ex = sd_create_qp_ex;
	c = &sc->vctx.context;
	c->device = device;
	c->abi_compat = __VERBS_ABI_IS_EXTENDED;
	c->cmd_fd = eventfd(0, EFD_CLOEXEC);
	c->async_fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
	c->num_comp_vectors = SD_NUM_COMP_VECTORS;
	pthread_mutex_init(&c->mutex, NULL);
	c->ops.poll_cq = sd_poll_cq;
	c->ops.req_notify_cq = sd_req_notify_cq;
	c->ops.post_send = sd_post_send;
	c->ops.post_recv = sd_post_recv;
	dlog("open_device %s -> ctx %p", device->name, (void *)c);
	return c;
}

EXPORT int ibv_close_device(struct ibv_context *c)
{
	struct sd_context *sc = ctx_of(c);

	dlog("close_device ctx %p", (void *)c);
	if (c->cmd_fd >= 0)
		close(c->cmd_fd);
	if (c->async_fd >= 0)
		close(c->async_fd);
	pthread_mutex_destroy(&sc->lock);
	free(sc);
	return 0;
}

EXPORT int ibv_query_device(struct ibv_context *c, struct ibv_device_attr *a)
{
	struct sd_context *sc = ctx_of(c);

	memset(a, 0, sizeof(*a));
	snprintf(a->fw_ver, sizeof(a->fw_ver), "0.0.0.0");
	a->node_guid = htobe64(0x02000000efa00000ULL | (uint64_t)sc->device->index);
	a->sys_image_guid = a->node_guid;
	a->max_mr_size = SD_MAX_MR_SIZE;
	a->page_size_cap = SD_PAGE_SIZE_CAP;
	a->vendor_id = SD_VENDOR_ID;
	a->vendor_part_id = SD_VENDOR_PART_ID;
	a->hw_ver = SD_HW_VER;
	a->max_qp = SD_MAX_QP;
	a->max_qp_wr = SD_MAX_QP_WR;
	a->max_sge = SD_MAX_SGE;
	a->max_sge_rd = SD_MAX_SGE_RD;
	a->max_cq = SD_MAX_CQ;
	a->max_cqe = SD_MAX_CQE;
	a->max_mr = SD_MAX_MR;
	a->max_pd = SD_MAX_PD;
	a->max_ah = SD_MAX_AH;
	a->max_pkeys = 1;
	a->atomic_cap = IBV_ATOMIC_NONE;
	a->phys_port_cnt = 1;
	return 0;
}

static int sd_query_port(struct ibv_context *c, uint8_t port, struct ibv_port_attr *a, size_t len)
{
	struct ibv_port_attr p;

	(void)c;
	if (port != 1)
		return EINVAL;
	memset(&p, 0, sizeof(p));
	p.state = IBV_PORT_ACTIVE;
	p.max_mtu = IBV_MTU_4096;
	p.active_mtu = IBV_MTU_4096;
	p.gid_tbl_len = 1;
	p.max_msg_sz = SD_PORT_MAX_MSG_SZ;
	p.pkey_tbl_len = 1;
	p.lmc = 1;
	p.max_vl_num = 1;
	p.link_layer = IBV_LINK_LAYER_UNSPECIFIED;
	memcpy(a, &p, len < sizeof(p) ? len : sizeof(p));
	return 0;
}

#undef ibv_query_port
EXPORT int ibv_query_port(struct ibv_context *c, uint8_t port, struct _compat_ibv_port_attr *a)
{
	return sd_query_port(c, port, (struct ibv_port_attr *)a, sizeof(struct ibv_port_attr));
}

EXPORT int ibv_query_gid(struct ibv_context *c, uint8_t port, int index, union ibv_gid *gid)
{
	struct sd_context *sc = ctx_of(c);

	if (port != 1 || index != 0)
		return -1;
	memset(gid, 0, sizeof(*gid));
	gid->raw[0] = 0xfe;
	gid->raw[1] = 0x80;
	gid->raw[8] = 0x02;
	gid->raw[13] = 0xef;
	gid->raw[14] = 0xa0;
	gid->raw[15] = (uint8_t)sc->device->index;
	return 0;
}

static int fork_initialized;

EXPORT int ibv_fork_init(void)
{
	fork_initialized = 1;
	return 0;
}

EXPORT enum ibv_fork_status ibv_is_fork_initialized(void)
{
	return fork_initialized ? IBV_FORK_ENABLED : IBV_FORK_DISABLED;
}

/* ---- PD and MR ---------------------------------------------------------- */

/*
 * Like the kernel, refuse to destroy an object that another live object
 * still refers to: a PD with MRs, QPs or AHs, a CQ with QPs, a completion
 * channel with CQs.
 */
struct sd_pd {
	struct ibv_pd pd;
	atomic_int users;
};

static void use(atomic_int *n, int d)
{
	atomic_fetch_add(n, d);
}

EXPORT struct ibv_pd *ibv_alloc_pd(struct ibv_context *c)
{
	struct sd_pd *p = calloc(1, sizeof(*p));

	if (!p) {
		errno = ENOMEM;
		return NULL;
	}
	p->pd.context = c;
	dlog("alloc_pd -> %p", (void *)p);
	return &p->pd;
}

EXPORT int ibv_dealloc_pd(struct ibv_pd *pd)
{
	struct sd_pd *p = (struct sd_pd *)pd;
	int n = atomic_load(&p->users);

	if (n) {
		dlog("dealloc_pd %p -> EBUSY (%d users)", (void *)pd, n);
		return EBUSY;
	}
	dlog("dealloc_pd %p", (void *)pd);
	free(p);
	return 0;
}

/*
 * rdma-core allocates every MR as struct verbs_mr (infiniband/driver.h, not
 * installed by distributions). The efa unit tests read its access field
 * through their own copy of the layout, so the stand-in uses the same one.
 */
struct sd_verbs_mr {
	struct ibv_mr ibv_mr;
	int mr_type;		/* enum ibv_mr_type in driver.h */
	int access;
};

EXPORT struct ibv_mr *ibv_reg_mr_iova2(struct ibv_pd *pd, void *addr, size_t length,
				       uint64_t iova, unsigned int access)
{
	struct sd_context *sc = ctx_of(pd->context);
	struct sd_verbs_mr *vmr;
	struct ibv_mr *mr;

	(void)iova;
	if (!length || length > SD_MAX_MR_SIZE) {
		errno = EINVAL;
		return NULL;
	}
	/* rdma-core hands out MRs as struct verbs_mr; callers may read access. */
	vmr = calloc(1, sizeof(*vmr));
	if (!vmr) {
		errno = ENOMEM;
		return NULL;
	}
	vmr->mr_type = 0;		/* IBV_MR_TYPE_MR */
	vmr->access = (int)access;
	mr = &vmr->ibv_mr;
	mr->context = pd->context;
	mr->pd = pd;
	mr->addr = addr;
	mr->length = length;
	use(&((struct sd_pd *)pd)->users, 1);
	pthread_mutex_lock(&sc->lock);
	mr->lkey = mr->rkey = mr->handle = sc->next_key++;
	pthread_mutex_unlock(&sc->lock);
	dlog("reg_mr %p+%zu access 0x%x -> key 0x%x", addr, length, access, mr->lkey);
	return mr;
}

EXPORT struct ibv_mr *ibv_reg_dmabuf_mr(struct ibv_pd *pd, uint64_t offset, size_t length,
					uint64_t iova, int fd, int access)
{
	(void)pd; (void)offset; (void)length; (void)iova; (void)fd; (void)access;
	dlog("reg_dmabuf_mr -> EOPNOTSUPP");
	errno = EOPNOTSUPP;
	return NULL;
}

EXPORT int ibv_dereg_mr(struct ibv_mr *mr)
{
	dlog("dereg_mr key 0x%x", mr->lkey);
	use(&((struct sd_pd *)mr->pd)->users, -1);
	free(mr);
	return 0;
}

EXPORT int efadv_query_mr(struct ibv_mr *ibvmr, struct efadv_mr_attr *attr, uint32_t inlen)
{
	(void)ibvmr;
	if (inlen < offsetof(struct efadv_mr_attr, ic_id_validity))
		return EINVAL;
	memset(attr, 0, inlen < sizeof(*attr) ? inlen : sizeof(*attr));
	return 0;
}

/* ---- completion channel and CQ ------------------------------------------ */

EXPORT struct ibv_comp_channel *ibv_create_comp_channel(struct ibv_context *c)
{
	struct ibv_comp_channel *ch = calloc(1, sizeof(*ch));

	if (!ch) {
		errno = ENOMEM;
		return NULL;
	}
	ch->context = c;
	ch->fd = eventfd(0, EFD_CLOEXEC);
	return ch;
}

EXPORT int ibv_destroy_comp_channel(struct ibv_comp_channel *ch)
{
	/* refcnt counts the CQs bound to the channel, as in rdma-core. */
	if (ch->refcnt) {
		dlog("destroy_comp_channel -> EBUSY (%d CQs)", ch->refcnt);
		return EBUSY;
	}
	if (ch->fd >= 0)
		close(ch->fd);
	free(ch);
	return 0;
}

struct sd_cq {
	struct ibv_cq_ex ex;
	struct efadv_cq dv;
	uint64_t wc_flags;
	atomic_int users;
};

static struct sd_cq *cq_of_ex(struct ibv_cq_ex *ex)
{
	return (struct sd_cq *)ex;
}

/* No data moves yet, so every CQ stays empty. */
static int sd_start_poll(struct ibv_cq_ex *cq, struct ibv_poll_cq_attr *attr)
{
	(void)cq; (void)attr;
	return ENOENT;
}

static int sd_next_poll(struct ibv_cq_ex *cq)
{
	(void)cq;
	return ENOENT;
}

static void sd_end_poll(struct ibv_cq_ex *cq)
{
	(void)cq;
}

static enum ibv_wc_opcode sd_read_opcode(struct ibv_cq_ex *cq) { (void)cq; return IBV_WC_SEND; }
static uint32_t sd_read_u32(struct ibv_cq_ex *cq) { (void)cq; return 0; }
static __be32 sd_read_imm(struct ibv_cq_ex *cq) { (void)cq; return 0; }
static unsigned int sd_read_wc_flags(struct ibv_cq_ex *cq) { (void)cq; return 0; }
static uint8_t sd_read_u8(struct ibv_cq_ex *cq) { (void)cq; return 0; }
static int sd_wc_read_sgid(struct efadv_cq *dv, union ibv_gid *gid) { (void)dv; memset(gid, 0, sizeof(*gid)); return 0; }
#if SD_HAVE_EFADV_WC_IS_UNSOLICITED
static bool sd_wc_is_unsolicited(struct efadv_cq *dv) { (void)dv; return false; }
#endif

static int sd_poll_cq(struct ibv_cq *cq, int n, struct ibv_wc *wc)
{
	(void)cq; (void)n; (void)wc;
	return 0;
}

static int sd_req_notify_cq(struct ibv_cq *cq, int solicited)
{
	(void)cq; (void)solicited;
	return 0;
}

static struct ibv_cq_ex *create_cq(struct ibv_context *c, struct ibv_cq_init_attr_ex *attr,
				   struct efadv_cq_init_attr *efa_attr, uint32_t inlen)
{
	struct sd_cq *cq;

	if (!attr->cqe || attr->cqe > SD_MAX_CQE) {
		errno = EINVAL;
		return NULL;
	}
	cq = calloc(1, sizeof(*cq));
	if (!cq) {
		errno = ENOMEM;
		return NULL;
	}
	cq->ex.context = c;
	cq->ex.channel = attr->channel;
	if (attr->channel) {
		pthread_mutex_lock(&ctx_of(c)->lock);
		attr->channel->refcnt++;
		pthread_mutex_unlock(&ctx_of(c)->lock);
	}
	cq->ex.cq_context = attr->cq_context;
	cq->ex.cqe = (int)attr->cqe;
	pthread_mutex_init(&cq->ex.mutex, NULL);
	pthread_cond_init(&cq->ex.cond, NULL);
	cq->ex.start_poll = sd_start_poll;
	cq->ex.next_poll = sd_next_poll;
	cq->ex.end_poll = sd_end_poll;
	cq->ex.read_opcode = sd_read_opcode;
	cq->ex.read_vendor_err = sd_read_u32;
	cq->ex.read_byte_len = sd_read_u32;
	cq->ex.read_imm_data = sd_read_imm;
	cq->ex.read_qp_num = sd_read_u32;
	cq->ex.read_src_qp = sd_read_u32;
	cq->ex.read_wc_flags = sd_read_wc_flags;
	cq->ex.read_slid = sd_read_u32;
	cq->ex.read_sl = sd_read_u8;
	cq->ex.read_dlid_path_bits = sd_read_u8;
	cq->dv.wc_read_sgid = sd_wc_read_sgid;
#if SD_HAVE_EFADV_WC_IS_UNSOLICITED
	cq->dv.wc_is_unsolicited = sd_wc_is_unsolicited;
#endif
	cq->wc_flags = attr->wc_flags;
	if (efa_attr && inlen >= offsetof(struct efadv_cq_init_attr, wc_flags) + sizeof(efa_attr->wc_flags))
		cq->wc_flags |= efa_attr->wc_flags << 32;
	dlog("create_cq cqe %u wc_flags 0x%llx -> %p", attr->cqe,
	     (unsigned long long)attr->wc_flags, (void *)cq);
	return &cq->ex;
}

static struct ibv_cq_ex *sd_create_cq_ex(struct ibv_context *c, struct ibv_cq_init_attr_ex *attr)
{
	return create_cq(c, attr, NULL, 0);
}

EXPORT struct ibv_cq_ex *efadv_create_cq(struct ibv_context *c, struct ibv_cq_init_attr_ex *attr,
					 struct efadv_cq_init_attr *efa_attr, uint32_t inlen)
{
	return create_cq(c, attr, efa_attr, inlen);
}

EXPORT struct efadv_cq *efadv_cq_from_ibv_cq_ex(struct ibv_cq_ex *ex)
{
	return &cq_of_ex(ex)->dv;
}

#if SD_HAVE_EFADV_QUERY_CQ
EXPORT int efadv_query_cq(struct ibv_cq *cq, struct efadv_cq_attr *attr, uint32_t inlen)
{
	(void)cq; (void)attr; (void)inlen;
	/* The direct data path needs the CQ ring; the stand-in has none. */
	return EOPNOTSUPP;
}
#endif

EXPORT int ibv_destroy_cq(struct ibv_cq *cq)
{
	struct sd_cq *c = (struct sd_cq *)cq;

	/* SD_MUTATION_CQ_IN_USE builds the stand-in wrong on purpose, for the CI's mutation control. */
#ifndef SD_MUTATION_CQ_IN_USE
	int n = atomic_load(&c->users);

	if (n) {
		dlog("destroy_cq %p -> EBUSY (%d QPs)", (void *)cq, n);
		return EBUSY;
	}
#endif
	dlog("destroy_cq %p", (void *)cq);
	if (c->ex.channel) {
		pthread_mutex_lock(&ctx_of(cq->context)->lock);
		c->ex.channel->refcnt--;
		pthread_mutex_unlock(&ctx_of(cq->context)->lock);
	}
	pthread_mutex_destroy(&c->ex.mutex);
	pthread_cond_destroy(&c->ex.cond);
	free(c);
	return 0;
}

EXPORT void ibv_ack_cq_events(struct ibv_cq *cq, unsigned int nevents)
{
	(void)cq; (void)nevents;
}

/* ---- QP ------------------------------------------------------------------- */

struct sd_qp {
	struct ibv_qp_ex ex;
	int extended;
};

static void wr_unsupported(struct ibv_qp_ex *qp)
{
	(void)qp;
}

static void sd_wr_start(struct ibv_qp_ex *qp) { (void)qp; }
static int sd_wr_complete(struct ibv_qp_ex *qp) { (void)qp; return ENOSYS; }
static void sd_wr_abort(struct ibv_qp_ex *qp) { (void)qp; }
static void sd_wr_send(struct ibv_qp_ex *qp) { wr_unsupported(qp); }
static void sd_wr_send_imm(struct ibv_qp_ex *qp, __be32 imm) { (void)imm; wr_unsupported(qp); }
static void sd_wr_rdma(struct ibv_qp_ex *qp, uint32_t rkey, uint64_t addr) { (void)rkey; (void)addr; wr_unsupported(qp); }
static void sd_wr_rdma_imm(struct ibv_qp_ex *qp, uint32_t rkey, uint64_t addr, __be32 imm) { (void)rkey; (void)addr; (void)imm; wr_unsupported(qp); }
static void sd_wr_set_ud_addr(struct ibv_qp_ex *qp, struct ibv_ah *ah, uint32_t qpn, uint32_t qkey) { (void)ah; (void)qpn; (void)qkey; wr_unsupported(qp); }
static void sd_wr_set_sge(struct ibv_qp_ex *qp, uint32_t lkey, uint64_t addr, uint32_t len) { (void)lkey; (void)addr; (void)len; wr_unsupported(qp); }
static void sd_wr_set_sge_list(struct ibv_qp_ex *qp, size_t n, const struct ibv_sge *sg) { (void)n; (void)sg; wr_unsupported(qp); }
static void sd_wr_set_inline_data(struct ibv_qp_ex *qp, void *addr, size_t len) { (void)addr; (void)len; wr_unsupported(qp); }
static void sd_wr_set_inline_data_list(struct ibv_qp_ex *qp, size_t n, const struct ibv_data_buf *b) { (void)n; (void)b; wr_unsupported(qp); }

static int sd_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad)
{
	(void)qp;
	if (bad)
		*bad = wr;
	return ENOSYS;
}

static int sd_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad)
{
	(void)qp;
	if (bad)
		*bad = wr;
	return ENOSYS;
}

static struct ibv_qp *create_qp(struct ibv_context *c, struct ibv_qp_init_attr_ex *attr, int extended)
{
	struct sd_context *sc = ctx_of(c);
	struct sd_qp *qp;

	if (attr->cap.max_send_wr > SD_EFA_MAX_SQ_WR || attr->cap.max_recv_wr > SD_EFA_MAX_RQ_WR ||
	    attr->cap.max_send_sge > SD_EFA_MAX_SQ_SGE || attr->cap.max_recv_sge > SD_EFA_MAX_RQ_SGE ||
	    attr->cap.max_inline_data > SD_EFA_INLINE_BUF_SIZE) {
		dlog("create_qp caps beyond device: wr %u/%u sge %u/%u inline %u",
		     attr->cap.max_send_wr, attr->cap.max_recv_wr, attr->cap.max_send_sge,
		     attr->cap.max_recv_sge, attr->cap.max_inline_data);
		errno = EINVAL;
		return NULL;
	}
	qp = calloc(1, sizeof(*qp));
	if (!qp) {
		errno = ENOMEM;
		return NULL;
	}
	qp->extended = extended;
	qp->ex.qp_base.context = c;
	qp->ex.qp_base.qp_context = attr->qp_context;
	qp->ex.qp_base.pd = attr->pd;
	qp->ex.qp_base.send_cq = attr->send_cq;
	qp->ex.qp_base.recv_cq = attr->recv_cq;
	qp->ex.qp_base.srq = attr->srq;
	qp->ex.qp_base.qp_type = attr->qp_type;
	qp->ex.qp_base.state = IBV_QPS_RESET;
	use(&((struct sd_pd *)attr->pd)->users, 1);
	if (attr->send_cq)
		use(&((struct sd_cq *)attr->send_cq)->users, 1);
	if (attr->recv_cq)
		use(&((struct sd_cq *)attr->recv_cq)->users, 1);
	pthread_mutex_init(&qp->ex.qp_base.mutex, NULL);
	pthread_cond_init(&qp->ex.qp_base.cond, NULL);
	pthread_mutex_lock(&sc->lock);
	qp->ex.qp_base.qp_num = sc->next_qpn++;
	pthread_mutex_unlock(&sc->lock);
	qp->ex.qp_base.handle = qp->ex.qp_base.qp_num;
	qp->ex.comp_mask = attr->comp_mask;
	qp->ex.wr_start = sd_wr_start;
	qp->ex.wr_complete = sd_wr_complete;
	qp->ex.wr_abort = sd_wr_abort;
	qp->ex.wr_send = sd_wr_send;
	qp->ex.wr_send_imm = sd_wr_send_imm;
	qp->ex.wr_rdma_read = sd_wr_rdma;
	qp->ex.wr_rdma_write = sd_wr_rdma;
	qp->ex.wr_rdma_write_imm = sd_wr_rdma_imm;
	qp->ex.wr_set_ud_addr = sd_wr_set_ud_addr;
	qp->ex.wr_set_sge = sd_wr_set_sge;
	qp->ex.wr_set_sge_list = sd_wr_set_sge_list;
	qp->ex.wr_set_inline_data = sd_wr_set_inline_data;
	qp->ex.wr_set_inline_data_list = sd_wr_set_inline_data_list;
	dlog("create_qp type %d send_wr %u recv_wr %u -> qpn %u", attr->qp_type,
	     attr->cap.max_send_wr, attr->cap.max_recv_wr, qp->ex.qp_base.qp_num);
	return &qp->ex.qp_base;
}

static struct ibv_qp *sd_create_qp_ex(struct ibv_context *c, struct ibv_qp_init_attr_ex *attr)
{
	return create_qp(c, attr, 1);
}

EXPORT struct ibv_qp *efadv_create_qp_ex(struct ibv_context *c, struct ibv_qp_init_attr_ex *attr,
					 struct efadv_qp_init_attr *efa_attr, uint32_t inlen)
{
	if (efa_attr && inlen >= offsetof(struct efadv_qp_init_attr, driver_qp_type) + sizeof(efa_attr->driver_qp_type) &&
	    efa_attr->driver_qp_type != EFADV_QP_DRIVER_TYPE_SRD) {
		errno = EOPNOTSUPP;
		return NULL;
	}
	return create_qp(c, attr, 1);
}

EXPORT struct ibv_qp_ex *ibv_qp_to_qp_ex(struct ibv_qp *qp)
{
	struct sd_qp *q = (struct sd_qp *)qp;

	return q->extended ? &q->ex : NULL;
}

EXPORT int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int mask)
{
	if (mask & IBV_QP_STATE)
		qp->state = attr->qp_state;
	dlog("modify_qp qpn %u mask 0x%x state %d", qp->qp_num, mask, qp->state);
	return 0;
}

EXPORT int ibv_destroy_qp(struct ibv_qp *qp)
{
	dlog("destroy_qp qpn %u", qp->qp_num);
	use(&((struct sd_pd *)qp->pd)->users, -1);
	if (qp->send_cq)
		use(&((struct sd_cq *)qp->send_cq)->users, -1);
	if (qp->recv_cq)
		use(&((struct sd_cq *)qp->recv_cq)->users, -1);
	pthread_mutex_destroy(&qp->mutex);
	pthread_cond_destroy(&qp->cond);
	free(qp);
	return 0;
}

#if SD_HAVE_IBV_QUERY_QP_DATA_IN_ORDER
EXPORT int ibv_query_qp_data_in_order(struct ibv_qp *qp, enum ibv_wr_opcode op, uint32_t flags)
{
	(void)qp; (void)op; (void)flags;
	/* SRD keeps no order between messages, and the stand-in promises none. */
	return 0;
}
#endif

#if SD_HAVE_EFADV_QUERY_QP_WQS
EXPORT int efadv_query_qp_wqs(struct ibv_qp *qp, struct efadv_wq_attr *sq, struct efadv_wq_attr *rq,
			      uint32_t inlen)
{
	(void)qp; (void)sq; (void)rq; (void)inlen;
	/* The direct data path needs the work queue rings; the stand-in has none. */
	return EOPNOTSUPP;
}
#endif

/* ---- AH ------------------------------------------------------------------- */

struct sd_ah {
	struct ibv_ah ah;
	uint16_t ahn;
	union ibv_gid gid;
};

EXPORT struct ibv_ah *ibv_create_ah(struct ibv_pd *pd, struct ibv_ah_attr *attr)
{
	struct sd_context *sc = ctx_of(pd->context);
	struct sd_ah *a = calloc(1, sizeof(*a));

	if (!a) {
		errno = ENOMEM;
		return NULL;
	}
	a->ah.context = pd->context;
	a->ah.pd = pd;
	a->gid = attr->grh.dgid;
	use(&((struct sd_pd *)pd)->users, 1);
	pthread_mutex_lock(&sc->lock);
	a->ahn = sc->next_ahn++;
	pthread_mutex_unlock(&sc->lock);
	a->ah.handle = a->ahn;
	dlog("create_ah -> ahn %u", a->ahn);
	return &a->ah;
}

EXPORT int ibv_destroy_ah(struct ibv_ah *ah)
{
	dlog("destroy_ah ahn %u", ((struct sd_ah *)ah)->ahn);
	use(&((struct sd_pd *)ah->pd)->users, -1);
	free(ah);
	return 0;
}

EXPORT int efadv_query_ah(struct ibv_ah *ibvah, struct efadv_ah_attr *attr, uint32_t inlen)
{
	if (inlen < offsetof(struct efadv_ah_attr, reserved))
		return EINVAL;
	memset(attr, 0, inlen < sizeof(*attr) ? inlen : sizeof(*attr));
	attr->ahn = ((struct sd_ah *)ibvah)->ahn;
	return 0;
}

/*
 * SRDSIM_EFA_DEVICE_CAPS overrides the capability bits, so that code which
 * behaves differently on devices with and without a capability can be run
 * both ways.
 */
static uint32_t device_caps(void)
{
	const char *v = getenv("SRDSIM_EFA_DEVICE_CAPS");

	return v && *v ? (uint32_t)strtoul(v, NULL, 0) : SD_EFA_DEVICE_CAPS;
}

/* ---- efadv device query ------------------------------------------------- */

EXPORT int efadv_query_device(struct ibv_context *c, struct efadv_device_attr *attr, uint32_t inlen)
{
	struct efadv_device_attr a;

	(void)c;
	if (inlen < offsetof(struct efadv_device_attr, device_caps))
		return EINVAL;
	memset(&a, 0, sizeof(a));
	a.max_sq_wr = SD_EFA_MAX_SQ_WR;
	a.max_rq_wr = SD_EFA_MAX_RQ_WR;
	a.max_sq_sge = SD_EFA_MAX_SQ_SGE;
	a.max_rq_sge = SD_EFA_MAX_RQ_SGE;
	a.inline_buf_size = SD_EFA_INLINE_BUF_SIZE;
	a.device_caps = device_caps();
	a.max_rdma_size = SD_EFA_MAX_RDMA_SIZE;
	memcpy(attr, &a, inlen < sizeof(a) ? inlen : sizeof(a));
	return 0;
}
