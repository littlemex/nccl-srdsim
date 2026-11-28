/* SPDX-License-Identifier: Apache-2.0 */
/*
 * A stand-in for the EFA device below the libfabric efa provider: the device,
 * its context, PDs, MRs and AHs. queue.c has the CQs, the QPs and the
 * movement of data.
 *
 * Built twice, as libibverbs.so.1 and libefa.so.1, exporting the symbols the
 * provider links against with the same symbol versions. Objects are laid out
 * with the real rdma-core headers, so the provider's inline verbs calls reach
 * the function pointers filled in here. Where the real libefa decides how a
 * device's answer looks to its caller, the stand-in answers the same way.
 *
 * Set SRDSIM_DEVICE_LOG=1 to log each call on stderr.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/eventfd.h>
#include <sys/stat.h>
#include <unistd.h>

#include "device.h"

#define dlog sd_log

bool sd_log_enabled(void)
{
	static int on = -1;

	if (on < 0) {
		const char *v = getenv("SRDSIM_DEVICE_LOG");

		on = v && *v && strcmp(v, "0");
	}
	return on;
}

void sd_log(const char *fmt, ...)
{
	va_list ap;

	if (!sd_log_enabled())
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

/*
 * Several processes build the same tree at once (a test forks, or two ranks start together), and each
 * may be reading what another is writing, so every file and link is made under a private name and
 * renamed into place, which replaces it in one step.
 */
static int write_file(const char *path, const char *text)
{
	char tmp[1100];
	FILE *f;

	snprintf(tmp, sizeof(tmp), "%s.%d", path, (int)getpid());
	f = fopen(tmp, "w");
	if (!f)
		return -1;
	fputs(text, f);
	if (fclose(f) || rename(tmp, path)) {
		unlink(tmp);
		return -1;
	}
	return 0;
}

static int link_to(const char *target, const char *path)
{
	char tmp[1100];

	snprintf(tmp, sizeof(tmp), "%s.%d", path, (int)getpid());
	unlink(tmp);
	if (symlink(target, tmp) || rename(tmp, path)) {
		unlink(tmp);
		return -1;
	}
	return 0;
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
		if (link_to(t, p))
			return NULL;
		snprintf(p, sizeof(p), "%s/class/infiniband/srdsim%d", sysfs_root, i);
		if (mkdirs(p))
			return NULL;
		snprintf(p, sizeof(p), "%s/class/infiniband/srdsim%d/device", sysfs_root, i);
		if (link_to(dev, p))
			return NULL;
	}
	if (!getenv("SYSFS_PATH")) {
		setenv("SYSFS_PATH", sysfs_root, 0);
		dlog("SYSFS_PATH=%s", sysfs_root);
	}
	return sysfs_root;
}

/* ---- device and context ---------------------------------------------- */

#define ctx_of sd_ctx

static int device_count(void)
{
	const char *v = getenv("SRDSIM_DEVICE_COUNT");
	int n = v ? atoi(v) : 1;

	return n < 0 ? 0 : n > 32 ? 32 : n;
}

/*
 * The devices, made once and kept for the life of the process. rdma-core does the same: a context keeps
 * pointing at its ibv_device, and the efa provider reads the device's name through it long after
 * ibv_free_device_list, which frees only the array.
 */
static struct sd_device *devices[32];
static int ndevices = -1;
static pthread_mutex_t devices_lock = PTHREAD_MUTEX_INITIALIZER;

static int make_devices(void)
{
	int n = device_count(), i;
	const char *root = sysfs_tree(n);

	if (!root) {
		dlog("cannot write the sysfs tree under %s", sysfs_root);
		errno = EIO;
		return -1;
	}
	for (i = 0; i < n; i++) {
		struct sd_device *d = calloc(1, sizeof(*d));

		if (!d) {
			errno = ENOMEM;
			return -1;
		}
		d->index = i;
		d->dev.node_type = IBV_NODE_UNSPECIFIED;
		d->dev.transport_type = IBV_TRANSPORT_UNSPECIFIED;
		snprintf(d->dev.name, sizeof(d->dev.name), "srdsim%d", i);
		snprintf(d->dev.dev_name, sizeof(d->dev.dev_name), "uverbs%d", i);
		snprintf(d->dev.dev_path, sizeof(d->dev.dev_path), "/nonexistent/srdsim/uverbs%d", i);
		snprintf(d->dev.ibdev_path, sizeof(d->dev.ibdev_path), "%s/class/infiniband/srdsim%d", root, i);
		devices[i] = d;
	}
	ndevices = n;
	return 0;
}

EXPORT struct ibv_device **ibv_get_device_list(int *num_devices)
{
	struct ibv_device **list;
	int i, err = 0;

	pthread_mutex_lock(&devices_lock);
	if (ndevices < 0 && make_devices())
		err = errno;
	pthread_mutex_unlock(&devices_lock);
	if (err) {
		errno = err;
		return NULL;
	}
	list = calloc((size_t)ndevices + 1, sizeof(*list));
	if (!list) {
		errno = ENOMEM;
		return NULL;
	}
	for (i = 0; i < ndevices; i++)
		list[i] = &devices[i]->dev;
	if (num_devices)
		*num_devices = ndevices;
	dlog("get_device_list -> %d", ndevices);
	return list;
}

EXPORT void ibv_free_device_list(struct ibv_device **list)
{
	free(list);
}

static int sd_query_port(struct ibv_context *c, uint8_t port, struct ibv_port_attr *a, size_t len);

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
	sc->next_ahn = 1;
	sc->next_key = 0x100;
	sc->vctx.sz = sizeof(sc->vctx);
	sc->vctx.query_port = sd_query_port;
	c = &sc->vctx.context;
	c->device = device;
	c->abi_compat = __VERBS_ABI_IS_EXTENDED;
	c->cmd_fd = eventfd(0, EFD_CLOEXEC);
	c->async_fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
	c->num_comp_vectors = SD_NUM_COMP_VECTORS;
	pthread_mutex_init(&c->mutex, NULL);
	if (sd_agent_start(sc)) {
		int err = errno;

		pthread_mutex_destroy(&sc->lock);
		close(c->cmd_fd);
		close(c->async_fd);
		free(sc);
		errno = err;
		return NULL;
	}
	dlog("open_device %s -> ctx %p", device->name, (void *)c);
	return c;
}

EXPORT int ibv_close_device(struct ibv_context *c)
{
	struct sd_context *sc = ctx_of(c);

	dlog("close_device ctx %p", (void *)c);
	sd_agent_stop(sc);
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
	sd_gid_of(sc->device->index, gid);
	return 0;
}

void sd_gid_of(int device, union ibv_gid *gid)
{
	memset(gid, 0, sizeof(*gid));
	gid->raw[0] = 0xfe;
	gid->raw[1] = 0x80;
	gid->raw[8] = 0x02;
	gid->raw[13] = 0xef;
	gid->raw[14] = 0xa0;
	gid->raw[15] = (uint8_t)device;
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

EXPORT struct ibv_mr *ibv_reg_mr_iova2(struct ibv_pd *pd, void *addr, size_t length,
				       uint64_t iova, unsigned int access)
{
	struct sd_context *sc = ctx_of(pd->context);
	struct sd_mr *m;

	if (!length || length > SD_MAX_MR_SIZE) {
		errno = EINVAL;
		return NULL;
	}
	m = calloc(1, sizeof(*m));
	if (!m) {
		errno = ENOMEM;
		return NULL;
	}
	m->mr_type = 0;
	m->access = (int)access;
	m->iova = iova;
	m->ibv_mr.context = pd->context;
	m->ibv_mr.pd = pd;
	m->ibv_mr.addr = addr;
	m->ibv_mr.length = length;
	use(&((struct sd_pd *)pd)->users, 1);
	pthread_mutex_lock(&sc->lock);
	m->ibv_mr.lkey = m->ibv_mr.rkey = m->ibv_mr.handle = sc->next_key++;
	m->next = sc->mrs;
	sc->mrs = m;
	pthread_mutex_unlock(&sc->lock);
	dlog("reg_mr %p+%zu iova 0x%llx access 0x%x -> key 0x%x", addr, length,
	     (unsigned long long)iova, access, m->ibv_mr.lkey);
	return &m->ibv_mr;
}

struct sd_mr *sd_mr_find(struct sd_context *ctx, uint32_t key)
{
	struct sd_mr *m;

	for (m = ctx->mrs; m; m = m->next)
		if (m->ibv_mr.lkey == key)
			return m;
	return NULL;
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
	struct sd_context *sc = ctx_of(mr->context);
	struct sd_mr **p;

	dlog("dereg_mr key 0x%x", mr->lkey);
	pthread_mutex_lock(&sc->lock);
	for (p = &sc->mrs; *p; p = &(*p)->next)
		if (&(*p)->ibv_mr == mr) {
			*p = (*p)->next;
			break;
		}
	pthread_mutex_unlock(&sc->lock);
	use(&((struct sd_pd *)mr->pd)->users, -1);
	free((struct sd_mr *)mr);
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

/* ---- AH ------------------------------------------------------------------- */

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
	/*
	 * assumed: the device has one AH number per destination GID in a PD, and every AH created for that GID
	 * gets it. A receive completion names its sender by that number (slid) and the sender's QP number,
	 * which the efa provider looks up together, so two numbers for one GID would leave the lookup to chance.
	 */
	{
		struct sd_ah *o;

		for (o = sc->ahs; o; o = o->next)
			if (o->ah.pd == pd && !memcmp(o->gid.raw, a->gid.raw, sizeof(a->gid.raw)))
				break;
		a->ahn = o ? o->ahn : sc->next_ahn++;
	}
	a->next = sc->ahs;
	sc->ahs = a;
	pthread_mutex_unlock(&sc->lock);
	a->ah.handle = a->ahn;
	dlog("create_ah -> ahn %u", a->ahn);
	return &a->ah;
}

uint16_t sd_ahn_of(struct sd_context *ctx, struct ibv_pd *pd, const union ibv_gid *gid)
{
	struct sd_ah *a;

	for (a = ctx->ahs; a; a = a->next)
		if (a->ah.pd == pd && !memcmp(a->gid.raw, gid->raw, sizeof(gid->raw)))
			return a->ahn;
	return 0xffff;
}

EXPORT int ibv_destroy_ah(struct ibv_ah *ah)
{
	struct sd_context *sc = ctx_of(ah->context);
	struct sd_ah **p;

	pthread_mutex_lock(&sc->lock);
	for (p = &sc->ahs; *p; p = &(*p)->next)
		if (&(*p)->ah == ah) {
			*p = (*p)->next;
			break;
		}
	pthread_mutex_unlock(&sc->lock);
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
