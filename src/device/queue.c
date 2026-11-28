/* SPDX-License-Identifier: Apache-2.0 */
/*
 * The CQs and QPs of the device stand-in, and how data moves between QPs.
 *
 * Every QP owns an AF_UNIX datagram socket in the abstract namespace, named by
 * its device and QP number, so QPs of any process on the host reach each
 * other by the (GID, QPN) the provider exchanges. QP numbers are unique on the
 * host because the name is: a QP takes the first free one.
 *
 * Each device context has one thread, the agent, that does everything the
 * hardware does on its own: it sends the frames of posted work, reading the
 * sender's memory only when it sends (so a buffer reused before its
 * completion is caught), applies incoming sends to posted receives and
 * incoming RDMA to registered memory, answers reads, and writes completions.
 * Sockets are non-blocking and only the agent sends, so two agents whose
 * queues are full never wait on each other.
 *
 * What a completion looks like follows rdma-core's libefa (providers/efa/
 * verbs.c), which turns the device's completion descriptors into what the
 * caller reads: vendor_err is the device status, a completed RDMA read reads
 * as IBV_WC_SEND, slid is the receiver's AH number for the sender or 0xffff,
 * and the source GID is given only when slid is 0xffff. Posting follows its
 * checks too: a send needs the QP in RTS, every WR must be signaled unless
 * sq_sig_all is set, only IBV_SEND_SIGNALED and IBV_SEND_INLINE are accepted,
 * and a full queue says ENOMEM.
 *
 * Not modelled yet: reordering (every frame between two QPs arrives in the
 * order sent), RNR retry limits (a send that finds no receive waits for one),
 * and CQ overrun (an entry that does not fit is dropped and logged).
 */
#define _GNU_SOURCE
#include <errno.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "device.h"

#define dlog sd_log
#define SD_CHUNK 32768u			/* bytes of RDMA data per frame */
#define SD_MAX_SGE_ANY 8

static void use(atomic_int *n, int d)
{
	atomic_fetch_add(n, d);
}

static void wake(struct sd_context *ctx);

/* ---- completion queues ------------------------------------------------- */

struct sd_cqe {
	uint64_t wr_id;
	enum ibv_wc_status status;
	uint32_t vendor_err;
	enum ibv_wc_opcode opcode;
	uint32_t byte_len;
	uint32_t imm;			/* host order, as the device writes it */
	bool has_imm;
	bool unsolicited;
	uint32_t qp_num;
	uint32_t src_qp;
	uint16_t slid;
	union ibv_gid sgid;
};

struct sd_cq {
	struct ibv_cq_ex ex;
	struct efadv_cq dv;
	atomic_int users;
	/*
	 * The ring. Taken only for the moment an entry goes in or comes out: the efa provider leaves a poll
	 * open across calls (start_poll once, end_poll later), and the device writes completions all the
	 * while, so nothing may hold this from start_poll to end_poll.
	 */
	pthread_mutex_t lock;
	struct sd_cqe *ring;
	uint32_t size, head, count;
	struct sd_cqe cur;
	bool armed;
	uint64_t overruns;
};

static struct sd_cq *cq_of(struct ibv_cq_ex *ex)
{
	return (struct sd_cq *)ex;
}

static enum ibv_wc_status to_ibv_status(enum sd_status st)
{
	switch (st) {
	case SD_ST_OK: return IBV_WC_SUCCESS;
	case SD_ST_FLUSHED: return IBV_WC_WR_FLUSH_ERR;
	case SD_ST_LOCAL_QP_INTERNAL_ERROR:
	case SD_ST_LOCAL_UNSUPPORTED_OP:
	case SD_ST_LOCAL_INVALID_AH: return IBV_WC_LOC_QP_OP_ERR;
	case SD_ST_LOCAL_INVALID_LKEY: return IBV_WC_LOC_PROT_ERR;
	case SD_ST_LOCAL_BAD_LENGTH: return IBV_WC_LOC_LEN_ERR;
	case SD_ST_REMOTE_ABORT: return IBV_WC_REM_ABORT_ERR;
	case SD_ST_REMOTE_RNR: return IBV_WC_RNR_RETRY_EXC_ERR;
	case SD_ST_REMOTE_BAD_DEST_QPN: return IBV_WC_REM_INV_RD_REQ_ERR;
	case SD_ST_REMOTE_BAD_STATUS: return IBV_WC_BAD_RESP_ERR;
	case SD_ST_REMOTE_BAD_LENGTH: return IBV_WC_REM_INV_REQ_ERR;
	case SD_ST_LOCAL_UNRESP_REMOTE:
	case SD_ST_LOCAL_UNREACH_REMOTE: return IBV_WC_RESP_TIMEOUT_ERR;
	case SD_ST_REMOTE_BAD_ADDRESS: return IBV_WC_REM_ACCESS_ERR;
	case SD_ST_REMOTE_UNKNOWN_PEER: return IBV_WC_REM_OP_ERR;
	}
	return IBV_WC_GENERAL_ERR;
}

/* Called by the agent, under ctx->lock. */
static void cq_push(struct ibv_cq *ibcq, const struct sd_cqe *e)
{
	struct sd_cq *cq = (struct sd_cq *)ibcq;
	struct ibv_comp_channel *ch;
	bool notify = false;

	pthread_mutex_lock(&cq->lock);
	if (cq->count == cq->size) {
		cq->overruns++;
		dlog("CQ %p overrun: completion for wr_id 0x%llx dropped", (void *)cq,
		     (unsigned long long)e->wr_id);
	} else {
		cq->ring[(cq->head + cq->count) % cq->size] = *e;
		cq->count++;
		notify = cq->armed;
		cq->armed = false;
	}
	ch = cq->ex.channel;
	pthread_mutex_unlock(&cq->lock);
	if (notify && ch) {
		uint64_t one = 1;

		if (write(ch->fd, &one, sizeof(one)) < 0)
			dlog("completion channel write: %s", strerror(errno));
	}
	dlog("cqe cq %p wr_id 0x%llx status %d vendor %u opcode %d len %u qp %u src %u slid 0x%x",
	     (void *)cq, (unsigned long long)e->wr_id, e->status, e->vendor_err, e->opcode,
	     e->byte_len, e->qp_num, e->src_qp, e->slid);
}

/* Under cq->lock. */
static int take(struct sd_cq *cq)
{
	if (!cq->count)
		return ENOENT;
	cq->cur = cq->ring[cq->head];
	cq->head = (cq->head + 1) % cq->size;
	cq->count--;
	cq->ex.wr_id = cq->cur.wr_id;
	cq->ex.status = cq->cur.status;
	return 0;
}

static int take_locked(struct sd_cq *cq)
{
	int r;

	pthread_mutex_lock(&cq->lock);
	r = take(cq);
	pthread_mutex_unlock(&cq->lock);
	return r;
}

static int sd_start_poll(struct ibv_cq_ex *ex, struct ibv_poll_cq_attr *attr)
{
	(void)attr;
	return take_locked(cq_of(ex));
}

static int sd_next_poll(struct ibv_cq_ex *ex)
{
	return take_locked(cq_of(ex));
}

static void sd_end_poll(struct ibv_cq_ex *ex)
{
	(void)ex;
}

static enum ibv_wc_opcode sd_read_opcode(struct ibv_cq_ex *ex) { return cq_of(ex)->cur.opcode; }
static uint32_t sd_read_vendor_err(struct ibv_cq_ex *ex) { return cq_of(ex)->cur.vendor_err; }
static uint32_t sd_read_byte_len(struct ibv_cq_ex *ex) { return cq_of(ex)->cur.byte_len; }
static __be32 sd_read_imm(struct ibv_cq_ex *ex) { return htobe32(cq_of(ex)->cur.imm); }
static uint32_t sd_read_qp_num(struct ibv_cq_ex *ex) { return cq_of(ex)->cur.qp_num; }
static uint32_t sd_read_src_qp(struct ibv_cq_ex *ex) { return cq_of(ex)->cur.src_qp; }
static uint32_t sd_read_slid(struct ibv_cq_ex *ex) { return cq_of(ex)->cur.slid; }
static uint8_t sd_read_zero8(struct ibv_cq_ex *ex) { (void)ex; return 0; }

static unsigned int sd_read_wc_flags(struct ibv_cq_ex *ex)
{
	return cq_of(ex)->cur.has_imm ? IBV_WC_WITH_IMM : 0;
}

static struct sd_cq *cq_of_dv(struct efadv_cq *dv)
{
	return (struct sd_cq *)((char *)dv - offsetof(struct sd_cq, dv));
}

static int sd_wc_read_sgid(struct efadv_cq *dv, union ibv_gid *sgid)
{
	struct sd_cq *cq = cq_of_dv(dv);

	if (cq->cur.slid != 0xffff)
		return -ENOENT;		/* as libefa: the GID only when the AH is unknown */
	*sgid = cq->cur.sgid;
	return 0;
}

#if SD_HAVE_EFADV_WC_IS_UNSOLICITED
static bool sd_wc_is_unsolicited(struct efadv_cq *dv)
{
	return cq_of_dv(dv)->cur.unsolicited;
}
#endif

static int sd_poll_cq(struct ibv_cq *ibcq, int n, struct ibv_wc *wc)
{
	struct sd_cq *cq = (struct sd_cq *)ibcq;
	int i;

	pthread_mutex_lock(&cq->lock);
	for (i = 0; i < n && !take(cq); i++) {
		memset(&wc[i], 0, sizeof(wc[i]));
		wc[i].wr_id = cq->cur.wr_id;
		wc[i].status = cq->cur.status;
		wc[i].opcode = cq->cur.opcode;
		wc[i].vendor_err = cq->cur.vendor_err;
		wc[i].byte_len = cq->cur.byte_len;
		wc[i].imm_data = htobe32(cq->cur.imm);
		wc[i].qp_num = cq->cur.qp_num;
		wc[i].src_qp = cq->cur.src_qp;
		wc[i].wc_flags = sd_read_wc_flags(&cq->ex);
		wc[i].slid = cq->cur.slid;
	}
	pthread_mutex_unlock(&cq->lock);
	return i;
}

static int sd_req_notify_cq(struct ibv_cq *ibcq, int solicited)
{
	struct sd_cq *cq = (struct sd_cq *)ibcq;

	(void)solicited;
	pthread_mutex_lock(&cq->lock);
	cq->armed = true;
	pthread_mutex_unlock(&cq->lock);
	return 0;
}

static struct ibv_cq_ex *create_cq(struct ibv_context *c, struct ibv_cq_init_attr_ex *attr)
{
	struct sd_cq *cq;

	if (!attr->cqe || attr->cqe > SD_MAX_CQE) {
		errno = EINVAL;
		return NULL;
	}
	cq = calloc(1, sizeof(*cq));
	if (cq)
		cq->ring = calloc(attr->cqe, sizeof(*cq->ring));
	if (!cq || !cq->ring) {
		free(cq);
		errno = ENOMEM;
		return NULL;
	}
	cq->size = attr->cqe;
	pthread_mutex_init(&cq->lock, NULL);
	cq->ex.context = c;
	cq->ex.channel = attr->channel;
	if (attr->channel) {
		pthread_mutex_lock(&sd_ctx(c)->lock);
		attr->channel->refcnt++;
		pthread_mutex_unlock(&sd_ctx(c)->lock);
	}
	cq->ex.cq_context = attr->cq_context;
	cq->ex.cqe = (int)attr->cqe;
	pthread_mutex_init(&cq->ex.mutex, NULL);
	pthread_cond_init(&cq->ex.cond, NULL);
	cq->ex.start_poll = sd_start_poll;
	cq->ex.next_poll = sd_next_poll;
	cq->ex.end_poll = sd_end_poll;
	cq->ex.read_opcode = sd_read_opcode;
	cq->ex.read_vendor_err = sd_read_vendor_err;
	cq->ex.read_byte_len = sd_read_byte_len;
	cq->ex.read_imm_data = sd_read_imm;
	cq->ex.read_qp_num = sd_read_qp_num;
	cq->ex.read_src_qp = sd_read_src_qp;
	cq->ex.read_wc_flags = sd_read_wc_flags;
	cq->ex.read_slid = sd_read_slid;
	cq->ex.read_sl = sd_read_zero8;
	cq->ex.read_dlid_path_bits = sd_read_zero8;
	cq->dv.wc_read_sgid = sd_wc_read_sgid;
#if SD_HAVE_EFADV_WC_IS_UNSOLICITED
	cq->dv.wc_is_unsolicited = sd_wc_is_unsolicited;
#endif
	dlog("create_cq cqe %u wc_flags 0x%llx -> %p", attr->cqe,
	     (unsigned long long)attr->wc_flags, (void *)cq);
	return &cq->ex;
}

static struct ibv_cq_ex *sd_create_cq_ex(struct ibv_context *c, struct ibv_cq_init_attr_ex *attr)
{
	return create_cq(c, attr);
}

EXPORT struct ibv_cq_ex *efadv_create_cq(struct ibv_context *c, struct ibv_cq_init_attr_ex *attr,
					 struct efadv_cq_init_attr *efa_attr, uint32_t inlen)
{
	(void)efa_attr; (void)inlen;
	return create_cq(c, attr);
}

EXPORT struct efadv_cq *efadv_cq_from_ibv_cq_ex(struct ibv_cq_ex *ex)
{
	return &cq_of(ex)->dv;
}

#if SD_HAVE_EFADV_QUERY_CQ
EXPORT int efadv_query_cq(struct ibv_cq *cq, struct efadv_cq_attr *attr, uint32_t inlen)
{
	(void)cq; (void)attr; (void)inlen;
	/* The direct data path needs the CQ ring; the stand-in has none. */
	return EOPNOTSUPP;
}
#endif

EXPORT int ibv_destroy_cq(struct ibv_cq *ibcq)
{
	struct sd_cq *cq = (struct sd_cq *)ibcq;

	/* SD_MUTATION_CQ_IN_USE builds the stand-in wrong on purpose, for the CI's mutation control. */
#ifndef SD_MUTATION_CQ_IN_USE
	int n = atomic_load(&cq->users);

	if (n) {
		dlog("destroy_cq %p -> EBUSY (%d QPs)", (void *)cq, n);
		return EBUSY;
	}
#endif
	dlog("destroy_cq %p (%llu overruns)", (void *)cq, (unsigned long long)cq->overruns);
	if (cq->ex.channel) {
		pthread_mutex_lock(&sd_ctx(ibcq->context)->lock);
		cq->ex.channel->refcnt--;
		pthread_mutex_unlock(&sd_ctx(ibcq->context)->lock);
	}
	pthread_mutex_destroy(&cq->lock);
	pthread_mutex_destroy(&cq->ex.mutex);
	pthread_cond_destroy(&cq->ex.cond);
	free(cq->ring);
	free(cq);
	return 0;
}

EXPORT void ibv_ack_cq_events(struct ibv_cq *cq, unsigned int nevents)
{
	(void)cq; (void)nevents;
}

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

/* ---- queue pairs and posted work ------------------------------------------ */

enum sd_op_kind { SD_OP_SEND, SD_OP_WRITE, SD_OP_READ };

/* A posted send queue WR, from posting until its completion. */
struct sd_op {
	struct sd_op *next;
	struct sd_qp *qp;
	uint64_t id;			/* names it in frames and their answers */
	uint64_t wr_id;
	enum sd_op_kind kind;
	bool has_imm;
	uint32_t imm;
	struct sd_ah *ah;
	uint32_t dst_qpn, qkey;
	uint64_t raddr;
	uint32_t rkey;
	int nsge;
	struct ibv_sge sge[SD_MAX_SGE_ANY];
	uint8_t *inline_data;		/* copied at posting, as inline data is */
	uint64_t len, sent, received;
	bool checked;			/* local keys checked, done once when sending starts */
};

struct sd_recv {
	struct sd_recv *next;
	uint64_t wr_id;
	int nsge;
	struct ibv_sge sge[SD_MAX_SGE_ANY];
};

/* A frame that arrived for a receive not posted yet. */
struct sd_held {
	struct sd_held *next;
	size_t size;
	unsigned char frame[];
};

struct sd_qp {
	struct ibv_qp_ex ex;
	struct sd_qp *next;		/* in ctx->qps */
	int fd;				/* its socket */
	uint32_t qkey;
	bool sq_sig_all;
	bool unsolicited_write_recv;
	struct ibv_qp_cap cap;
	uint32_t sq_outstanding;	/* posted and not completed */
	struct sd_op *batch, **batch_tail;	/* between wr_start and wr_complete */
	int batch_count, batch_err;
	struct sd_recv *rq, **rq_tail;
	uint32_t rq_count;
	struct sd_held *held, **held_tail;
};

static struct sd_qp *qp_of(struct ibv_qp *qp)
{
	return (struct sd_qp *)qp;
}

static struct sd_qp *qp_of_ex(struct ibv_qp_ex *ex)
{
	return (struct sd_qp *)ex;
}

static void free_op(struct sd_op *op)
{
	free(op->inline_data);
	free(op);
}

static void sd_wr_start(struct ibv_qp_ex *ex)
{
	struct sd_qp *qp = qp_of_ex(ex);

	qp->batch = NULL;
	qp->batch_tail = &qp->batch;
	qp->batch_count = 0;
	qp->batch_err = 0;
}

static struct sd_op *batch_last(struct sd_qp *qp)
{
	struct sd_op *op = qp->batch;

	while (op && op->next)
		op = op->next;
	return op;
}

static void new_op(struct ibv_qp_ex *ex, enum sd_op_kind kind, bool has_imm, uint32_t imm,
		   uint32_t rkey, uint64_t raddr)
{
	struct sd_qp *qp = qp_of_ex(ex);
	struct sd_op *op;

	dlog("wr kind %d qpn %u wr_id 0x%llx flags 0x%x", kind, ex->qp_base.qp_num,
	     (unsigned long long)ex->wr_id, ex->wr_flags);
	if (qp->batch_err)
		return;
	if (!(ex->wr_flags & IBV_SEND_SIGNALED) && !qp->sq_sig_all) {
		dlog("qpn %u: non-signaled WRs are not supported", ex->qp_base.qp_num);
		qp->batch_err = EINVAL;
		return;
	}
	if (ex->wr_flags & ~(unsigned int)(IBV_SEND_SIGNALED | IBV_SEND_INLINE)) {
		dlog("qpn %u: unsupported wr_flags 0x%x", ex->qp_base.qp_num, ex->wr_flags);
		qp->batch_err = EINVAL;
		return;
	}
	op = calloc(1, sizeof(*op));
	if (!op) {
		qp->batch_err = ENOMEM;
		return;
	}
	op->qp = qp;
	op->wr_id = ex->wr_id;
	op->kind = kind;
	op->has_imm = has_imm;
	op->imm = imm;
	op->rkey = rkey;
	op->raddr = raddr;
	*qp->batch_tail = op;
	qp->batch_tail = &op->next;
	qp->batch_count++;
}

static void sd_wr_send(struct ibv_qp_ex *ex) { new_op(ex, SD_OP_SEND, false, 0, 0, 0); }
static void sd_wr_send_imm(struct ibv_qp_ex *ex, __be32 imm) { new_op(ex, SD_OP_SEND, true, be32toh(imm), 0, 0); }
static void sd_wr_rdma_write(struct ibv_qp_ex *ex, uint32_t rkey, uint64_t addr) { new_op(ex, SD_OP_WRITE, false, 0, rkey, addr); }
static void sd_wr_rdma_write_imm(struct ibv_qp_ex *ex, uint32_t rkey, uint64_t addr, __be32 imm) { new_op(ex, SD_OP_WRITE, true, be32toh(imm), rkey, addr); }
static void sd_wr_rdma_read(struct ibv_qp_ex *ex, uint32_t rkey, uint64_t addr) { new_op(ex, SD_OP_READ, false, 0, rkey, addr); }

static void sd_wr_set_ud_addr(struct ibv_qp_ex *ex, struct ibv_ah *ah, uint32_t qpn, uint32_t qkey)
{
	struct sd_op *op = batch_last(qp_of_ex(ex));

	if (!op)
		return;
	op->ah = (struct sd_ah *)ah;
	op->dst_qpn = qpn;
	op->qkey = qkey;
}

static void sd_wr_set_sge_list(struct ibv_qp_ex *ex, size_t n, const struct ibv_sge *sg)
{
	struct sd_qp *qp = qp_of_ex(ex);
	struct sd_op *op = batch_last(qp);
	size_t i;

	if (!op || qp->batch_err)
		return;
	if (n > qp->cap.max_send_sge || n > SD_MAX_SGE_ANY) {
		qp->batch_err = EINVAL;
		return;
	}
	op->nsge = (int)n;
	op->len = 0;
	for (i = 0; i < n; i++) {
		op->sge[i] = sg[i];
		op->len += sg[i].length;
	}
}

static void sd_wr_set_sge(struct ibv_qp_ex *ex, uint32_t lkey, uint64_t addr, uint32_t len)
{
	struct ibv_sge sg = { .addr = addr, .length = len, .lkey = lkey };

	sd_wr_set_sge_list(ex, 1, &sg);
}

static void sd_wr_set_inline_data_list(struct ibv_qp_ex *ex, size_t n, const struct ibv_data_buf *b)
{
	struct sd_qp *qp = qp_of_ex(ex);
	struct sd_op *op = batch_last(qp);
	size_t i, total = 0;

	if (!op || qp->batch_err)
		return;
	for (i = 0; i < n; i++)
		total += b[i].length;
	if (total > SD_EFA_INLINE_BUF_SIZE) {
		qp->batch_err = EINVAL;
		return;
	}
	op->inline_data = malloc(total ? total : 1);
	if (!op->inline_data) {
		qp->batch_err = ENOMEM;
		return;
	}
	op->len = 0;
	for (i = 0; i < n; i++) {
		memcpy(op->inline_data + op->len, b[i].addr, b[i].length);
		op->len += b[i].length;
	}
}

static void sd_wr_set_inline_data(struct ibv_qp_ex *ex, void *addr, size_t len)
{
	struct ibv_data_buf b = { .addr = addr, .length = len };

	sd_wr_set_inline_data_list(ex, 1, &b);
}

static void sd_wr_abort(struct ibv_qp_ex *ex)
{
	struct sd_qp *qp = qp_of_ex(ex);

	while (qp->batch) {
		struct sd_op *op = qp->batch;

		qp->batch = op->next;
		free_op(op);
	}
	sd_wr_start(ex);
}

static void tx_append(struct sd_context *ctx, struct sd_op *first);

static int sd_wr_complete(struct ibv_qp_ex *ex)
{
	struct sd_qp *qp = qp_of_ex(ex);
	struct sd_context *ctx = sd_ctx(ex->qp_base.context);
	int err = qp->batch_err;

	if (!err && ex->qp_base.state != IBV_QPS_RTS && ex->qp_base.state != IBV_QPS_SQD) {
		dlog("qpn %u: posting in state %d", ex->qp_base.qp_num, ex->qp_base.state);
		err = EINVAL;
	}
	pthread_mutex_lock(&ctx->lock);
	if (!err && qp->sq_outstanding + (uint32_t)qp->batch_count > qp->cap.max_send_wr)
		err = ENOMEM;
	if (!err) {
		qp->sq_outstanding += (uint32_t)qp->batch_count;
		tx_append(ctx, qp->batch);
		qp->batch = NULL;
	}
	pthread_mutex_unlock(&ctx->lock);
	if (err) {
		dlog("wr_complete qpn %u: %d WRs refused with %d", ex->qp_base.qp_num, qp->batch_count, err);
		sd_wr_abort(ex);
		return err;
	}
	dlog("wr_complete qpn %u: %d WRs", ex->qp_base.qp_num, qp->batch_count);
	sd_wr_start(ex);
	wake(ctx);
	return 0;
}

static int sd_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad)
{
	(void)qp;
	/* The efa provider posts sends through ibv_wr_*; this path is not used. */
	if (bad)
		*bad = wr;
	return EOPNOTSUPP;
}

static int sd_post_recv(struct ibv_qp *ibqp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad)
{
	struct sd_qp *qp = qp_of(ibqp);
	struct sd_context *ctx = sd_ctx(ibqp->context);
	int err = 0, i;

	pthread_mutex_lock(&ctx->lock);
	for (; wr; wr = wr->next) {
		struct sd_recv *r;

		if (wr->num_sge < 0 || (uint32_t)wr->num_sge > qp->cap.max_recv_sge ||
		    wr->num_sge > SD_MAX_SGE_ANY) {
			err = EINVAL;
			break;
		}
		if (qp->rq_count >= qp->cap.max_recv_wr) {
			err = ENOMEM;
			break;
		}
		r = calloc(1, sizeof(*r));
		if (!r) {
			err = ENOMEM;
			break;
		}
		r->wr_id = wr->wr_id;
		r->nsge = wr->num_sge;
		for (i = 0; i < wr->num_sge; i++)
			r->sge[i] = wr->sg_list[i];
		*qp->rq_tail = r;
		qp->rq_tail = &r->next;
		qp->rq_count++;
	}
	pthread_mutex_unlock(&ctx->lock);
	if (sd_log_enabled())
		dlog("post_recv qpn %u -> %d, %u posted", ibqp->qp_num, err, qp->rq_count);
	if (err && bad)
		*bad = wr;
	if (qp->held)
		wake(ctx);
	return err;
}

static socklen_t qp_name(int device, uint32_t qpn, struct sockaddr_un *a)
{
	int n;

	memset(a, 0, sizeof(*a));
	a->sun_family = AF_UNIX;
	/* Abstract namespace: sun_path[0] is NUL, the name follows, with no trailing NUL. */
	n = snprintf(a->sun_path + 1, sizeof(a->sun_path) - 1, "srdsim-device-%d-qp-%u", device, qpn);
	return (socklen_t)(offsetof(struct sockaddr_un, sun_path) + 1 + (size_t)n);
}

static int agent_watch(struct sd_context *ctx, int fd, bool add);

/*
 * A socket bound to the first free QP number at or after a start that differs between processes. The
 * efa provider carries a QP number in 16 bits of its address, as EFA's QP numbers fit, so these do too.
 */
#define SD_QPN_MAX 0xffffu

static int bind_qp(int device, uint32_t *qpn)
{
	int fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
	uint32_t start = ((uint32_t)getpid() * 2654435761u) % SD_QPN_MAX, i;
	int buf = 4 << 20;

	if (fd < 0)
		return -1;
	setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof(buf));
	setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof(buf));
	for (i = 0; i < SD_QPN_MAX; i++) {
		struct sockaddr_un a;
		uint32_t n = (start + i) % SD_QPN_MAX + 1;
		socklen_t len = qp_name(device, n, &a);

		if (!bind(fd, (struct sockaddr *)&a, len)) {
			*qpn = n;
			return fd;
		}
		if (errno != EADDRINUSE)
			break;
	}
	close(fd);
	return -1;
}

static struct ibv_qp *create_qp(struct ibv_context *c, struct ibv_qp_init_attr_ex *attr, bool unsolicited)
{
	struct sd_context *ctx = sd_ctx(c);
	struct sd_qp *qp;
	uint32_t qpn;

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
	qp->fd = bind_qp(ctx->device->index, &qpn);
	if (qp->fd < 0) {
		int err = errno;

		free(qp);
		errno = err ? err : EBUSY;
		return NULL;
	}
	qp->cap = attr->cap;
	qp->sq_sig_all = attr->sq_sig_all;
	qp->unsolicited_write_recv = unsolicited;
	qp->rq_tail = &qp->rq;
	qp->held_tail = &qp->held;
	qp->ex.qp_base.context = c;
	qp->ex.qp_base.qp_context = attr->qp_context;
	qp->ex.qp_base.pd = attr->pd;
	qp->ex.qp_base.send_cq = attr->send_cq;
	qp->ex.qp_base.recv_cq = attr->recv_cq;
	qp->ex.qp_base.srq = attr->srq;
	qp->ex.qp_base.qp_type = attr->qp_type;
	qp->ex.qp_base.state = IBV_QPS_RESET;
	qp->ex.qp_base.qp_num = qpn;
	qp->ex.qp_base.handle = qpn;
	pthread_mutex_init(&qp->ex.qp_base.mutex, NULL);
	pthread_cond_init(&qp->ex.qp_base.cond, NULL);
	qp->ex.comp_mask = attr->comp_mask;
	qp->ex.wr_start = sd_wr_start;
	qp->ex.wr_complete = sd_wr_complete;
	qp->ex.wr_abort = sd_wr_abort;
	qp->ex.wr_send = sd_wr_send;
	qp->ex.wr_send_imm = sd_wr_send_imm;
	qp->ex.wr_rdma_read = sd_wr_rdma_read;
	qp->ex.wr_rdma_write = sd_wr_rdma_write;
	qp->ex.wr_rdma_write_imm = sd_wr_rdma_write_imm;
	qp->ex.wr_set_ud_addr = sd_wr_set_ud_addr;
	qp->ex.wr_set_sge = sd_wr_set_sge;
	qp->ex.wr_set_sge_list = sd_wr_set_sge_list;
	qp->ex.wr_set_inline_data = sd_wr_set_inline_data;
	qp->ex.wr_set_inline_data_list = sd_wr_set_inline_data_list;
	sd_wr_start(&qp->ex);

	use(&((struct sd_pd *)attr->pd)->users, 1);
	if (attr->send_cq)
		use(&((struct sd_cq *)attr->send_cq)->users, 1);
	if (attr->recv_cq)
		use(&((struct sd_cq *)attr->recv_cq)->users, 1);
	pthread_mutex_lock(&ctx->lock);
	qp->next = ctx->qps;
	ctx->qps = qp;
	agent_watch(ctx, qp->fd, true);
	pthread_mutex_unlock(&ctx->lock);
	dlog("create_qp type %d send_wr %u recv_wr %u sig_all %d unsolicited %d -> qpn %u",
	     attr->qp_type, attr->cap.max_send_wr, attr->cap.max_recv_wr, attr->sq_sig_all,
	     unsolicited, qpn);
	return &qp->ex.qp_base;
}

static struct ibv_qp *sd_create_qp_ex(struct ibv_context *c, struct ibv_qp_init_attr_ex *attr)
{
	return create_qp(c, attr, false);
}

EXPORT struct ibv_qp *efadv_create_qp_ex(struct ibv_context *c, struct ibv_qp_init_attr_ex *attr,
					 struct efadv_qp_init_attr *efa_attr, uint32_t inlen)
{
	bool unsolicited = false;

	if (efa_attr && inlen >= offsetof(struct efadv_qp_init_attr, driver_qp_type) +
				 sizeof(efa_attr->driver_qp_type) &&
	    efa_attr->driver_qp_type != EFADV_QP_DRIVER_TYPE_SRD) {
		errno = EOPNOTSUPP;
		return NULL;
	}
#if SD_HAVE_EFADV_QP_FLAGS
	if (efa_attr && inlen >= offsetof(struct efadv_qp_init_attr, flags) + sizeof(efa_attr->flags))
		unsolicited = efa_attr->flags & EFADV_QP_FLAGS_UNSOLICITED_WRITE_RECV;
#endif
	return create_qp(c, attr, unsolicited);
}

EXPORT struct ibv_qp_ex *ibv_qp_to_qp_ex(struct ibv_qp *qp)
{
	return &qp_of(qp)->ex;
}

EXPORT int ibv_modify_qp(struct ibv_qp *ibqp, struct ibv_qp_attr *attr, int mask)
{
	struct sd_qp *qp = qp_of(ibqp);
	struct sd_context *ctx = sd_ctx(ibqp->context);

	pthread_mutex_lock(&ctx->lock);
	if (mask & IBV_QP_QKEY)
		qp->qkey = attr->qkey;
	if (mask & IBV_QP_STATE)
		ibqp->state = attr->qp_state;
	pthread_mutex_unlock(&ctx->lock);
	dlog("modify_qp qpn %u mask 0x%x state %d qkey 0x%x", ibqp->qp_num, mask, ibqp->state, qp->qkey);
	return 0;
}

static void tx_forget_qp(struct sd_context *ctx, struct sd_qp *qp);

EXPORT int ibv_destroy_qp(struct ibv_qp *ibqp)
{
	struct sd_qp *qp = qp_of(ibqp);
	struct sd_context *ctx = sd_ctx(ibqp->context);
	struct sd_qp **p;

	dlog("destroy_qp qpn %u", ibqp->qp_num);
	pthread_mutex_lock(&ctx->lock);
	for (p = &ctx->qps; *p; p = &(*p)->next)
		if (*p == qp) {
			*p = qp->next;
			break;
		}
	agent_watch(ctx, qp->fd, false);
	tx_forget_qp(ctx, qp);
	pthread_mutex_unlock(&ctx->lock);
	close(qp->fd);
	while (qp->rq) {
		struct sd_recv *r = qp->rq;

		qp->rq = r->next;
		free(r);
	}
	while (qp->held) {
		struct sd_held *h = qp->held;

		qp->held = h->next;
		free(h);
	}
	sd_wr_abort(&qp->ex);
	use(&((struct sd_pd *)ibqp->pd)->users, -1);
	if (ibqp->send_cq)
		use(&((struct sd_cq *)ibqp->send_cq)->users, -1);
	if (ibqp->recv_cq)
		use(&((struct sd_cq *)ibqp->recv_cq)->users, -1);
	pthread_mutex_destroy(&ibqp->mutex);
	pthread_cond_destroy(&ibqp->cond);
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

/* ---- frames and the agent --------------------------------------------- */

#define SD_MAGIC 0x53524446u		/* "SRDF" */

enum sd_frame_type {
	SD_F_SEND = 1,		/* a whole message */
	SD_F_WRITE,		/* a chunk of an RDMA write */
	SD_F_READ_REQ,		/* a read of [raddr, raddr + total) under rkey */
	SD_F_READ_RESP,		/* a chunk of the answer to a read, or its status */
	SD_F_ACK,		/* the target finished a send or a write; status says how */
};

#define SD_FF_IMM 1u

struct sd_frame {
	uint32_t magic;
	uint8_t type, flags;
	uint16_t status;
	uint32_t src_dev, src_qpn, dst_qpn, qkey, rkey, imm, len;
	uint64_t op_id, raddr, total, offset;
	unsigned char data[];
};

#define SD_FRAME_MAX (sizeof(struct sd_frame) + (SD_CHUNK > SD_PORT_MAX_MSG_SZ ? SD_CHUNK : SD_PORT_MAX_MSG_SZ))

enum sd_tx_kind { SD_TX_OP, SD_TX_ACK, SD_TX_RESP };

/* Something the agent has to send: the frames of an op, an acknowledgement, or the answer to a read. */
struct sd_tx {
	struct sd_tx *next;
	enum sd_tx_kind kind;
	struct sd_qp *from;
	struct sd_op *op;		/* SD_TX_OP */
	int dev;			/* SD_TX_ACK and SD_TX_RESP: where to */
	uint32_t dst_qpn;
	uint64_t op_id;
	uint16_t status;
	const unsigned char *src;	/* SD_TX_RESP: the memory read */
	uint64_t total, sent;
};

/* An RDMA write arriving at this context, until its last byte. */
struct sd_rxw {
	struct sd_rxw *next;
	uint32_t src_dev, src_qpn;
	uint64_t op_id, got;
	uint16_t status;
};

struct sd_agent {
	pthread_t thread;
	int epfd, wakefd;
	bool stop;
	struct sd_tx *tx, **tx_tail;
	struct sd_op *inflight;		/* sent, waiting for their answer */
	struct sd_rxw *rxw;
	uint64_t next_op_id;
	unsigned char buf[SD_FRAME_MAX];
};

static void wake(struct sd_context *ctx)
{
	uint64_t one = 1;

	if (write(ctx->agent->wakefd, &one, sizeof(one)) < 0 && errno != EAGAIN)
		dlog("wake: %s", strerror(errno));
}

static int agent_watch(struct sd_context *ctx, int fd, bool add)
{
	struct epoll_event ev = { .events = EPOLLIN, .data.fd = fd };

	return epoll_ctl(ctx->agent->epfd, add ? EPOLL_CTL_ADD : EPOLL_CTL_DEL, fd, &ev);
}

static void tx_push(struct sd_agent *ag, struct sd_tx *t)
{
	t->next = NULL;
	*ag->tx_tail = t;
	ag->tx_tail = &t->next;
}

/* Under ctx->lock. */
static void tx_append(struct sd_context *ctx, struct sd_op *op)
{
	struct sd_agent *ag = ctx->agent;

	while (op) {
		struct sd_op *next = op->next;
		struct sd_tx *t = calloc(1, sizeof(*t));

		op->next = NULL;
		op->id = ++ag->next_op_id;
		if (!t) {
			free_op(op);	/* never completes; the caller's queue depth stays taken */
			op = next;
			continue;
		}
		t->kind = SD_TX_OP;
		t->from = op->qp;
		t->op = op;
		tx_push(ag, t);
		op = next;
	}
}

static void complete_op(struct sd_op *op, enum sd_status st)
{
	struct sd_cqe e = {
		.wr_id = op->wr_id,
		.status = to_ibv_status(st),
		.vendor_err = st,
		/* as libefa: only a write reads as a write; a read reads as a send */
		.opcode = op->kind == SD_OP_WRITE ? IBV_WC_RDMA_WRITE : IBV_WC_SEND,
		.qp_num = op->qp->ex.qp_base.qp_num,
		.slid = 0xffff,
	};

	cq_push(op->qp->ex.qp_base.send_cq, &e);
	op->qp->sq_outstanding--;
	free_op(op);
}

/* Under ctx->lock: drop everything queued for or waiting on qp, which is going away. */
static void tx_forget_qp(struct sd_context *ctx, struct sd_qp *qp)
{
	struct sd_agent *ag = ctx->agent;
	struct sd_tx **t = &ag->tx;
	struct sd_op **o = &ag->inflight;

	while (*t) {
		if ((*t)->from == qp) {
			struct sd_tx *x = *t;

			*t = x->next;
			if (x->op)
				free_op(x->op);
			free(x);
		} else {
			t = &(*t)->next;
		}
	}
	ag->tx_tail = &ag->tx;
	while (*ag->tx_tail)
		ag->tx_tail = &(*ag->tx_tail)->next;
	while (*o) {
		if ((*o)->qp == qp) {
			struct sd_op *x = *o;

			*o = x->next;
			free_op(x);
		} else {
			o = &(*o)->next;
		}
	}
}

/* Copies len bytes at offset off of the buffer an SGE list describes, in either direction. */
static void sge_copy(const struct ibv_sge *sge, int n, uint64_t off, void *buf, uint64_t len, bool into_sges)
{
	unsigned char *b = buf;
	int i;

	for (i = 0; i < n && len; i++) {
		uint64_t l = sge[i].length, k;

		if (off >= l) {
			off -= l;
			continue;
		}
		k = l - off < len ? l - off : len;
		if (into_sges)
			memcpy((unsigned char *)(uintptr_t)sge[i].addr + off, b, k);
		else
			memcpy(b, (unsigned char *)(uintptr_t)sge[i].addr + off, k);
		b += k;
		len -= k;
		off = 0;
	}
}

/* Under ctx->lock: whether every SGE lies inside a local MR of this context. */
static bool sges_valid(struct sd_context *ctx, const struct ibv_sge *sge, int n)
{
	int i;

	for (i = 0; i < n; i++) {
		struct sd_mr *m = sd_mr_find(ctx, sge[i].lkey);
		uintptr_t base;

		if (!m)
			return false;
		base = (uintptr_t)m->ibv_mr.addr;
		if (sge[i].addr < base || sge[i].addr + sge[i].length > base + m->ibv_mr.length)
			return false;
	}
	return true;
}

/* Under ctx->lock: the local address of [raddr, raddr + len) under rkey with access, or NULL. */
static unsigned char *remote_range(struct sd_context *ctx, uint32_t rkey, uint64_t raddr, uint64_t len,
				   int access)
{
	struct sd_mr *m = sd_mr_find(ctx, rkey);

	if (!m || !(m->access & access) || raddr < m->iova || len > m->ibv_mr.length ||
	    raddr - m->iova > m->ibv_mr.length - len)
		return NULL;
	return (unsigned char *)m->ibv_mr.addr + (raddr - m->iova);
}

/* Sends one frame from qp's socket; 0, EAGAIN, or the status the op ends with. */
static int send_frame(struct sd_qp *from, int dev, uint32_t dst_qpn, struct sd_frame *f, size_t size)
{
	struct sockaddr_un a;
	socklen_t alen = qp_name(dev, dst_qpn, &a);

	if (sendto(from->fd, f, size, 0, (struct sockaddr *)&a, alen) == (ssize_t)size)
		return 0;
	if (errno == EAGAIN || errno == EWOULDBLOCK || errno == ENOBUFS)
		return EAGAIN;
	dlog("qpn %u -> %d/%u: %s", from->ex.qp_base.qp_num, dev, dst_qpn, strerror(errno));
	/* assumed: no QP of that number answers */
	return errno == ECONNREFUSED || errno == ENOENT ? SD_ST_REMOTE_BAD_DEST_QPN : SD_ST_LOCAL_UNREACH_REMOTE;
}

static void frame_init(struct sd_frame *f, struct sd_context *ctx, struct sd_qp *from, uint8_t type,
		       uint32_t dst_qpn, uint64_t op_id)
{
	memset(f, 0, sizeof(*f));
	f->magic = SD_MAGIC;
	f->type = type;
	f->src_dev = (uint32_t)ctx->device->index;
	f->src_qpn = from->ex.qp_base.qp_num;
	f->dst_qpn = dst_qpn;
	f->op_id = op_id;
}

/*
 * Under ctx->lock: sends what remains of an op. 0 when its frames are all out (the op then waits in
 * inflight for its answer, or has completed with an error), EAGAIN when the destination is full.
 */
static int pump_op(struct sd_context *ctx, struct sd_op *op)
{
	struct sd_agent *ag = ctx->agent;
	struct sd_frame *f = (struct sd_frame *)ag->buf;
	int dev, st;

	if (!op->checked) {
		op->checked = true;
		if (!op->ah) {
			complete_op(op, SD_ST_LOCAL_INVALID_AH);
			return 0;
		}
		if ((op->kind == SD_OP_SEND && op->len > SD_PORT_MAX_MSG_SZ) ||
		    (op->kind != SD_OP_SEND && op->len > SD_EFA_MAX_RDMA_SIZE)) {
			complete_op(op, SD_ST_LOCAL_BAD_LENGTH);
			return 0;
		}
		if (!op->inline_data && !sges_valid(ctx, op->sge, op->nsge)) {
			complete_op(op, SD_ST_LOCAL_INVALID_LKEY);
			return 0;
		}
	}
	dev = sd_gid_device(&op->ah->gid);
	do {
		uint64_t n = 0;

		switch (op->kind) {
		case SD_OP_SEND:
			frame_init(f, ctx, op->qp, SD_F_SEND, op->dst_qpn, op->id);
			n = op->len;
			break;
		case SD_OP_WRITE:
			frame_init(f, ctx, op->qp, SD_F_WRITE, op->dst_qpn, op->id);
			n = op->len - op->sent < SD_CHUNK ? op->len - op->sent : SD_CHUNK;
			break;
		case SD_OP_READ:
			frame_init(f, ctx, op->qp, SD_F_READ_REQ, op->dst_qpn, op->id);
			break;
		}
		f->qkey = op->qkey;
		f->rkey = op->rkey;
		f->raddr = op->raddr + op->sent;
		f->total = op->len;
		f->offset = op->sent;
		f->len = (uint32_t)n;
		if (op->has_imm) {
			f->flags |= SD_FF_IMM;
			f->imm = op->imm;
		}
		/* The sender's memory is read now, when the frame leaves, not when the WR was posted. */
		if (op->inline_data)
			memcpy(f->data, op->inline_data + op->sent, n);
		else
			sge_copy(op->sge, op->nsge, op->sent, f->data, n, false);
		st = send_frame(op->qp, dev, op->dst_qpn, f, sizeof(*f) + n);
		if (st == EAGAIN)
			return EAGAIN;
		if (st) {
			complete_op(op, (enum sd_status)st);
			return 0;
		}
		op->sent += n;
	} while (op->kind == SD_OP_WRITE && op->sent < op->len);
	op->next = ag->inflight;
	ag->inflight = op;
	return 0;
}

static int pump_reply(struct sd_context *ctx, struct sd_tx *t)
{
	struct sd_frame *f = (struct sd_frame *)ctx->agent->buf;
	int st;

	do {
		uint64_t n = 0;

		frame_init(f, ctx, t->from, t->kind == SD_TX_ACK ? SD_F_ACK : SD_F_READ_RESP, t->dst_qpn,
			   t->op_id);
		f->status = t->status;
		f->total = t->total;
		f->offset = t->sent;
		if (t->kind == SD_TX_RESP && !t->status) {
			n = t->total - t->sent < SD_CHUNK ? t->total - t->sent : SD_CHUNK;
			memcpy(f->data, t->src + t->sent, n);	/* the target's memory, read now */
		}
		f->len = (uint32_t)n;
		st = send_frame(t->from, t->dev, t->dst_qpn, f, sizeof(*f) + n);
		if (st == EAGAIN)
			return EAGAIN;
		if (st)
			return 0;	/* the requester is gone; nothing waits for this answer */
		t->sent += n;
	} while (t->kind == SD_TX_RESP && !t->status && t->sent < t->total);
	return 0;
}

/* Under ctx->lock: sends queued work in order until a destination is full. Whether it stopped early. */
static bool pump(struct sd_context *ctx)
{
	struct sd_agent *ag = ctx->agent;

	while (ag->tx) {
		struct sd_tx *t = ag->tx;
		int r = t->kind == SD_TX_OP ? pump_op(ctx, t->op) : pump_reply(ctx, t);

		if (r == EAGAIN)
			return true;
		ag->tx = t->next;
		if (!ag->tx)
			ag->tx_tail = &ag->tx;
		free(t);
	}
	return false;
}

static void reply(struct sd_context *ctx, struct sd_qp *from, const struct sd_frame *req, enum sd_tx_kind kind,
		  enum sd_status st, const unsigned char *src, uint64_t total)
{
	struct sd_tx *t = calloc(1, sizeof(*t));

	if (!t) {
		dlog("no memory for an answer to qpn %u", req->src_qpn);
		return;
	}
	t->kind = kind;
	t->from = from;
	t->dev = (int)req->src_dev;
	t->dst_qpn = req->src_qpn;
	t->op_id = req->op_id;
	t->status = (uint16_t)st;
	t->src = src;
	t->total = total;
	tx_push(ctx->agent, t);
}

#define SD_FF_WRITE_DONE 2u	/* on a held frame: the write has landed and waits for a receive */

static struct sd_op *inflight_take(struct sd_agent *ag, struct sd_qp *qp, uint64_t op_id)
{
	struct sd_op **p;

	for (p = &ag->inflight; *p; p = &(*p)->next)
		if ((*p)->qp == qp && (*p)->id == op_id) {
			struct sd_op *op = *p;

			*p = op->next;
			op->next = NULL;
			return op;
		}
	return NULL;
}

static struct sd_recv *rq_pop(struct sd_qp *qp)
{
	struct sd_recv *r = qp->rq;

	if (r) {
		qp->rq = r->next;
		if (!qp->rq)
			qp->rq_tail = &qp->rq;
		qp->rq_count--;
	}
	return r;
}

static void recv_cqe(struct sd_context *ctx, struct sd_qp *qp, const struct sd_frame *f, uint64_t wr_id,
		     enum sd_status st, enum ibv_wc_opcode opcode, uint32_t len, bool unsolicited)
{
	struct sd_cqe e = {
		.wr_id = wr_id,
		.status = to_ibv_status(st),
		.vendor_err = st,
		.opcode = opcode,
		.byte_len = len,
		.has_imm = f->flags & SD_FF_IMM,
		.imm = f->imm,
		.unsolicited = unsolicited,
		.qp_num = qp->ex.qp_base.qp_num,
		.src_qp = f->src_qpn,
	};

	sd_gid_of((int)f->src_dev, &e.sgid);
	e.slid = sd_ahn_of(ctx, &e.sgid);
	cq_push(qp->ex.qp_base.recv_cq, &e);
}

static void deliver_send(struct sd_context *ctx, struct sd_qp *qp, const struct sd_frame *f, struct sd_recv *r)
{
	uint64_t cap = 0;
	int i;

	for (i = 0; i < r->nsge; i++)
		cap += r->sge[i].length;
	if (f->len > cap) {
		recv_cqe(ctx, qp, f, r->wr_id, SD_ST_LOCAL_BAD_LENGTH, IBV_WC_RECV, f->len, false);
		reply(ctx, qp, f, SD_TX_ACK, SD_ST_REMOTE_BAD_LENGTH, NULL, 0);
	} else if (!sges_valid(ctx, r->sge, r->nsge)) {
		recv_cqe(ctx, qp, f, r->wr_id, SD_ST_LOCAL_INVALID_LKEY, IBV_WC_RECV, f->len, false);
		reply(ctx, qp, f, SD_TX_ACK, SD_ST_REMOTE_ABORT, NULL, 0);	/* assumed */
	} else {
		sge_copy(r->sge, r->nsge, 0, (void *)f->data, f->len, true);
		recv_cqe(ctx, qp, f, r->wr_id, SD_ST_OK, IBV_WC_RECV, f->len, false);
		reply(ctx, qp, f, SD_TX_ACK, SD_ST_OK, NULL, 0);
	}
	free(r);
}

static void deliver_write_imm(struct sd_context *ctx, struct sd_qp *qp, const struct sd_frame *f, struct sd_recv *r)
{
	recv_cqe(ctx, qp, f, r ? r->wr_id : 0, SD_ST_OK, IBV_WC_RECV_RDMA_WITH_IMM, (uint32_t)f->total, !r);
	reply(ctx, qp, f, SD_TX_ACK, SD_ST_OK, NULL, 0);
	free(r);
}

static void hold(struct sd_qp *qp, const struct sd_frame *f, size_t size, uint8_t extra_flags)
{
	struct sd_held *h = malloc(sizeof(*h) + size);

	if (!h) {
		dlog("qpn %u: no memory to hold a frame", qp->ex.qp_base.qp_num);
		return;
	}
	h->next = NULL;
	h->size = size;
	memcpy(h->frame, f, size);
	((struct sd_frame *)h->frame)->flags |= extra_flags;
	*qp->held_tail = h;
	qp->held_tail = &h->next;
}

/* Under ctx->lock: hands held frames to receives posted since, in arrival order. */
static void retry_held(struct sd_context *ctx)
{
	struct sd_qp *qp;

	for (qp = ctx->qps; qp; qp = qp->next) {
		while (qp->held && qp->rq) {
			struct sd_held *h = qp->held;
			struct sd_frame *f = (struct sd_frame *)h->frame;

			qp->held = h->next;
			if (!qp->held)
				qp->held_tail = &qp->held;
			if (f->flags & SD_FF_WRITE_DONE)
				deliver_write_imm(ctx, qp, f, rq_pop(qp));
			else
				deliver_send(ctx, qp, f, rq_pop(qp));
			free(h);
		}
	}
}

static struct sd_rxw *rxw_get(struct sd_agent *ag, const struct sd_frame *f, bool take)
{
	struct sd_rxw **p, *w;

	for (p = &ag->rxw; *p; p = &(*p)->next) {
		w = *p;
		if (w->src_dev == f->src_dev && w->src_qpn == f->src_qpn && w->op_id == f->op_id) {
			if (take)
				*p = w->next;
			return w;
		}
	}
	if (take)
		return NULL;
	w = calloc(1, sizeof(*w));
	if (!w)
		return NULL;
	w->src_dev = f->src_dev;
	w->src_qpn = f->src_qpn;
	w->op_id = f->op_id;
	w->next = ag->rxw;
	ag->rxw = w;
	return w;
}

/* Under ctx->lock: one frame that arrived on qp's socket. */
static void handle(struct sd_context *ctx, struct sd_qp *qp, struct sd_frame *f, size_t size)
{
	struct sd_agent *ag = ctx->agent;
	struct sd_op *op;

	if (size < sizeof(*f) || f->magic != SD_MAGIC || f->len > size - sizeof(*f)) {
		dlog("qpn %u: malformed frame of %zu bytes dropped", qp->ex.qp_base.qp_num, size);
		return;
	}
	dlog("frame type %u %u/%u -> qpn %u op %llu len %u total %llu status %u", f->type, f->src_dev,
	     f->src_qpn, qp->ex.qp_base.qp_num, (unsigned long long)f->op_id, f->len,
	     (unsigned long long)f->total, f->status);
	switch (f->type) {
	case SD_F_SEND:
		if (qp->ex.qp_base.state < IBV_QPS_RTR || f->qkey != qp->qkey) {
			reply(ctx, qp, f, SD_TX_ACK, SD_ST_REMOTE_BAD_DEST_QPN, NULL, 0);	/* assumed */
			return;
		}
		if (qp->held || !qp->rq)
			hold(qp, f, size, 0);	/* assumed: the device retries until a receive is posted */
		else
			deliver_send(ctx, qp, f, rq_pop(qp));
		return;
	case SD_F_WRITE: {
		struct sd_rxw *w = rxw_get(ag, f, false);
		uint16_t st;

		if (!w)
			return;
		if (!w->status) {
			unsigned char *dst = remote_range(ctx, f->rkey, f->raddr, f->len, IBV_ACCESS_REMOTE_WRITE);

			if (dst)
				memcpy(dst, f->data, f->len);
			else
				w->status = SD_ST_REMOTE_BAD_ADDRESS;
		}
		w->got += f->len;
		if (w->got < f->total)
			return;
		st = w->status;
		rxw_get(ag, f, true);
		free(w);
		if (st || !(f->flags & SD_FF_IMM))
			reply(ctx, qp, f, SD_TX_ACK, (enum sd_status)st, NULL, 0);
		else if (qp->unsolicited_write_recv)
			deliver_write_imm(ctx, qp, f, NULL);
		else if (qp->held || !qp->rq)
			hold(qp, f, sizeof(*f), SD_FF_WRITE_DONE);
		else
			deliver_write_imm(ctx, qp, f, rq_pop(qp));
		return;
	}
	case SD_F_READ_REQ: {
		unsigned char *src = remote_range(ctx, f->rkey, f->raddr, f->total, IBV_ACCESS_REMOTE_READ);

		reply(ctx, qp, f, SD_TX_RESP, src ? SD_ST_OK : SD_ST_REMOTE_BAD_ADDRESS, src, src ? f->total : 0);
		return;
	}
	case SD_F_READ_RESP:
		op = inflight_take(ag, qp, f->op_id);
		if (!op)
			return;
		if (f->status) {
			complete_op(op, (enum sd_status)f->status);
			return;
		}
		sge_copy(op->sge, op->nsge, f->offset, f->data, f->len, true);
		op->received += f->len;
		if (op->received >= op->len) {
			complete_op(op, SD_ST_OK);
			return;
		}
		op->next = ag->inflight;	/* more to come */
		ag->inflight = op;
		return;
	case SD_F_ACK:
		op = inflight_take(ag, qp, f->op_id);
		if (op)
			complete_op(op, (enum sd_status)f->status);
		return;
	}
	dlog("qpn %u: frame of unknown type %u dropped", qp->ex.qp_base.qp_num, f->type);
}

static struct sd_qp *qp_by_fd(struct sd_context *ctx, int fd)
{
	struct sd_qp *qp;

	for (qp = ctx->qps; qp; qp = qp->next)
		if (qp->fd == fd)
			return qp;
	return NULL;
}

static void *agent_main(void *arg)
{
	struct sd_context *ctx = arg;
	struct sd_agent *ag = ctx->agent;
	bool blocked = false;

	for (;;) {
		struct epoll_event ev[16];
		int n = epoll_wait(ag->epfd, ev, 16, blocked ? 1 : -1), i;

		pthread_mutex_lock(&ctx->lock);
		if (ag->stop) {
			pthread_mutex_unlock(&ctx->lock);
			return NULL;
		}
		for (i = 0; i < n; i++) {
			struct sd_qp *qp;
			ssize_t got;

			if (ev[i].data.fd == ag->wakefd) {
				uint64_t v;

				if (read(ag->wakefd, &v, sizeof(v)) < 0 && errno != EAGAIN)
					dlog("wake read: %s", strerror(errno));
				continue;
			}
			qp = qp_by_fd(ctx, ev[i].data.fd);
			if (!qp)
				continue;
			while ((got = recv(qp->fd, ag->buf, sizeof(ag->buf), 0)) >= 0)
				handle(ctx, qp, (struct sd_frame *)ag->buf, (size_t)got);
		}
		retry_held(ctx);
		blocked = pump(ctx);
		pthread_mutex_unlock(&ctx->lock);
	}
}

int sd_agent_start(struct sd_context *ctx)
{
	struct sd_agent *ag = calloc(1, sizeof(*ag));
	struct epoll_event ev = { .events = EPOLLIN };
	struct ibv_context *c = &ctx->vctx.context;

	if (!ag) {
		errno = ENOMEM;
		return -1;
	}
	ag->tx_tail = &ag->tx;
	ag->epfd = epoll_create1(EPOLL_CLOEXEC);
	ag->wakefd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
	ev.data.fd = ag->wakefd;
	if (ag->epfd < 0 || ag->wakefd < 0 || epoll_ctl(ag->epfd, EPOLL_CTL_ADD, ag->wakefd, &ev)) {
		int err = errno;

		if (ag->epfd >= 0)
			close(ag->epfd);
		if (ag->wakefd >= 0)
			close(ag->wakefd);
		free(ag);
		errno = err;
		return -1;
	}
	ctx->agent = ag;
	ctx->vctx.create_cq_ex = sd_create_cq_ex;
	ctx->vctx.create_qp_ex = sd_create_qp_ex;
	c->ops.poll_cq = sd_poll_cq;
	c->ops.req_notify_cq = sd_req_notify_cq;
	c->ops.post_send = sd_post_send;
	c->ops.post_recv = sd_post_recv;
	if (pthread_create(&ag->thread, NULL, agent_main, ctx)) {
		close(ag->epfd);
		close(ag->wakefd);
		free(ag);
		ctx->agent = NULL;
		errno = EAGAIN;
		return -1;
	}
	return 0;
}

void sd_agent_stop(struct sd_context *ctx)
{
	struct sd_agent *ag = ctx->agent;

	pthread_mutex_lock(&ctx->lock);
	ag->stop = true;
	pthread_mutex_unlock(&ctx->lock);
	wake(ctx);
	pthread_join(ag->thread, NULL);
	while (ag->tx) {
		struct sd_tx *t = ag->tx;

		ag->tx = t->next;
		if (t->op)
			free_op(t->op);
		free(t);
	}
	while (ag->inflight) {
		struct sd_op *op = ag->inflight;

		ag->inflight = op->next;
		free_op(op);
	}
	while (ag->rxw) {
		struct sd_rxw *w = ag->rxw;

		ag->rxw = w->next;
		free(w);
	}
	close(ag->epfd);
	close(ag->wakefd);
	free(ag);
	ctx->agent = NULL;
}
