/* SPDX-License-Identifier: Apache-2.0 */
/*
 * The transfer engine of an endpoint: how sends, receives, RMA writes and reads travel between endpoints on
 * one host, and in what order they take effect.
 *
 * Each endpoint listens on a TCP socket on the loopback interface. An endpoint that sends to a peer dials it
 * once, and everything it sends to that peer travels on that one stream, in the order the endpoint released
 * it. One thread per endpoint (the agent) owns every socket: it accepts, reads and applies incoming frames,
 * and writes outgoing ones, so no other thread blocks on a socket.
 *
 * Ordering is decided when an operation is released, in the endpoint's progress (fi_cq_read), and follows
 * SRD as the efa provider exposes it:
 *   - Sends and receives keep their order per peer, as the efa provider restores it in software.
 *   - An RMA write is split into chunks (SRDSIM_CHUNK_BYTES), and under SRDSIM_MODE=srd (the default) the
 *     chunks of all pending writes are released in a random order, seeded by SRDSIM_SEED. Under
 *     SRDSIM_MODE=inorder everything is released in submission order.
 *   - Without FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES, the target writes each chunk back to front, so a
 *     reader polling the end of a buffer can see it before the start; with it, front to back.
 *   - A write's remote completion with CQ data (fi_writedata) is raised only once every byte of that write
 *     has landed, as on EFA. SRDSIM_FI_CQ_DATA_EARLY=1 raises it on the first chunk instead, which breaks that
 *     guarantee on purpose, so a test can show it would see the break.
 *   - Under srd, each chunk of a write also waits at the target for a random time of up to
 *     SRDSIM_FI_JITTER_US (default 200) before it is applied, as packets that take different paths through the
 *     fabric arrive at different times. Without it, a consumer polling the target would see every chunk the
 *     sender had released at once.
 * A transmit completion is reported once the target has applied the whole operation.
 *
 * Completions are queued by the agent and written to the CQs by the endpoint's progress, on the consumer's
 * thread, so the CQs need no lock of their own.
 */
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "srdsim.h"

#define SRDSIM_DEFAULT_CHUNK 8192
#define SRDSIM_RELEASE_BUDGET 256	/* units released per progress call */

enum srdsim_frame_type {
	SRDSIM_F_HELLO = 1,	/* payload: the sender's struct srdsim_addr */
	SRDSIM_F_MSG,		/* a piece of a send, in order */
	SRDSIM_F_WRITE,		/* a chunk of an RMA write */
	SRDSIM_F_READ_REQ,	/* a read of [addr, addr + total) under key */
	SRDSIM_F_READ_RESP,	/* a chunk of the answer to a read */
	SRDSIM_F_ACK,		/* the target applied the whole operation; data holds a status */
};

#define SRDSIM_FR_TAGGED (1u << 0)
#define SRDSIM_FR_CQ_DATA (1u << 1)
#define SRDSIM_FR_IN_ORDER (1u << 2)	/* write each 128-byte block in order */

struct srdsim_frame {
	uint32_t type;
	uint32_t flags;
	uint64_t op_id;
	uint64_t offset;	/* of this piece within the operation */
	uint64_t len;		/* payload bytes that follow */
	uint64_t total;		/* bytes of the whole operation */
	uint64_t addr;		/* target address (RMA) */
	uint64_t key;		/* target key (RMA) */
	uint64_t data;		/* CQ data, or the status of an ACK */
	uint64_t tag;
};

enum srdsim_mode { SRDSIM_MODE_SRD, SRDSIM_MODE_INORDER };

struct srdsim_buf {
	char *p;
	size_t len, cap, off;	/* off: bytes already written out (outbound) */
};

/* An inbound message being received on a connection (sends arrive whole and in order). */
struct srdsim_inmsg {
	bool active;
	uint64_t op_id, total, received, tag, data;
	uint32_t flags;
	struct srdsim_recv *recv;	/* the receive it matched, or */
	struct srdsim_unexp *unexp;	/* the unexpected message it is kept in */
};

struct srdsim_conn {
	struct dlist_entry entry;
	int fd;
	bool dialed;			/* by this endpoint, to send on */
	bool peer_known;
	struct srdsim_addr peer;
	struct srdsim_buf in, out;
	struct srdsim_inmsg inmsg;
};

/* A receive the consumer posted. */
struct srdsim_recv {
	struct dlist_entry entry;
	struct iovec iov[SRDSIM_IOV_MAX];
	void *desc[SRDSIM_IOV_MAX];
	size_t iov_count, len;
	bool directed;
	struct srdsim_addr src;
	uint64_t tag, ignore;
	void *context;
	uint64_t flags;			/* FI_COMPLETION etc. */
};

/* A message that arrived before a receive matched it. */
struct srdsim_unexp {
	struct dlist_entry entry;
	struct srdsim_addr src;
	uint64_t tag, data;
	uint32_t flags;
	char *buf;
	size_t total, received;
	struct srdsim_recv *claimed;	/* a receive that matched it while it was still arriving */
};

/* An inbound RMA write, tracked until every byte of it has landed. */
struct srdsim_inwrite {
	struct dlist_entry entry;
	struct srdsim_conn *conn;
	uint64_t op_id, total, landed, data;
	uint32_t flags;
	int status;
	bool cq_data_raised;
};

enum srdsim_op_kind { SRDSIM_OP_SEND, SRDSIM_OP_WRITE, SRDSIM_OP_READ };

/* An operation this endpoint started. */
struct srdsim_op {
	struct dlist_entry entry;	/* in pending (unreleased units) or inflight */
	enum srdsim_op_kind kind;
	uint64_t id;
	struct srdsim_conn *conn;
	struct iovec iov[SRDSIM_IOV_MAX];
	void *desc[SRDSIM_IOV_MAX];
	size_t iov_count, len;
	char *staged;			/* an injected source, copied at submission */
	uint64_t addr, key, data, tag;
	uint32_t frame_flags;
	void *context;
	uint64_t cq_flags;		/* the flags of its completion */
	bool completion;		/* whether it reports one */
	size_t nchunks, next;		/* units released so far, in `order` */
	uint32_t *order;
	uint64_t landed;		/* reads: bytes of the answer applied */
};

enum srdsim_event_kind { SRDSIM_EV_TX, SRDSIM_EV_RX };

struct srdsim_event {
	struct dlist_entry entry;
	enum srdsim_event_kind kind;
	void *context;
	uint64_t flags, data, tag;
	size_t len;
	void *buf;
	int err;			/* a positive FI_ errno, or 0 */
	size_t olen;			/* bytes that did not fit (truncation) */
	bool has_src;
	struct srdsim_addr src;
};

/* A write chunk waiting at the target until `due`. */
struct srdsim_delayed {
	struct dlist_entry entry;
	struct srdsim_conn *conn;
	uint64_t due_ns;
	struct srdsim_frame f;
	char payload[];
};

struct srdsim_xfer {
	struct srdsim_ep *ep;
	pthread_mutex_t mu;
	pthread_t agent;
	bool agent_running;
	bool stop;
	int listen_fd;
	int wake_fd;
	enum srdsim_mode mode;
	size_t chunk;
	bool cq_data_early;
	size_t jitter_us;
	bool direct;			/* the efa-direct fabric: sends are not ordered either */
	bool rx_cq_data;		/* FI_RX_CQ_DATA: remote CQ data consumes a posted receive */
	struct dlist_entry cq_data_waiting;	/* remote CQ data that found no receive posted yet */
	struct dlist_entry delayed;	/* in order of `due` */
	uint64_t rng;			/* the consumer thread's choices: release order */
	uint64_t agent_rng;		/* the agent's: delays, the order of a read's answer */
	char *scratch;			/* one chunk, for the consumer thread */
	size_t outstanding;		/* operations submitted and not yet complete */
	size_t tx_size;
	uint64_t next_op_id;
	struct dlist_entry conns;
	struct dlist_entry pending;	/* ops with units still to release, in submission order */
	struct dlist_entry inflight;	/* ops released, waiting for their ACK or answer */
	struct dlist_entry recvs, trecvs;	/* posted receives, untagged and tagged */
	struct dlist_entry unexp, tunexp;	/* unexpected messages, untagged and tagged */
	struct dlist_entry inwrites;
	struct dlist_entry events;
};

/* ---- Configuration and helpers ---------------------------------------------------------------------- */

static size_t env_size(const char *name, size_t dflt, size_t lo, size_t hi)
{
	const char *v = getenv(name);
	char *end = NULL;
	unsigned long long n;

	if (!v || !*v)
		return dflt;
	errno = 0;
	n = strtoull(v, &end, 10);
	if (errno || *end || n < lo || n > hi) {
		SRDSIM_WARN(FI_LOG_EP_CTRL, "%s=%s is not between %zu and %zu; using %zu\n", name, v, lo, hi, dflt);
		return dflt;
	}
	return (size_t) n;
}

static uint64_t srdsim_rand(uint64_t *state)
{
	/* splitmix64: the same seed gives the same sequence. */
	uint64_t z = (*state += 0x9E3779B97F4A7C15ull);

	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
	return z ^ (z >> 31);
}

static bool srdsim_addr_equal(const struct srdsim_addr *a, const struct srdsim_addr *b)
{
	return a->nonce == b->nonce && a->port == b->port && a->ipv4 == b->ipv4;
}

static int srdsim_buf_append(struct srdsim_buf *b, const void *p, size_t len)
{
	char *n;
	size_t cap;

	if (b->len + len > b->cap) {
		cap = MAX(b->cap * 2, b->len + len + 4096);
		n = realloc(b->p, cap);
		if (!n)
			return -FI_ENOMEM;
		b->p = n;
		b->cap = cap;
	}
	memcpy(b->p + b->len, p, len);
	b->len += len;
	return 0;
}

static void srdsim_wake(struct srdsim_xfer *x)
{
	uint64_t one = 1;

	if (write(x->wake_fd, &one, sizeof(one)) < 0 && errno != EAGAIN)
		SRDSIM_WARN(FI_LOG_EP_CTRL, "cannot wake the agent: %s\n", strerror(errno));
}

/* Queues a frame and its payload on a connection; the agent writes it out. Called with x->mu held. */
static int srdsim_send_frame(struct srdsim_conn *c, const struct srdsim_frame *f, const void *payload)
{
	int ret = srdsim_buf_append(&c->out, f, sizeof(*f));

	if (!ret && f->len)
		ret = srdsim_buf_append(&c->out, payload, f->len);
	return ret;
}

/* Copies between a buffer of any memory kind (host or device, by its descriptor) and host memory. */
static int srdsim_copy_from(void *dst, const struct iovec *iov, void **desc, size_t count, size_t offset, size_t len)
{
	size_t i, done = 0, skip = offset, n;
	struct ofi_mr *mr;
	int ret;

	for (i = 0; i < count && done < len; i++) {
		if (skip >= iov[i].iov_len) {
			skip -= iov[i].iov_len;
			continue;
		}
		n = MIN(iov[i].iov_len - skip, len - done);
		mr = desc ? desc[i] : NULL;
		if (mr && mr->iface != FI_HMEM_SYSTEM) {
			ret = ofi_copy_from_hmem(mr->iface, mr->device, (char *) dst + done,
						 (char *) iov[i].iov_base + skip, n);
			if (ret)
				return ret;
		} else {
			memcpy((char *) dst + done, (char *) iov[i].iov_base + skip, n);
		}
		done += n;
		skip = 0;
	}
	return done == len ? 0 : -FI_ETRUNC;
}

static int srdsim_copy_to(const struct iovec *iov, void **desc, size_t count, size_t offset, const void *src,
			  size_t len)
{
	size_t i, done = 0, skip = offset, n;
	struct ofi_mr *mr;
	int ret;

	for (i = 0; i < count && done < len; i++) {
		if (skip >= iov[i].iov_len) {
			skip -= iov[i].iov_len;
			continue;
		}
		n = MIN(iov[i].iov_len - skip, len - done);
		mr = desc ? desc[i] : NULL;
		if (mr && mr->iface != FI_HMEM_SYSTEM) {
			ret = ofi_copy_to_hmem(mr->iface, mr->device, (char *) iov[i].iov_base + skip,
					       (const char *) src + done, n);
			if (ret)
				return ret;
		} else {
			memcpy((char *) iov[i].iov_base + skip, (const char *) src + done, n);
		}
		done += n;
		skip = 0;
	}
	return done == len ? 0 : -FI_ETRUNC;
}

static size_t srdsim_iov_len(const struct iovec *iov, size_t count)
{
	size_t i, n = 0;

	for (i = 0; i < count; i++)
		n += iov[i].iov_len;
	return n;
}

static void srdsim_event(struct srdsim_xfer *x, enum srdsim_event_kind kind, void *context, uint64_t flags,
			 size_t len, void *buf, uint64_t data, uint64_t tag, int err, size_t olen,
			 const struct srdsim_addr *src)
{
	struct srdsim_event *e = calloc(1, sizeof(*e));

	if (!e) {
		SRDSIM_WARN(FI_LOG_CQ, "out of memory: a completion is lost\n");
		return;
	}
	e->kind = kind;
	e->context = context;
	e->flags = flags;
	e->len = len;
	e->buf = buf;
	e->data = data;
	e->tag = tag;
	e->err = err;
	e->olen = olen;
	if (src) {
		e->has_src = true;
		e->src = *src;
	}
	dlist_insert_tail(&e->entry, &x->events);
}

/* ---- Connections ------------------------------------------------------------------------------------ */

static struct srdsim_conn *srdsim_conn_new(struct srdsim_xfer *x, int fd, bool dialed)
{
	struct srdsim_conn *c = calloc(1, sizeof(*c));
	int one = 1;

	if (!c)
		return NULL;
	c->fd = fd;
	c->dialed = dialed;
	(void) setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
	dlist_insert_tail(&c->entry, &x->conns);
	return c;
}

static void srdsim_conn_free(struct srdsim_conn *c)
{
	dlist_remove(&c->entry);
	close(c->fd);
	free(c->in.p);
	free(c->out.p);
	free(c);
}

/* The connection this endpoint sends to `peer` on, dialled on first use. Called with x->mu held. */
static struct srdsim_conn *srdsim_conn_to(struct srdsim_xfer *x, const struct srdsim_addr *peer)
{
	struct srdsim_frame hello = { .type = SRDSIM_F_HELLO, .len = sizeof(struct srdsim_addr) };
	struct sockaddr_in sin = { .sin_family = AF_INET };
	struct srdsim_conn *c;
	int fd;

	dlist_foreach_container(&x->conns, struct srdsim_conn, c, entry)
		if (c->dialed && srdsim_addr_equal(&c->peer, peer))
			return c;

	fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
	if (fd < 0)
		return NULL;
	sin.sin_port = peer->port;
	sin.sin_addr.s_addr = peer->ipv4;
	/* A blocking connect to the loopback interface completes or fails at once. */
	if (connect(fd, (struct sockaddr *) &sin, sizeof(sin))) {
		SRDSIM_WARN(FI_LOG_EP_DATA, "cannot connect to port %u: %s\n", ntohs(peer->port), strerror(errno));
		close(fd);
		return NULL;
	}
	if (fcntl(fd, F_SETFL, O_NONBLOCK)) {
		close(fd);
		return NULL;
	}
	c = srdsim_conn_new(x, fd, true);
	if (!c) {
		close(fd);
		return NULL;
	}
	c->peer = *peer;
	c->peer_known = true;
	if (srdsim_send_frame(c, &hello, &x->ep->addr)) {
		srdsim_conn_free(c);
		return NULL;
	}
	return c;
}

/* ---- Receives -------------------------------------------------------------------------------------- */

static bool srdsim_recv_matches(const struct srdsim_recv *r, bool tagged, uint64_t tag, const struct srdsim_addr *src)
{
	if (r->directed && !srdsim_addr_equal(&r->src, src))
		return false;
	return !tagged || ((r->tag ^ tag) & ~r->ignore) == 0;
}

/* Reports a receive done with the message it got. Called with x->mu held. */
static void srdsim_recv_complete(struct srdsim_xfer *x, struct srdsim_recv *r, bool tagged, uint64_t total,
				 uint64_t tag, uint64_t data, uint32_t fflags, const struct srdsim_addr *src)
{
	uint64_t flags = FI_RECV | (tagged ? FI_TAGGED : FI_MSG);
	size_t got = MIN(total, r->len);

	if (fflags & SRDSIM_FR_CQ_DATA)
		flags |= FI_REMOTE_CQ_DATA;
	if (total > r->len)
		srdsim_event(x, SRDSIM_EV_RX, r->context, flags, got, NULL, data, tag, FI_ETRUNC, total - r->len, src);
	else if (r->flags & FI_COMPLETION)
		srdsim_event(x, SRDSIM_EV_RX, r->context, flags, got, NULL, data, tag, 0, 0, src);
	free(r);
}

/* ---- What the agent applies ------------------------------------------------------------------------- */

static void srdsim_ack(struct srdsim_conn *c, uint64_t op_id, int status)
{
	struct srdsim_frame ack = { .type = SRDSIM_F_ACK, .op_id = op_id, .data = (uint64_t) (int64_t) status };

	if (srdsim_send_frame(c, &ack, NULL))
		SRDSIM_WARN(FI_LOG_EP_DATA, "out of memory: an acknowledgement is lost\n");
}

/*
 * Resolves a remote access to [addr, addr + len) under key into a local address and its registration. On
 * success the domain lock is held, so the registration cannot be closed while the caller copies; the caller
 * releases it with srdsim_verify_done().
 */
static int srdsim_verify(struct srdsim_xfer *x, uint64_t addr, uint64_t len, uint64_t key, uint64_t access,
			 void **local, struct ofi_mr **mr)
{
	struct util_domain *domain = x->ep->util_ep.domain;
	uintptr_t io = (uintptr_t) addr;
	void *ctx = NULL;
	int ret;

	ofi_genlock_lock(&domain->lock);
	ret = ofi_mr_map_verify(&domain->mr_map, &io, len, key, access, &ctx);
	if (ret) {
		ofi_genlock_unlock(&domain->lock);
		return ret;
	}
	*local = (void *) io;
	*mr = ctx;
	return 0;
}

static void srdsim_verify_done(struct srdsim_xfer *x)
{
	ofi_genlock_unlock(&x->ep->util_ep.domain->lock);
}

static int srdsim_store(struct ofi_mr *mr, void *dst, const char *src, size_t len, bool in_order)
{
	size_t step = 128, n;

	if (mr && mr->iface != FI_HMEM_SYSTEM)
		return ofi_copy_to_hmem(mr->iface, mr->device, dst, src, len);
	if (in_order) {
		/* Front to back, a block at a time, each visible before the next is written. */
		while (len) {
			n = MIN(len, step);
			memcpy(dst, src, n);
			__atomic_thread_fence(__ATOMIC_RELEASE);
			dst = (char *) dst + n;
			src += n;
			len -= n;
		}
		return 0;
	}
	/* Back to front, a block at a time: the end of the chunk lands first. */
	while (len) {
		n = len % step ? len % step : step;
		len -= n;
		memcpy((char *) dst + len, src + len, n);
		/* A barrier between blocks, so another core can see the later ones before the earlier ones. */
		__atomic_thread_fence(__ATOMIC_RELEASE);
	}
	return 0;
}

/* Remote CQ data waiting, under FI_RX_CQ_DATA, for a receive to consume. */
struct srdsim_cq_data {
	struct dlist_entry entry;
	uint64_t data;
	bool has_src;
	struct srdsim_addr src;
};

/*
 * A write with CQ data has landed. Under FI_RX_CQ_DATA it consumes the oldest posted receive, whose context
 * its completion carries, and waits for one if none is posted, as the device retries a receiver that is not
 * ready; otherwise it completes on its own. Called with x->mu held.
 */
static void srdsim_raise_cq_data(struct srdsim_xfer *x, uint64_t data, const struct srdsim_addr *src)
{
	const uint64_t flags = FI_RMA | FI_REMOTE_WRITE | FI_REMOTE_CQ_DATA;
	struct srdsim_recv *r;
	struct srdsim_cq_data *d;

	if (!x->rx_cq_data) {
		srdsim_event(x, SRDSIM_EV_RX, NULL, flags, 0, NULL, data, 0, 0, 0, src);
		return;
	}
	if (!dlist_empty(&x->recvs) && dlist_empty(&x->cq_data_waiting)) {
		dlist_pop_front(&x->recvs, struct srdsim_recv, r, entry);
		srdsim_event(x, SRDSIM_EV_RX, r->context, flags, 0, NULL, data, 0, 0, 0, src);
		free(r);
		return;
	}
	d = calloc(1, sizeof(*d));
	if (!d) {
		SRDSIM_WARN(FI_LOG_EP_DATA, "out of memory: remote CQ data is lost\n");
		return;
	}
	d->data = data;
	if (src) {
		d->has_src = true;
		d->src = *src;
	}
	dlist_insert_tail(&d->entry, &x->cq_data_waiting);
}

static void srdsim_on_write(struct srdsim_xfer *x, struct srdsim_conn *c, const struct srdsim_frame *f,
			    const char *payload)
{
	struct srdsim_inwrite *w = NULL, *cur;
	struct ofi_mr *mr = NULL;
	void *dst;
	int ret;

	dlist_foreach_container(&x->inwrites, struct srdsim_inwrite, cur, entry)
		if (cur->conn == c && cur->op_id == f->op_id) {
			w = cur;
			break;
		}
	if (!w) {
		w = calloc(1, sizeof(*w));
		if (!w) {
			SRDSIM_WARN(FI_LOG_EP_DATA, "out of memory: a write is dropped\n");
			return;
		}
		w->conn = c;
		w->op_id = f->op_id;
		w->total = f->total;
		w->data = f->data;
		w->flags = f->flags;
		dlist_insert_tail(&w->entry, &x->inwrites);
	}

	ret = srdsim_verify(x, f->addr, f->len, f->key, FI_REMOTE_WRITE, &dst, &mr);
	if (!ret) {
		ret = srdsim_store(mr, dst, payload, f->len, f->flags & SRDSIM_FR_IN_ORDER);
		srdsim_verify_done(x);
	}
	if (ret && !w->status)
		w->status = ret;
	w->landed += f->len;

	if ((w->flags & SRDSIM_FR_CQ_DATA) && !w->cq_data_raised && !w->status &&
	    (w->landed == w->total || x->cq_data_early)) {
		srdsim_raise_cq_data(x, w->data, c->peer_known ? &c->peer : NULL);
		w->cq_data_raised = true;
	}
	if (w->landed >= w->total) {
		srdsim_ack(c, w->op_id, w->status);
		dlist_remove(&w->entry);
		free(w);
	}
}

static void srdsim_on_read_req(struct srdsim_xfer *x, struct srdsim_conn *c, const struct srdsim_frame *f)
{
	struct srdsim_frame r = { .type = SRDSIM_F_READ_RESP, .op_id = f->op_id, .total = f->total };
	struct ofi_mr *mr = NULL;
	size_t nchunks, i, j, k, off, len;
	uint32_t *order, t;
	char *tmp;
	void *src;
	int ret;

	ret = f->total ? srdsim_verify(x, f->addr, f->total, f->key, FI_REMOTE_READ, &src, &mr) : 0;
	if (ret || f->total == 0) {
		r.data = (uint64_t) (int64_t) ret;
		(void) srdsim_send_frame(c, &r, NULL);
		return;
	}
	nchunks = (f->total + x->chunk - 1) / x->chunk;
	order = malloc(nchunks * sizeof(*order));
	tmp = malloc(x->chunk);
	if (!order || !tmp) {
		srdsim_verify_done(x);
		r.data = (uint64_t) (int64_t) -FI_ENOMEM;
		(void) srdsim_send_frame(c, &r, NULL);
		free(order);
		free(tmp);
		return;
	}
	for (i = 0; i < nchunks; i++)
		order[i] = (uint32_t) i;
	if (x->mode == SRDSIM_MODE_SRD)
		for (i = nchunks; i > 1; i--) {
			j = srdsim_rand(&x->agent_rng) % i;
			t = order[i - 1];
			order[i - 1] = order[j];
			order[j] = t;
		}
	for (k = 0; k < nchunks; k++) {
		off = (size_t) order[k] * x->chunk;
		len = MIN(x->chunk, f->total - off);
		if (mr && mr->iface != FI_HMEM_SYSTEM)
			ret = ofi_copy_from_hmem(mr->iface, mr->device, tmp, (char *) src + off, len);
		else
			memcpy(tmp, (char *) src + off, len);
		r.offset = off;
		r.len = len;
		r.data = (uint64_t) (int64_t) ret;
		if (srdsim_send_frame(c, &r, tmp))
			break;
	}
	srdsim_verify_done(x);
	free(order);
	free(tmp);
}

static struct srdsim_op *srdsim_inflight_op(struct srdsim_xfer *x, uint64_t id)
{
	struct srdsim_op *op;

	dlist_foreach_container(&x->inflight, struct srdsim_op, op, entry)
		if (op->id == id)
			return op;
	return NULL;
}

static void srdsim_op_free(struct srdsim_op *op)
{
	free(op->order);
	free(op->staged);
	free(op);
}

/* Reports an operation this endpoint started as done, and forgets it. Called with x->mu held. */
static void srdsim_op_done(struct srdsim_xfer *x, struct srdsim_op *op, int status)
{
	dlist_remove(&op->entry);
	x->outstanding--;
	if (status)
		srdsim_event(x, SRDSIM_EV_TX, op->context, op->cq_flags, 0, NULL, 0, 0,
			     status == -FI_EACCES || status == -FI_EINVAL ? FI_EACCES : -status, 0, NULL);
	else if (op->completion)
		srdsim_event(x, SRDSIM_EV_TX, op->context, op->cq_flags, 0, NULL, 0, 0, 0, 0, NULL);
	srdsim_op_free(op);
}

static void srdsim_on_read_resp(struct srdsim_xfer *x, const struct srdsim_frame *f, const char *payload)
{
	struct srdsim_op *op = srdsim_inflight_op(x, f->op_id);
	int status = (int) (int64_t) f->data;

	if (!op || op->kind != SRDSIM_OP_READ)
		return;
	if (!status && f->len)
		status = srdsim_copy_to(op->iov, op->desc, op->iov_count, f->offset, payload, f->len);
	op->landed += f->len;
	if (status || op->landed >= f->total)
		srdsim_op_done(x, op, status);
}

static void srdsim_on_ack(struct srdsim_xfer *x, const struct srdsim_frame *f)
{
	struct srdsim_op *op = srdsim_inflight_op(x, f->op_id);

	if (op)
		srdsim_op_done(x, op, (int) (int64_t) f->data);
}

/*
 * Abandons a message whose pieces stopped coming (the sender failed to release the rest): its receive, if it
 * had one, completes with FI_EIO, and an unexpected copy is dropped. Called with x->mu held.
 */
static void srdsim_inmsg_abort(struct srdsim_xfer *x, struct srdsim_inmsg *m, bool tagged)
{
	struct srdsim_recv *r = m->recv;

	OFI_UNUSED(tagged);
	if (m->unexp) {
		r = m->unexp->claimed;
		dlist_remove(&m->unexp->entry);
		free(m->unexp->buf);
		free(m->unexp);
	}
	if (r) {
		srdsim_event(x, SRDSIM_EV_RX, r->context, FI_RECV, 0, NULL, 0, 0, FI_EIO, 0, NULL);
		free(r);
	}
	memset(m, 0, sizeof(*m));
}

/* A piece of a send. Pieces of one connection arrive in order, one message after another. */
static void srdsim_on_msg(struct srdsim_xfer *x, struct srdsim_conn *c, const struct srdsim_frame *f,
			  const char *payload)
{
	struct srdsim_inmsg *m = &c->inmsg;
	bool tagged = f->flags & SRDSIM_FR_TAGGED;
	struct dlist_entry *list = tagged ? &x->trecvs : &x->recvs;
	struct srdsim_recv *r, *found = NULL;
	struct srdsim_unexp *u;
	size_t room;

	if (!c->peer_known)
		return;
	if (m->active && (f->op_id != m->op_id || f->offset != m->received))
		srdsim_inmsg_abort(x, m, tagged);
	if (!m->active) {
		if (f->offset != 0)
			return;	/* the rest of a message whose start was not received */
		memset(m, 0, sizeof(*m));
		m->active = true;
		m->op_id = f->op_id;
		m->total = f->total;
		m->tag = f->tag;
		m->data = f->data;
		m->flags = f->flags;
		dlist_foreach_container(list, struct srdsim_recv, r, entry)
			if (srdsim_recv_matches(r, tagged, f->tag, &c->peer)) {
				found = r;
				break;
			}
		if (found) {
			dlist_remove(&found->entry);
			m->recv = found;
		} else {
			u = calloc(1, sizeof(*u));
			if (u)
				u->buf = malloc(f->total ? f->total : 1);
			if (!u || !u->buf) {
				SRDSIM_WARN(FI_LOG_EP_DATA, "out of memory: a message is dropped\n");
				free(u);
				m->active = false;
				return;
			}
			u->src = c->peer;
			u->tag = f->tag;
			u->data = f->data;
			u->flags = f->flags;
			u->total = f->total;
			dlist_insert_tail(&u->entry, tagged ? &x->tunexp : &x->unexp);
			m->unexp = u;
		}
	}

	if (m->recv) {
		room = f->offset < m->recv->len ? MIN(f->len, m->recv->len - f->offset) : 0;
		if (room)
			(void) srdsim_copy_to(m->recv->iov, m->recv->desc, m->recv->iov_count, f->offset, payload, room);
	} else {
		memcpy(m->unexp->buf + f->offset, payload, f->len);
		m->unexp->received += f->len;
	}
	m->received += f->len;
	if (m->received < m->total)
		return;

	if (m->recv) {
		srdsim_recv_complete(x, m->recv, tagged, m->total, m->tag, m->data, m->flags, &c->peer);
	} else if (m->unexp->claimed) {
		/* A receive matched it while it was arriving. */
		r = m->unexp->claimed;
		(void) srdsim_copy_to(r->iov, r->desc, r->iov_count, 0, m->unexp->buf, MIN(m->total, r->len));
		dlist_remove(&m->unexp->entry);
		free(m->unexp->buf);
		free(m->unexp);
		srdsim_recv_complete(x, r, tagged, m->total, m->tag, m->data, m->flags, &c->peer);
	}
	srdsim_ack(c, m->op_id, 0);
	m->active = false;
}

static uint64_t srdsim_now_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

static void srdsim_delay(struct srdsim_xfer *x, struct srdsim_conn *c, const struct srdsim_frame *f,
			 const char *payload)
{
	struct srdsim_delayed *d = malloc(sizeof(*d) + f->len), *cur;
	struct dlist_entry *it;

	if (!d) {
		srdsim_on_write(x, c, f, payload);
		return;
	}
	d->conn = c;
	d->f = *f;
	memcpy(d->payload, payload, f->len);
	d->due_ns = srdsim_now_ns() + (srdsim_rand(&x->agent_rng) % (x->jitter_us + 1)) * 1000;
	for (it = x->delayed.prev; it != &x->delayed; it = it->prev) {
		cur = container_of(it, struct srdsim_delayed, entry);
		if (cur->due_ns <= d->due_ns) {
			dlist_insert_after(&d->entry, it);
			return;
		}
	}
	dlist_insert_head(&d->entry, &x->delayed);
}

/* Applies the delayed chunks that are due; returns the milliseconds until the next one, or -1. */
static int srdsim_apply_due(struct srdsim_xfer *x)
{
	struct srdsim_delayed *d;
	uint64_t now = srdsim_now_ns();

	while (!dlist_empty(&x->delayed)) {
		d = container_of(x->delayed.next, struct srdsim_delayed, entry);
		if (d->due_ns > now)
			return (int) ((d->due_ns - now + 999999) / 1000000);
		dlist_remove(&d->entry);
		srdsim_on_write(x, d->conn, &d->f, d->payload);
		free(d);
	}
	return -1;
}

static void srdsim_on_frame(struct srdsim_xfer *x, struct srdsim_conn *c, const struct srdsim_frame *f,
			    const char *payload)
{
	switch (f->type) {
	case SRDSIM_F_HELLO:
		if (f->len == sizeof(struct srdsim_addr)) {
			memcpy(&c->peer, payload, sizeof(c->peer));
			c->peer_known = true;
		}
		break;
	case SRDSIM_F_MSG:
		srdsim_on_msg(x, c, f, payload);
		break;
	case SRDSIM_F_WRITE:
		if (x->jitter_us)
			srdsim_delay(x, c, f, payload);
		else
			srdsim_on_write(x, c, f, payload);
		break;
	case SRDSIM_F_READ_REQ:
		srdsim_on_read_req(x, c, f);
		break;
	case SRDSIM_F_READ_RESP:
		srdsim_on_read_resp(x, f, payload);
		break;
	case SRDSIM_F_ACK:
		srdsim_on_ack(x, f);
		break;
	default:
		SRDSIM_WARN(FI_LOG_EP_DATA, "unknown frame type %u\n", f->type);
	}
}

/* Whether a frame's sizes are consistent, so that applying it stays inside what it describes. */
static bool srdsim_frame_valid(const struct srdsim_frame *f)
{
	switch (f->type) {
	case SRDSIM_F_HELLO:
		return f->len == sizeof(struct srdsim_addr);
	case SRDSIM_F_MSG:
	case SRDSIM_F_WRITE:
	case SRDSIM_F_READ_RESP:
		return f->len <= f->total && f->offset <= f->total - f->len;
	case SRDSIM_F_READ_REQ:
	case SRDSIM_F_ACK:
		return f->len == 0;
	default:
		return false;
	}
}

/* Applies every whole frame buffered on a connection. False when one is malformed. */
static bool srdsim_conn_parse(struct srdsim_xfer *x, struct srdsim_conn *c)
{
	struct srdsim_frame f;
	size_t used = 0;
	bool ok = true;

	while (c->in.len - used >= sizeof(f)) {
		memcpy(&f, c->in.p + used, sizeof(f));
		if (!srdsim_frame_valid(&f)) {
			SRDSIM_WARN(FI_LOG_EP_DATA, "a malformed frame (type %u); closing the connection\n", f.type);
			ok = false;
			break;
		}
		if (c->in.len - used - sizeof(f) < f.len)
			break;
		srdsim_on_frame(x, c, &f, c->in.p + used + sizeof(f));
		used += sizeof(f) + f.len;
	}
	memmove(c->in.p, c->in.p + used, c->in.len - used);
	c->in.len -= used;
	return ok;
}

/*
 * Reads what a connection has, applying frames as they complete, so a fast sender cannot make the agent
 * buffer without bound. False when the peer closed it or sent something malformed.
 */
static bool srdsim_conn_read(struct srdsim_xfer *x, struct srdsim_conn *c)
{
	char tmp[65536];
	ssize_t n;
	int rounds;

	for (rounds = 0; rounds < 64; rounds++) {
		n = recv(c->fd, tmp, sizeof(tmp), 0);
		if (n > 0) {
			if (srdsim_buf_append(&c->in, tmp, (size_t) n) || !srdsim_conn_parse(x, c))
				return false;
			continue;
		}
		if (n == 0)
			return false;
		if (errno == EINTR)
			continue;
		return errno == EAGAIN || errno == EWOULDBLOCK;
	}
	return true;
}

/* Drops what has been written from the front of an outbound buffer that is never fully drained. */
static void srdsim_out_compact(struct srdsim_buf *b)
{
	if (b->off > (1u << 20) && b->off * 2 > b->len) {
		memmove(b->p, b->p + b->off, b->len - b->off);
		b->len -= b->off;
		b->off = 0;
	}
}

static bool srdsim_conn_write(struct srdsim_conn *c)
{
	ssize_t n;

	while (c->out.off < c->out.len) {
		n = send(c->fd, c->out.p + c->out.off, c->out.len - c->out.off, MSG_NOSIGNAL);
		if (n > 0) {
			c->out.off += (size_t) n;
			continue;
		}
		if (n < 0 && errno == EINTR)
			continue;
		if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
			srdsim_out_compact(&c->out);
			return true;
		}
		return false;
	}
	c->out.len = c->out.off = 0;
	return true;
}

/* A connection broke: what was travelling on it fails, and it is closed. Called with x->mu held. */
static void srdsim_conn_lost(struct srdsim_xfer *x, struct srdsim_conn *c)
{
	struct dlist_entry *lists[2] = { &x->pending, &x->inflight };
	struct srdsim_op *op, *otmp;
	struct srdsim_inwrite *w, *wtmp;
	size_t i;

	for (i = 0; i < 2; i++)
		dlist_foreach_container_safe(lists[i], struct srdsim_op, op, entry, otmp)
			if (op->conn == c)
				srdsim_op_done(x, op, -FI_EIO);
	struct srdsim_delayed *d, *dtmp;

	dlist_foreach_container_safe(&x->delayed, struct srdsim_delayed, d, entry, dtmp)
		if (d->conn == c) {
			dlist_remove(&d->entry);
			free(d);
		}
	dlist_foreach_container_safe(&x->inwrites, struct srdsim_inwrite, w, entry, wtmp)
		if (w->conn == c) {
			dlist_remove(&w->entry);
			free(w);
		}
	if (c->inmsg.active)
		srdsim_inmsg_abort(x, &c->inmsg, c->inmsg.flags & SRDSIM_FR_TAGGED);
	srdsim_conn_free(c);
}

static void *srdsim_agent(void *arg)
{
	struct srdsim_xfer *x = arg;
	struct pollfd *pfd = NULL;
	struct srdsim_conn **who = NULL, *c, *tmp;
	size_t n, cap = 0, i;
	uint64_t drain;
	int fd, timeout;

	pthread_mutex_lock(&x->mu);
	while (!x->stop) {
		/* Delayed chunks due now are applied first, so what they queue is written out in this round. */
		timeout = srdsim_apply_due(x);
		n = 2;
		dlist_foreach_container(&x->conns, struct srdsim_conn, c, entry)
			n++;
		if (n > cap) {
			cap = n * 2;
			pfd = realloc(pfd, cap * sizeof(*pfd));
			who = realloc(who, cap * sizeof(*who));
			if (!pfd || !who)
				break;
		}
		pfd[0] = (struct pollfd) { .fd = x->wake_fd, .events = POLLIN };
		pfd[1] = (struct pollfd) { .fd = x->listen_fd, .events = POLLIN };
		i = 2;
		dlist_foreach_container(&x->conns, struct srdsim_conn, c, entry) {
			pfd[i] = (struct pollfd) { .fd = c->fd, .events = POLLIN | (c->out.len > c->out.off ? POLLOUT : 0) };
			who[i++] = c;
		}
		pthread_mutex_unlock(&x->mu);
		(void) poll(pfd, n, timeout < 0 || timeout > 100 ? 100 : timeout);
		pthread_mutex_lock(&x->mu);
		(void) srdsim_apply_due(x);
		if (pfd[0].revents & POLLIN)
			(void) !read(x->wake_fd, &drain, sizeof(drain));
		if (pfd[1].revents & POLLIN) {
			fd = accept4(x->listen_fd, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
			if (fd >= 0 && !srdsim_conn_new(x, fd, false))
				close(fd);
		}
		for (i = 2; i < n; i++) {
			c = who[i];
			if ((pfd[i].revents & (POLLIN | POLLHUP | POLLERR)) && !srdsim_conn_read(x, c)) {
				srdsim_conn_lost(x, c);
				continue;
			}
			if (!srdsim_conn_write(c))
				srdsim_conn_lost(x, c);
		}
		/* Frames queued while handling the ones above go out now, not after the next poll. */
		dlist_foreach_container_safe(&x->conns, struct srdsim_conn, c, entry, tmp)
			if (c->out.len > c->out.off && !srdsim_conn_write(c))
				srdsim_conn_lost(x, c);
	}
	pthread_mutex_unlock(&x->mu);
	free(pfd);
	free(who);
	return NULL;
}

/* ---- What the consumer's thread does ---------------------------------------------------------------- */

static struct srdsim_xfer *srdsim_xfer_of(struct fid_ep *ep_fid)
{
	return container_of(ep_fid, struct srdsim_ep, util_ep.ep_fid)->xfer;
}

/* The peer behind an fi_addr_t, from the address vector. */
static int srdsim_peer(struct srdsim_xfer *x, fi_addr_t fi_addr, struct srdsim_addr *out)
{
	struct util_av *av = x->ep->util_ep.av;
	int ret = 0;

	if (!av)
		return -FI_ENOAV;
	ofi_genlock_lock(&av->lock);
	if (fi_addr == FI_ADDR_UNSPEC || !ofi_bufpool_ibuf_is_valid(av->av_entry_pool, fi_addr))
		ret = -FI_EINVAL;
	else
		memcpy(out, ofi_av_get_addr(av, fi_addr), sizeof(*out));
	ofi_genlock_unlock(&av->lock);
	return ret;
}

/* Submits an operation; its units are released by the endpoint's progress. */
static ssize_t srdsim_submit(struct fid_ep *ep_fid, enum srdsim_op_kind kind, const struct iovec *iov, void **desc,
			     size_t count, fi_addr_t dest, uint64_t addr, uint64_t key, uint64_t data, uint64_t tag,
			     uint32_t frame_flags, void *context, uint64_t cq_flags, uint64_t op_flags)
{
	struct srdsim_xfer *x = srdsim_xfer_of(ep_fid);
	struct srdsim_ep *ep = x->ep;
	struct srdsim_addr peer;
	struct srdsim_op *op;
	size_t i, j;
	uint32_t t;
	int ret;

	if (!ep->enabled)
		return -FI_EOPBADSTATE;
	if (count > SRDSIM_IOV_MAX)
		return -FI_EINVAL;
	ret = srdsim_peer(x, dest, &peer);
	if (ret)
		return ret;

	op = calloc(1, sizeof(*op));
	if (!op)
		return -FI_ENOMEM;
	op->kind = kind;
	op->iov_count = count;
	for (i = 0; i < count; i++) {
		op->iov[i] = iov[i];
		op->desc[i] = desc ? desc[i] : NULL;
	}
	op->len = srdsim_iov_len(iov, count);
	op->addr = addr;
	op->key = key;
	op->data = data;
	op->tag = tag;
	op->frame_flags = frame_flags | (ep->write_in_order_128 ? SRDSIM_FR_IN_ORDER : 0);
	op->context = context;
	op->cq_flags = cq_flags;
	op->completion = op_flags & FI_COMPLETION;
	if ((op_flags & FI_INJECT) && op->len > ep->inject_size) {
		srdsim_op_free(op);
		return -FI_EINVAL;
	}
	if (op_flags & FI_INJECT) {
		/* The source may be reused on return: take a copy now. */
		op->staged = malloc(op->len ? op->len : 1);
		if (!op->staged || srdsim_copy_from(op->staged, op->iov, op->desc, op->iov_count, 0, op->len)) {
			srdsim_op_free(op);
			return -FI_ENOMEM;
		}
	}
	op->nchunks = kind == SRDSIM_OP_READ ? 1 : MAX((size_t) 1, (op->len + x->chunk - 1) / x->chunk);
	op->order = malloc(op->nchunks * sizeof(*op->order));
	if (!op->order) {
		srdsim_op_free(op);
		return -FI_ENOMEM;
	}
	for (i = 0; i < op->nchunks; i++)
		op->order[i] = (uint32_t) i;

	pthread_mutex_lock(&x->mu);
	/* The chunks of a write are released in a random order; a send's pieces always in order. */
	if (kind == SRDSIM_OP_WRITE && x->mode == SRDSIM_MODE_SRD)
		for (i = op->nchunks; i > 1; i--) {
			j = srdsim_rand(&x->rng) % i;
			t = op->order[i - 1];
			op->order[i - 1] = op->order[j];
			op->order[j] = t;
		}
	op->conn = srdsim_conn_to(x, &peer);
	if (!op->conn) {
		pthread_mutex_unlock(&x->mu);
		srdsim_op_free(op);
		return -FI_EHOSTUNREACH;
	}
	/* As a full send queue does: the consumer retries after reading completions. */
	if (x->outstanding >= x->tx_size) {
		pthread_mutex_unlock(&x->mu);
		srdsim_op_free(op);
		return -FI_EAGAIN;
	}
	x->outstanding++;
	op->id = ++x->next_op_id;
	dlist_insert_tail(&op->entry, &x->pending);
	pthread_mutex_unlock(&x->mu);
	return 0;
}

/* Releases one unit of `op`. Called with x->mu held. */
static int srdsim_release_one(struct srdsim_xfer *x, struct srdsim_op *op, char *tmp)
{
	struct srdsim_frame f = { .op_id = op->id, .total = op->len, .flags = op->frame_flags, .data = op->data,
				  .tag = op->tag, .key = op->key };
	size_t off, len;
	int ret;

	if (op->kind == SRDSIM_OP_READ) {
		f.type = SRDSIM_F_READ_REQ;
		f.addr = op->addr;
		ret = srdsim_send_frame(op->conn, &f, NULL);
		op->next = op->nchunks;
		return ret;
	}
	off = (size_t) op->order[op->next] * x->chunk;
	len = op->len ? MIN(x->chunk, op->len - off) : 0;
	if (op->staged)
		memcpy(tmp, op->staged + off, len);
	else if (len && (ret = srdsim_copy_from(tmp, op->iov, op->desc, op->iov_count, off, len)))
		return ret;
	f.type = op->kind == SRDSIM_OP_SEND ? SRDSIM_F_MSG : SRDSIM_F_WRITE;
	f.offset = off;
	f.len = len;
	f.addr = op->addr + off;
	ret = srdsim_send_frame(op->conn, &f, tmp);
	if (!ret)
		op->next++;
	return ret;
}

/* Whether `op` may release its next unit now: a send waits for every earlier send to the same peer. */
static bool srdsim_eligible(struct srdsim_xfer *x, struct srdsim_op *op)
{
	struct srdsim_op *e;

	if (op->kind != SRDSIM_OP_SEND || x->direct)
		return true;
	dlist_foreach_container(&x->pending, struct srdsim_op, e, entry) {
		if (e == op)
			return true;
		if (e->kind == SRDSIM_OP_SEND && e->conn == op->conn)
			return false;
	}
	return true;
}

static void srdsim_release(struct srdsim_xfer *x)
{
	struct srdsim_op *op, *pick, *cand[64];
	size_t budget, ncand;
	char *tmp = x->scratch;
	int ret;

	for (budget = 0; budget < SRDSIM_RELEASE_BUDGET && !dlist_empty(&x->pending); budget++) {
		ncand = 0;
		dlist_foreach_container(&x->pending, struct srdsim_op, op, entry) {
			if (ncand == ARRAY_SIZE(cand))
				break;
			if (srdsim_eligible(x, op))
				cand[ncand++] = op;
			if (x->mode == SRDSIM_MODE_INORDER && ncand)
				break;
		}
		if (!ncand)
			break;
		pick = cand[x->mode == SRDSIM_MODE_SRD ? srdsim_rand(&x->rng) % ncand : 0];
		ret = srdsim_release_one(x, pick, tmp);
		if (ret) {
			srdsim_op_done(x, pick, ret);
			continue;
		}
		if (pick->next == pick->nchunks) {
			dlist_remove(&pick->entry);
			dlist_insert_tail(&pick->entry, &x->inflight);
		}
	}
	if (budget)
		srdsim_wake(x);
}

/* Writes the completions the agent queued, on the consumer's thread. */
static void srdsim_drain_events(struct srdsim_xfer *x)
{
	struct util_ep *uep = &x->ep->util_ep;
	struct srdsim_event *e;
	struct fi_cq_err_entry err;
	struct util_cq *cq;
	fi_addr_t src;

	while (!dlist_empty(&x->events)) {
		dlist_pop_front(&x->events, struct srdsim_event, e, entry);
		cq = e->kind == SRDSIM_EV_TX ? uep->tx_cq : uep->rx_cq;
		src = FI_ADDR_NOTAVAIL;
		if (e->has_src && uep->av && (uep->caps & FI_SOURCE))
			src = ofi_av_lookup_fi_addr(uep->av, &e->src);
		if (e->err) {
			memset(&err, 0, sizeof(err));
			err.op_context = e->context;
			err.flags = e->flags;
			err.len = e->len;
			err.data = e->data;
			err.tag = e->tag;
			err.olen = e->olen;
			err.err = e->err;
			err.prov_errno = -e->err;
			err.src_addr = src;
			(void) ofi_cq_write_error(cq, &err);
		} else if (uep->caps & FI_SOURCE) {
			(void) ofi_cq_write_src(cq, e->context, e->flags, e->len, e->buf, e->data, e->tag, src);
		} else {
			(void) ofi_cq_write(cq, e->context, e->flags, e->len, e->buf, e->data, e->tag);
		}
		free(e);
	}
}

void srdsim_xfer_progress(struct util_ep *util_ep)
{
	struct srdsim_xfer *x = container_of(util_ep, struct srdsim_ep, util_ep)->xfer;

	if (!x)
		return;
	pthread_mutex_lock(&x->mu);
	srdsim_release(x);
	srdsim_drain_events(x);
	pthread_mutex_unlock(&x->mu);
}

/* Posts a receive: it takes the oldest unexpected message it matches, else it waits for one. */
static ssize_t srdsim_post_recv(struct fid_ep *ep_fid, bool tagged, const struct iovec *iov, void **desc,
				size_t count, fi_addr_t src, uint64_t tag, uint64_t ignore, void *context,
				uint64_t flags)
{
	struct srdsim_xfer *x = srdsim_xfer_of(ep_fid);
	struct srdsim_ep *ep = x->ep;
	struct srdsim_recv *r;
	struct srdsim_unexp *u, *found = NULL;
	size_t i;
	int ret;

	if (!ep->enabled)
		return -FI_EOPBADSTATE;
	if (count > SRDSIM_IOV_MAX)
		return -FI_EINVAL;
	r = calloc(1, sizeof(*r));
	if (!r)
		return -FI_ENOMEM;
	r->iov_count = count;
	for (i = 0; i < count; i++) {
		r->iov[i] = iov[i];
		r->desc[i] = desc ? desc[i] : NULL;
	}
	r->len = srdsim_iov_len(iov, count);
	r->tag = tag;
	r->ignore = ignore;
	r->context = context;
	r->flags = flags;
	if ((ep->util_ep.caps & FI_DIRECTED_RECV) && src != FI_ADDR_UNSPEC) {
		ret = srdsim_peer(x, src, &r->src);
		if (ret) {
			free(r);
			return ret;
		}
		r->directed = true;
	}

	pthread_mutex_lock(&x->mu);
	if (!tagged && !dlist_empty(&x->cq_data_waiting)) {
		struct srdsim_cq_data *d;

		dlist_pop_front(&x->cq_data_waiting, struct srdsim_cq_data, d, entry);
		srdsim_event(x, SRDSIM_EV_RX, r->context, FI_RMA | FI_REMOTE_WRITE | FI_REMOTE_CQ_DATA, 0, NULL,
			     d->data, 0, 0, 0, d->has_src ? &d->src : NULL);
		free(d);
		free(r);
		pthread_mutex_unlock(&x->mu);
		return 0;
	}
	dlist_foreach_container(tagged ? &x->tunexp : &x->unexp, struct srdsim_unexp, u, entry)
		if (!u->claimed && srdsim_recv_matches(r, tagged, u->tag, &u->src)) {
			found = u;
			break;
		}
	if (!found) {
		dlist_insert_tail(&r->entry, tagged ? &x->trecvs : &x->recvs);
	} else if (found->received < found->total) {
		found->claimed = r;
	} else {
		(void) srdsim_copy_to(r->iov, r->desc, r->iov_count, 0, found->buf, MIN(found->total, r->len));
		dlist_remove(&found->entry);
		srdsim_recv_complete(x, r, tagged, found->total, found->tag, found->data, found->flags, &found->src);
		free(found->buf);
		free(found);
	}
	pthread_mutex_unlock(&x->mu);
	return 0;
}

/* ---- Op tables ------------------------------------------------------------------------------------- */

#define SRDSIM_EP(e) container_of(e, struct srdsim_ep, util_ep.ep_fid)
/* Only FI_COMPLETION of the default op flags applies to the calls without flags. */
#define SRDSIM_TXF(e) (SRDSIM_EP(e)->util_ep.tx_op_flags & FI_COMPLETION)
#define SRDSIM_RXF(e) (SRDSIM_EP(e)->util_ep.rx_op_flags & FI_COMPLETION)
#define SRDSIM_IOV1(name, b, l) struct iovec name = { .iov_base = (void *) (b), .iov_len = (l) }

static ssize_t srdsim_recv(struct fid_ep *e, void *buf, size_t len, void *desc, fi_addr_t src, void *context)
{
	SRDSIM_IOV1(iov, buf, len);
	return srdsim_post_recv(e, false, &iov, &desc, 1, src, 0, 0, context, SRDSIM_RXF(e));
}

static ssize_t srdsim_recvv(struct fid_ep *e, const struct iovec *iov, void **desc, size_t count, fi_addr_t src,
			    void *context)
{
	return srdsim_post_recv(e, false, iov, desc, count, src, 0, 0, context, SRDSIM_RXF(e));
}

static ssize_t srdsim_recvmsg(struct fid_ep *e, const struct fi_msg *m, uint64_t flags)
{
	return srdsim_post_recv(e, false, m->msg_iov, m->desc, m->iov_count, m->addr, 0, 0, m->context,
				flags | SRDSIM_EP(e)->util_ep.rx_msg_flags);
}

static ssize_t srdsim_send_common(struct fid_ep *e, bool tagged, const struct iovec *iov, void **desc, size_t count,
				  fi_addr_t dest, uint64_t data, bool has_data, uint64_t tag, void *context,
				  uint64_t flags)
{
	return srdsim_submit(e, SRDSIM_OP_SEND, iov, desc, count, dest, 0, 0, data, tag,
			     (tagged ? SRDSIM_FR_TAGGED : 0) | (has_data ? SRDSIM_FR_CQ_DATA : 0), context,
			     FI_SEND | (tagged ? FI_TAGGED : FI_MSG), flags);
}

static ssize_t srdsim_send(struct fid_ep *e, const void *buf, size_t len, void *desc, fi_addr_t dest, void *context)
{
	SRDSIM_IOV1(iov, buf, len);
	return srdsim_send_common(e, false, &iov, &desc, 1, dest, 0, false, 0, context, SRDSIM_TXF(e));
}

static ssize_t srdsim_sendv(struct fid_ep *e, const struct iovec *iov, void **desc, size_t count, fi_addr_t dest,
			    void *context)
{
	return srdsim_send_common(e, false, iov, desc, count, dest, 0, false, 0, context, SRDSIM_TXF(e));
}

static ssize_t srdsim_sendmsg(struct fid_ep *e, const struct fi_msg *m, uint64_t flags)
{
	return srdsim_send_common(e, false, m->msg_iov, m->desc, m->iov_count, m->addr, m->data,
				  flags & FI_REMOTE_CQ_DATA, 0, m->context, flags | SRDSIM_EP(e)->util_ep.tx_msg_flags);
}

static ssize_t srdsim_inject(struct fid_ep *e, const void *buf, size_t len, fi_addr_t dest)
{
	SRDSIM_IOV1(iov, buf, len);
	return srdsim_send_common(e, false, &iov, NULL, 1, dest, 0, false, 0, NULL, FI_INJECT);
}

static ssize_t srdsim_senddata(struct fid_ep *e, const void *buf, size_t len, void *desc, uint64_t data,
			       fi_addr_t dest, void *context)
{
	SRDSIM_IOV1(iov, buf, len);
	return srdsim_send_common(e, false, &iov, &desc, 1, dest, data, true, 0, context, SRDSIM_TXF(e));
}

static ssize_t srdsim_injectdata(struct fid_ep *e, const void *buf, size_t len, uint64_t data, fi_addr_t dest)
{
	SRDSIM_IOV1(iov, buf, len);
	return srdsim_send_common(e, false, &iov, NULL, 1, dest, data, true, 0, NULL, FI_INJECT);
}

struct fi_ops_msg srdsim_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = srdsim_recv,
	.recvv = srdsim_recvv,
	.recvmsg = srdsim_recvmsg,
	.send = srdsim_send,
	.sendv = srdsim_sendv,
	.sendmsg = srdsim_sendmsg,
	.inject = srdsim_inject,
	.senddata = srdsim_senddata,
	.injectdata = srdsim_injectdata,
};

static ssize_t srdsim_trecv(struct fid_ep *e, void *buf, size_t len, void *desc, fi_addr_t src, uint64_t tag,
			    uint64_t ignore, void *context)
{
	SRDSIM_IOV1(iov, buf, len);
	return srdsim_post_recv(e, true, &iov, &desc, 1, src, tag, ignore, context, SRDSIM_RXF(e));
}

static ssize_t srdsim_trecvv(struct fid_ep *e, const struct iovec *iov, void **desc, size_t count, fi_addr_t src,
			     uint64_t tag, uint64_t ignore, void *context)
{
	return srdsim_post_recv(e, true, iov, desc, count, src, tag, ignore, context, SRDSIM_RXF(e));
}

static ssize_t srdsim_trecvmsg(struct fid_ep *e, const struct fi_msg_tagged *m, uint64_t flags)
{
	if (flags & (FI_PEEK | FI_CLAIM))
		return -FI_ENOSYS;
	return srdsim_post_recv(e, true, m->msg_iov, m->desc, m->iov_count, m->addr, m->tag, m->ignore, m->context,
				flags | SRDSIM_EP(e)->util_ep.rx_msg_flags);
}

static ssize_t srdsim_tsend(struct fid_ep *e, const void *buf, size_t len, void *desc, fi_addr_t dest, uint64_t tag,
			    void *context)
{
	SRDSIM_IOV1(iov, buf, len);
	return srdsim_send_common(e, true, &iov, &desc, 1, dest, 0, false, tag, context, SRDSIM_TXF(e));
}

static ssize_t srdsim_tsendv(struct fid_ep *e, const struct iovec *iov, void **desc, size_t count, fi_addr_t dest,
			     uint64_t tag, void *context)
{
	return srdsim_send_common(e, true, iov, desc, count, dest, 0, false, tag, context, SRDSIM_TXF(e));
}

static ssize_t srdsim_tsendmsg(struct fid_ep *e, const struct fi_msg_tagged *m, uint64_t flags)
{
	return srdsim_send_common(e, true, m->msg_iov, m->desc, m->iov_count, m->addr, m->data,
				  flags & FI_REMOTE_CQ_DATA, m->tag, m->context, flags | SRDSIM_EP(e)->util_ep.tx_msg_flags);
}

static ssize_t srdsim_tinject(struct fid_ep *e, const void *buf, size_t len, fi_addr_t dest, uint64_t tag)
{
	SRDSIM_IOV1(iov, buf, len);
	return srdsim_send_common(e, true, &iov, NULL, 1, dest, 0, false, tag, NULL, FI_INJECT);
}

static ssize_t srdsim_tsenddata(struct fid_ep *e, const void *buf, size_t len, void *desc, uint64_t data,
				fi_addr_t dest, uint64_t tag, void *context)
{
	SRDSIM_IOV1(iov, buf, len);
	return srdsim_send_common(e, true, &iov, &desc, 1, dest, data, true, tag, context, SRDSIM_TXF(e));
}

static ssize_t srdsim_tinjectdata(struct fid_ep *e, const void *buf, size_t len, uint64_t data, fi_addr_t dest,
				  uint64_t tag)
{
	SRDSIM_IOV1(iov, buf, len);
	return srdsim_send_common(e, true, &iov, NULL, 1, dest, data, true, tag, NULL, FI_INJECT);
}

struct fi_ops_tagged srdsim_tagged_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = srdsim_trecv,
	.recvv = srdsim_trecvv,
	.recvmsg = srdsim_trecvmsg,
	.send = srdsim_tsend,
	.sendv = srdsim_tsendv,
	.sendmsg = srdsim_tsendmsg,
	.inject = srdsim_tinject,
	.senddata = srdsim_tsenddata,
	.injectdata = srdsim_tinjectdata,
};

static ssize_t srdsim_rma(struct fid_ep *e, enum srdsim_op_kind kind, const struct iovec *iov, void **desc,
			  size_t count, fi_addr_t dest, uint64_t addr, uint64_t key, uint64_t data, bool has_data,
			  void *context, uint64_t flags)
{
	return srdsim_submit(e, kind, iov, desc, count, dest, addr, key, data, 0, has_data ? SRDSIM_FR_CQ_DATA : 0,
			     context, FI_RMA | (kind == SRDSIM_OP_READ ? FI_READ : FI_WRITE), flags);
}

static ssize_t srdsim_read(struct fid_ep *e, void *buf, size_t len, void *desc, fi_addr_t src, uint64_t addr,
			   uint64_t key, void *context)
{
	SRDSIM_IOV1(iov, buf, len);
	return srdsim_rma(e, SRDSIM_OP_READ, &iov, &desc, 1, src, addr, key, 0, false, context, SRDSIM_TXF(e));
}

static ssize_t srdsim_readv(struct fid_ep *e, const struct iovec *iov, void **desc, size_t count, fi_addr_t src,
			    uint64_t addr, uint64_t key, void *context)
{
	return srdsim_rma(e, SRDSIM_OP_READ, iov, desc, count, src, addr, key, 0, false, context, SRDSIM_TXF(e));
}

static ssize_t srdsim_readmsg(struct fid_ep *e, const struct fi_msg_rma *m, uint64_t flags)
{
	if (m->rma_iov_count != 1)
		return -FI_EINVAL;
	return srdsim_rma(e, SRDSIM_OP_READ, m->msg_iov, m->desc, m->iov_count, m->addr, m->rma_iov[0].addr,
			  m->rma_iov[0].key, 0, false, m->context, flags | SRDSIM_EP(e)->util_ep.tx_msg_flags);
}

static ssize_t srdsim_write(struct fid_ep *e, const void *buf, size_t len, void *desc, fi_addr_t dest,
			    uint64_t addr, uint64_t key, void *context)
{
	SRDSIM_IOV1(iov, buf, len);
	return srdsim_rma(e, SRDSIM_OP_WRITE, &iov, &desc, 1, dest, addr, key, 0, false, context, SRDSIM_TXF(e));
}

static ssize_t srdsim_writev(struct fid_ep *e, const struct iovec *iov, void **desc, size_t count, fi_addr_t dest,
			     uint64_t addr, uint64_t key, void *context)
{
	return srdsim_rma(e, SRDSIM_OP_WRITE, iov, desc, count, dest, addr, key, 0, false, context, SRDSIM_TXF(e));
}

static ssize_t srdsim_writemsg(struct fid_ep *e, const struct fi_msg_rma *m, uint64_t flags)
{
	if (m->rma_iov_count != 1)
		return -FI_EINVAL;
	return srdsim_rma(e, SRDSIM_OP_WRITE, m->msg_iov, m->desc, m->iov_count, m->addr, m->rma_iov[0].addr,
			  m->rma_iov[0].key, m->data, flags & FI_REMOTE_CQ_DATA, m->context,
			  flags | SRDSIM_EP(e)->util_ep.tx_msg_flags);
}

static ssize_t srdsim_rma_inject(struct fid_ep *e, const void *buf, size_t len, fi_addr_t dest, uint64_t addr,
				 uint64_t key)
{
	SRDSIM_IOV1(iov, buf, len);
	return srdsim_rma(e, SRDSIM_OP_WRITE, &iov, NULL, 1, dest, addr, key, 0, false, NULL, FI_INJECT);
}

static ssize_t srdsim_writedata(struct fid_ep *e, const void *buf, size_t len, void *desc, uint64_t data,
				fi_addr_t dest, uint64_t addr, uint64_t key, void *context)
{
	SRDSIM_IOV1(iov, buf, len);
	return srdsim_rma(e, SRDSIM_OP_WRITE, &iov, &desc, 1, dest, addr, key, data, true, context, SRDSIM_TXF(e));
}

static ssize_t srdsim_rma_injectdata(struct fid_ep *e, const void *buf, size_t len, uint64_t data, fi_addr_t dest,
				     uint64_t addr, uint64_t key)
{
	SRDSIM_IOV1(iov, buf, len);
	return srdsim_rma(e, SRDSIM_OP_WRITE, &iov, NULL, 1, dest, addr, key, data, true, NULL, FI_INJECT);
}

struct fi_ops_rma srdsim_rma_ops = {
	.size = sizeof(struct fi_ops_rma),
	.read = srdsim_read,
	.readv = srdsim_readv,
	.readmsg = srdsim_readmsg,
	.write = srdsim_write,
	.writev = srdsim_writev,
	.writemsg = srdsim_writemsg,
	.inject = srdsim_rma_inject,
	.writedata = srdsim_writedata,
	.injectdata = srdsim_rma_injectdata,
};

/* ---- Lifetime -------------------------------------------------------------------------------------- */

/* Opens the listening socket, so the endpoint has its address before it is enabled. */
static uint64_t srdsim_endpoints_opened;

int srdsim_xfer_open(struct srdsim_ep *ep)
{
	uint64_t seed, index;
	struct sockaddr_in sin = { .sin_family = AF_INET, .sin_addr.s_addr = htonl(INADDR_LOOPBACK) };
	socklen_t len = sizeof(sin);
	struct srdsim_xfer *x;
	const char *mode = getenv("SRDSIM_MODE");
	const char *early = getenv("SRDSIM_FI_CQ_DATA_EARLY");

	x = calloc(1, sizeof(*x));
	if (!x)
		return -FI_ENOMEM;
	x->ep = ep;
	x->listen_fd = x->wake_fd = -1;
	dlist_init(&x->conns);
	dlist_init(&x->pending);
	dlist_init(&x->inflight);
	dlist_init(&x->recvs);
	dlist_init(&x->trecvs);
	dlist_init(&x->unexp);
	dlist_init(&x->tunexp);
	dlist_init(&x->inwrites);
	dlist_init(&x->events);
	dlist_init(&x->delayed);
	dlist_init(&x->cq_data_waiting);
	pthread_mutex_init(&x->mu, NULL);

	x->mode = SRDSIM_MODE_SRD;
	if (mode && !strcmp(mode, "inorder"))
		x->mode = SRDSIM_MODE_INORDER;
	else if (mode && *mode && strcmp(mode, "srd"))
		SRDSIM_WARN(FI_LOG_EP_CTRL, "SRDSIM_MODE=%s: the provider knows srd and inorder; using srd\n", mode);
	x->chunk = env_size("SRDSIM_CHUNK_BYTES", SRDSIM_DEFAULT_CHUNK, 128, 64u << 20);
	/*
	 * Seeded from SRDSIM_SEED and the order in which the process opens endpoints, not from anything random,
	 * so the same program with the same seed makes the same choices where its calls arrive in the same order.
	 */
	seed = env_size("SRDSIM_SEED", 1, 0, SIZE_MAX);
	index = __atomic_fetch_add(&srdsim_endpoints_opened, 1, __ATOMIC_RELAXED);
	x->rng = seed * 0x9E3779B97F4A7C15ull + index;
	x->agent_rng = ~x->rng;
	x->tx_size = ep->tx_size;
	x->direct = ep->direct;
	x->rx_cq_data = ep->rx_cq_data;
	x->scratch = malloc(x->chunk);
	if (!x->scratch) {
		ep->xfer = x;
		srdsim_xfer_close(ep);
		return -FI_ENOMEM;
	}
	x->cq_data_early = early && !strcmp(early, "1");
	x->jitter_us = x->mode == SRDSIM_MODE_SRD ? env_size("SRDSIM_FI_JITTER_US", 200, 0, 1000000) : 0;
	if (x->cq_data_early)
		SRDSIM_WARN(FI_LOG_EP_CTRL, "SRDSIM_FI_CQ_DATA_EARLY=1: remote CQ data is raised before its write "
			    "has landed, which breaks EFA's guarantee on purpose\n");

	x->listen_fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
	x->wake_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
	if (x->listen_fd < 0 || x->wake_fd < 0 || bind(x->listen_fd, (struct sockaddr *) &sin, sizeof(sin)) ||
	    listen(x->listen_fd, 128) || getsockname(x->listen_fd, (struct sockaddr *) &sin, &len)) {
		SRDSIM_WARN(FI_LOG_EP_CTRL, "cannot listen on the loopback interface: %s\n", strerror(errno));
		ep->xfer = x;
		srdsim_xfer_close(ep);
		return -FI_EOTHER;
	}
	ep->addr.port = sin.sin_port;
	ep->addr.ipv4 = sin.sin_addr.s_addr;
	ep->xfer = x;
	return 0;
}

int srdsim_xfer_start(struct srdsim_ep *ep)
{
	struct srdsim_xfer *x = ep->xfer;

	if (pthread_create(&x->agent, NULL, srdsim_agent, x))
		return -FI_EOTHER;
	x->agent_running = true;
	return 0;
}

void srdsim_xfer_close(struct srdsim_ep *ep)
{
	struct srdsim_xfer *x = ep->xfer;
	struct srdsim_conn *c;
	struct srdsim_op *op;
	struct srdsim_recv *r;
	struct srdsim_unexp *u;
	struct srdsim_inwrite *w;
	struct srdsim_event *e;
	struct dlist_entry *lists[2];
	bool pending;
	size_t i;

	if (!x)
		return;
	if (x->agent_running) {
		/*
		 * What is queued (acknowledgements, injected sends) goes out before the sockets close, for up to a
		 * second, so peers do not see the close as the loss of operations that had been delivered.
		 */
		for (i = 0; i < 1000; i++) {
			pthread_mutex_lock(&x->mu);
			pending = false;
			dlist_foreach_container(&x->conns, struct srdsim_conn, c, entry)
				pending = pending || c->out.len > c->out.off;
			pthread_mutex_unlock(&x->mu);
			if (!pending)
				break;
			srdsim_wake(x);
			usleep(1000);
		}
		pthread_mutex_lock(&x->mu);
		x->stop = true;
		pthread_mutex_unlock(&x->mu);
		srdsim_wake(x);
		pthread_join(x->agent, NULL);
	}
	while (!dlist_empty(&x->conns)) {
		c = container_of(x->conns.next, struct srdsim_conn, entry);
		srdsim_conn_lost(x, c);
	}
	lists[0] = &x->pending;
	lists[1] = &x->inflight;
	for (i = 0; i < 2; i++)
		while (!dlist_empty(lists[i])) {
			dlist_pop_front(lists[i], struct srdsim_op, op, entry);
			srdsim_op_free(op);
		}
	lists[0] = &x->recvs;
	lists[1] = &x->trecvs;
	for (i = 0; i < 2; i++)
		while (!dlist_empty(lists[i])) {
			dlist_pop_front(lists[i], struct srdsim_recv, r, entry);
			free(r);
		}
	lists[0] = &x->unexp;
	lists[1] = &x->tunexp;
	for (i = 0; i < 2; i++)
		while (!dlist_empty(lists[i])) {
			dlist_pop_front(lists[i], struct srdsim_unexp, u, entry);
			free(u->claimed);
			free(u->buf);
			free(u);
		}
	while (!dlist_empty(&x->inwrites)) {
		dlist_pop_front(&x->inwrites, struct srdsim_inwrite, w, entry);
		free(w);
	}
	while (!dlist_empty(&x->events)) {
		dlist_pop_front(&x->events, struct srdsim_event, e, entry);
		free(e);
	}
	while (!dlist_empty(&x->cq_data_waiting)) {
		struct srdsim_cq_data *q = container_of(x->cq_data_waiting.next, struct srdsim_cq_data, entry);

		dlist_remove(&q->entry);
		free(q);
	}
	while (!dlist_empty(&x->delayed)) {
		struct srdsim_delayed *d = container_of(x->delayed.next, struct srdsim_delayed, entry);

		dlist_remove(&d->entry);
		free(d);
	}
	if (x->listen_fd >= 0)
		close(x->listen_fd);
	if (x->wake_fd >= 0)
		close(x->wake_fd);
	pthread_mutex_destroy(&x->mu);
	free(x->scratch);
	free(x);
	ep->xfer = NULL;
}
