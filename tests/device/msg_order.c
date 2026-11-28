/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Checks EFA-MSG-001: on the efa fabric's RDM endpoints, messages from one endpoint to another are
 * received in the order they were sent, although SRD below delivers them in any order (EFA-MSG-002).
 *
 *   msg_order [COUNT]
 *
 * One process opens two endpoints on the same domain, sends COUNT numbered messages from the first to
 * the second, sizes alternating between one that fits a packet and one that does not, and checks that
 * the i-th receive holds message i. Two endpoints on one host would use shm unless
 * FI_EFA_ENABLE_SHM_TRANSFER=0, which the caller sets so that the messages cross the device.
 * Exits 0 when every message arrived in order, 1 when one did not, 2 on any other failure.
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>

#define SMALL 64
#define LARGE (64 * 1024)
#define WINDOW 64

#define CHECK(call) do { int r_ = (int)(call); if (r_) { \
	fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, fi_strerror(-r_)); exit(2); } } while (0)

struct side {
	struct fid_ep *ep;
	struct fid_cq *cq;
	struct fid_av *av;
};

static void open_side(struct fid_domain *dom, struct fi_info *info, struct side *s)
{
	struct fi_cq_attr cq_attr = { .format = FI_CQ_FORMAT_CONTEXT, .size = 4 * WINDOW };
	struct fi_av_attr av_attr = { .type = FI_AV_TABLE };

	CHECK(fi_endpoint(dom, info, &s->ep, NULL));
	CHECK(fi_cq_open(dom, &cq_attr, &s->cq, NULL));
	CHECK(fi_av_open(dom, &av_attr, &s->av, NULL));
	CHECK(fi_ep_bind(s->ep, &s->cq->fid, FI_SEND | FI_RECV));
	CHECK(fi_ep_bind(s->ep, &s->av->fid, 0));
	CHECK(fi_enable(s->ep));
}

static size_t size_of(long i)
{
	return i % 2 ? LARGE : SMALL;
}

/* Returns the number of completions read, after reporting an error completion. */
static int drain(struct fid_cq *cq, long *done)
{
	struct fi_cq_entry e[16];
	ssize_t n = fi_cq_read(cq, e, 16);

	if (n == -FI_EAVAIL) {
		struct fi_cq_err_entry err = { 0 };

		fi_cq_readerr(cq, &err, 0);
		fprintf(stderr, "completion error: %s (prov %d)\n", fi_strerror(err.err), err.prov_errno);
		exit(2);
	}
	if (n > 0)
		*done += n;
	return n > 0 ? (int)n : 0;
}

int main(int argc, char **argv)
{
	long count = argc > 1 ? atol(argv[1]) : 2000, sent = 0, send_done = 0, posted = 0, recv_done = 0;
	struct fi_info *hints = fi_allocinfo(), *info;
	struct fid_fabric *fab;
	struct fid_domain *dom;
	struct side a, b;
	char name[256];
	size_t len = sizeof(name);
	fi_addr_t to_b;
	char **rbuf, **sbuf;
	long i, bad = 0;

	hints->caps = FI_MSG;
	hints->ep_attr->type = FI_EP_RDM;
	hints->fabric_attr->prov_name = strdup("efa");
	hints->fabric_attr->name = strdup("efa");
	hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
	hints->mode = FI_CONTEXT;
	CHECK(fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0, hints, &info));
	CHECK(fi_fabric(info->fabric_attr, &fab, NULL));
	CHECK(fi_domain(fab, info, &dom, NULL));
	open_side(dom, info, &a);
	open_side(dom, info, &b);
	CHECK(fi_getname(&b.ep->fid, name, &len));
	if (fi_av_insert(a.av, name, 1, &to_b, 0, NULL) != 1) {
		fprintf(stderr, "fi_av_insert failed\n");
		return 2;
	}

	/* A buffer per message, so that up to WINDOW sends are in flight and SRD can reorder them. */
	rbuf = calloc((size_t)count, sizeof(*rbuf));
	sbuf = calloc((size_t)count, sizeof(*sbuf));
	if (!rbuf || !sbuf)
		return 2;
	for (i = 0; i < count; i++) {
		rbuf[i] = calloc(1, size_of(i));
		sbuf[i] = calloc(1, size_of(i));
		if (!rbuf[i] || !sbuf[i])
			return 2;
		memcpy(sbuf[i], &i, sizeof(i));
		memcpy(sbuf[i] + size_of(i) - sizeof(i), &i, sizeof(i));
	}

	while (recv_done < count || send_done < count) {
		while (posted < count && posted - recv_done < WINDOW) {
			ssize_t r = fi_recv(b.ep, rbuf[posted], size_of(posted), NULL, FI_ADDR_UNSPEC, NULL);

			if (r == -FI_EAGAIN)
				break;
			CHECK(r);
			posted++;
		}
		while (sent < count && sent - send_done < WINDOW && sent < posted) {
			ssize_t r = fi_send(a.ep, sbuf[sent], size_of(sent), NULL, to_b, NULL);

			if (r == -FI_EAGAIN)
				break;
			CHECK(r);
			sent++;
		}
		drain(a.cq, &send_done);
		drain(b.cq, &recv_done);
	}

	for (i = 0; i < count; i++) {
		long head, tail;

		memcpy(&head, rbuf[i], sizeof(head));
		memcpy(&tail, rbuf[i] + size_of(i) - sizeof(tail), sizeof(tail));
		if (head != i || tail != i) {
			if (bad < 5)
				fprintf(stderr, "receive %ld holds message %ld (tail %ld)\n", i, head, tail);
			bad++;
		}
	}
	printf("%s %ld of %ld messages out of order\n", bad ? "FAIL" : "PASS", bad, count);
	CHECK(fi_close(&a.ep->fid));
	CHECK(fi_close(&b.ep->fid));
	CHECK(fi_close(&a.av->fid));
	CHECK(fi_close(&b.av->fid));
	CHECK(fi_close(&a.cq->fid));
	CHECK(fi_close(&b.cq->fid));
	CHECK(fi_close(&dom->fid));
	CHECK(fi_close(&fab->fid));
	fi_freeinfo(info);
	fi_freeinfo(hints);
	for (i = 0; i < count; i++) {
		free(rbuf[i]);
		free(sbuf[i]);
	}
	free(rbuf);
	free(sbuf);
	return bad ? 1 : 0;
}
