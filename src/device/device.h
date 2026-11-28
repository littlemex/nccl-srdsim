/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Types shared by the two halves of the device stand-in: device.c (the device,
 * PD, MR and AH) and queue.c (CQ, QP and the movement of data).
 */
#ifndef SRDSIM_DEVICE_H
#define SRDSIM_DEVICE_H

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

#include <infiniband/verbs.h>
#include <infiniband/efadv.h>

#include "profile.h"

#define EXPORT __attribute__((visibility("default")))

/* The completion status a device writes, as rdma-core's efa provider defines it (efa_io_defs.h). */
enum sd_status {
	SD_ST_OK = 0,
	SD_ST_FLUSHED = 1,
	SD_ST_LOCAL_QP_INTERNAL_ERROR = 2,
	SD_ST_LOCAL_UNSUPPORTED_OP = 3,
	SD_ST_LOCAL_INVALID_AH = 4,
	SD_ST_LOCAL_INVALID_LKEY = 5,
	SD_ST_LOCAL_BAD_LENGTH = 6,
	SD_ST_REMOTE_BAD_ADDRESS = 7,
	SD_ST_REMOTE_ABORT = 8,
	SD_ST_REMOTE_BAD_DEST_QPN = 9,
	SD_ST_REMOTE_RNR = 10,
	SD_ST_REMOTE_BAD_LENGTH = 11,
	SD_ST_REMOTE_BAD_STATUS = 12,
	SD_ST_LOCAL_UNRESP_REMOTE = 13,
	SD_ST_REMOTE_UNKNOWN_PEER = 14,
	SD_ST_LOCAL_UNREACH_REMOTE = 15,
};

struct sd_device {
	struct ibv_device dev;
	int index;
};

struct sd_agent;

struct sd_context {
	struct verbs_context vctx;
	struct sd_device *device;
	pthread_mutex_t lock;		/* the lists below, and every QP's queues */
	uint16_t next_ahn;
	uint32_t next_key;
	struct sd_mr *mrs;
	struct sd_ah *ahs;
	struct sd_qp *qps;
	struct sd_agent *agent;
};

struct sd_pd {
	struct ibv_pd pd;
	atomic_int users;
};

/*
 * rdma-core allocates every MR as struct verbs_mr (infiniband/driver.h, not
 * installed by distributions). The efa unit tests read its access field
 * through their own copy of the layout, so the stand-in starts with the same.
 */
struct sd_mr {
	struct ibv_mr ibv_mr;
	int mr_type;			/* enum ibv_mr_type; 0 is IBV_MR_TYPE_MR */
	int access;
	uint64_t iova;
	struct sd_mr *next;
};

struct sd_ah {
	struct ibv_ah ah;
	uint16_t ahn;
	union ibv_gid gid;
	struct sd_ah *next;
};

void sd_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
bool sd_log_enabled(void);

static inline struct sd_context *sd_ctx(struct ibv_context *c)
{
	return (struct sd_context *)((char *)c - offsetof(struct verbs_context, context));
}

/* The device index a GID names; the stand-in's GIDs end in it. */
static inline int sd_gid_device(const union ibv_gid *gid)
{
	return gid->raw[15];
}

void sd_gid_of(int device, union ibv_gid *gid);

/* Under ctx->lock: the MR whose key is key, or NULL. */
struct sd_mr *sd_mr_find(struct sd_context *ctx, uint32_t key);

/* Under ctx->lock: the AH number this context gave gid, or 0xffff when it has none. */
uint16_t sd_ahn_of(struct sd_context *ctx, const union ibv_gid *gid);

/* queue.c */
int sd_agent_start(struct sd_context *ctx);
void sd_agent_stop(struct sd_context *ctx);

#endif
