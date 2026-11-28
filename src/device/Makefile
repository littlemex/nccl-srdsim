# SPDX-License-Identifier: Apache-2.0
# Builds the device stand-in as libibverbs.so.1 and libefa.so.1 in OUT.
#
#   make -C src/device OUT=/path/to/dir [MUTATION=-DSD_MUTATION_...]
#
# The efadv and verbs headers on the build host decide which calls the efa
# provider is compiled to make, so the same headers decide which of those
# calls the stand-in defines. Each probe compiles one line against them.
CC ?= cc
CFLAGS ?= -O2 -g
OUT ?= lib
SRC := $(dir $(lastword $(MAKEFILE_LIST)))

probe = $(shell printf '#include <infiniband/verbs.h>\n#include <infiniband/efadv.h>\n%s\n' '$(1)' \
	| $(CC) -fsyntax-only -Werror -x c - 2>/dev/null && echo 1 || echo 0)

FEATURES := \
	-DSD_HAVE_EFADV_QUERY_QP_WQS=$(call probe,int f(struct efadv_wq_attr *a) { return efadv_query_qp_wqs(0, a, a, sizeof(*a)); }) \
	-DSD_HAVE_EFADV_QUERY_CQ=$(call probe,int f(struct efadv_cq_attr *a) { return efadv_query_cq(0, a, sizeof(*a)); }) \
	-DSD_HAVE_EFADV_WC_IS_UNSOLICITED=$(call probe,int f(struct efadv_cq *c) { return c->wc_is_unsolicited != 0; }) \
	-DSD_HAVE_IBV_QUERY_QP_DATA_IN_ORDER=$(call probe,int f(struct ibv_qp *q) { return ibv_query_qp_data_in_order(q, IBV_WR_SEND, 0); })

ALL_CFLAGS := $(CFLAGS) $(MUTATION) -std=gnu11 -fPIC -Wall -Wextra -Werror -fvisibility=hidden $(FEATURES)

all: $(OUT)/libibverbs.so.1 $(OUT)/libefa.so.1

$(OUT):
	mkdir -p $@

$(OUT)/libibverbs.so.1: $(SRC)device.c $(SRC)profile.h $(SRC)libibverbs.map | $(OUT)
	$(CC) $(ALL_CFLAGS) -shared -Wl,-soname,libibverbs.so.1 -Wl,--version-script=$(SRC)libibverbs.map \
		-o $@ $(SRC)device.c -lpthread

$(OUT)/libefa.so.1: $(SRC)device.c $(SRC)profile.h $(SRC)libefa.map | $(OUT)
	$(CC) $(ALL_CFLAGS) -shared -Wl,-soname,libefa.so.1 -Wl,--version-script=$(SRC)libefa.map \
		-o $@ $(SRC)device.c -lpthread

features:
	@echo $(FEATURES)

.PHONY: all features
