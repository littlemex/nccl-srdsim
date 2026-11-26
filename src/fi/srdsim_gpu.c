/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Copies to and from GPU memory for the srdsim provider, made from a CUDA context of the provider's own.
 *
 * WHY NOT ofi_copy_{to,from}_hmem
 * libfabric's CUDA copy runs in the device's primary context. An application whose kernels are resident in that
 * context and spinning for the bytes this provider delivers (DeepEP's GIN kernels) holds the copy until the kernel
 * gives up: the kernel waits for the copy and the copy waits for the kernel. A module load or a free in the
 * primary context holds every other driver call in it the same way. A second context of the same process is not
 * held by that lock, and its non-blocking stream has no implicit dependency on the application's streams.
 *
 * So the provider copies from a context it creates itself, through a non-blocking stream, out of pinned host
 * memory (a pageable source makes the driver stage through its own pool and wait on the device). This is the
 * same design as the plugin's delivery context in src/nccl_rma_srdsim.cc.
 *
 * The CUDA driver is loaded with dlopen so the provider builds inside a libfabric tree with no CUDA headers and
 * loads on a host with no GPU. The types below are the driver API's, by value: they are ABI, not private.
 *
 * What the own context cannot address (an allocation that is not a VMM allocation, so its mapping belongs to
 * the context that made it) is refused with -FI_ENOSYS before any copy is submitted, and the caller falls back
 * to libfabric's copy, which is the path that can deadlock. That is said once on stderr, not silently, and
 * SRDSIM_DELIVERY_CONTEXT=strict fails the transfer instead. SRDSIM_DELIVERY_CONTEXT=primary takes the libfabric
 * path always, to reproduce the deadlock on purpose.
 *
 * Return values: 0; -FI_ENOSYS when nothing was submitted and libfabric's copy may be used; any other error is
 * final and must not be retried through libfabric (a failed copy, or a process that is exiting).
 */
#include "srdsim.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef int cu_result;
typedef int cu_device;
typedef unsigned long long cu_deviceptr;
typedef void *cu_context;
typedef void *cu_stream;
typedef unsigned long long cu_alloc_handle;

#define CU_SUCCESS 0
#define CU_STREAM_NON_BLOCKING 1
#define CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL 9

/* One staging buffer for the process, used under gpu_copy_mu (there is one stream, so concurrent copies would only
 * wait for each other's synchronise), and a copy is split to fit it. Never freed: a free is a driver call that
 * waits behind the application's resident kernels. */
#define SRDSIM_GPU_STAGE ((size_t) 4 << 20)

/* The symbols carry their _v2 names: without cuda.h their unversioned names are the pre-3.2 ABI. */
struct srdsim_cuda {
	cu_result (*cuInit)(unsigned int);
	cu_result (*cuDeviceGet)(cu_device *, int);
	cu_result (*cuCtxCreate_v4)(cu_context *, void *, unsigned int, cu_device);
	cu_result (*cuCtxCreate_v2)(cu_context *, unsigned int, cu_device);
	cu_result (*cuCtxGetCurrent)(cu_context *);
	cu_result (*cuCtxPushCurrent_v2)(cu_context);
	cu_result (*cuCtxPopCurrent_v2)(cu_context *);
	cu_result (*cuStreamCreate)(cu_stream *, unsigned int);
	cu_result (*cuStreamSynchronize)(cu_stream);
	cu_result (*cuMemHostAlloc)(void **, size_t, unsigned int);
	cu_result (*cuMemcpyHtoDAsync_v2)(cu_deviceptr, const void *, size_t, cu_stream);
	cu_result (*cuMemcpyDtoHAsync_v2)(void *, cu_deviceptr, size_t, cu_stream);
	cu_result (*cuPointerGetAttribute)(void *, int, cu_deviceptr);
	cu_result (*cuMemRetainAllocationHandle)(cu_alloc_handle *, void *);
	cu_result (*cuMemRelease)(cu_alloc_handle);
};

static struct srdsim_cuda cu;
static pthread_mutex_t gpu_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t gpu_copy_mu = PTHREAD_MUTEX_INITIALIZER;
static int gpu_state;		/* 0 not tried, 1 usable, -1 not usable */
static cu_context gpu_ctx;
static cu_stream gpu_stream;
static void *gpu_stage;
static int gpu_ordinal = -1;
static atomic_int gpu_exiting;
static atomic_int gpu_in_driver;
static atomic_int gpu_said_fallback;

static void srdsim_gpu_say(const char *what, int rc)
{
	fprintf(stderr, "srdsim-fi: %s (%d)\n", what, rc);
}

static int srdsim_gpu_env_is(const char *name, const char *value)
{
	const char *e = getenv(name);

	return e && !strcmp(e, value);
}

static int srdsim_gpu_trace(void)
{
	return srdsim_gpu_env_is("SRDSIM_FI_TRACE", "1");
}

static int srdsim_gpu_strict(void)
{
	return srdsim_gpu_env_is("SRDSIM_DELIVERY_CONTEXT", "strict");
}

static int srdsim_gpu_primary_requested(void)
{
	return srdsim_gpu_env_is("SRDSIM_DELIVERY_CONTEXT", "primary");
}

/* After this runs no thread enters the driver again: driver teardown at exit frees structures a copy in flight
 * would land in. Handlers run in reverse order of registration and this one is registered after the CUDA
 * runtime initialised, so it runs first. */
static void srdsim_gpu_at_exit(void)
{
	int i;

	atomic_store(&gpu_exiting, 1);
	for (i = 0; i < 2000 && atomic_load(&gpu_in_driver) != 0; i++)
		usleep(1000);
}

#define LOAD(lib, name) ((cu.name = dlsym(lib, #name)) != NULL)

static int srdsim_gpu_load(void)
{
	void *lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);

	if (!lib)
		return -1;
	if (!(LOAD(lib, cuInit) && LOAD(lib, cuDeviceGet) && LOAD(lib, cuCtxGetCurrent) &&
	      LOAD(lib, cuCtxPushCurrent_v2) && LOAD(lib, cuCtxPopCurrent_v2) && LOAD(lib, cuStreamCreate) &&
	      LOAD(lib, cuStreamSynchronize) && LOAD(lib, cuMemHostAlloc) && LOAD(lib, cuMemcpyHtoDAsync_v2) &&
	      LOAD(lib, cuMemcpyDtoHAsync_v2) && LOAD(lib, cuPointerGetAttribute) &&
	      LOAD(lib, cuMemRetainAllocationHandle) && LOAD(lib, cuMemRelease)))
		return -1;
	cu.cuCtxCreate_v4 = dlsym(lib, "cuCtxCreate_v4");
	cu.cuCtxCreate_v2 = dlsym(lib, "cuCtxCreate_v2");
	return (cu.cuCtxCreate_v4 || cu.cuCtxCreate_v2) ? 0 : -1;
}

/* Called with gpu_mu held. Makes the context and the stream for `ordinal`, and puts the calling thread's context
 * back as it found it: cuCtxCreate makes the new context current on a thread that belongs to the application. */
static int srdsim_gpu_open_locked(int ordinal)
{
	cu_device dev;
	cu_context ctx = NULL, popped = NULL;
	cu_stream stream = NULL;
	cu_result rc;

	if (gpu_state != 0)
		return gpu_state == 1 ? 0 : -FI_ENOSYS;
	gpu_state = -1;
	/* Registered whatever the outcome, so that a process that exits while the agents still copy through libfabric's
	 * path (SRDSIM_DELIVERY_CONTEXT=primary, or a window the own context cannot address) is stopped as well. */
	atexit(srdsim_gpu_at_exit);
	if (srdsim_gpu_primary_requested())
		return -FI_ENOSYS;
	if (srdsim_gpu_load() || cu.cuInit(0) != CU_SUCCESS || cu.cuDeviceGet(&dev, ordinal) != CU_SUCCESS)
		return -FI_ENOSYS;
	rc = cu.cuCtxCreate_v4 ? cu.cuCtxCreate_v4(&ctx, NULL, 0, dev) : cu.cuCtxCreate_v2(&ctx, 0, dev);
	if (rc != CU_SUCCESS) {
		srdsim_gpu_say("cannot create the provider's own CUDA context", rc);
		return -FI_ENOSYS;
	}
	rc = cu.cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
	if (cu.cuCtxPopCurrent_v2(&popped) != CU_SUCCESS || popped != ctx) {
		/* The application's thread would stay in our context; nothing safe is left to do. */
		srdsim_gpu_say("cannot restore the registering thread's CUDA context; GPU copies disabled", 0);
		return -FI_ENOSYS;
	}
	if (rc != CU_SUCCESS) {
		srdsim_gpu_say("cannot create the delivery stream", rc);
		return -FI_ENOSYS;
	}
	gpu_ctx = ctx;
	gpu_stream = stream;
	gpu_ordinal = ordinal;
	gpu_state = 1;
	return 0;
}

/* Whether the own context can address `ptr`: only a VMM allocation's mapping belongs to the process rather
 * than to the context that made it. */
static int srdsim_gpu_addressable(void *ptr)
{
	cu_alloc_handle h;

	if (cu.cuMemRetainAllocationHandle(&h, ptr) != CU_SUCCESS)
		return 0;
	cu.cuMemRelease(h);
	return 1;
}

/* A GPU window the own context cannot use. By default libfabric's copy takes over, said once on stderr because that
 * path waits for kernels resident in the application's context; with SRDSIM_DELIVERY_CONTEXT=strict the transfer
 * fails instead, so a run never silently depends on the path that can hang. */
static int srdsim_gpu_refuse(const char *why)
{
	if (srdsim_gpu_strict()) {
		fprintf(stderr, "srdsim-fi: %s (SRDSIM_DELIVERY_CONTEXT=strict: the transfer fails)\n", why);
		return -FI_EINVAL;
	}
	if (!atomic_exchange(&gpu_said_fallback, 1))
		fprintf(stderr, "srdsim-fi: %s; falling back to libfabric's CUDA copy, which waits for any kernel "
				"resident in the application's context. Allocate windows with ncclMemAlloc "
				"(NCCL_CUMEM_ENABLE=1), or set SRDSIM_DELIVERY_CONTEXT=strict to fail instead.\n", why);
	return -FI_ENOSYS;
}

/* One scope: the own context current on this thread, and the previous one restored on the way out. */
struct srdsim_gpu_scope {
	int pushed;
};

static int srdsim_gpu_enter(struct srdsim_gpu_scope *s)
{
	cu_context current = NULL;

	s->pushed = 0;
	atomic_fetch_add(&gpu_in_driver, 1);
	if (atomic_load(&gpu_exiting))
		return -FI_ESHUTDOWN;
	if (cu.cuCtxGetCurrent(&current) == CU_SUCCESS && current == gpu_ctx)
		return 0;
	if (cu.cuCtxPushCurrent_v2(gpu_ctx) != CU_SUCCESS)
		return -FI_EOTHER;
	s->pushed = 1;
	return 0;
}

static void srdsim_gpu_leave(struct srdsim_gpu_scope *s)
{
	cu_context popped = NULL;

	if (s->pushed && (cu.cuCtxPopCurrent_v2(&popped) != CU_SUCCESS || popped != gpu_ctx))
		srdsim_gpu_say("restoring this thread's CUDA context failed; its later CUDA calls may run in the "
			       "wrong context", 0);
	atomic_fetch_sub(&gpu_in_driver, 1);
}

static int srdsim_gpu_copy(uint64_t device, void *dev_ptr, void *host, size_t len, int to_device)
{
	struct srdsim_gpu_scope scope;
	struct timespec t0, t1;
	size_t done = 0, n;
	cu_result rc = CU_SUCCESS;
	int ret, ordinal = -1;

	if (len == 0)
		return 0;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	pthread_mutex_lock(&gpu_mu);
	ret = srdsim_gpu_open_locked((int) (uint32_t) device);
	pthread_mutex_unlock(&gpu_mu);
	if (ret == -FI_ENOSYS && !srdsim_gpu_primary_requested())
		return srdsim_gpu_refuse("the provider's own CUDA context is not available");
	if (ret)
		return ret;
	if (gpu_ordinal != (int) (uint32_t) device)
		return srdsim_gpu_refuse("a GPU window on a second device; one device per process is supported");

	ret = srdsim_gpu_enter(&scope);
	if (ret) {
		srdsim_gpu_leave(&scope);
		return ret;
	}
	/* Both ends: a VMM reservation can hold several mappings. */
	if (cu.cuPointerGetAttribute(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (cu_deviceptr) (uintptr_t) dev_ptr) !=
		    CU_SUCCESS || ordinal != gpu_ordinal || !srdsim_gpu_addressable(dev_ptr) ||
	    !srdsim_gpu_addressable((char *) dev_ptr + len - 1)) {
		srdsim_gpu_leave(&scope);
		return srdsim_gpu_refuse("a GPU window is not a VMM allocation of the provider's device, so its own "
					 "context cannot address it");
	}
	pthread_mutex_lock(&gpu_copy_mu);
	if (!gpu_stage && cu.cuMemHostAlloc(&gpu_stage, SRDSIM_GPU_STAGE, 0) != CU_SUCCESS) {
		gpu_stage = NULL;
		pthread_mutex_unlock(&gpu_copy_mu);
		srdsim_gpu_leave(&scope);
		return srdsim_gpu_refuse("cannot pin staging memory");
	}
	while (done < len && rc == CU_SUCCESS) {
		n = len - done < SRDSIM_GPU_STAGE ? len - done : SRDSIM_GPU_STAGE;
		if (to_device) {
			memcpy(gpu_stage, (const char *) host + done, n);
			rc = cu.cuMemcpyHtoDAsync_v2((cu_deviceptr) (uintptr_t) ((char *) dev_ptr + done), gpu_stage, n,
						     gpu_stream);
		} else {
			rc = cu.cuMemcpyDtoHAsync_v2(gpu_stage, (cu_deviceptr) (uintptr_t) ((char *) dev_ptr + done), n,
						     gpu_stream);
		}
		if (rc == CU_SUCCESS)
			rc = cu.cuStreamSynchronize(gpu_stream);
		if (rc == CU_SUCCESS && !to_device)
			memcpy((char *) host + done, gpu_stage, n);
		done += n;
	}
	pthread_mutex_unlock(&gpu_copy_mu);
	srdsim_gpu_leave(&scope);
	if (srdsim_gpu_trace()) {
		clock_gettime(CLOCK_MONOTONIC, &t1);
		fprintf(stderr, "srdsim-fi: own-context copy %s %zu bytes, %ld us, rc %d\n", to_device ? "to" : "from",
			len, (long) ((t1.tv_sec - t0.tv_sec) * 1000000 + (t1.tv_nsec - t0.tv_nsec) / 1000), (int) rc);
	}
	if (rc != CU_SUCCESS) {
		srdsim_gpu_say(to_device ? "a copy into GPU memory failed" : "a copy out of GPU memory failed", rc);
		return -FI_EIO;
	}
	return 0;
}

int srdsim_gpu_copy_to(uint64_t device, void *dst, const void *src, size_t len)
{
	return srdsim_gpu_copy(device, dst, (void *) src, len, 1);
}

int srdsim_gpu_copy_from(uint64_t device, void *dst, const void *src, size_t len)
{
	return srdsim_gpu_copy(device, (void *) src, dst, len, 0);
}

int srdsim_gpu_exiting(void)
{
	return atomic_load(&gpu_exiting);
}
