#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Assemble the NCCL plugin headers needed to type-check `nccl_rma_srdsim.cc` on a machine that
# has neither a built NCCL nor CUDA.
#
# On a real build host none of this is needed: use the installed NCCL's include directory (the
# `nvidia-nccl-cu13` wheel ships it) and the CUDA toolkit. This exists so the ABI can be checked
# from a laptop, which is where most of the editing happens.
#
# Usage: NCCL_SRC=/path/to/nccl ./prepare_nccl_headers.sh /path/to/output
set -euo pipefail

NCCL_SRC="${NCCL_SRC:-$HOME/tmp/deepep/nccl}"
OUT="${1:-$HOME/tmp/deepep/nccl-inc}"
STUB="$OUT/../cuda-stub"

if [ ! -f "$NCCL_SRC/src/nccl.h.in" ]; then
    echo "ERROR: $NCCL_SRC does not look like an NCCL checkout (set NCCL_SRC)" >&2
    exit 2
fi

mkdir -p "$OUT" "$STUB" "$OUT/nccl_device"

# nccl.h is generated from a template at configure time. The version numbers only matter for
# NCCL_VERSION_CODE comparisons, and 2.31.2 is the floor for the GIN device API.
sed -e 's/${nccl:Major}/2/'   -e 's/${nccl:Minor}/31/' \
    -e 's/${nccl:Patch}/2/'   -e 's/${nccl:Suffix}/-1/' \
    -e 's/${nccl:Version}/23102/' \
    "$NCCL_SRC/src/nccl.h.in" > "$OUT/nccl.h"

cp -R "$NCCL_SRC/src/include/plugin/." "$OUT/"
cp "$NCCL_SRC/src/include/nccl_common.h" "$OUT/"
cp "$NCCL_SRC/src/include/nccl_device/net_device.h" "$OUT/nccl_device/"

# Type-only CUDA stubs, in two parts.
#
# The first part is what nccl.h's own declarations need. The second declares every driver-API and
# NVRTC entry point the plugin calls, so that the CUDA path type-checks here too. That second part was
# added after a private member access inside `#ifdef SRDSIM_WITH_CUDA` compiled fine on a laptop -- the
# block was skipped -- and failed on the target twelve minutes later. Code that only ever compiles on
# the machine it runs on is code the local suite cannot check, and this suite exists to check it.
#
# Declarations only: nothing here links or runs. `make plugin-cuda` compiles with `-fsyntax-only`.
cat > "$STUB/cuda_runtime.h" <<'EOF'
#pragma once
#include <stddef.h>
#include <stdint.h>
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;   // nccl.h 2.32 has one in ncclCollConfig_t
typedef int cudaError_t;
EOF
# The driver API, for type-checking the plugin's device path (-DSRDSIM_WITH_CUDA) off a GPU host.
cat > "$STUB/cuda.h" <<'EOF'
#pragma once
#include <stddef.h>
#include <stdint.h>
typedef unsigned long long CUdeviceptr;
typedef int CUresult;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUstream_st* CUstream;
typedef int CUdevice;
#define CUDA_SUCCESS 0
// 600 in the real header. Named here because the health probe has to tell "still working" from
// "broken", and treating a pending stream as an error would report a fault on every busy run.
#define CUDA_ERROR_NOT_READY 600
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_INVALID_CONTEXT 201
#define CUDA_ERROR_LAUNCH_FAILED 719
#define CU_STREAM_NON_BLOCKING 1
#define CU_MEMHOSTALLOC_PORTABLE 1
#define CU_EVENT_DISABLE_TIMING 2
typedef struct CUevent_st* CUevent;
#define CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL 9
typedef unsigned long long CUmemGenericAllocationHandle;
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76
// Green contexts (CUDA 12.4+), for SRDSIM_DELIVERY_CONTEXT=green. Only the members the plugin touches.
typedef struct CUgreenCtx_st* CUgreenCtx;
typedef struct CUdevResourceDesc_st* CUdevResourceDesc;
typedef int CUdevResourceType;
#define CU_DEV_RESOURCE_TYPE_SM 1
#define CU_GREEN_CTX_DEFAULT_STREAM 1
typedef struct { unsigned int smCount; } CUdevSmResource;
typedef struct { CUdevResourceType type; unsigned char _internal_padding[92]; CUdevSmResource sm; } CUdevResource;
#ifdef __cplusplus
extern "C" {
#endif
CUresult cuDeviceGetDevResource(CUdevice dev, CUdevResource* result, CUdevResourceType type);
CUresult cuDevSmResourceSplitByCount(CUdevResource* result, unsigned int* nbGroups, const CUdevResource* input,
                                     CUdevResource* remaining, unsigned int useFlags, unsigned int minCount);
CUresult cuDevResourceGenerateDesc(CUdevResourceDesc* desc, CUdevResource* resources, unsigned int nbResources);
CUresult cuGreenCtxCreate(CUgreenCtx* ctx, CUdevResourceDesc desc, CUdevice dev, unsigned int flags);
CUresult cuCtxFromGreenCtx(CUcontext* ctx, CUgreenCtx green);
CUresult cuGreenCtxStreamCreate(CUstream* stream, CUgreenCtx green, unsigned int flags, int priority);
CUresult cuGreenCtxDestroy(CUgreenCtx green);
CUresult cuMemcpyHtoD(CUdeviceptr dst, const void* src, size_t bytes);
CUresult cuMemcpyDtoH(void* dst, CUdeviceptr src, size_t bytes);
CUresult cuMemcpyHtoDAsync(CUdeviceptr dst, const void* src, size_t bytes, CUstream stream);
CUresult cuMemcpyDtoHAsync(void* dst, CUdeviceptr src, size_t bytes, CUstream stream);
CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytes);
CUresult cuMemAllocHost(void** pp, size_t bytes);
CUresult cuMemHostAlloc(void** pp, size_t bytes, unsigned int flags);
CUresult cuMemFreeHost(void* p);
CUresult cuCtxGetCurrent(CUcontext* ctx);
CUresult cuCtxSetCurrent(CUcontext ctx);
CUresult cuCtxGetDevice(CUdevice* dev);
CUresult cuDeviceGet(CUdevice* dev, int ordinal);
CUresult cuDeviceGetAttribute(int* value, int attribute, CUdevice dev);
CUresult cuDevicePrimaryCtxRetain(CUcontext* ctx, CUdevice dev);
CUresult cuCtxCreate(CUcontext* ctx, unsigned int flags, CUdevice dev);
CUresult cuCtxPushCurrent(CUcontext ctx);
CUresult cuCtxPopCurrent(CUcontext* ctx);
CUresult cuEventCreate(CUevent* e, unsigned int flags);
CUresult cuEventRecord(CUevent e, CUstream s);
CUresult cuEventSynchronize(CUevent e);
CUresult cuEventQuery(CUevent e);
CUresult cuEventDestroy(CUevent e);
CUresult cuPointerGetAttribute(void* data, int attribute, CUdeviceptr ptr);
CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* h, void* addr);
CUresult cuMemRelease(CUmemGenericAllocationHandle h);
CUresult cuStreamCreate(CUstream* stream, unsigned int flags);
CUresult cuStreamSynchronize(CUstream stream);
CUresult cuStreamQuery(CUstream stream);
CUresult cuModuleLoadData(CUmodule* module, const void* image);
CUresult cuModuleGetFunction(CUfunction* func, CUmodule module, const char* name);
CUresult cuLaunchKernel(CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz,
                        unsigned int bx, unsigned int by, unsigned int bz,
                        unsigned int shared, CUstream stream, void** params, void** extra);
CUresult cuGetErrorName(CUresult error, const char** str);
#ifdef __cplusplus
}
#endif
EOF

# NVRTC, declarations only. The plugin compiles its one-line signal kernel at run time, so these are
# the entry points that have to exist for the CUDA path to type-check.
cat > "$STUB/nvrtc.h" <<'EOF'
#pragma once
#include <stddef.h>
typedef struct _nvrtcProgram* nvrtcProgram;
typedef int nvrtcResult;
#define NVRTC_SUCCESS 0
#ifdef __cplusplus
extern "C" {
#endif
nvrtcResult nvrtcCreateProgram(nvrtcProgram* prog, const char* src, const char* name,
                               int num_headers, const char* const* headers,
                               const char* const* include_names);
nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int num_options, const char* const* options);
nvrtcResult nvrtcDestroyProgram(nvrtcProgram* prog);
nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* size);
nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char* log);
nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t* size);
nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char* ptx);
nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog, size_t* size);
nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, char* cubin);
#ifdef __cplusplus
}
#endif
EOF
# GDRCopy, declarations only. The plugin writes device windows through a BAR mapping rather than with a
# CUDA copy, because a CUDA copy does not complete while DeepEP's kernels are resident. Stubbed here for
# the same reason the CUDA driver API is: so the path that only ever runs on a GPU host can still be
# type-checked where the editing happens.
cat > "$STUB/gdrapi.h" <<'EOF'
#pragma once
#include <stddef.h>
#include <stdint.h>
#define GPU_PAGE_SIZE 65536
typedef struct gdr* gdr_t;
typedef struct gdr_mh_s { unsigned long h; } gdr_mh_t;
typedef struct gdr_info { uint64_t va; uint64_t mapped_size; uint32_t page_size;
                          uint64_t tm_cycles; uint32_t cycles_per_ms; unsigned mapped:1;
                          unsigned wc_mapping:1; } gdr_info_t;
#ifdef __cplusplus
extern "C" {
#endif
gdr_t gdr_open(void);
int gdr_close(gdr_t g);
int gdr_pin_buffer(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token,
                   uint32_t va_space, gdr_mh_t* handle);
int gdr_unpin_buffer(gdr_t g, gdr_mh_t handle);
int gdr_map(gdr_t g, gdr_mh_t handle, void** va, size_t size);
int gdr_unmap(gdr_t g, gdr_mh_t handle, void* va, size_t size);
int gdr_get_info(gdr_t g, gdr_mh_t handle, gdr_info_t* info);
int gdr_copy_to_mapping(gdr_mh_t handle, void* map_d_ptr, const void* h_ptr, size_t size);
int gdr_copy_from_mapping(gdr_mh_t handle, void* h_ptr, const void* map_d_ptr, size_t size);
void gdr_runtime_get_version(int* major, int* minor);
int gdr_driver_get_version(gdr_t g, int* major, int* minor);
#ifdef __cplusplus
}
#endif
EOF
: > "$STUB/cuda_fp16.h"
: > "$STUB/cuda_bf16.h"
: > "$STUB/cuda_fp8.h"

echo "include tree:  $OUT"
echo "cuda stubs:    $STUB"
echo
echo "type-check with:"
echo "  c++ -std=c++17 -O2 -fPIC -Wall -Wextra -I $OUT -I $STUB -I \$(dirname \$0) \\"
echo "      -c \$(dirname \$0)/nccl_rma_srdsim.cc -o /tmp/srdsim.o"
