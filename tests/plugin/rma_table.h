// SPDX-License-Identifier: Apache-2.0
// The plugin's op table, loaded the way NCCL loads it: dlopen the library, look up ncclRmaPlugin_vN for
// one version N, and call through it. The wrappers give every version the same calls, so a scenario runs
// unchanged on v14, v15 and v16; what a newer table adds (the optFlags of v15, getRmaProperties of v16) is
// passed or skipped by version.
#pragma once

#include "nccl_rma.h"

#include <dlfcn.h>

#include <cstdio>
#include <string>

// Calls to a field that only v16 and later define, and to the v15 form of a call, by the loaded table.
#if defined(RMA_V16_H_)
#define V16(expr) (v16_->expr)
#define V15UP(expr) (v15_ ? (v15_->expr) : (v16_->expr))
#else
#define V16(expr) (ncclInternalError)
#define V15UP(expr) (v15_->expr)
#endif

namespace harness {

class RmaTable {
public:
    // Loads `path` and looks up ncclRmaPlugin_v<version>. Returns false, with `why`, when either fails.
    bool load(const std::string& path, int version, std::string* why) {
        lib_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (lib_ == nullptr) { *why = std::string("dlopen: ") + dlerror(); return false; }
        const std::string sym = "ncclRmaPlugin_v" + std::to_string(version);
        void* t = dlsym(lib_, sym.c_str());
        if (t == nullptr) { *why = "no symbol " + sym; return false; }
        version_ = version;
        switch (version) {
            case 14: v14_ = static_cast<ncclRma_v14_t*>(t); break;
            case 15: v15_ = static_cast<ncclRma_v15_t*>(t); break;
#if defined(RMA_V16_H_)
            case 16: v16_ = static_cast<ncclRma_v16_t*>(t); break;
#endif
            default: *why = "the harness does not know table version " + std::to_string(version); return false;
        }
        return true;
    }

    int version() const { return version_; }

    // Every call below goes to the same field of whichever table was loaded.
#define SRDSIM_TABLE_CALL(expr) \
    (v14_ ? (v14_->expr) : v15_ ? (v15_->expr) : V16(expr))

    ncclResult_t init(void** ctx, uint64_t commId, ncclDebugLogger_t log) { return SRDSIM_TABLE_CALL(init(ctx, commId, log)); }
    ncclResult_t devices(int* n) { return SRDSIM_TABLE_CALL(devices(n)); }
    ncclResult_t getProperties(int dev, ncclNetProperties_v12_t* p) { return SRDSIM_TABLE_CALL(getProperties(dev, p)); }
    ncclResult_t listen(void* ctx, int dev, void* handle, void** listenComm) {
        return SRDSIM_TABLE_CALL(listen(ctx, dev, handle, listenComm));
    }
    ncclResult_t connect(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm) {
        return SRDSIM_TABLE_CALL(connect(ctx, handles, nranks, rank, listenComm, collComm));
    }
    ncclResult_t createContext(void* collComm, int nContexts, void** rmaCtx) {
        ncclRmaConfig_v14_t config{};
        config.nContexts = nContexts;
        config.trafficClass = 0;
        config.rankStride = 1;
        return SRDSIM_TABLE_CALL(createContext(collComm, &config, rmaCtx));
    }
    ncclResult_t regMrSym(void* collComm, void* data, size_t size, int type, uint64_t mrFlags, void** mh) {
        return SRDSIM_TABLE_CALL(regMrSym(collComm, data, size, type, mrFlags, mh));
    }
    ncclResult_t deregMrSym(void* collComm, void* mh) { return SRDSIM_TABLE_CALL(deregMrSym(collComm, mh)); }
    ncclResult_t destroyContext(void* rmaCtx) { return SRDSIM_TABLE_CALL(destroyContext(rmaCtx)); }
    ncclResult_t closeColl(void* collComm) { return SRDSIM_TABLE_CALL(closeColl(collComm)); }
    ncclResult_t closeListen(void* listenComm) { return SRDSIM_TABLE_CALL(closeListen(listenComm)); }
    ncclResult_t test(void* collComm, void* request, int* done) { return SRDSIM_TABLE_CALL(test(collComm, request, done)); }
    ncclResult_t progress(void* rmaCtx) { return SRDSIM_TABLE_CALL(rmaProgress(rmaCtx)); }
    ncclResult_t queryLastError(void* rmaCtx, bool* hasError) { return SRDSIM_TABLE_CALL(queryLastError(rmaCtx, hasError)); }
    ncclResult_t finalize(void* ctx) { return SRDSIM_TABLE_CALL(finalize(ctx)); }

    ncclResult_t iput(void* rmaCtx, int context, uint64_t srcOff, void* srcMh, size_t size, uint64_t dstOff,
                      void* dstMh, uint32_t rank, uint32_t optFlags, void** request) {
        if (v14_) return v14_->iput(rmaCtx, context, srcOff, srcMh, size, dstOff, dstMh, rank, request);
        return V15UP(iput(rmaCtx, context, srcOff, srcMh, size, dstOff, dstMh, rank, optFlags, request));
    }
    ncclResult_t iputSignal(void* rmaCtx, int context, uint64_t srcOff, void* srcMh, size_t size, uint64_t dstOff,
                            void* dstMh, uint32_t rank, uint64_t signalOff, void* signalMh, uint64_t signalValue,
                            uint32_t signalOp, bool strong, uint32_t optFlags, void** request) {
        if (v14_)
            return v14_->iputSignal(rmaCtx, context, srcOff, srcMh, size, dstOff, dstMh, rank, signalOff, signalMh,
                                    signalValue, signalOp, strong, request);
        return V15UP(iputSignal(rmaCtx, context, srcOff, srcMh, size, dstOff, dstMh, rank, signalOff, signalMh,
                                signalValue, signalOp, strong, optFlags, request));
    }
    ncclResult_t iget(void* rmaCtx, int context, uint64_t remoteOff, void* remoteMh, size_t size, uint64_t localOff,
                      void* localMh, uint32_t rank, uint32_t optFlags, void** request) {
        if (v14_) return v14_->iget(rmaCtx, context, remoteOff, remoteMh, size, localOff, localMh, rank, request);
        return V15UP(iget(rmaCtx, context, remoteOff, remoteMh, size, localOff, localMh, rank, optFlags, request));
    }
    ncclResult_t iflush(void* rmaCtx, int context, void* mh, uint32_t rank, void** request) {
        return SRDSIM_TABLE_CALL(iflush(rmaCtx, context, mh, rank, request));
    }
#undef SRDSIM_TABLE_CALL

private:
    void* lib_ = nullptr;
    int version_ = 0;
    ncclRma_v14_t* v14_ = nullptr;
    ncclRma_v15_t* v15_ = nullptr;
#if defined(RMA_V16_H_)
    ncclRma_v16_t* v16_ = nullptr;
#endif
};

}  // namespace harness

#undef V16
#undef V15UP
