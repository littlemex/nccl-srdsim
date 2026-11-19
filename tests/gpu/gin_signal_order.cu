// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// A GIN user that depends on exactly one ordering guarantee: invariant I1, a weak signal becomes
// visible only after its own put has landed. Ranks form a ring of blocks. In every round each block
// puts a buffer to the next block in chunks, each chunk carrying a weak signal increment; the receiver
// waits until the signal has counted every chunk of the round and then checks every element. Nothing
// else orders the data, so a signal that overtakes its own put shows up as a wrong element.
//
// Under a delivery that keeps I1 (srd, inorder, ...) no element may be wrong. Under violate-i1 some
// seed should find one; a run that never does would mean the check is not reading what it guards.
//
// Usage: gin_signal_order RANK NRANKS ID_FILE [ROUNDS] [ELEMENTS] [CHUNKS]
//   Every rank is a separate process. Rank 0 writes NCCL's unique id to ID_FILE; the others wait for it.
// Exit status: 0 when no element was wrong, 1 when some were, 2 on a setup error.

#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>

#define CHECK_CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(2); } } while (0)
#define CHECK_NCCL(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, ncclGetErrorString(r_)); std::exit(2); } } while (0)

__global__ void ring(ncclDevComm comm, ncclDevResourceHandle hbuf, int rounds, int elts, int chunks,
                     unsigned long long* wrong) {
    const ncclTeam world = ncclTeamWorld(comm);
    const int blocks = gridDim.x;
    const int me = world.rank * blocks + blockIdx.x;
    const int all = world.nRanks * blocks;
    const int up = (me + all - 1) % all, down = (me + 1) % all;
    const int up_rank = up / blocks, up_block = up % blocks;
    const int down_rank = down / blocks, down_block = down % blocks;

    // Layout of every rank's buffer: `blocks` send slots, then `blocks` receive slots.
    const ncclSymPtr<int> base = ncclGetResourceBuffer(comm, hbuf);
    int* local = static_cast<int*>(ncclGetResourceBufferLocalPointer(comm, hbuf));
    int* send = local + blockIdx.x * elts;
    const int* recv = local + (blocks + blockIdx.x) * elts;
    const ncclSymPtr<int> my_send = base + blockIdx.x * elts;
    const ncclSymPtr<int> their_recv = base + (blocks + down_block) * elts;

    // Signal ids: [0, blocks) count received chunks per block, [blocks, 2 blocks) count free slots.
    const ncclGinSignal_t data_sig = blockIdx.x, free_sig = blocks + blockIdx.x;
    ncclGin gin(comm, 0);
    ncclCoopCta cta;
    const int chunk_elts = (elts + chunks - 1) / chunks;
    unsigned long long local_wrong = 0;

    for (int r = 0; r < rounds; ++r) {
        // The downstream block has checked round r - 1, so its receive slot may be overwritten.
        gin.waitSignal(cta, free_sig, r);
        for (int i = threadIdx.x; i < elts; i += blockDim.x) send[i] = r + 1;
        __syncthreads();
        for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
            const int first = c * chunk_elts;
            if (first >= elts) break;
            const int n = min(chunk_elts, elts - first);
            gin.put(world, down_rank, their_recv + first, my_send + first, n,
                    ncclGin_WeakSignalInc{static_cast<ncclGinSignal_t>(down_block)});
        }
        // Every chunk of round r from upstream has signalled. Under I1 each has also landed.
        gin.waitSignal(cta, data_sig, static_cast<uint64_t>(r + 1) * chunks);
        for (int i = threadIdx.x; i < elts; i += blockDim.x)
            if (recv[i] != r + 1) ++local_wrong;
        __syncthreads();
        if (threadIdx.x == 0)
            gin.signal(world, up_rank, ncclGin_StrongSignalInc{static_cast<ncclGinSignal_t>(blocks + up_block)});
        gin.flush(cta);
    }
    if (local_wrong) atomicAdd(wrong, local_wrong);
}

static void exchange_id(int rank, const std::string& path, ncclUniqueId* id) {
    if (rank == 0) {
        CHECK_NCCL(ncclGetUniqueId(id));
        const std::string tmp = path + ".tmp";
        std::ofstream(tmp, std::ios::binary).write(reinterpret_cast<const char*>(id), sizeof(*id));
        std::rename(tmp.c_str(), path.c_str());
        return;
    }
    for (int waited = 0; waited < 600; ++waited) {
        std::ifstream in(path, std::ios::binary);
        if (in.read(reinterpret_cast<char*>(id), sizeof(*id))) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::fprintf(stderr, "no unique id in %s after 60 s\n", path.c_str());
    std::exit(2);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s RANK NRANKS ID_FILE [ROUNDS] [ELEMENTS] [CHUNKS]\n", argv[0]);
        return 2;
    }
    const int rank = std::atoi(argv[1]), nranks = std::atoi(argv[2]);
    const int rounds = argc > 4 ? std::atoi(argv[4]) : 50;
    const int elts = argc > 5 ? std::atoi(argv[5]) : 1 << 16;
    const int chunks = argc > 6 ? std::atoi(argv[6]) : 16;
    const int blocks = 4, threads = 256;

    ncclUniqueId id;
    exchange_id(rank, argv[3], &id);
    CHECK_CUDA(cudaSetDevice(0));
    ncclComm_t comm;
    CHECK_NCCL(ncclCommInitRank(&comm, nranks, id, rank));

    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.ginSignalCount = 2 * blocks;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    reqs.ginStrongSignalsRequired = true;
    ncclDevResourceHandle hbuf;
    ncclDevResourceRequirements buf = {};
    buf.bufferSize = size_t(2) * blocks * elts * sizeof(int);
    buf.outBufferHandle = &hbuf;
    buf.next = reqs.resourceRequirementsList;
    reqs.resourceRequirementsList = &buf;
    ncclDevComm dcomm;
    CHECK_NCCL(ncclDevCommCreate(comm, &reqs, &dcomm));

    unsigned long long* wrong;
    CHECK_CUDA(cudaMalloc(&wrong, sizeof(*wrong)));
    CHECK_CUDA(cudaMemset(wrong, 0, sizeof(*wrong)));
    ring<<<blocks, threads>>>(dcomm, hbuf, rounds, elts, chunks, wrong);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    unsigned long long host_wrong = 0;
    CHECK_CUDA(cudaMemcpy(&host_wrong, wrong, sizeof(host_wrong), cudaMemcpyDeviceToHost));

    std::printf("rank %d: %d rounds, %d elements in %d chunks per block, %llu wrong elements\n",
                rank, rounds, elts, chunks, host_wrong);
    CHECK_NCCL(ncclDevCommDestroy(comm, &dcomm));
    CHECK_NCCL(ncclCommFinalize(comm));
    CHECK_NCCL(ncclCommDestroy(comm));
    return host_wrong == 0 ? 0 : 1;
}
