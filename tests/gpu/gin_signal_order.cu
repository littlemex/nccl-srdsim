// SPDX-License-Identifier: Apache-2.0

// A GIN user whose correctness rests on invariant I1: a weak signal becomes visible only after its own
// put has landed. Ranks form a ring. In every round, block b of each rank puts a buffer to block b of the
// next rank in chunks, and every chunk carries a weak signal on a signal id of its own. The receiver
// repeatedly scans the chunks it has not checked and checks the first whose signal is visible, so a chunk
// never waits for the signals of the chunks before it, only for one scan. Every element encodes the round,
// the sending rank, the block and its own index, so a chunk that lands in the wrong place, from the
// wrong sender or in the wrong round is also caught. The receiver then returns a weak signal to the
// sender, which waits for it before refilling the slot; that signal only has to arrive, so it puts no
// further ordering requirement on the fabric.
//
// Under a delivery that keeps I1 no element may be wrong. Under violate-i1 some seed should find one;
// a run that never does would mean the check is not reading what its signal guards.
//
// Usage: gin_signal_order RANK NRANKS ID_FILE [ROUNDS] [ELEMENTS] [CHUNKS] [TIMEOUT_S]
//   Every rank is a separate process. Rank 0 writes NCCL's unique id to ID_FILE; the others wait for it.
// Exit status: 0 when no element was wrong, 1 when some were, 2 on a setup error, 3 when a wait timed out
// (then without NCCL teardown).

#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>

#define CHECK_CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(2); } } while (0)
#define CHECK_NCCL(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, ncclGetErrorString(r_)); std::exit(2); } } while (0)

struct Result {
    unsigned long long wrong;     // elements that did not hold the expected value
    unsigned long long timeouts;  // blocks that gave up waiting
    int first_timeout_round;      // the round of the first block that gave up, or -1
    int first_timeout_kind;       // 1 waiting for a free slot, 2 for a chunk's signal, 3 for a flush
};

__device__ __forceinline__ uint32_t expected(int round, int sender, int block, size_t index) {
    return (uint32_t(round + 1) * 0x9E3779B1u) ^ (uint32_t(sender) * 0x85EBCA6Bu) ^
           (uint32_t(block) * 0xC2B2AE35u) ^ uint32_t(index);
}

// Signals are read with absolute values (round r needs r or r + 1), so the test assumes they start at
// zero: one launch per communicator.
__global__ void ring(ncclDevComm comm, ncclDevResourceHandle hbuf, int rounds, size_t elts, int nchunks,
                     size_t chunk_elts, long long timeout_cycles, Result* result) {
    const ncclTeam world = ncclTeamWorld(comm);
    const int blocks = gridDim.x, b = blockIdx.x;
    const int down = (world.rank + 1) % world.nRanks;
    const int up = (world.rank + world.nRanks - 1) % world.nRanks;

    // Every rank's buffer holds `blocks` send slots, then `blocks` receive slots.
    const ncclSymPtr<uint32_t> base = ncclGetResourceBuffer(comm, hbuf);
    uint32_t* local = static_cast<uint32_t*>(ncclGetResourceBufferLocalPointer(comm, hbuf));
    uint32_t* send = local + size_t(b) * elts;
    const volatile uint32_t* recv = local + (size_t(blocks) + b) * elts;
    const ncclSymPtr<uint32_t> my_send = base + size_t(b) * elts;
    const ncclSymPtr<uint32_t> their_recv = base + (size_t(blocks) + b) * elts;

    // Signal ids: chunk c of block b is b * nchunks + c; block b's free slot is blocks * nchunks + b.
    const ncclGinSignal_t chunk_sig0 = ncclGinSignal_t(b) * nchunks;
    const ncclGinSignal_t free_sig = ncclGinSignal_t(blocks) * nchunks + b;
    ncclGin gin(comm, 0);
    ncclCoopCta cta;

    // Thread 0 decides every signal wait and publishes the decision here, so the whole block takes the same
    // branch. The flush result is evaluated by every thread and merged here the same way.
    __shared__ unsigned long long wrong;
    __shared__ int stop;     // 0 to go on, else the kind of wait that expired
    __shared__ int next;     // the chunk whose signal thread 0 found visible
    __shared__ unsigned checked_bits[32];
    if (threadIdx.x == 0) { wrong = 0; stop = 0; }
    __syncthreads();
    // Every thread reads `stop` before any thread can pass the barrier and write it again.
    auto stopped = [&]() { const int s = stop; __syncthreads(); return s != 0; };

    int r = 0;
    for (; r < rounds; ++r) {
        // The downstream block has checked round r - 1, so its receive slot may be overwritten.
        if (threadIdx.x == 0) {
            const long long deadline = clock64() + timeout_cycles;
            while (gin.readSignal(free_sig) < uint64_t(r))
                if (clock64() > deadline) { stop = 1; break; }
            for (int w = 0; w < 32; ++w) checked_bits[w] = 0;
        }
        __syncthreads();
        if (stopped()) break;
        for (size_t i = threadIdx.x; i < elts; i += blockDim.x) send[i] = expected(r, world.rank, b, i);
        __syncthreads();
        for (int c = threadIdx.x; c < nchunks; c += blockDim.x) {
            const size_t first = size_t(c) * chunk_elts;
            const size_t n = min(chunk_elts, elts - first);
            gin.put(world, down, their_recv + first, my_send + first, n,
                    ncclGin_WeakSignalInc{chunk_sig0 + c});
        }
        // Check whichever unchecked chunk thread 0's scan finds visible first, rather than in chunk order, so
        // a chunk whose signal is visible never waits for the signals of the chunks before it.
        for (int done = 0; done < nchunks; ++done) {
            if (threadIdx.x == 0) {
                const long long deadline = clock64() + timeout_cycles;
                next = -1;
                while (next < 0) {
                    for (int c = 0; c < nchunks && next < 0; ++c)
                        if (!(checked_bits[c / 32] >> (c % 32) & 1u) &&
                            gin.readSignal(chunk_sig0 + c) >= uint64_t(r + 1)) next = c;
                    if (next < 0 && clock64() > deadline) { stop = 2; break; }
                }
                if (next >= 0) checked_bits[next / 32] |= 1u << (next % 32);
            }
            __syncthreads();
            const int c = next;
            if (stopped()) break;
            // Thread 0's acquire is not relied on to order the other threads' reads: each does its own.
            (void)gin.readSignal(chunk_sig0 + c);
            const size_t first = size_t(c) * chunk_elts, last = min(first + chunk_elts, elts);
            unsigned long long bad = 0;
            for (size_t i = first + threadIdx.x; i < last; i += blockDim.x)
                if (recv[i] != expected(r, up, b, i)) ++bad;
            if (bad) atomicAdd(&wrong, bad);
            __syncthreads();
        }
        if (stopped()) break;
        if (threadIdx.x == 0) gin.signal(world, up, ncclGin_WeakSignalInc{free_sig});
        // The puts of this round have read their source before the slot is refilled next round.
        if (gin.flush(cta, cuda::memory_order_acquire, ncclGin_None{}, uint64_t(timeout_cycles)) != ncclSuccess)
            stop = 3;
        __syncthreads();
        if (stopped()) break;
    }
    if (threadIdx.x == 0) {
        if (wrong) atomicAdd(&result->wrong, wrong);
        if (stop && atomicAdd(&result->timeouts, 1ull) == 0) {
            result->first_timeout_round = r;
            result->first_timeout_kind = stop;
        }
    }
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

static long integer(const char* s, const char* what, long least) {
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || v < least) {
        std::fprintf(stderr, "%s must be an integer of at least %ld, got '%s'\n", what, least, s);
        std::exit(2);
    }
    return v;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s RANK NRANKS ID_FILE [ROUNDS] [ELEMENTS] [CHUNKS] [TIMEOUT_S]\n", argv[0]);
        return 2;
    }
    // A log is a file, so stdout would be fully buffered and lost when the driver stops this rank.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    // Two ranks at least: with one, every put would go to the sender's own rank.
    const int nranks = int(integer(argv[2], "NRANKS", 2));
    const int rank = int(integer(argv[1], "RANK", 0));
    if (rank >= nranks) { std::fprintf(stderr, "RANK must be less than NRANKS\n"); return 2; }
    const int rounds = argc > 4 ? int(integer(argv[4], "ROUNDS", 1)) : 50;
    const size_t elts = argc > 5 ? size_t(integer(argv[5], "ELEMENTS", 1)) : size_t(1) << 16;
    const int chunks = argc > 6 ? int(integer(argv[6], "CHUNKS", 1)) : 16;
    const int timeout_s = argc > 7 ? int(integer(argv[7], "TIMEOUT_S", 1)) : 60;
    const int blocks = 4, threads = 256;
    // The number of chunks actually sent, which is what each receiver waits for.
    const size_t chunk_elts = (elts + chunks - 1) / chunks;
    const int nchunks = int((elts + chunk_elts - 1) / chunk_elts);
    if (nchunks > 1024) { std::fprintf(stderr, "at most 1024 chunks per block\n"); return 2; }

    ncclUniqueId id;
    exchange_id(rank, argv[3], &id);
    CHECK_CUDA(cudaSetDevice(0));
    int clock_khz = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    const long long timeout_cycles = (long long)clock_khz * 1000 * timeout_s;

    ncclComm_t comm;
    CHECK_NCCL(ncclCommInitRank(&comm, nranks, id, rank));
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.ginSignalCount = blocks * nchunks + blocks;
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    ncclDevResourceHandle hbuf;
    ncclDevResourceRequirements buf = {};
    buf.bufferSize = size_t(2) * blocks * elts * sizeof(uint32_t);
    buf.bufferAlign = 4096;
    buf.outBufferHandle = &hbuf;
    buf.next = reqs.resourceRequirementsList;
    reqs.resourceRequirementsList = &buf;
    ncclDevComm dcomm;
    CHECK_NCCL(ncclDevCommCreate(comm, &reqs, &dcomm));

    Result* result;
    CHECK_CUDA(cudaMalloc(&result, sizeof(*result)));
    const Result init = {0, 0, -1, 0};
    CHECK_CUDA(cudaMemcpy(result, &init, sizeof(init), cudaMemcpyHostToDevice));
    ring<<<blocks, threads>>>(dcomm, hbuf, rounds, elts, nchunks, chunk_elts, timeout_cycles, result);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    Result host;
    CHECK_CUDA(cudaMemcpy(&host, result, sizeof(host), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(result));

    if (host.timeouts)
        std::printf("rank %d: %llu block(s) timed out after %d s, the first in round %d of %d waiting for %s; "
                    "%llu wrong elements in what was checked before that\n", rank, host.timeouts, timeout_s,
                    host.first_timeout_round, rounds,
                    host.first_timeout_kind == 1 ? "a free slot" : host.first_timeout_kind == 2 ? "a chunk's signal" : "a flush",
                    host.wrong);
    else
        std::printf("rank %d: %d rounds, %zu elements in %d chunks per block, %llu wrong elements\n",
                    rank, rounds, elts, nchunks, host.wrong);
    // After a timeout, operations may still be pending and peers may be gone, so a teardown could fail or
    // hang and replace the status that says what happened. Leave without it.
    if (host.timeouts) { std::fflush(stdout); std::_Exit(3); }
    CHECK_NCCL(ncclDevCommDestroy(comm, &dcomm));
    CHECK_NCCL(ncclCommFinalize(comm));
    CHECK_NCCL(ncclCommDestroy(comm));
    return host.wrong == 0 ? 0 : 1;
}
