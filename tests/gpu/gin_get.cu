// SPDX-License-Identifier: Apache-2.0

// A GIN user that reads remote memory: device-initiated gets, made visible with flushAsync and wait. Ranks form a
// ring. In every round, block b of each rank fills its source slot and signals the next rank that the slot is ready;
// that rank gets the slot into its own destination slot in chunks, one get per chunk, waits for a flush of its gets
// to the previous rank, checks every element, and signals back that the source may be refilled. Every element encodes
// the round, the owning rank, the block and its own index, so a chunk that lands in the wrong place, from the wrong
// owner or from the wrong round is caught.
//
// The flush is what makes a get's data visible (NCCL's GIN proxy tracks issued and visible gets per peer and
// completes the flush only when the plugin reports every get landed). With GIN_GET_CHECK_EARLY=1 the receiver checks
// without waiting for the flush: a negative control that some run should fail, which shows that the check reads what
// the flush guards and that the gets went through the plugin.
//
// Usage: gin_get RANK NRANKS ID_FILE [ROUNDS] [ELEMENTS] [CHUNKS] [TIMEOUT_S]
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
#include <cstring>
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
    int first_timeout_kind;       // 1 waiting for the source to be ready, 2 for the flush, 3 for the source to be free
    int first_wrong;              // set once the first wrong element below is recorded
    int wrong_round, wrong_block;
    unsigned long long wrong_index;
    uint32_t wrong_got, wrong_want;
};

__device__ __forceinline__ uint32_t expected(int round, int owner, int block, size_t index) {
    return (uint32_t(round + 1) * 0x9E3779B1u) ^ (uint32_t(owner) * 0x85EBCA6Bu) ^
           (uint32_t(block) * 0xC2B2AE35u) ^ uint32_t(index);
}

// Signals are read with absolute values, so the test assumes they start at zero: one launch per communicator.
__global__ void ring(ncclDevComm comm, ncclDevResourceHandle hbuf, int rounds, size_t elts, int nchunks,
                     size_t chunk_elts, long long timeout_cycles, bool check_early, Result* result) {
    const ncclTeam world = ncclTeamWorld(comm);
    const int blocks = gridDim.x, b = blockIdx.x;
    const int down = (world.rank + 1) % world.nRanks;
    const int up = (world.rank + world.nRanks - 1) % world.nRanks;

    // Every rank's buffer holds `blocks` source slots, then `blocks` destination slots.
    const ncclSymPtr<uint32_t> base = ncclGetResourceBuffer(comm, hbuf);
    uint32_t* local = static_cast<uint32_t*>(ncclGetResourceBufferLocalPointer(comm, hbuf));
    uint32_t* src = local + size_t(b) * elts;
    const volatile uint32_t* dst = local + (size_t(blocks) + b) * elts;
    const size_t src_off = base.offset + size_t(b) * elts * sizeof(uint32_t);
    const size_t dst_off = base.offset + (size_t(blocks) + b) * elts * sizeof(uint32_t);

    // Signal ids: block b's source is ready (raised by the owner upstream) is b; it is free again (raised by the
    // reader downstream) is blocks + b.
    const ncclGinSignal_t ready_sig = ncclGinSignal_t(b), free_sig = ncclGinSignal_t(blocks) + b;
    ncclGin gin(comm, 0);

    __shared__ unsigned long long wrong;
    __shared__ int stop;     // 0 to go on, else the kind of wait that expired
    if (threadIdx.x == 0) { wrong = 0; stop = 0; }
    __syncthreads();
    auto stopped = [&]() { const int s = stop; __syncthreads(); return s != 0; };
    const auto wait_signal = [&](ncclGinSignal_t sig, uint64_t value, int kind) {
        if (threadIdx.x == 0) {
            const long long deadline = clock64() + timeout_cycles;
            while (gin.readSignal(sig) < value)
                if (clock64() > deadline) { stop = kind; break; }
        }
        __syncthreads();
    };

    int r = 0;
    for (; r < rounds; ++r) {
        // The reader downstream has checked round r - 1, so this block's source may be overwritten.
        wait_signal(free_sig, uint64_t(r), 3);
        if (stopped()) break;
        for (size_t i = threadIdx.x; i < elts; i += blockDim.x) src[i] = expected(r, world.rank, b, i);
        // The stores reach memory the plugin's host thread reads before the reader is told the slot is ready.
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) gin.signal(world, down, ncclGin_WeakSignalInc{ready_sig});

        // Read the upstream owner's slot once it says the slot holds round r.
        wait_signal(ready_sig, uint64_t(r + 1), 1);
        if (stopped()) break;
        for (int c = threadIdx.x; c < nchunks; c += blockDim.x) {
            const size_t first = size_t(c) * chunk_elts;
            const size_t n = min(chunk_elts, elts - first);
            gin.get(world, up, base.window, src_off + first * sizeof(uint32_t), base.window,
                    dst_off + first * sizeof(uint32_t), n * sizeof(uint32_t));
        }
        __syncthreads();
        if (not check_early) {
            if (threadIdx.x == 0) {
                ncclGinRequest_t req;
                gin.flushAsync(world, uint32_t(up), &req);
                if (gin.wait(req, ncclCoopThread{}, ncclGin_None{}, cuda::memory_order_acquire, uint64_t(timeout_cycles)) !=
                    ncclSuccess)
                    stop = 2;
            }
            __syncthreads();
            if (stopped()) break;
        }
        unsigned long long bad = 0;
        for (size_t i = threadIdx.x; i < elts; i += blockDim.x) {
            const uint32_t got = dst[i], want = expected(r, up, b, i);
            if (got == want) continue;
            ++bad;
            if (atomicCAS(&result->first_wrong, 0, 1) == 0) {
                result->wrong_round = r; result->wrong_block = b; result->wrong_index = i;
                result->wrong_got = got; result->wrong_want = want;
            }
        }
        if (bad) atomicAdd(&wrong, bad);
        __syncthreads();
        if (check_early) {
            // Still wait for the gets before the destination is reused and before the owner refills the source.
            if (threadIdx.x == 0) {
                ncclGinRequest_t req;
                gin.flushAsync(world, uint32_t(up), &req);
                if (gin.wait(req, ncclCoopThread{}, ncclGin_None{}, cuda::memory_order_acquire, uint64_t(timeout_cycles)) !=
                    ncclSuccess)
                    stop = 2;
            }
            __syncthreads();
            if (stopped()) break;
        }
        if (threadIdx.x == 0) gin.signal(world, up, ncclGin_WeakSignalInc{free_sig});
    }
    // The reader downstream may still be getting this block's last source: the rank stays, with its memory registered,
    // until that reader says it is done. Otherwise its last get could fail, and a failed get also completes the flush.
    if (r == rounds) {
        wait_signal(free_sig, uint64_t(rounds), 3);
        (void)stopped();
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
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const int nranks = int(integer(argv[2], "NRANKS", 2));
    const int rank = int(integer(argv[1], "RANK", 0));
    if (rank >= nranks) { std::fprintf(stderr, "RANK must be less than NRANKS\n"); return 2; }
    const int rounds = argc > 4 ? int(integer(argv[4], "ROUNDS", 1)) : 50;
    const size_t elts = argc > 5 ? size_t(integer(argv[5], "ELEMENTS", 1)) : size_t(1) << 16;
    const int chunks = argc > 6 ? int(integer(argv[6], "CHUNKS", 1)) : 16;
    const int timeout_s = argc > 7 ? int(integer(argv[7], "TIMEOUT_S", 1)) : 60;
    const char* early = std::getenv("GIN_GET_CHECK_EARLY");
    const bool check_early = early != nullptr && std::strcmp(early, "1") == 0;
    const int blocks = 4, threads = 256;
    const size_t chunk_elts = (elts + chunks - 1) / chunks;
    const int nchunks = int((elts + chunk_elts - 1) / chunk_elts);

    ncclUniqueId id;
    exchange_id(rank, argv[3], &id);
    CHECK_CUDA(cudaSetDevice(0));
    int clock_khz = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    const long long timeout_cycles = (long long)clock_khz * 1000 * timeout_s;

    ncclComm_t comm;
    CHECK_NCCL(ncclCommInitRank(&comm, nranks, id, rank));
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.ginSignalCount = 2 * blocks;
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
    const Result init = {0, 0, -1, 0, 0, 0, 0, 0, 0, 0};
    CHECK_CUDA(cudaMemcpy(result, &init, sizeof(init), cudaMemcpyHostToDevice));
    ring<<<blocks, threads>>>(dcomm, hbuf, rounds, elts, nchunks, chunk_elts, timeout_cycles, check_early, result);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    Result host;
    CHECK_CUDA(cudaMemcpy(&host, result, sizeof(host), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(result));

    const char* kinds[] = {"", "the source to be ready", "the flush", "the source to be free"};
    if (host.timeouts)
        std::printf("rank %d: %llu block(s) timed out after %d s, the first in round %d of %d waiting for %s; "
                    "%llu wrong elements in what was checked before that\n", rank, host.timeouts, timeout_s,
                    host.first_timeout_round, rounds, kinds[host.first_timeout_kind], host.wrong);
    else
        std::printf("rank %d: %d rounds, %zu elements in %d gets per block%s, %llu wrong elements\n", rank, rounds, elts,
                    nchunks, check_early ? " checked before the flush" : "", host.wrong);
    if (host.first_wrong) {
        // What the wrong element held: the previous round's value, another block's or owner's, or something else.
        std::printf("rank %d: the first wrong element was round %d block %d index %llu: held %#x, expected %#x",
                    rank, host.wrong_round, host.wrong_block, host.wrong_index, host.wrong_got, host.wrong_want);
        bool named = false;
        for (int rr = -1; rr <= rounds and not named; ++rr)
            for (int o = 0; o < nranks and not named; ++o)
                for (int bb = 0; bb < blocks and not named; ++bb) {
                    const uint32_t v = (uint32_t(rr + 1) * 0x9E3779B1u) ^ (uint32_t(o) * 0x85EBCA6Bu) ^
                                       (uint32_t(bb) * 0xC2B2AE35u) ^ uint32_t(host.wrong_index);
                    if (v == host.wrong_got) { std::printf(" (round %d's value of rank %d block %d)", rr, o, bb); named = true; }
                }
        std::printf("\n");
    }
    if (host.timeouts) { std::fflush(stdout); std::_Exit(3); }
    CHECK_NCCL(ncclDevCommDestroy(comm, &dcomm));
    CHECK_NCCL(ncclCommFinalize(comm));
    CHECK_NCCL(ncclCommDestroy(comm));
    return host.wrong == 0 ? 0 : 1;
}
