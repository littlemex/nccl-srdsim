// SPDX-License-Identifier: Apache-2.0
// What the GPU tests share: error checks, the exchange of NCCL's unique id through a file, and argument parsing.
#pragma once

#include <cuda_runtime.h>
#include <nccl.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>

#define CHECK_CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(2); } } while (0)
#define CHECK_NCCL(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, ncclGetErrorString(r_)); std::exit(2); } } while (0)

// Rank 0 writes NCCL's unique id to `path`; the others wait for it, for up to 60 s.
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
