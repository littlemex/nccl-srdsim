// SPDX-License-Identifier: Apache-2.0
//
// A write into GPU memory completes while a kernel that is resident on the target's GPU is spinning for it.
//
// This is the shape of DeepEP's GIN kernels: the kernel on the receiving GPU waits for the bytes a peer writes, and
// the provider that applies the write must not wait for the kernel. libfabric's own CUDA copy runs in the
// application's primary context and does exactly that, so the kernel waits for the write and the write waits for the
// kernel. The provider copies from a context of its own instead (src/fi/srdsim_gpu.c).
//
// Rank 1 allocates a GPU window with the driver's virtual memory API (an allocation the provider's own context can
// address), registers it, launches a kernel that spins on its last word for up to kKernelSeconds, and checks the
// kernel saw the word and the whole window holds what rank 0 wrote. Rank 0 writes the window with one fi_write.
// Run with SRDSIM_DELIVERY_CONTEXT=primary it must fail: that is the control showing the test can see the deadlock.
//
// usage: test_fi_gpu_write PROVIDER_SO
#include <cuda.h>
#include <cuda_runtime.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>

#include <arpa/inet.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_rank = 0;
int g_failures = 0;
int g_to_peer = -1, g_from_peer = -1;

void check(bool ok, const char* what) {
    std::printf("rank %d: %s %s\n", g_rank, ok ? "PASS" : "FAIL", what);
    std::fflush(stdout);
    if (not ok) ++g_failures;
}

[[noreturn]] void die(const char* what, int rc) {
    std::printf("rank %d: FAIL %s: %d (%s)\n", g_rank, what, rc, fi_strerror(-rc));
    std::fflush(stdout);
    std::_Exit(1);
}

void put(const void* p, size_t n) {
    if (write(g_to_peer, p, n) != ssize_t(n)) die("pipe write", -FI_EIO);
}
void get(void* p, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(g_from_peer, static_cast<char*>(p) + got, n - got);
        if (r <= 0) die("pipe read", -FI_EIO);
        got += size_t(r);
    }
}

struct Endpoint {
    fi_info* info = nullptr;
    fid_fabric* fabric = nullptr;
    fid_domain* domain = nullptr;
    fid_av* av = nullptr;
    fid_cq* cq = nullptr;
    fid_ep* ep = nullptr;
    fi_addr_t peer = FI_ADDR_NOTAVAIL;
};

void open_endpoint(Endpoint* e) {
    fi_info* hints = fi_allocinfo();
    hints->caps = FI_MSG | FI_TAGGED | FI_RMA | FI_HMEM | FI_LOCAL_COMM | FI_REMOTE_COMM;
    hints->ep_attr->type = FI_EP_RDM;
    hints->domain_attr->mr_mode = FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_HMEM;
    hints->domain_attr->threading = FI_THREAD_COMPLETION;
    hints->fabric_attr->prov_name = strdup("efa");
    hints->tx_attr->msg_order = FI_ORDER_SAS;
    hints->rx_attr->msg_order = FI_ORDER_SAS;
    int rc = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0, hints, &e->info);
    fi_freeinfo(hints);
    if (rc) die("fi_getinfo", rc);
    if (std::strcmp(e->info->domain_attr->name, "srdsim0-rdm") != 0) die("not the srdsim provider", -FI_ENODATA);
    if ((rc = fi_fabric(e->info->fabric_attr, &e->fabric, nullptr))) die("fi_fabric", rc);
    if ((rc = fi_domain(e->fabric, e->info, &e->domain, nullptr))) die("fi_domain", rc);
    fi_av_attr av_attr = {};
    av_attr.type = FI_AV_TABLE;
    if ((rc = fi_av_open(e->domain, &av_attr, &e->av, nullptr))) die("fi_av_open", rc);
    fi_cq_attr cq_attr = {};
    cq_attr.format = FI_CQ_FORMAT_TAGGED;
    if ((rc = fi_cq_open(e->domain, &cq_attr, &e->cq, nullptr))) die("fi_cq_open", rc);
    if ((rc = fi_endpoint(e->domain, e->info, &e->ep, nullptr))) die("fi_endpoint", rc);
    if ((rc = fi_ep_bind(e->ep, &e->av->fid, 0))) die("bind av", rc);
    if ((rc = fi_ep_bind(e->ep, &e->cq->fid, FI_TRANSMIT | FI_RECV))) die("bind cq", rc);
    if ((rc = fi_enable(e->ep))) die("fi_enable", rc);
    char mine[64], theirs[64];
    size_t len = sizeof mine;
    if ((rc = fi_getname(&e->ep->fid, mine, &len))) die("fi_getname", rc);
    put(mine, 32);
    get(theirs, 32);
    if (fi_av_insert(e->av, theirs, 1, &e->peer, 0, nullptr) != 1) die("fi_av_insert", -FI_EINVAL);
}

void close_endpoint(Endpoint* e) {
    fi_close(&e->ep->fid);
    fi_close(&e->cq->fid);
    fi_close(&e->av->fid);
    fi_close(&e->domain->fid);
    fi_close(&e->fabric->fid);
    fi_freeinfo(e->info);
}

// Waits for one completion; returns 0, or the error's positive errno with *err filled in.
int wait_cq(Endpoint* e, fi_cq_tagged_entry* out, fi_cq_err_entry* err = nullptr, int timeout_s = 30) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    while (std::chrono::steady_clock::now() < deadline) {
        ssize_t n = fi_cq_read(e->cq, out, 1);
        if (n == 1) return 0;
        if (n == -FI_EAVAIL) {
            fi_cq_err_entry tmp = {};
            fi_cq_readerr(e->cq, &tmp, 0);
            if (err) *err = tmp;
            return tmp.err;
        }
        if (n != -FI_EAGAIN) die("fi_cq_read", int(n));
    }
    die("no completion within the timeout", -FI_ETIMEDOUT);
}


void barrier() {
    char b = 1;
    put(&b, 1);
    get(&b, 1);
}

constexpr size_t kLen = 256 << 10;
constexpr uint64_t kMagic = 0x5352445349464c47ull;
constexpr unsigned kKernelSeconds = 15;

__global__ void spin_for_word(volatile uint64_t* word, unsigned long long budget_ns, int* got) {
    auto now = [] { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; };
    const unsigned long long start = now();
    while (*word != kMagic) {
        if (now() - start > budget_ns) { *got = 0; return; }
    }
    *got = 1;
}

#define CU(call) do { CUresult r_ = (call); if (r_ != CUDA_SUCCESS) die(#call, int(r_)); } while (0)

uint8_t pattern(size_t i) { return uint8_t((i * 7 + (i >> 9) + 1) & 0xff); }

void gpu_write() {
    Endpoint e;
    open_endpoint(&e);
    CUdeviceptr base = 0;
    int ordinal = 0;
    if (g_rank == 1) {
        CU(cuInit(0));
        CUdevice dev;
        CU(cuDeviceGet(&dev, ordinal));
        if (cudaSetDevice(ordinal) != cudaSuccess) die("cudaSetDevice", -FI_EIO);
        cudaFree(nullptr);
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = dev;
        size_t gran = 0;
        CU(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        const size_t size = (kLen + gran - 1) / gran * gran;
        CUmemGenericAllocationHandle h;
        CU(cuMemCreate(&h, size, &prop, 0));
        CU(cuMemAddressReserve(&base, size, 0, 0, 0));
        CU(cuMemMap(base, size, 0, h, 0));
        CUmemAccessDesc acc = {};
        acc.location = prop.location;
        acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU(cuMemSetAccess(base, size, &acc, 1));
        CU(cuMemsetD8(base, 0, size));
        iovec iov = {reinterpret_cast<void*>(base), kLen};
        fi_mr_attr attr = {};
        attr.mr_iov = &iov;
        attr.iov_count = 1;
        attr.access = FI_REMOTE_WRITE | FI_REMOTE_READ;
        attr.iface = FI_HMEM_CUDA;
        attr.device.cuda = ordinal;
        fid_mr* mr = nullptr;
        int rc = fi_mr_regattr(e.domain, &attr, 0, &mr);
        if (rc) die("fi_mr_regattr", rc);
        uint64_t desc[2] = {uint64_t(base), fi_mr_key(mr)};
        put(desc, sizeof desc);

        int* got = nullptr;
        cudaMallocManaged(&got, sizeof(int));
        *got = -1;
        spin_for_word<<<1, 1>>>(reinterpret_cast<volatile uint64_t*>(base + kLen - 8), kKernelSeconds * 1000000000ull, got);
        const auto t0 = std::chrono::steady_clock::now();
        cudaError_t err = cudaDeviceSynchronize();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        check(err == cudaSuccess, "the kernel finished");
        check(*got == 1, "the resident kernel saw the written word before its budget ran out");
        std::printf("rank 1: the kernel returned after %.2f s\n", s);
        barrier();
        std::vector<uint8_t> host(kLen);
        CU(cuMemcpyDtoH(host.data(), base, kLen));
        bool whole = true;
        for (size_t i = 0; i + 8 < kLen and whole; ++i) whole = host[i] == pattern(i);
        uint64_t last;
        std::memcpy(&last, host.data() + kLen - 8, 8);
        check(whole and last == kMagic, "the whole window holds what rank 0 wrote");
        fi_close(&mr->fid);
    } else {
        uint64_t desc[2];
        get(desc, sizeof desc);
        std::vector<uint8_t> buf(kLen);
        for (size_t i = 0; i < kLen; ++i) buf[i] = pattern(i);
        std::memcpy(buf.data() + kLen - 8, &kMagic, 8);
        usleep(500000);   // after the kernel is spinning
        int rc = int(fi_write(e.ep, buf.data(), kLen, nullptr, e.peer, desc[0], desc[1], nullptr));
        if (rc) die("fi_write", rc);
        fi_cq_tagged_entry c;
        fi_cq_err_entry err = {};
        const auto w0 = std::chrono::steady_clock::now();
        const bool done = wait_cq(&e, &c, &err, 40) == 0;
        std::printf("rank 0: the write completed %.2f s after it was submitted\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count());
        check(done, "the write completed while the kernel spun");
        barrier();
    }
    barrier();
    close_endpoint(&e);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s PROVIDER_SO\n", argv[0]); return 2; }
    setenv("FI_PROVIDER_PATH", (std::string("+") + argv[1]).c_str(), 1);
    int p01[2], p10[2];
    if (pipe(p01) or pipe(p10)) return 2;
    std::fflush(nullptr);
    pid_t child = fork();
    if (child < 0) return 2;
    g_rank = child == 0 ? 1 : 0;
    g_to_peer = g_rank == 0 ? p01[1] : p10[1];
    g_from_peer = g_rank == 0 ? p10[0] : p01[0];
    alarm(120);
    gpu_write();
    if (g_rank == 1) std::_Exit(g_failures ? 1 : 0);
    int st = 0;
    waitpid(child, &st, 0);
    const bool child_ok = WIFEXITED(st) and WEXITSTATUS(st) == 0;
    std::printf("gpu_write: %s\n", g_failures == 0 and child_ok ? "passed" : "failed");
    return g_failures == 0 and child_ok ? 0 : 1;
}
