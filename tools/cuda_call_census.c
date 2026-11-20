// SPDX-License-Identifier: Apache-2.0

// Which CUDA driver calls this process has entered and not returned from.
//
// WHY THIS EXISTS
// A thread stuck inside `cuMemcpyHtoDAsync` for six minutes, on a stream that `cuStreamQuery` reports
// idle, is blocked before its work becomes pending on that stream. That places the block inside the
// driver, and from outside the driver there is exactly one thing worth knowing: which other call is
// also in there. Every candidate explanation -- a module load waiting for the context to go quiet, a
// virtual-memory mapping change, a pinned allocation, a lock inversion -- predicts a *different* second
// call that has been entered and has not returned. So the measurement that separates them is a census:
// at the moment of the freeze, list every driver call in flight, with its thread and its age.
//
// A stack dump nearly answers this and kept not answering it. It shows the blocked thread reliably and
// the holder only if the sample lands in the right window, and five attempts produced two useful
// captures. A census is not a sample: an entry that has no matching exit stays in the table until it
// returns, so the holder cannot be missed by timing.
//
// WHY AN LD_PRELOAD SHIM RATHER THAN INSTRUMENTING THE PLUGIN
// The interesting calls are not the plugin's. DeepEP loads modules and maps memory on the dispatch
// path, PyTorch allocates, NCCL creates streams, and none of that passes through code this project
// owns. Interposing on the library everybody shares is the only place all of them are visible.
//
// WHY THERE IS ALSO A CUPTI FRONT END
// The shim sees only callers that link libcuda's exported symbols. The CUDA runtime -- inside PyTorch and
// NCCL -- does not: it obtains every driver entry point through `cuGetProcAddress`, and intercepting that
// lookup breaks PyTorch ("context is destroyed"), because the runtime asks the lookup for itself and depends
// on what comes back. So a count of calls in flight from the shim alone is a lower bound. Built with -DCENSUS_WITH_CUPTI and loaded through CUDA_INJECTION64_PATH, the same census is fed by
// CUPTI's driver-API callbacks instead, which the driver raises for every call whoever made it. The two
// front ends are alternatives: load the library one way or the other, not both.
//
// WHAT IT DOES NOT DO
// It cannot name the driver's internal mutexes, so it does not prove that two calls contend for one
// lock. It establishes which calls are simultaneously in flight, which is what the competing
// explanations disagree about. Attributing the wait to a particular lock needs NVIDIA's symbols and is
// out of reach here.
//
// Build:
//   cc -O2 -fPIC -shared -o libcuda_census.so cuda_call_census.c -ldl -lpthread
//   cc -O2 -fPIC -shared -DCENSUS_WITH_CUPTI -I<cupti include> -o libcuda_census_cupti.so \
//      cuda_call_census.c -L<cupti lib> -lcupti -ldl -lpthread
// Use:
//   CUDA_CENSUS_LOG=/path/to/census.log LD_PRELOAD=/path/to/libcuda_census.so <program>
//   CUDA_CENSUS_LOG=/path/to/census.log CUDA_INJECTION64_PATH=/path/to/libcuda_census_cupti.so <program>
// Report:
//   kill -USR2 <pid>     writes the outstanding table to the log
// and the table is written automatically for any call that stays in flight longer than
// CUDA_CENSUS_WARN_MS (default 2000), which is what makes it useful without knowing when to look.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

// Bounded and preallocated. A census that allocates while reporting can deadlock against an allocator
// the frozen thread already holds, and the whole point is to work when the process is stuck.
#define CENSUS_SLOTS 512

#define CENSUS_FRAMES 24

struct slot {
    // Zero means free. Written last on entry and first on exit, so a reader that sees a name also sees
    // a valid timestamp.
    uint64_t entered_ms;
    long tid;
    const char* name;
    int warned;
    // The caller's stack, captured on entry for the calls that free memory. A free that waits for a
    // resident kernel blocks every other submission in the process, and the question is always *who*
    // freed: the stack answers it where a name cannot.
    int nframes;
    void* frames[CENSUS_FRAMES];
};

static struct slot g_slots[CENSUS_SLOTS];
static FILE* g_log = NULL;
static pthread_mutex_t g_log_mu = PTHREAD_MUTEX_INITIALIZER;
static long g_warn_ms = 2000;

static uint64_t now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)(ts.tv_nsec / 1000000);
}

static long self_tid(void) {
#ifdef __linux__
    return (long)syscall(SYS_gettid);
#else
    return 0;   // only ever loaded on Linux; this keeps the file syntax-checkable elsewhere
#endif
}

static void census_open(void) {
    const char* path = getenv("CUDA_CENSUS_LOG");
    g_log = (path && *path) ? fopen(path, "a") : stderr;
    if (!g_log) g_log = stderr;
    const char* w = getenv("CUDA_CENSUS_WARN_MS");
    if (w && *w) {
        char* end = NULL;
        long v = strtol(w, &end, 10);
        if (end && *end == '\0' && v > 0) g_warn_ms = v;
    }
}

static void say(const char* fmt, ...) {
    pthread_mutex_lock(&g_log_mu);
    if (!g_log) census_open();
    va_list ap;
    va_start(ap, fmt);
    vfprintf(g_log, fmt, ap);
    va_end(ap);
    fputc('\n', g_log);
    fflush(g_log);
    pthread_mutex_unlock(&g_log_mu);
}

// The whole table, oldest first is not worth sorting for: the ages are printed, and a reader looking for
// a holder wants every entry anyway.
static void census_report(const char* why) {
    const uint64_t now = now_ms();
    int found = 0;
    say("== CUDA calls entered and not returned (%s)", why);
    for (int i = 0; i < CENSUS_SLOTS; ++i) {
        const uint64_t at = __atomic_load_n(&g_slots[i].entered_ms, __ATOMIC_ACQUIRE);
        if (at == 0) continue;
        const char* name = __atomic_load_n(&g_slots[i].name, __ATOMIC_RELAXED);
        say("   tid %ld: %s, in flight for %llu ms",
            __atomic_load_n(&g_slots[i].tid, __ATOMIC_RELAXED), name ? name : "(unnamed)",
            (unsigned long long)(now >= at ? now - at : 0));
        ++found;
    }
    if (found == 0) say("   nothing is in flight");
}

static void on_usr2(int sig) { (void)sig; census_report("asked by SIGUSR2"); }

static void check_overdue(void);   // defined below, beside the table it sweeps

static void* watchdog(void* ignored) {
    (void)ignored;
    for (;;) {
        // Half the warning interval, so an overdue call is noticed within one interval of becoming one.
        struct timespec ts;
        ts.tv_sec = (time_t)(g_warn_ms / 2000);
        ts.tv_nsec = (long)((g_warn_ms % 2000) * 500000L);
        if (ts.tv_sec == 0 && ts.tv_nsec == 0) ts.tv_nsec = 100000000L;
        nanosleep(&ts, NULL);
        check_overdue();
    }
    return NULL;
}

__attribute__((constructor)) static void census_init(void) {
    census_open();
    // SIGUSR2 rather than SIGUSR1: PyTorch and CUDA both use SIGUSR1 in places, and stealing a signal
    // another library relies on would make this tool the cause of a new failure.
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_usr2;
    sa.sa_flags = SA_RESTART;
    sigaction(SIGUSR2, &sa, NULL);
    pthread_t th;
    if (pthread_create(&th, NULL, watchdog, NULL) == 0)
        pthread_detach(th);
    else
        say("== census could not start its watchdog thread; overdue calls will only be noticed when "
            "another CUDA call is made, which is exactly when they are not");
    say("== census armed, warning at %ld ms, pid %d", g_warn_ms, (int)getpid());
}

// Claims a slot. Returns -1 when the table is full, and the wrapper then simply does not track that
// call rather than blocking: losing one entry is better than adding a wait to the path being measured.
static int claim(const char* name) {
    const long tid = self_tid();
    const uint64_t at = now_ms();
    for (int i = 0; i < CENSUS_SLOTS; ++i) {
        uint64_t expected = 0;
        if (__atomic_compare_exchange_n(&g_slots[i].entered_ms, &expected, at, 0,
                                        __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
            __atomic_store_n(&g_slots[i].tid, tid, __ATOMIC_RELAXED);
            __atomic_store_n(&g_slots[i].name, name, __ATOMIC_RELAXED);
            __atomic_store_n(&g_slots[i].warned, 0, __ATOMIC_RELAXED);
            g_slots[i].nframes = 0;
            if (strstr(name, "Free") || strstr(name, "Unmap") || strstr(name, "Release") ||
                strstr(name, "Unregister") || strstr(name, "Unload") || strstr(name, "Load") ||
                strstr(name, "Trim"))
                g_slots[i].nframes = backtrace(g_slots[i].frames, CENSUS_FRAMES);
            return i;
        }
    }
    return -1;
}

static void release(int i) {
    if (i < 0) return;
    __atomic_store_n(&g_slots[i].entered_ms, 0, __ATOMIC_RELEASE);
}

// Swept by a thread of its own. Sweeping from inside the wrappers assumes that a call which hangs is always
// accompanied by other calls still arriving, which is false in exactly the case this tool is for: when the
// threads that make CUDA calls are the ones that are stuck, no wrapper is entered, nothing sweeps, and the
// census stays silent about a table full of overdue entries.
//
// A thread that can fail to start is the lesser risk, and it says so when it does.
static void check_overdue(void) {
    const uint64_t now = now_ms();
    // Swept at most once per warning interval. Scanning five hundred slots inside every copy would put
    // measurable cost on the one path under investigation, which is how an instrument becomes a cause.
    static uint64_t last_sweep = 0;
    uint64_t previous = __atomic_load_n(&last_sweep, __ATOMIC_RELAXED);
    if (now < previous + (uint64_t)g_warn_ms) return;
    if (!__atomic_compare_exchange_n(&last_sweep, &previous, now, 0,
                                     __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
        return;
    for (int i = 0; i < CENSUS_SLOTS; ++i) {
        const uint64_t at = __atomic_load_n(&g_slots[i].entered_ms, __ATOMIC_ACQUIRE);
        if (at == 0 || now < at + (uint64_t)g_warn_ms) continue;
        int expected = 0;
        // Reported once per call, not once per check, or a six-minute hang writes thousands of lines and
        // buries the census under itself.
        if (!__atomic_compare_exchange_n(&g_slots[i].warned, &expected, 1, 0,
                                         __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
            continue;
        const char* name = __atomic_load_n(&g_slots[i].name, __ATOMIC_RELAXED);
        say("== overdue: tid %ld has been inside %s for %llu ms",
            __atomic_load_n(&g_slots[i].tid, __ATOMIC_RELAXED), name ? name : "(unnamed)",
            (unsigned long long)(now - at));
        if (g_slots[i].nframes > 0) {
            say("   called from:");
            pthread_mutex_lock(&g_log_mu);
            backtrace_symbols_fd(g_slots[i].frames, g_slots[i].nframes, fileno(g_log ? g_log : stderr));
            fflush(g_log ? g_log : stderr);
            pthread_mutex_unlock(&g_log_mu);
        }
        census_report("an overdue call");
    }
}

#ifndef CENSUS_WITH_CUPTI   // the exported-symbol front end; the CUPTI one below replaces it
// Finds the real entry point, and it takes two tries rather than one.
//
// `RTLD_NEXT` searches only the objects that follow this one in the load order, and libcuda is not
// necessarily among them: NCCL and PyTorch bring it in with `dlopen` into a local scope, after the
// preload. So the first lookup can return NULL for calls this shim must forward, and calling that NULL
// crashes the process. Opening the library by name and asking its handle finds it wherever it lives.
//
// `RTLD_NOLOAD` first, so this never *causes* libcuda to load in a process that was not going to use it.
static void* real(const char* name) {
    void* p = dlsym(RTLD_NEXT, name);
    if (p) return p;
    static void* cuda = NULL;
    if (!cuda) {
        cuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
        if (!cuda) cuda = dlopen("libcuda.so.1", RTLD_LAZY);
    }
    if (cuda) p = dlsym(cuda, name);
    if (!p) {
        // Fatal and said out loud. Returning a made-up error code would let the process continue with a
        // call that silently did nothing, and an instrument that alters the run it is measuring is worse
        // than no instrument, and crashing quietly is worse still.
        say("== census cannot resolve %s anywhere; refusing to guess. Unset LD_PRELOAD to run without "
            "the census.", name);
        abort();
    }
    return p;
}

// One wrapper shape for every call that takes no more than six arguments, which covers all of them here.
// Written as a macro because the alternative is fifty near-identical functions, and a difference between
// two of them would be a bug nobody would see.
#define WRAP(ret, fn, params, args)                                     \
    ret fn params {                                                     \
        static ret (*next) params = NULL;                               \
        if (!next) next = real(#fn);                                    \
        check_overdue();                                                \
        const int slot = claim(#fn);                                    \
        ret r = next args;                                              \
        release(slot);                                                  \
        return r;                                                       \
    }

typedef int CUresult_;
typedef unsigned long long CUdeviceptr_;

// The set is chosen to cover every explanation that is still open: module management, pinned and device
// allocation, virtual memory mapping, transfers, launches, and the context-wide operations. A call left
// out here is an explanation that cannot be ruled out.
WRAP(CUresult_, cuModuleLoad, (void* m, const char* f), (m, f))
WRAP(CUresult_, cuModuleLoadData, (void* m, const void* i), (m, i))
WRAP(CUresult_, cuModuleLoadDataEx, (void* m, const void* i, unsigned n, int* o, void** v),
     (m, i, n, o, v))
WRAP(CUresult_, cuModuleLoadFatBinary, (void* m, const void* i), (m, i))
WRAP(CUresult_, cuModuleUnload, (void* m), (m))
WRAP(CUresult_, cuModuleGetFunction, (void* f, void* m, const char* n), (f, m, n))
WRAP(CUresult_, cuLibraryLoadData,
     (void* l, const void* i, void** jo, void** jv, unsigned jn, void** lo, void** lv, unsigned ln),
     (l, i, jo, jv, jn, lo, lv, ln))
WRAP(CUresult_, cuMemAlloc_v2, (CUdeviceptr_* p, size_t b), (p, b))
WRAP(CUresult_, cuMemFree_v2, (CUdeviceptr_ p), (p))
WRAP(CUresult_, cuMemAllocHost_v2, (void** p, size_t b), (p, b))
WRAP(CUresult_, cuMemFreeHost, (void* p), (p))
WRAP(CUresult_, cuMemHostRegister_v2, (void* p, size_t b, unsigned f), (p, b, f))
WRAP(CUresult_, cuMemHostUnregister, (void* p), (p))
WRAP(CUresult_, cuMemAddressReserve,
     (CUdeviceptr_* p, size_t s, size_t a, CUdeviceptr_ f, unsigned long long fl), (p, s, a, f, fl))
WRAP(CUresult_, cuMemCreate, (void* h, size_t s, const void* p, unsigned long long f), (h, s, p, f))
WRAP(CUresult_, cuMemMap, (CUdeviceptr_ p, size_t s, size_t o, void* h, unsigned long long f),
     (p, s, o, h, f))
WRAP(CUresult_, cuMemUnmap, (CUdeviceptr_ p, size_t s), (p, s))
WRAP(CUresult_, cuMemSetAccess, (CUdeviceptr_ p, size_t s, const void* d, size_t c), (p, s, d, c))
WRAP(CUresult_, cuMemRelease, (void* h), (h))
WRAP(CUresult_, cuMemcpyHtoDAsync_v2, (CUdeviceptr_ d, const void* s, size_t b, void* st),
     (d, s, b, st))
WRAP(CUresult_, cuMemcpyDtoHAsync_v2, (void* d, CUdeviceptr_ s, size_t b, void* st), (d, s, b, st))
WRAP(CUresult_, cuMemcpyHtoD_v2, (CUdeviceptr_ d, const void* s, size_t b), (d, s, b))
WRAP(CUresult_, cuMemcpyDtoH_v2, (void* d, CUdeviceptr_ s, size_t b), (d, s, b))
WRAP(CUresult_, cuLaunchKernel,
     (void* f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
      unsigned shared, void* stream, void** params, void** extra),
     (f, gx, gy, gz, bx, by, bz, shared, stream, params, extra))
WRAP(CUresult_, cuStreamCreate, (void** s, unsigned f), (s, f))
WRAP(CUresult_, cuStreamDestroy_v2, (void* s), (s))
WRAP(CUresult_, cuStreamSynchronize, (void* s), (s))
WRAP(CUresult_, cuCtxSynchronize, (void), ())
#endif  // !CENSUS_WITH_CUPTI

#ifdef CENSUS_WITH_CUPTI
#include <cupti.h>

// Every driver call, entered and left. The slot travels in CUPTI's per-call correlation word, so exit finds
// exactly the slot entry claimed without a lookup.
static void CUPTIAPI on_driver_call(void* userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                                    const void* data) {
    (void)userdata; (void)domain; (void)cbid;
    const CUpti_CallbackData* cb = (const CUpti_CallbackData*)data;
    if (cb->callbackSite == CUPTI_API_ENTER) {
        // CUDA_CENSUS_TRACE_CALLS=<substring> logs every entered call whose name contains it: the evidence
        // that a given call is seen at all, which an empty in-flight table cannot give.
        static const char* trace = (const char*)1;
        if (trace == (const char*)1) trace = getenv("CUDA_CENSUS_TRACE_CALLS");
        if (trace && *trace && strstr(cb->functionName, trace)) say("== census: entered %s", cb->functionName);
        *cb->correlationData = (uint64_t)(int64_t)claim(cb->functionName);
    } else {
        release((int)(int64_t)*cb->correlationData);
    }
}

// Called by the driver when CUDA_INJECTION64_PATH names this library, before the first driver call returns.
int InitializeInjection(void) {
    CUpti_SubscriberHandle subscriber;
    const CUptiResult r = cuptiSubscribe(&subscriber, on_driver_call, NULL);
    if (r != CUPTI_SUCCESS) {
        say("== census: cuptiSubscribe failed (%d); driver calls are NOT counted", (int)r);
        return 0;
    }
    if (cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API) != CUPTI_SUCCESS) {
        say("== census: enabling the driver-API domain failed; driver calls are NOT counted");
        return 0;
    }
    say("== census: counting every driver call through CUPTI");
    return 1;
}
#endif
