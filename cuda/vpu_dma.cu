// vpu_dma.cu: the GPU as a client of the Xeon Phi VPU worker. The worker's
// host window (/dev/shm/phi-hostmem) is pinned for the GPU, so CUDA kernels
// write activations into it, post the request, wait for the reply and read
// the results, all through the GPU's own PCIe traffic; the card moves the
// same pages with its DMA channel 7. No host thread copies data or rings
// the doorbell. Protocol: Intel-Phi-AVX-512-C card/vpu/vpu_proto.h and
// vpu_matmul.h (included below, so their layout asserts run here too).
// See vpu_dma.md.
#include <cuda_runtime.h>
#include <dirent.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <vector>

#define _Static_assert(c, m) static_assert(c, m)
#include "vpu_proto.h"
#include "vpu_matmul.h"
#undef _Static_assert

// The window ranges of phi-vpu matmul-check (host/crates/phi-vpu/src/matmul.rs)
static const uint64_t OFF_A = 128ull << 20, A_MAX = 512ull << 20;
static const uint64_t OFF_B = 640ull << 20, B_MAX = 64ull << 20;
static const uint64_t OFF_D = 704ull << 20, D_MAX = 64ull << 20;
static const uint64_t WIN_LEN = 768ull << 20;
static const char *WINDOW = "/dev/shm/phi-hostmem";

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "vpu_dma: %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

static uint64_t round_block(uint64_t x) { return (x + VPU_BLOCK - 1) & ~(uint64_t) (VPU_BLOCK - 1); }

// a whole-argument unsigned number, or exit: getopt hands "-m 256 -k 8" to -m as one string
static uint64_t num(const char *s, char opt) {
    char *e;
    uint64_t v = strtoull(s, &e, 0);
    if (e == s || *e) { fprintf(stderr, "vpu_dma: -%c wants a number, got \"%s\"\n", opt, s); exit(2); }
    return v;
}

// ---- device side ----

__device__ __forceinline__ uint64_t ld_acquire_sys(const volatile uint64_t *p) {
    uint64_t v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void st_release_sys(volatile uint64_t *p, uint64_t v) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ uint64_t gtimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

struct call_out {
    int status;          // the reply's, or -100 for no answer in time
    int threads;
    uint64_t seq, wait_ns, compute_ns, total_ns, pull_ns, push_ns;
};

// Descriptor and request fields, then the doorbell; the caller has fenced
// any data it wrote. One thread.
__device__ uint64_t ring(unsigned char *win, const vpu_matmul &mm, uint32_t kernel, uint32_t threads) {
    volatile uint64_t *d = (volatile uint64_t *) (win + VPU_OFF_MATMUL);
    const uint64_t *s = (const uint64_t *) &mm;
    for (int i = 0; i < (int) (sizeof(vpu_matmul) / 8); i++) d[i] = s[i];
    volatile vpu_request *rq = (volatile vpu_request *) (win + VPU_OFF_REQ);
    rq->kernel = kernel;
    rq->threads = threads;
    rq->n = 1;
    rq->in_off = 0; rq->out_off = 0; rq->aux_off = 0; rq->aux_len = 0;
    uint64_t seq = rq->seq + 1;
    __threadfence_system();
    st_release_sys(&rq->seq, seq);
    return seq;
}

__device__ void await(unsigned char *win, uint64_t seq, uint64_t t0, uint64_t timeout_ns, call_out *o) {
    volatile vpu_reply *rp = (volatile vpu_reply *) (win + VPU_OFF_REPLY);
    for (;;) {
        if (ld_acquire_sys(&rp->seq) == seq) break;
        if (gtimer() - t0 > timeout_ns) { o->status = -100; o->seq = seq; return; }
    }
    o->wait_ns = gtimer() - t0;
    o->seq = seq;
    o->status = rp->status;
    o->threads = rp->threads;
    o->compute_ns = rp->compute_ns;
    o->total_ns = rp->total_ns;
    o->pull_ns = rp->pull_ns;
    o->push_ns = rp->push_ns;
}

__global__ void k_call(unsigned char *win, vpu_matmul mm, uint32_t kernel, uint32_t threads,
                       uint64_t timeout_ns, call_out *o) {
    uint64_t seq = ring(win, mm, kernel, threads);
    await(win, seq, gtimer(), timeout_ns, o);
}

// deterministic values in [-scale, scale)
__global__ void k_fill(float *p, size_t n, uint32_t seed, float scale) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t) i * 0x9E3779B1u ^ seed;
        h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
        p[i] = ((float) (h >> 8) * (1.0f / 8388608.0f) - 1.0f) * scale;
    }
}

// rows of k floats into the window at stride nb (bytes); GPU stores over PCIe
__global__ void k_put(unsigned char *dst, const float *src, uint64_t rows, uint64_t k, uint64_t nb) {
    const uint64_t q = k / 4;
    for (uint64_t r = blockIdx.x; r < rows; r += gridDim.x) {
        float4 *o = (float4 *) (dst + r * nb);
        const float4 *s = (const float4 *) (src + r * k);
        for (uint64_t i = threadIdx.x; i < q; i += blockDim.x) o[i] = s[i];
    }
    __threadfence_system();
}

// rows of m floats out of the window; __ldcv fetches from memory every time
__global__ void k_get(float *dst, const unsigned char *src, uint64_t rows, uint64_t m, uint64_t nb) {
    const uint64_t q = m / 4;
    for (uint64_t r = blockIdx.x; r < rows; r += gridDim.x) {
        const float4 *s = (const float4 *) (src + r * nb);
        float4 *o = (float4 *) (dst + r * m);
        for (uint64_t i = threadIdx.x; i < q; i += blockDim.x) o[i] = __ldcv(s + i);
    }
}

// fp64 reference: d[c][r] = a[r] . b[c]
__global__ void k_ref(double *d, const float *a, const float *b, uint64_t m, uint64_t n, uint64_t k) {
    uint64_t idx = blockIdx.x * (uint64_t) blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    uint64_t c = idx / m, r = idx % m;
    double s = 0;
    for (uint64_t i = 0; i < k; i++) s += (double) a[r * k + i] * (double) b[c * k + i];
    d[c * m + r] = s;
}

// The whole round trip in one block, reps times: activations into the
// window, every thread's stores fenced, the doorbell, the reply, the
// results back into GPU memory. lat[r] is each round trip in ns.
__global__ void k_fused(unsigned char *win, vpu_matmul mm, uint32_t threads, const float *b, float *d,
                        int reps, uint64_t timeout_ns, uint64_t *lat, call_out *sum) {
    __shared__ int stop;
    const uint64_t qb = mm.k / 4, qd = mm.m / 4;
    if (threadIdx.x == 0) { stop = 0; *sum = call_out{}; }
    __syncthreads();
    for (int r = 0; r < reps; r++) {
        uint64_t t0 = gtimer();
        for (uint64_t c = 0; c < mm.n; c++) {
            float4 *o = (float4 *) (win + mm.b_off + c * mm.nb_b);
            const float4 *s = (const float4 *) (b + c * mm.k);
            for (uint64_t i = threadIdx.x; i < qb; i += blockDim.x) o[i] = s[i];
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            call_out o{};
            uint64_t seq = ring(win, mm, VPU_K_MATMUL, threads);
            await(win, seq, t0, timeout_ns, &o);
            if (o.status != VPU_OK) { stop = 1; *sum = o; }
            else {
                sum->seq = o.seq; sum->threads = o.threads;
                sum->compute_ns += o.compute_ns; sum->total_ns += o.total_ns;
                sum->pull_ns += o.pull_ns; sum->push_ns += o.push_ns;
            }
        }
        __syncthreads();
        if (stop) return;
        for (uint64_t c = 0; c < mm.n; c++) {
            const float4 *s = (const float4 *) (win + mm.d_off + c * mm.m * 4);
            float4 *o = (float4 *) (d + c * mm.m);
            for (uint64_t i = threadIdx.x; i < qd; i += blockDim.x) o[i] = __ldcv(s + i);
        }
        __syncthreads();
        if (threadIdx.x == 0) lat[r] = gtimer() - t0;
    }
}

// ---- host side ----

// the "one program per card" rule (scripts/card-users.sh): nobody but the
// stack's daemon and monitor may map the window
static int card_users(void) {
    int found = 0;
    DIR *dir = opendir("/proc");
    if (!dir) return 0;
    struct dirent *e;
    char path[300], line[4096];
    while ((e = readdir(dir))) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9' || atoi(e->d_name) == (int) getpid()) continue;
        snprintf(path, sizeof path, "/proc/%s/comm", e->d_name);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        char comm[64] = "";
        if (!fgets(comm, sizeof comm, f)) comm[0] = 0;
        fclose(f);
        comm[strcspn(comm, "\n")] = 0;
        if (!strcmp(comm, "phictl") || !strcmp(comm, "phitop")) continue;
        snprintf(path, sizeof path, "/proc/%s/maps", e->d_name);
        if (!(f = fopen(path, "r"))) continue;
        while (fgets(line, sizeof line, f)) {
            size_t L = strlen(line);
            if (L > 21 && !strcmp(line + L - 21, "/dev/shm/phi-hostmem\n")) {
                fprintf(stderr, "vpu_dma: pid %s (%s) maps the window\n", e->d_name, comm);
                found = 1;
                break;
            }
        }
        fclose(f);
    }
    closedir(dir);
    return found;
}

static double secs(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    return ms / 1000.0;
}

static void call(unsigned char *dwin, const vpu_matmul &mm, uint32_t kernel, uint32_t threads, call_out *dout,
                 const char *what) {
    k_call<<<1, 1>>>(dwin, mm, kernel, threads, 10000000000ull, dout);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    call_out o;
    CK(cudaMemcpy(&o, dout, sizeof o, cudaMemcpyDeviceToHost));
    if (o.status != VPU_OK) {
        fprintf(stderr, "vpu_dma: %s (request %llu): status %d%s\n", what ? what : "request", (unsigned long long) o.seq,
                o.status, o.status == -100 ? " (no answer within 10 s)" : "");
        exit(1);
    }
    if (!what) return;   // a repeated request: only failures print
    printf("%-8s seq %llu: wait %.3f ms, card total %.3f ms (pull %.3f, compute %.3f, push %.3f), %d threads\n",
           what, (unsigned long long) o.seq, o.wait_ns / 1e6, o.total_ns / 1e6, o.pull_ns / 1e6, o.compute_ns / 1e6,
           o.push_ns / 1e6, o.threads);
}

static void check(const char *what, const float *d_dev, const double *ref_dev, uint64_t cnt) {
    std::vector<float> d(cnt);
    std::vector<double> ref(cnt);
    CK(cudaMemcpy(d.data(), d_dev, cnt * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ref.data(), ref_dev, cnt * 8, cudaMemcpyDeviceToHost));
    double max_abs = 0, max_ref = 0;
    for (uint64_t i = 0; i < cnt; i++) {
        max_abs = std::max(max_abs, fabs((double) d[i] - ref[i]));
        max_ref = std::max(max_ref, fabs(ref[i]));
    }
    printf("%-8s %llu results vs fp64: max |err| %.3e, max |ref| %.3e, ratio %.3e\n", what,
           (unsigned long long) cnt, max_abs, max_ref, max_abs / max_ref);
}

static void pct(const char *what, std::vector<uint64_t> v) {
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (uint64_t x : v) sum += (double) x;
    size_t n = v.size();
    printf("%-26s %zu round trips: min %.1f us, p50 %.1f, p99 %.1f, max %.1f, mean %.1f\n", what, n, v[0] / 1e3,
           v[n / 2] / 1e3, v[std::min(n - 1, n * 99 / 100)] / 1e3, v[n - 1] / 1e3, sum / n / 1e3);
}

int main(int argc, char **argv) {
    uint64_t m = 4096, k = 2048, n = 1, id = 0x4750555F444D41ull;   // "GPU_DMA"
    int reps = 2000, threads = 120, free_all = 0, dl_reps = 0, opt;
    while ((opt = getopt(argc, argv, "m:k:n:r:t:FD:")) != -1) {
        switch (opt) {
        case 'F': free_all = 1; break;
        case 'D': dl_reps = (int) num(optarg, 'D'); break;
        case 'm': m = num(optarg, 'm'); break;
        case 'k': k = num(optarg, 'k'); break;
        case 'n': n = num(optarg, 'n'); break;
        case 'r': reps = (int) num(optarg, 'r'); break;
        case 't': threads = (int) num(optarg, 't'); break;
        default: fprintf(stderr, "usage: vpu_dma [-m rows] [-k width] [-n columns] [-r reps] [-t card threads] | -F\n"); return 2;
        }
    }
    const uint64_t nb_a = k * 4, nb_b = (k * 4 + 63) & ~63ull;
    if (m % 4 || k % 16 || reps < 1 || m * nb_a > A_MAX || n * nb_b > B_MAX || n * m * 4 > D_MAX) {
        fprintf(stderr, "vpu_dma: m a multiple of 4, k of 16, and the operands within the window ranges\n");
        return 2;
    }
    if (card_users()) { fprintf(stderr, "vpu_dma: the card is in use; one program at a time\n"); return 1; }

    int fd = open(WINDOW, O_RDWR);
    if (fd < 0) { perror(WINDOW); return 1; }
    unsigned char *win = (unsigned char *) mmap(0, WIN_LEN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (win == MAP_FAILED) { perror("mmap"); return 1; }
    close(fd);
    if (*(volatile uint64_t *) (win + VPU_OFF_READY) != VPU_MAGIC) { fprintf(stderr, "vpu_dma: no worker polling\n"); return 1; }

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU %s, window %s pinned for it: %llu MiB\n", prop.name, WINDOW, (unsigned long long) (WIN_LEN >> 20));
    CK(cudaHostRegister(win, WIN_LEN, cudaHostRegisterMapped | cudaHostRegisterPortable));
    unsigned char *dwin;
    CK(cudaHostGetDevicePointer((void **) &dwin, win, 0));

    if (free_all) {   // -F: VPU_K_FREE with a_id 0 drops every resident tensor on the card
        call_out *o;
        CK(cudaMalloc(&o, sizeof(call_out)));
        vpu_matmul fr = {};
        call(dwin, fr, VPU_K_FREE, threads, o, "free all");
        CK(cudaHostUnregister(win));
        munmap(win, WIN_LEN);
        return 0;
    }

    float *a, *b, *d;
    double *ref;
    call_out *dout;
    uint64_t *lat;
    CK(cudaMalloc(&a, m * k * 4));
    CK(cudaMalloc(&b, n * k * 4));
    CK(cudaMalloc(&d, n * m * 4));
    CK(cudaMalloc(&ref, n * m * 8));
    CK(cudaMalloc(&dout, sizeof(call_out)));
    CK(cudaMalloc(&lat, reps * 8ull));
    k_fill<<<256, 256>>>(a, m * k, 0x1234567u, 1.0f);
    k_fill<<<256, 256>>>(b, n * k, 0x7654321u, 1.0f);
    k_ref<<<(unsigned) ((m * n + 255) / 256), 256>>>(ref, a, b, m, n, k);
    CK(cudaDeviceSynchronize());

    vpu_stats st0;
    memcpy(&st0, (const void *) (win + VPU_OFF_STATS), sizeof st0);
    uint64_t cd0[4];
    memcpy(cd0, (const void *) (win + VPU_OFF_CDMA_STATS), 32);

    // the weights into the window: the copy engine, then the SMs' own stores
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    CK(cudaMemcpy(win + OFF_A, a, m * nb_a, cudaMemcpyDeviceToHost));
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    printf("weights %.1f MiB into the window: copy engine %.2f GB/s", m * nb_a / 1048576.0, m * nb_a / secs(e0, e1) / 1e9);
    CK(cudaEventRecord(e0));
    k_put<<<1024, 256>>>(dwin + OFF_A, a, m, k, nb_a);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
    printf(", kernel stores %.2f GB/s\n", m * nb_a / secs(e0, e1) / 1e9);

    vpu_matmul mm = {};
    mm.a_id = id; mm.a_off = OFF_A; mm.bytes = round_block(m * nb_a);
    call(dwin, mm, VPU_K_UPLOAD, threads, dout, "upload");

    if (dl_reps > 0) {   // -D N: the resident matrix back to the window N times (VPU_K_DOWNLOAD), checked, timed
        const uint64_t bytes = round_block(m * nb_a);
        if (bytes > B_MAX) { fprintf(stderr, "vpu_dma: -D wants the matrix within %llu bytes\n", (unsigned long long) B_MAX); return 2; }
        vpu_matmul dl = {};
        dl.a_id = id; dl.a_off = 0; dl.bytes = bytes; dl.d_off = OFF_B;
        memset(win + OFF_B, 0, bytes);
        timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int r = 0; r < dl_reps; r++) call(dwin, dl, VPU_K_DOWNLOAD, threads, dout, r ? NULL : "download");
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const double s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        const int same = memcmp(win + OFF_B, win + OFF_A, m * nb_a) == 0;
        printf("download %d x %.1f MiB: %.3f s, %.2f GB/s card -> window, bytes %s the uploaded matrix\n", dl_reps,
               bytes / 1048576.0, s, dl_reps * (double) bytes / s / 1e9, same ? "IDENTICAL to" : "DIFFER from");
        float *gd;
        CK(cudaMalloc(&gd, bytes));
        CK(cudaEventRecord(e0));
        for (int r = 0; r < dl_reps; r++) CK(cudaMemcpy(gd, win + OFF_B, bytes, cudaMemcpyHostToDevice));
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        printf("window -> GPU (copy engine): %.2f GB/s\n", dl_reps * (double) bytes / secs(e0, e1) / 1e9);
        CK(cudaFree(gd));
        mm = vpu_matmul{};
        mm.a_id = id;
        call(dwin, mm, VPU_K_FREE, threads, dout, "free");
        CK(cudaHostUnregister(win));
        munmap(win, WIN_LEN);
        return same ? 0 : 1;
    }

    mm = vpu_matmul{};
    mm.a_id = id; mm.a_type = VPU_MM_F32; mm.b_type = 0;
    mm.m = m; mm.n = n; mm.k = k; mm.nb_a = nb_a; mm.nb_b = nb_b;
    mm.b_off = OFF_B; mm.d_off = OFF_D;

    // one multiply, three kernels: put, call, get
    k_put<<<(unsigned) std::min<uint64_t>(n, 1024), 256>>>(dwin + OFF_B, b, n, k, nb_b);
    CK(cudaGetLastError());
    call(dwin, mm, VPU_K_MATMUL, threads, dout, "matmul");
    CK(cudaMemset(d, 0, n * m * 4));
    k_get<<<(unsigned) std::min<uint64_t>(n, 1024), 256>>>(d, dwin + OFF_D, n, m, m * 4);
    CK(cudaDeviceSynchronize());
    check("matmul", d, ref, n * m);

    // the GPU-driven loop
    CK(cudaMemset(d, 0, n * m * 4));
    k_fused<<<1, 512>>>(dwin, mm, threads, b, d, reps, 10000000000ull, lat, dout);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    call_out s;
    CK(cudaMemcpy(&s, dout, sizeof s, cudaMemcpyDeviceToHost));
    if (s.status != VPU_OK) { fprintf(stderr, "vpu_dma: fused loop stopped at request %llu: status %d\n", (unsigned long long) s.seq, s.status); return 1; }
    std::vector<uint64_t> v(reps);
    CK(cudaMemcpy(v.data(), lat, reps * 8ull, cudaMemcpyDeviceToHost));
    check("fused", d, ref, n * m);
    pct("GPU-driven (fused kernel)", v);
    printf("%-26s card per request: total %.1f us (pull %.1f, compute %.1f, push %.1f)\n", "", s.total_ns / 1e3 / reps,
           s.pull_ns / 1e3 / reps, s.compute_ns / 1e3 / reps, s.push_ns / 1e3 / reps);

    // the host-orchestrated loop: copy engine in, CPU doorbell, copy engine out
    volatile vpu_request *rq = (volatile vpu_request *) (win + VPU_OFF_REQ);
    volatile vpu_reply *rp = (volatile vpu_reply *) (win + VPU_OFF_REPLY);
    memcpy((void *) (win + VPU_OFF_MATMUL), &mm, sizeof mm);
    CK(cudaMemset(d, 0, n * m * 4));
    for (int r = 0; r < reps; r++) {
        timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CK(cudaMemcpy2D(win + OFF_B, nb_b, b, k * 4, k * 4, n, cudaMemcpyDeviceToHost));
        rq->kernel = VPU_K_MATMUL; rq->threads = threads; rq->n = 1;
        uint64_t seq = rq->seq + 1;
        __sync_synchronize();
        rq->seq = seq;
        __sync_synchronize();
        while (rp->seq != seq) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            if ((t1.tv_sec - t0.tv_sec) > 10) { fprintf(stderr, "vpu_dma: host loop: no answer to %llu\n", (unsigned long long) seq); return 1; }
        }
        if (rp->status != VPU_OK) { fprintf(stderr, "vpu_dma: host loop: status %d\n", rp->status); return 1; }
        CK(cudaMemcpy(d, win + OFF_D, n * m * 4, cudaMemcpyHostToDevice));
        clock_gettime(CLOCK_MONOTONIC, &t1);
        v[r] = (uint64_t) ((t1.tv_sec - t0.tv_sec) * 1000000000ll + (t1.tv_nsec - t0.tv_nsec));
    }
    check("host", d, ref, n * m);
    pct("host-orchestrated", v);

    mm = vpu_matmul{};
    mm.a_id = id;
    call(dwin, mm, VPU_K_FREE, threads, dout, "free");

    vpu_stats st1;
    memcpy(&st1, (const void *) (win + VPU_OFF_STATS), sizeof st1);
    uint64_t cd1[4];
    memcpy(cd1, (const void *) (win + VPU_OFF_CDMA_STATS), 32);
    printf("worker: %llu requests (%llu errors); card DMA channel 7: %.2f MB in over %llu copies, %.2f MB out over %llu copies\n",
           (unsigned long long) (st1.requests - st0.requests), (unsigned long long) (st1.errors - st0.errors),
           (cd1[0] - cd0[0]) / 1e6, (unsigned long long) (cd1[2] - cd0[2]), (cd1[1] - cd0[1]) / 1e6,
           (unsigned long long) (cd1[3] - cd0[3]));

    CK(cudaHostUnregister(win));
    munmap(win, WIN_LEN);
    return 0;
}
