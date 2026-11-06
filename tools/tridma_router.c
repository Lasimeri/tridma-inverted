// tridma_router.c: the host side of the tridma DMA router, an LD_PRELOAD library
// (libtridma.so) for a llama-server that serves a GGUF with libggml_phi.so offloading part
// of every weight matrix to a Xeon Phi. Two jobs:
//
// 1. No model file is read whole into memory: for mappings of ".gguf" files, MAP_POPULATE
//    is cleared and (POSIX_)MADV_WILLNEED dropped (llama.cpp's loader asks for both).
//
// 2. The card's rows are backed by the card, not by the disk: after libggml_phi.so uploads
//    a matrix's card rows it calls tridma_back(), which puts an anonymous range under
//    userfaultfd over those rows' pages. Whatever reads them later (llama.cpp's CUDA op
//    offload copying the weight to the GPU for a prompt) faults here; the handler thread
//    fetches the chunk from the card's memory with K_DOWNLOAD (the card's DMA into the
//    window), copies it in with UFFDIO_COPY and reads ahead; a budget returns the oldest
//    chunks with MADV_DONTNEED, so the card's rows never pile up in host RAM. Requests go
//    through the window's doorbell lock (VPU_OFF_LOCK), so they never meet the backend's.
//    The faults are user-mode (CUDA reads pageable memory with ordinary loads), which is
//    what an unprivileged userfaultfd (UFFD_USER_MODE_ONLY) handles.
//
// Environment: TRIDMA_CHUNK (bytes per fetch, default 8 MiB), TRIDMA_AHEAD (chunks read
// ahead, default 2), TRIDMA_BUDGET (bytes resident at most, default 512 MiB),
// TRIDMA_STAGE (window offset of the staging area, default 1 GiB), TRIDMA_VERBOSE=1.
// See tridma_router.md.
//
// build: gcc -O2 -Wall -Wextra -shared -fPIC -I<Intel-Phi-AVX-512-C>/card/vpu
//        -o libtridma.so tridma_router.c -ldl -lpthread
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <poll.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "vpu_proto.h"
#include "vpu_matmul.h"

#define PG 4096ull
#define MAX_GGUF 64
#define MAX_RANGES 4096
#define MAX_CHUNKS 65536

// ---- 1. no populate for GGUF mappings ----

static struct { uintptr_t a, b; } gguf[MAX_GGUF];
static pthread_mutex_t gguf_mu = PTHREAD_MUTEX_INITIALIZER;
static void *(*real_mmap)(void *, size_t, int, int, int, off_t);
static int (*real_munmap)(void *, size_t);
static int (*real_madvise)(void *, size_t, int);
static int (*real_posix_madvise)(void *, size_t, int);
static int verbose = -1;

static int vb(void) {
    if (verbose < 0) { const char *v = getenv("TRIDMA_VERBOSE"); verbose = v && *v == '1'; }
    return verbose;
}

static int fd_is_gguf(int fd) {
    char link[64], path[4096];
    snprintf(link, sizeof link, "/proc/self/fd/%d", fd);
    ssize_t n = readlink(link, path, sizeof path - 1);
    if (n < 5) return 0;
    path[n] = 0;
    return !strcmp(path + n - 5, ".gguf");
}

static int in_gguf(const void *p, size_t len) {
    uintptr_t a = (uintptr_t) p, b = a + len;
    int hit = 0;
    pthread_mutex_lock(&gguf_mu);
    for (int i = 0; i < MAX_GGUF && !hit; i++) hit = gguf[i].b && a < gguf[i].b && b > gguf[i].a;
    pthread_mutex_unlock(&gguf_mu);
    return hit;
}

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    if (!real_mmap) real_mmap = (void *(*)(void *, size_t, int, int, int, off_t)) dlsym(RTLD_NEXT, "mmap");
    const int g = fd >= 0 && fd_is_gguf(fd);
    if (g && (flags & MAP_POPULATE)) {
        flags &= ~MAP_POPULATE;
        if (vb()) fprintf(stderr, "tridma: MAP_POPULATE cleared on a %zu-byte GGUF mapping\n", len);
    }
    void *r = real_mmap(addr, len, prot, flags, fd, off);
    if (g && r != MAP_FAILED) {
        pthread_mutex_lock(&gguf_mu);
        for (int i = 0; i < MAX_GGUF; i++)
            if (!gguf[i].b) { gguf[i].a = (uintptr_t) r; gguf[i].b = (uintptr_t) r + len; break; }
        pthread_mutex_unlock(&gguf_mu);
    }
    return r;
}
void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off_t off) __attribute__((alias("mmap")));

int munmap(void *addr, size_t len) {
    if (!real_munmap) real_munmap = (int (*)(void *, size_t)) dlsym(RTLD_NEXT, "munmap");
    pthread_mutex_lock(&gguf_mu);
    for (int i = 0; i < MAX_GGUF; i++)
        if (gguf[i].b && gguf[i].a == (uintptr_t) addr) gguf[i].a = gguf[i].b = 0;
    pthread_mutex_unlock(&gguf_mu);
    return real_munmap(addr, len);
}

int madvise(void *addr, size_t len, int advice) {
    if (!real_madvise) real_madvise = (int (*)(void *, size_t, int)) dlsym(RTLD_NEXT, "madvise");
    if (advice == MADV_WILLNEED && in_gguf(addr, len)) return 0;
    return real_madvise(addr, len, advice);
}

int posix_madvise(void *addr, size_t len, int advice) {
    if (!real_posix_madvise) real_posix_madvise = (int (*)(void *, size_t, int)) dlsym(RTLD_NEXT, "posix_madvise");
    if (advice == POSIX_MADV_WILLNEED && in_gguf(addr, len)) return 0;
    return real_posix_madvise(addr, len, advice);
}

// ---- 2. card-backed pages ----

struct range {
    uintptr_t a, b;        // the anonymous range, page aligned
    uint64_t id;           // the card's resident tensor
    uint64_t card_off;     // the card byte of a
    uint32_t chunk0;       // this range's first index in chunk_state
};

static struct range ranges[MAX_RANGES];
static int nranges;
static unsigned char chunk_state[MAX_CHUNKS];   // 1 = populated
static uint32_t nchunks;
static uint32_t fifo[MAX_CHUNKS];               // populated chunks, oldest first
static uint32_t fifo_head, fifo_len;
static uint64_t resident;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static int uffd = -1;
static unsigned char *win;                      // the card's window (the backend's mapping)
static uint32_t card_threads;
static uint64_t opt_chunk, opt_budget, opt_stage, opt_ahead;
static uint64_t st_faults, st_fetched, st_bytes, st_evicted, st_ns;

static uint64_t env_u64(const char *n, uint64_t d) {
    const char *v = getenv(n);
    if (!v || !*v) return d;
    char *e;
    uint64_t x = strtoull(v, &e, 0);
    return *e ? d : x;
}

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t) t.tv_sec * 1000000000ull + (uint64_t) t.tv_nsec;
}

// the doorbell lock: the same test-and-set as the backend's lock.S
static void lock_take(volatile uint32_t *l) {
    while (__atomic_exchange_n(l, 1, __ATOMIC_ACQUIRE))
        while (__atomic_load_n(l, __ATOMIC_RELAXED)) __builtin_ia32_pause();
}
static void lock_drop(volatile uint32_t *l) { __atomic_store_n(l, 0, __ATOMIC_RELEASE); }

// K_DOWNLOAD: bytes [off, off + bytes) of resident id into the window at d_off; 0 or the status
static int card_download(uint64_t id, uint64_t off, uint64_t bytes, uint64_t d_off) {
    volatile uint32_t *lk = (volatile uint32_t *) (win + VPU_OFF_LOCK);
    lock_take(lk);
    struct vpu_matmul mm;
    memset(&mm, 0, sizeof mm);
    mm.a_id = id;
    mm.a_off = off;
    mm.bytes = bytes;
    mm.d_off = d_off;
    memcpy((void *) (win + VPU_OFF_MATMUL), &mm, sizeof mm);
    volatile struct vpu_request *rq = (volatile struct vpu_request *) (win + VPU_OFF_REQ);
    const uint64_t seq = rq->seq + 1;
    rq->kernel = VPU_K_DOWNLOAD;
    rq->threads = card_threads;
    rq->n = 1;
    rq->in_off = 0; rq->out_off = 0; rq->aux_off = 0; rq->aux_len = 0;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    rq->seq = seq;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    volatile struct vpu_reply *rp = (volatile struct vpu_reply *) (win + VPU_OFF_REPLY);
    const uint64_t t0 = now_ns();
    int status = -100;
    for (;;) {
        if (rp->seq == seq) { __atomic_thread_fence(__ATOMIC_ACQUIRE); status = rp->status; break; }
        if (now_ns() - t0 > 30000000000ull) break;
        __builtin_ia32_pause();
    }
    lock_drop(lk);
    return status;
}

static struct range *find_range(uintptr_t x) {
    int lo = 0, hi = nranges - 1;   // ranges are kept sorted by address
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (x < ranges[mid].a) hi = mid - 1;
        else if (x >= ranges[mid].b) lo = mid + 1;
        else return &ranges[mid];
    }
    return NULL;
}

// one chunk of a range in: from the card into the staging area, then into the pages
static int fill_chunk(struct range *r, uint64_t k) {
    const uint32_t ci = r->chunk0 + (uint32_t) k;
    if (chunk_state[ci]) return 0;
    const uintptr_t a = r->a + k * opt_chunk;
    if (a >= r->b) return 0;
    const uint64_t len = (r->b - a) < opt_chunk ? (r->b - a) : opt_chunk;
    const uint64_t coff = r->card_off + (a - r->a);
    const uint64_t off0 = coff & ~(PG - 1), delta = coff - off0;
    const uint64_t bytes = (delta + len + PG - 1) & ~(PG - 1);
    const uint64_t t0 = now_ns();
    int s = card_download(r->id, off0, bytes, opt_stage);
    if (s) {
        fprintf(stderr, "tridma: K_DOWNLOAD of id %llu, %llu bytes at %llu failed: status %d; aborting rather than "
                        "giving the GPU wrong weights\n", (unsigned long long) r->id, (unsigned long long) bytes,
                (unsigned long long) off0, s);
        abort();
    }
    struct uffdio_copy c = {.dst = a, .src = (uintptr_t) (win + opt_stage + delta), .len = len, .mode = 0};
    if (ioctl(uffd, UFFDIO_COPY, &c) && errno != EEXIST) {
        fprintf(stderr, "tridma: UFFDIO_COPY at %#lx: %s\n", (unsigned long) a, strerror(errno));
        abort();
    }
    chunk_state[ci] = 1;
    fifo[(fifo_head + fifo_len) % MAX_CHUNKS] = ci;
    fifo_len++;
    resident += len;
    st_fetched++;
    st_bytes += len;
    st_ns += now_ns() - t0;
    return 1;
}

// the oldest chunks out while over budget (their pages fault again if read again)
static void evict(void) {
    while (resident > opt_budget && fifo_len) {
        const uint32_t ci = fifo[fifo_head];
        fifo_head = (fifo_head + 1) % MAX_CHUNKS;
        fifo_len--;
        if (!chunk_state[ci]) continue;
        for (int i = 0; i < nranges; i++) {
            struct range *r = &ranges[i];
            const uint64_t n = (r->b - r->a + opt_chunk - 1) / opt_chunk;
            if (ci < r->chunk0 || ci >= r->chunk0 + n) continue;
            const uintptr_t a = r->a + (uint64_t) (ci - r->chunk0) * opt_chunk;
            const uint64_t len = (r->b - a) < opt_chunk ? (r->b - a) : opt_chunk;
            if (!real_madvise) real_madvise = (int (*)(void *, size_t, int)) dlsym(RTLD_NEXT, "madvise");
            real_madvise((void *) a, len, MADV_DONTNEED);
            resident -= len;
            st_evicted++;
            break;
        }
        chunk_state[ci] = 0;
    }
}

static void *handler(void *arg) {
    (void) arg;
    for (;;) {
        struct pollfd p = {.fd = uffd, .events = POLLIN};
        if (poll(&p, 1, -1) <= 0) continue;
        struct uffd_msg m;
        if (read(uffd, &m, sizeof m) != (ssize_t) sizeof m) continue;
        if (m.event != UFFD_EVENT_PAGEFAULT) continue;
        const uintptr_t x = (uintptr_t) m.arg.pagefault.address;
        pthread_mutex_lock(&mu);
        st_faults++;
        struct range *r = find_range(x);
        if (!r) {
            pthread_mutex_unlock(&mu);
            fprintf(stderr, "tridma: a fault at %#lx outside every range\n", (unsigned long) x);
            abort();
        }
        const uint64_t k = (x - r->a) / opt_chunk;
        if (!fill_chunk(r, k)) {   // already in: a racing fault; wake the thread
            struct uffdio_range w = {.start = x & ~(PG - 1), .len = PG};
            ioctl(uffd, UFFDIO_WAKE, &w);
        }
        for (uint64_t j = 1; j <= opt_ahead; j++) fill_chunk(r, k + j);
        evict();
        if (vb() && st_fetched % 256 == 0)
            fprintf(stderr, "tridma: %llu faults, %llu chunks fetched (%.2f GB, %.2f GB/s), %llu evicted, %.0f MiB resident\n",
                    (unsigned long long) st_faults, (unsigned long long) st_fetched, st_bytes / 1e9,
                    st_ns ? st_bytes / (st_ns * 1e-9) / 1e9 : 0, (unsigned long long) st_evicted, resident / 1048576.0);
        pthread_mutex_unlock(&mu);
    }
    return NULL;
}

static int router_init(unsigned char *window, uint32_t threads) {
    if (uffd >= 0) return 0;
    opt_chunk = env_u64("TRIDMA_CHUNK", 8ull << 20) & ~(PG - 1);
    opt_ahead = env_u64("TRIDMA_AHEAD", 2);
    opt_budget = env_u64("TRIDMA_BUDGET", 512ull << 20);
    opt_stage = env_u64("TRIDMA_STAGE", 1ull << 30);
    if (!opt_chunk || opt_stage % PG) return -1;
    uffd = (int) syscall(SYS_userfaultfd, O_CLOEXEC | UFFD_USER_MODE_ONLY);
    if (uffd < 0) { fprintf(stderr, "tridma: userfaultfd: %s\n", strerror(errno)); return -1; }
    struct uffdio_api api = {.api = UFFD_API, .features = 0};
    if (ioctl(uffd, UFFDIO_API, &api)) { fprintf(stderr, "tridma: UFFDIO_API: %s\n", strerror(errno)); return -1; }
    win = window;
    card_threads = threads;
    pthread_t t;
    if (pthread_create(&t, NULL, handler, NULL)) return -1;
    pthread_detach(t);
    fprintf(stderr, "tridma: card-backed pages on: chunk %llu KiB, %llu ahead, budget %llu MiB, staging at %llu MiB of the window\n",
            (unsigned long long) (opt_chunk >> 10), (unsigned long long) opt_ahead,
            (unsigned long long) (opt_budget >> 20), (unsigned long long) (opt_stage >> 20));
    return 0;
}

// Called by libggml_phi.so (back_or_drop) right after it uploaded a dense matrix's card
// rows [addr, addr + len) as resident id, card byte a_off; the bytes now card-backed, or 0
// (the caller then drops the pages as before).
size_t tridma_back(void *addr, size_t len, uint64_t id, uint64_t a_off, unsigned char *window, uint32_t threads) {
    pthread_mutex_lock(&mu);
    size_t done = 0;
    if (router_init(window, threads) == 0 && nranges < MAX_RANGES) {
        const uintptr_t a = ((uintptr_t) addr + PG - 1) & ~(PG - 1);
        const uintptr_t b = ((uintptr_t) addr + len) & ~(PG - 1);
        const uint64_t n = b > a ? (b - a + opt_chunk - 1) / opt_chunk : 0;
        if (n && nchunks + n <= MAX_CHUNKS) {
            if (!real_mmap) real_mmap = (void *(*)(void *, size_t, int, int, int, off_t)) dlsym(RTLD_NEXT, "mmap");
            void *r = real_mmap((void *) a, b - a, PROT_READ | PROT_WRITE,
                                MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (r == (void *) a) {
                if (!real_madvise) real_madvise = (int (*)(void *, size_t, int)) dlsym(RTLD_NEXT, "madvise");
                real_madvise((void *) a, b - a, MADV_NOHUGEPAGE);
                struct uffdio_register reg = {.range = {.start = a, .len = b - a}, .mode = UFFDIO_REGISTER_MODE_MISSING};
                if (ioctl(uffd, UFFDIO_REGISTER, &reg) == 0) {
                    int i = nranges++;   // insert sorted by address
                    while (i > 0 && ranges[i - 1].a > a) { ranges[i] = ranges[i - 1]; i--; }
                    ranges[i] = (struct range){a, b, id, a_off + (a - (uintptr_t) addr), nchunks};
                    nchunks += (uint32_t) n;
                    done = b - a;
                } else {
                    fprintf(stderr, "tridma: UFFDIO_REGISTER: %s\n", strerror(errno));
                }
            } else {
                fprintf(stderr, "tridma: mmap over the card rows failed: %s\n", strerror(errno));
            }
        }
    }
    pthread_mutex_unlock(&mu);
    if (done && vb()) fprintf(stderr, "tridma: id %llu: %zu bytes card-backed\n", (unsigned long long) id, done);
    return done;
}
