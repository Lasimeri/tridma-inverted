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
//    offload copying the weight to the GPU for a prompt) faults here. Three threads route it:
//    the fault thread queues the chunk, the fetch thread keeps the card's link busy with
//    K_DOWNLOAD after K_DOWNLOAD (the card's DMA into one of four staging slots of the
//    window), demand first, then read-ahead that runs on into the matrix read next (an order
//    learned from the first prompt), and the fill thread copies a finished slot in with
//    UFFDIO_COPY while the card sends the next; a budget returns the oldest
//    chunks with MADV_DONTNEED, so the card's rows never pile up in host RAM. Requests go
//    through the window's doorbell lock (VPU_OFF_LOCK), so they never meet the backend's.
//    The faults are user-mode (CUDA reads pageable memory with ordinary loads), which is
//    what an unprivileged userfaultfd (UFFD_USER_MODE_ONLY) handles.
//
// Environment: TRIDMA_CHUNK (bytes per fetch, default 8 MiB), TRIDMA_AHEAD (chunks read
// ahead past the last fault, default 32), TRIDMA_BUDGET (bytes resident at most, default 512 MiB),
// TRIDMA_STAGE (window offset of the staging area, default 1 GiB), TRIDMA_VERBOSE=1,
// TRIDMA_VERIFY=N (a test: every Nth chunk compared with the GGUF file, which it reads).
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

static struct { uintptr_t a, b; uint64_t off, seq; char path[4096]; } gguf[MAX_GGUF];
static uint64_t gguf_seq;
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
            if (!gguf[i].b) {
                gguf[i].a = (uintptr_t) r; gguf[i].b = (uintptr_t) r + len; gguf[i].off = (uint64_t) off; gguf[i].seq = ++gguf_seq;
                char link[64];
                snprintf(link, sizeof link, "/proc/self/fd/%d", fd);
                const ssize_t k = readlink(link, gguf[i].path, sizeof gguf[i].path - 1);
                gguf[i].path[k > 0 ? k : 0] = 0;
                break;
            }
        pthread_mutex_unlock(&gguf_mu);
    }
    return r;
}
void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off_t off) __attribute__((alias("mmap")));

int munmap(void *addr, size_t len) {
    if (!real_munmap) real_munmap = (int (*)(void *, size_t)) dlsym(RTLD_NEXT, "munmap");
    pthread_mutex_lock(&gguf_mu);
    for (int i = 0; i < MAX_GGUF; i++)
        // only the whole mapping: llama.cpp unmaps fragments of it (unmap_fragment), the first
        // of which can start at its first byte
        if (gguf[i].b && (uintptr_t) addr <= gguf[i].a && (uintptr_t) addr + len >= gguf[i].b) gguf[i].a = gguf[i].b = 0;
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

// ---- 2. card-backed pages, routed in three stages ----
//
// fault thread  reads userfaultfd; a chunk not yet asked for goes on the demand queue and
//               moves the read-ahead cursor there; it never waits for the card.
// fetch thread  owns the card channel: demand chunks first, then read-ahead along the
//               cursor, K_DOWNLOAD after K_DOWNLOAD into NSLOT staging slots of the
//               window, so the card's link never waits for the host's copies.
// fill thread   UFFDIO_COPY from a finished slot into the pages (which wakes the threads
//               blocked on them), the slot back to the fetcher, the budget kept.
//
// The cursor crosses from the end of one range into the range that followed it last time
// (succ[], learned from the demand faults of the first prompt: the graph reads the weights
// in the same order every time), so read-ahead runs from one matrix into the next.

#define NSLOT 4
#define DQ_CAP 1024

struct range {
    uintptr_t a, b;        // the anonymous range, page aligned
    uint64_t id;           // the card's resident tensor
    uint64_t card_off;     // the card byte of a
    uint32_t chunk0;       // this range's first index in chunk_state
    uint32_t n;            // its chunks
};

static struct range rg[MAX_RANGES];              // in registration order, never moved
static int sorted[MAX_RANGES];                   // indices of rg by address
static int succ[MAX_RANGES];                     // the range read after this one, or -1
static int nranges;
static unsigned char chunk_state[MAX_CHUNKS];   // 0 absent, 1 asked for, 2 resident
static uint16_t chunk_rg[MAX_CHUNKS];
static uint32_t nchunks;
static uint32_t fifo[MAX_CHUNKS];               // resident chunks, oldest first
static uint32_t fifo_head, fifo_len;
static uint64_t resident;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t work_cv = PTHREAD_COND_INITIALIZER, fill_cv = PTHREAD_COND_INITIALIZER;
static int uffd = -1;
static unsigned char *win;                      // the card's window (the backend's mapping)
static uint32_t card_threads;
static uint64_t opt_chunk, opt_budget, opt_stage, opt_ahead, slot_bytes;

static uint32_t dq[DQ_CAP];                     // demand chunks, oldest first
static uint32_t dq_head, dq_len;
static int pf_rg = -1;                          // the read-ahead cursor: range, chunk
static uint32_t pf_k, pf_count;                 // and chunks read ahead since the last fault
static int tail_rg = -1;                        // the range the stream is in

static int slot_free[NSLOT], nfree;             // staging slots: free stack
static struct { uint32_t ci; uint64_t delta, len; } ready[NSLOT];
static int ready_slot[NSLOT], ready_head, ready_len;

static uint64_t st_faults, st_demand, st_prefetch, st_bytes, st_evicted, st_dl_ns;
static uint64_t opt_verify, st_vseen, st_vok, st_vbad;          // TRIDMA_VERIFY: a test of the data path
static uint64_t b_bytes, b_t0, b_t1, b_demand, b_prefetch;   // the current burst

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

static int find_range(uintptr_t x) {
    int lo = 0, hi = nranges - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) / 2;
        const struct range *r = &rg[sorted[mid]];
        if (x < r->a) hi = mid - 1;
        else if (x >= r->b) lo = mid + 1;
        else return sorted[mid];
    }
    return -1;
}

// the next chunk to read ahead, or -1 (mu held)
static int64_t next_prefetch(void) {
    for (uint32_t steps = 0; pf_rg >= 0 && pf_count < opt_ahead && steps <= nchunks; steps++) {
        if (pf_k >= rg[pf_rg].n) {
            pf_rg = succ[pf_rg];
            pf_k = 0;
            if (pf_rg >= 0) tail_rg = pf_rg;
            continue;
        }
        const uint32_t ci = rg[pf_rg].chunk0 + pf_k++;
        if (chunk_state[ci] == 0) { pf_count++; return ci; }
    }
    return -1;
}

// the oldest chunks out while over budget (their pages fault again if read again; mu held)
static void evict(void) {
    while (resident > opt_budget && fifo_len) {
        const uint32_t ci = fifo[fifo_head];
        fifo_head = (fifo_head + 1) % MAX_CHUNKS;
        fifo_len--;
        if (chunk_state[ci] != 2) continue;
        const struct range *r = &rg[chunk_rg[ci]];
        const uintptr_t a = r->a + (uint64_t) (ci - r->chunk0) * opt_chunk;
        const uint64_t len = (r->b - a) < opt_chunk ? (r->b - a) : opt_chunk;
        if (!real_madvise) real_madvise = (int (*)(void *, size_t, int)) dlsym(RTLD_NEXT, "madvise");
        real_madvise((void *) a, len, MADV_DONTNEED);
        resident -= len;
        st_evicted++;
        chunk_state[ci] = 0;
    }
}

static void *fetcher(void *arg) {
    (void) arg;
    pthread_mutex_lock(&mu);
    for (;;) {
        int64_t ci = -1;
        int pre = 0;
        while (ci < 0) {
            if (nfree) {
                while (dq_len && ci < 0) {
                    const uint32_t c = dq[dq_head];
                    dq_head = (dq_head + 1) % DQ_CAP;
                    dq_len--;
                    if (chunk_state[c] == 0) ci = c;
                }
                if (ci < 0 && (ci = next_prefetch()) >= 0) pre = 1;
            }
            if (ci >= 0) break;
            struct timespec t;
            clock_gettime(CLOCK_REALTIME, &t);
            t.tv_nsec += 200000000L;
            if (t.tv_nsec >= 1000000000L) { t.tv_sec++; t.tv_nsec -= 1000000000L; }
            if (pthread_cond_timedwait(&work_cv, &mu, &t) && b_bytes && vb() && !ready_len && nfree == NSLOT) {
                char v[96] = "";
                if (opt_verify)
                    snprintf(v, sizeof v, "; verified %llu chunks equal to the GGUF, %llu differ",
                             (unsigned long long) st_vok, (unsigned long long) st_vbad);
                fprintf(stderr, "tridma: burst %.3f GB in %.3f s (%.2f GB/s): %llu chunks on demand, %llu read ahead; "
                                "%llu evicted, %.0f MiB resident%s\n", b_bytes / 1e9, (b_t1 - b_t0) / 1e9,
                        b_bytes / ((b_t1 - b_t0) * 1e-9) / 1e9, (unsigned long long) b_demand,
                        (unsigned long long) b_prefetch, (unsigned long long) st_evicted, resident / 1048576.0, v);
                b_bytes = b_demand = b_prefetch = 0;
            }
        }
        chunk_state[ci] = 1;
        const int s = slot_free[--nfree];
        const struct range *r = &rg[chunk_rg[ci]];
        const uintptr_t a = r->a + (uint64_t) (ci - r->chunk0) * opt_chunk;
        const uint64_t len = (r->b - a) < opt_chunk ? (r->b - a) : opt_chunk;
        const uint64_t coff = r->card_off + (a - r->a);
        const uint64_t off0 = coff & ~(PG - 1), delta = coff - off0;
        const uint64_t bytes = (delta + len + PG - 1) & ~(PG - 1);
        const uint64_t id = r->id;
        if (pre) st_prefetch++, b_prefetch++; else st_demand++, b_demand++;
        pthread_mutex_unlock(&mu);

        const uint64_t t0 = now_ns();
        const int st = card_download(id, off0, bytes, opt_stage + (uint64_t) s * slot_bytes);
        const uint64_t t1 = now_ns();
        if (st) {
            fprintf(stderr, "tridma: K_DOWNLOAD of id %llu, %llu bytes at %llu failed: status %d; aborting rather than "
                            "giving the GPU wrong weights\n", (unsigned long long) id, (unsigned long long) bytes,
                    (unsigned long long) off0, st);
            abort();
        }

        pthread_mutex_lock(&mu);
        st_dl_ns += t1 - t0;
        if (!b_bytes) b_t0 = t0;
        b_t1 = t1;
        b_bytes += len;
        ready[s].ci = (uint32_t) ci;
        ready[s].delta = delta;
        ready[s].len = len;
        ready_slot[(ready_head + ready_len) % NSLOT] = s;
        ready_len++;
        pthread_cond_signal(&fill_cv);
    }
    return NULL;
}

// TRIDMA_VERIFY=N: every Nth chunk filled is compared, in the pages themselves, with the GGUF
// file at the same offset (read for the test only and dropped from the page cache after).
static void verify_chunk(uintptr_t a, uint64_t len) {
    static int vfd[MAX_GGUF] = {[0 ... MAX_GGUF - 1] = -1};
    static unsigned char *buf;
    int gi = -1;
    pthread_mutex_lock(&gguf_mu);
    // the newest mapping that covers it: a later file can be mapped into a hole llama.cpp
    // left by unmapping fragments of an earlier one (the draft model inside the main model's span)
    for (int i = 0; i < MAX_GGUF; i++)
        if (gguf[i].b && a >= gguf[i].a && a + len <= gguf[i].b && (gi < 0 || gguf[i].seq > gguf[gi].seq)) gi = i;
    pthread_mutex_unlock(&gguf_mu);
    if (gi < 0) { st_vbad++; fprintf(stderr, "tridma: verify: %#lx in no GGUF mapping\n", (unsigned long) a); return; }
    if (vfd[gi] < 0) vfd[gi] = open(gguf[gi].path, O_RDONLY | O_CLOEXEC);
    if (!buf) buf = malloc(opt_chunk);
    const off_t fo = (off_t) (gguf[gi].off + (a - gguf[gi].a));
    if (vfd[gi] < 0 || !buf || pread(vfd[gi], buf, len, fo) != (ssize_t) len) {
        st_vbad++;
        fprintf(stderr, "tridma: verify: cannot read %s at %lld\n", gguf[gi].path, (long long) fo);
        return;
    }
    posix_fadvise(vfd[gi], fo, (off_t) len, POSIX_FADV_DONTNEED);
    if (memcmp(buf, (const void *) a, len)) {
        if (st_vbad < 3) fprintf(stderr, "tridma: verify: chunk at %#lx (file %lld, %llu bytes) DIFFERS from the GGUF\n",
                                 (unsigned long) a, (long long) fo, (unsigned long long) len);
        st_vbad++;
    } else {
        st_vok++;
    }
}

static void *filler(void *arg) {
    (void) arg;
    pthread_mutex_lock(&mu);
    for (;;) {
        while (!ready_len) pthread_cond_wait(&fill_cv, &mu);
        const int s = ready_slot[ready_head];
        ready_head = (ready_head + 1) % NSLOT;
        ready_len--;
        const uint32_t ci = ready[s].ci;
        const struct range *r = &rg[chunk_rg[ci]];
        const uintptr_t a = r->a + (uint64_t) (ci - r->chunk0) * opt_chunk;
        const uint64_t len = ready[s].len;
        const uintptr_t src = (uintptr_t) (win + opt_stage + (uint64_t) s * slot_bytes + ready[s].delta);
        pthread_mutex_unlock(&mu);

        struct uffdio_copy c = {.dst = a, .src = src, .len = len, .mode = 0};
        if (ioctl(uffd, UFFDIO_COPY, &c) && errno != EEXIST) {
            fprintf(stderr, "tridma: UFFDIO_COPY at %#lx: %s\n", (unsigned long) a, strerror(errno));
            abort();
        }
        if (opt_verify && st_vseen++ % opt_verify == 0) verify_chunk(a, len);

        pthread_mutex_lock(&mu);
        chunk_state[ci] = 2;
        fifo[(fifo_head + fifo_len) % MAX_CHUNKS] = ci;
        fifo_len++;
        resident += len;
        st_bytes += len;
        evict();
        slot_free[nfree++] = s;
        pthread_cond_signal(&work_cv);
        if (vb() && (st_demand + st_prefetch) % 512 == 0)
            fprintf(stderr, "tridma: %llu faults, %llu chunks on demand, %llu read ahead (%.2f GB, card %.2f GB/s), "
                            "%llu evicted, %.0f MiB resident\n", (unsigned long long) st_faults,
                    (unsigned long long) st_demand, (unsigned long long) st_prefetch, st_bytes / 1e9,
                    st_dl_ns ? st_bytes / (st_dl_ns * 1e-9) / 1e9 : 0, (unsigned long long) st_evicted,
                    resident / 1048576.0);
    }
    return NULL;
}

static void *faults(void *arg) {
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
        const int j = find_range(x);
        if (j < 0) {
            pthread_mutex_unlock(&mu);
            fprintf(stderr, "tridma: a fault at %#lx outside every range\n", (unsigned long) x);
            abort();
        }
        const uint32_t k = (uint32_t) ((x - rg[j].a) / opt_chunk), ci = rg[j].chunk0 + k;
        if (chunk_state[ci] == 2) {           // in already: a racing fault, wake it
            struct uffdio_range w = {.start = x & ~(PG - 1), .len = PG};
            ioctl(uffd, UFFDIO_WAKE, &w);
        } else if (chunk_state[ci] == 0) {    // a miss: first in line, read ahead from here
            if (k == 0 && tail_rg >= 0 && tail_rg != j && succ[tail_rg] < 0) succ[tail_rg] = j;
            tail_rg = j;
            if (dq_len < DQ_CAP) { dq[(dq_head + dq_len) % DQ_CAP] = ci; dq_len++; }
            pf_rg = j;
            pf_k = k + 1;
            pf_count = 0;
        } else {                              // asked for already: the reader caught up
            pf_count = 0;
        }
        pthread_cond_signal(&work_cv);
        pthread_mutex_unlock(&mu);
    }
    return NULL;
}

static int router_init(unsigned char *window, uint32_t threads) {
    if (uffd >= 0) return 0;
    opt_chunk = env_u64("TRIDMA_CHUNK", 8ull << 20) & ~(PG - 1);
    opt_ahead = env_u64("TRIDMA_AHEAD", 32);
    opt_budget = env_u64("TRIDMA_BUDGET", 512ull << 20);
    opt_stage = env_u64("TRIDMA_STAGE", 1ull << 30);
    opt_verify = env_u64("TRIDMA_VERIFY", 0);
    if (!opt_chunk || opt_stage % PG) return -1;
    if (opt_budget < (opt_ahead + NSLOT + 2) * opt_chunk) {
        fprintf(stderr, "tridma: TRIDMA_BUDGET under (TRIDMA_AHEAD + %d) chunks would evict what was read ahead; "
                        "raising it\n", NSLOT + 2);
        opt_budget = (opt_ahead + NSLOT + 2) * opt_chunk;
    }
    slot_bytes = opt_chunk + 2 * PG;
    for (int i = 0; i < MAX_RANGES; i++) succ[i] = -1;
    for (int i = 0; i < NSLOT; i++) slot_free[nfree++] = NSLOT - 1 - i;
    uffd = (int) syscall(SYS_userfaultfd, O_CLOEXEC | UFFD_USER_MODE_ONLY);
    if (uffd < 0) { fprintf(stderr, "tridma: userfaultfd: %s\n", strerror(errno)); return -1; }
    struct uffdio_api api = {.api = UFFD_API, .features = 0};
    if (ioctl(uffd, UFFDIO_API, &api)) { fprintf(stderr, "tridma: UFFDIO_API: %s\n", strerror(errno)); return -1; }
    win = window;
    card_threads = threads;
    void *(*fn[3])(void *) = {faults, fetcher, filler};
    for (int i = 0; i < 3; i++) {
        pthread_t t;
        if (pthread_create(&t, NULL, fn[i], NULL)) return -1;
        pthread_detach(t);
    }
    fprintf(stderr, "tridma: card-backed pages on: chunk %llu KiB, %llu read ahead, %d staging slots, budget %llu MiB, "
                    "staging at %llu MiB of the window\n", (unsigned long long) (opt_chunk >> 10),
            (unsigned long long) opt_ahead, NSLOT, (unsigned long long) (opt_budget >> 20),
            (unsigned long long) (opt_stage >> 20));
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
                    const int idx = nranges++;
                    rg[idx] = (struct range){a, b, id, a_off + (a - (uintptr_t) addr), nchunks, (uint32_t) n};
                    for (uint32_t c = 0; c < n; c++) chunk_rg[nchunks + c] = (uint16_t) idx;
                    nchunks += (uint32_t) n;
                    int i = idx;   // the address index, kept sorted
                    while (i > 0 && rg[sorted[i - 1]].a > a) { sorted[i] = sorted[i - 1]; i--; }
                    sorted[i] = idx;
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
