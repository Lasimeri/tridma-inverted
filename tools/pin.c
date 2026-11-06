// pin: the residency half of the DMA router. For a GGUF model served with libggml_phi.so
// under PHI_GGML_OFFLOAD=1 (the card keeps the last rows of every matrix, the host the
// first), keep the host's bytes resident in RAM and the card's out of it:
//   - every matrix's host rows and every tensor the card never takes: mlock'd through
//     this process's own read-only shared mapping (the page cache is shared, so the
//     server's mapping sees the same pinned pages and never faults to the disk)
//   - every matrix's card rows: posix_fadvise(DONTNEED), out of the page cache
// The split is the backend's own (backend.S card_rows, one card): rows = round(m * f),
// capped so one slice stays within A_MAX (512 MiB), host = (m - rows) & ~63, card the
// rest, from the top. Stays in the foreground holding the locks until SIGINT or SIGTERM.
// See pin.md.
//
// usage: pin [-n] MODEL.gguf SHARE      (SHARE: the card's fraction, the backend's "keeps
//                                        X% of every dense matrix's rows", e.g. 0.469)
//   -n   plan only: print the split and the byte counts, lock nothing
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define A_MAX (512ull << 20)

static const unsigned char *g_p, *g_end;
static int bad;

static uint64_t rd(int n) {
    uint64_t v = 0;
    if (g_p + n > g_end) { bad = 1; return 0; }
    memcpy(&v, g_p, (size_t) n);
    g_p += n;
    return v;
}
static void skip_str(void) { uint64_t n = rd(8); if (g_p + n > g_end) bad = 1; else g_p += n; }

static int scalar_size(uint32_t t) {   // GGUF value types 0..12
    static const int sz[13] = {1, 1, 2, 2, 4, 4, 4, 1, -1, -1, 8, 8, 8};
    return t < 13 ? sz[t] : -1;
}
static void skip_val(uint32_t t) {
    if (t == 8) { skip_str(); return; }
    if (t == 9) {
        uint32_t et = (uint32_t) rd(4);
        uint64_t n = rd(8);
        for (uint64_t i = 0; i < n && !bad; i++) skip_val(et);
        return;
    }
    int s = scalar_size(t);
    if (s < 0) { bad = 1; return; }
    g_p += s;
}

// ggml type: bytes per block and elements per block (ggml.c type_traits)
static int type_size(uint32_t t, uint64_t *bs, uint64_t *be) {
    switch (t) {
    case 0: *bs = 4; *be = 1; return 1;      // F32
    case 1: *bs = 2; *be = 1; return 1;      // F16
    case 2: *bs = 18; *be = 32; return 1;    // Q4_0
    case 3: *bs = 20; *be = 32; return 1;    // Q4_1
    case 8: *bs = 34; *be = 32; return 1;    // Q8_0
    case 10: *bs = 84; *be = 256; return 1;  // Q2_K
    case 11: *bs = 110; *be = 256; return 1; // Q3_K
    case 12: *bs = 144; *be = 256; return 1; // Q4_K
    case 13: *bs = 176; *be = 256; return 1; // Q5_K
    case 14: *bs = 210; *be = 256; return 1; // Q6_K
    case 23: *bs = 136; *be = 256; return 1; // IQ4_XS
    case 30: *bs = 2; *be = 1; return 1;     // BF16
    default: return 0;
    }
}

static volatile sig_atomic_t stop;
static void on_sig(int s) { (void) s; stop = 1; }

int main(int argc, char **argv) {
    int plan_only = 0, ai = 1;
    if (argc > 1 && !strcmp(argv[1], "-n")) { plan_only = 1; ai = 2; }
    if (argc - ai != 2) { fprintf(stderr, "usage: pin [-n] MODEL.gguf SHARE\n"); return 2; }
    char *e;
    const double f = strtod(argv[ai + 1], &e);
    if (*e || !(f >= 0 && f < 1)) { fprintf(stderr, "pin: SHARE is a fraction in [0, 1)\n"); return 2; }
    int fd = open(argv[ai], O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) < 0) { perror(argv[ai]); return 1; }
    const size_t size = (size_t) st.st_size;
    const unsigned char *base = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    g_p = base; g_end = base + size;
    if (rd(4) != 0x46554747u) { fprintf(stderr, "pin: not a GGUF file\n"); return 1; }
    rd(4);                                   // version
    const uint64_t n_tensors = rd(8), n_kv = rd(8);
    uint64_t align = 32;
    for (uint64_t i = 0; i < n_kv && !bad; i++) {
        uint64_t kl = rd(8);
        const unsigned char *k = g_p;
        g_p += kl;
        uint32_t t = (uint32_t) rd(4);
        if (kl == 17 && !memcmp(k, "general.alignment", 17) && t == 4) { align = rd(4); continue; }
        skip_val(t);
    }
    struct tinfo { const unsigned char *name; uint64_t nl, ne0, ne1, nel, off; uint32_t type, nd; } *ti =
        calloc(n_tensors, sizeof *ti);
    for (uint64_t i = 0; i < n_tensors && !bad; i++) {
        ti[i].nl = rd(8); ti[i].name = g_p; g_p += ti[i].nl;
        ti[i].nd = (uint32_t) rd(4);
        ti[i].nel = 1; ti[i].ne1 = 1;
        for (uint32_t d = 0; d < ti[i].nd; d++) {
            uint64_t ne = rd(8);
            if (d == 0) ti[i].ne0 = ne;
            if (d == 1) ti[i].ne1 = ne;
            ti[i].nel *= ne;
        }
        ti[i].type = (uint32_t) rd(4);
        ti[i].off = rd(8);
    }
    if (bad) { fprintf(stderr, "pin: truncated header\n"); return 1; }
    const uint64_t data = ((uint64_t) (g_p - base) + align - 1) / align * align;

    // ranges to lock (host) and to drop (card), page aligned inward for the drop,
    // outward for the lock (a shared edge page stays locked: the host's)
    const long pg = sysconf(_SC_PAGESIZE);
    const char *only = getenv("PIN_ONLY");
    long lay_lo = -1, lay_hi = -1;   // PIN_LAYERS=lo-hi: only those layers are split host/card; the rest are on the GPU
    if (getenv("PIN_LAYERS") && sscanf(getenv("PIN_LAYERS"), "%ld-%ld", &lay_lo, &lay_hi) != 2) { fprintf(stderr, "pin: PIN_LAYERS is lo-hi\n"); return 2; }
    const size_t only_len = only ? strlen(only) : 0;
    uint64_t lock_b = 0, drop_b = 0, split_n = 0, whole_n = 0;
    unsigned char (*act)[1] = NULL;
    (void) act;
    // the header itself stays locked too (the loader reads it)
    // the model's own MTP block (blk.N with nextn tensors): unused by the main model, so dropped
    long mtp = -1;
    for (uint64_t i = 0; i < n_tensors; i++) {
        if (ti[i].nl > 4 && !memcmp(ti[i].name, "blk.", 4) && memmem(ti[i].name, ti[i].nl, "nextn", 5))
            mtp = strtol((const char *) ti[i].name + 4, NULL, 10);
    }
    char mtp_prefix[32] = "";
    if (mtp >= 0) snprintf(mtp_prefix, sizeof mtp_prefix, "blk.%ld.", mtp);
    const size_t mtp_len = strlen(mtp_prefix);
    uint64_t mtp_b = 0;
    struct rg { uint64_t a, b; int lock; } *rgs = calloc(2 * n_tensors + 1, sizeof *rgs);
    size_t nr = 0;
    rgs[nr++] = (struct rg){0, data, 1};
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint64_t bs, be;
        if (!type_size(ti[i].type, &bs, &be)) {
            fprintf(stderr, "pin: %.*s: ggml type %u not in the table\n", (int) ti[i].nl, ti[i].name, ti[i].type);
            return 1;
        }
        const uint64_t a = data + ti[i].off, bytes = ti[i].nel / be * bs;
        if (only_len) {   // PIN_ONLY=prefix: those tensors locked whole, nothing else touched
            if (ti[i].nl >= only_len && !memcmp(ti[i].name, only, only_len)) { rgs[nr++] = (struct rg){a, a + bytes, 1}; lock_b += bytes; whole_n++; }
            continue;
        }
        const int is_embd = ti[i].nl >= 10 && !memcmp(ti[i].name, "token_embd", 10);
        if (f > 0 && mtp_len && ti[i].nl > mtp_len && !memcmp(ti[i].name, mtp_prefix, mtp_len)) {   // share 0 locks a draft model whole
            rgs[nr++] = (struct rg){a, a + bytes, 0};
            drop_b += bytes; mtp_b += bytes;
            continue;
        }
        if (lay_lo >= 0 && ti[i].nd >= 2 && ti[i].ne1 >= 2 && !is_embd) {   // a matrix outside the card layers is on the GPU: out of the page cache
            long lay = -1;
            if (ti[i].nl > 4 && !memcmp(ti[i].name, "blk.", 4)) lay = strtol((const char *) ti[i].name + 4, NULL, 10);
            if (lay < lay_lo || lay > lay_hi) { rgs[nr++] = (struct rg){a, a + bytes, 0}; drop_b += bytes; continue; }
        }
        if (ti[i].nd < 2 || ti[i].ne1 < 2 || is_embd || f == 0) {   // never on the card
            rgs[nr++] = (struct rg){a, a + bytes, 1};
            lock_b += bytes; whole_n++;
            continue;
        }
        const uint64_t m = ti[i].nel / ti[i].ne0, row = ti[i].ne0 / be * bs;
        uint64_t rows = (uint64_t) llround((double) m * f);
        if (row && rows * row > A_MAX) rows = A_MAX / row;
        if (rows > m) rows = m;
        const uint64_t host = (m - rows) & ~63ull;
        rgs[nr++] = (struct rg){a, a + host * row, 1};
        rgs[nr++] = (struct rg){a + host * row, a + bytes, 0};
        lock_b += host * row; drop_b += (m - host) * row; split_n++;
    }
    if (mtp >= 0) printf("the model's MTP block %s: %.2f GB dropped with the card rows\n", mtp_prefix, mtp_b / 1e9);
    printf("%s: %llu tensors (%llu split, %llu whole), share %.4f: host %.2f GB locked, card %.2f GB dropped\n",
           argv[ai], (unsigned long long) n_tensors, (unsigned long long) split_n, (unsigned long long) whole_n, f,
           lock_b / 1e9, drop_b / 1e9);
    if (plan_only) return 0;

    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);
    uint64_t locked = 0, dropped = 0;
    for (size_t r = 0; r < nr; r++) {
        uint64_t a = rgs[r].a, b = rgs[r].b;
        if (b <= a) continue;
        if (rgs[r].lock) {
            a = a / pg * pg;
            b = (b + pg - 1) / pg * pg;
            if (b > size) b = size;
            if (mlock(base + a, b - a)) { fprintf(stderr, "pin: mlock: %s\n", strerror(errno)); return 1; }
            locked += b - a;
        } else {
            uint64_t a2 = (a + pg - 1) / pg * pg, b2 = b / pg * pg;
            if (b2 > a2 && posix_fadvise(fd, (off_t) a2, (off_t) (b2 - a2), POSIX_FADV_DONTNEED) == 0) dropped += b2 - a2;
        }
    }
    printf("locked %.2f GB resident, dropped %.2f GB of card rows from the page cache; holding (SIGTERM ends)\n",
           locked / 1e9, dropped / 1e9);
    fflush(stdout);
    while (!stop) pause();
    return 0;
}
