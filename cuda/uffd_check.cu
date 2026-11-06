// uffd_check.cu: does a CUDA pageable host-to-device copy work from a user-mode-only
// userfaultfd range (the unprivileged kind), with every page filled by a handler thread on
// fault? The premise of tools/tridma_router.c. See uffd_check.md.
#include <cuda_runtime.h>
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
#include <unistd.h>
static int uffd; static char *src; static size_t PG = 4096; static long faults;
static void *handler(void *) {
    for (;;) {
        pollfd p = {uffd, POLLIN, 0};
        if (poll(&p, 1, 2000) <= 0) return NULL;
        uffd_msg m; if (read(uffd, &m, sizeof m) != sizeof m) continue;
        if (m.event != UFFD_EVENT_PAGEFAULT) continue;
        uintptr_t a = m.arg.pagefault.address & ~(PG - 1);
        uint32_t *page = (uint32_t *) src;
        for (size_t i = 0; i < PG / 4; i++) page[i] = (uint32_t) (a / 4 + i);   // value = word index
        uffdio_copy c = {a, (uintptr_t) src, PG, 0, 0};
        if (ioctl(uffd, UFFDIO_COPY, &c)) perror("UFFDIO_COPY");
        faults++;
    }
}
int main() {
    uffd = (int) syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY);
    if (uffd < 0) { perror("userfaultfd"); return 1; }
    uffdio_api api = {UFFD_API, 0, 0};
    if (ioctl(uffd, UFFDIO_API, &api)) { perror("UFFDIO_API"); return 1; }
    const size_t len = 64ull << 20;
    char *r = (char *) mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    madvise(r, len, MADV_NOHUGEPAGE);
    uffdio_register reg = {{(uintptr_t) r, len}, UFFDIO_REGISTER_MODE_MISSING, 0};
    if (ioctl(uffd, UFFDIO_REGISTER, &reg)) { perror("UFFDIO_REGISTER"); return 1; }
    src = (char *) aligned_alloc(PG, PG);
    pthread_t t; pthread_create(&t, NULL, handler, NULL);
    void *d; cudaMalloc(&d, len);
    cudaError_t e = cudaMemcpy(d, r, len, cudaMemcpyHostToDevice);
    printf("cudaMemcpy H2D from a user-mode-only userfaultfd range: %s, %ld faults handled\n", cudaGetErrorString(e), faults);
    uint32_t *h = (uint32_t *) malloc(len);
    cudaMemcpy(h, d, len, cudaMemcpyDeviceToHost);
    size_t bad = 0; for (size_t i = 0; i < len / 4; i++) if (h[i] != (uint32_t) (((uintptr_t) r) / 4 + i)) bad++;
    printf("GPU copy checked: %zu of %zu words wrong\n", bad, len / 4);
    return e != cudaSuccess || bad;
}
