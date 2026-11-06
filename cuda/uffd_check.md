# uffd_check.cu

The premise of the DMA router ([`tools/tridma_router.c`](../tools/tridma_router.md))
tested on its own: llama.cpp's CUDA prefill copies weights with pageable
host-to-device copies, and the router serves the card's rows by userfaultfd. On a
host where `vm.unprivileged_userfaultfd` is 0, an unprivileged process only gets
`UFFD_USER_MODE_ONLY`, which handles faults raised by user-mode loads; a copy that
pinned the pages through the kernel would fail instead.

The program maps 64 MiB anonymous, registers it with a user-mode-only userfaultfd
(missing mode), and runs a handler thread that fills each faulting page with its
own word indices (`UFFDIO_COPY`). Then one `cudaMemcpy` host to device of the whole
range, a copy back, and every word compared.

Measured on the RTX 3090 Ti (driver of 2026-11, CUDA 13.4): no error, 16,384 faults
handled (one per 4 KiB page), 0 of 16,777,216 words wrong. CUDA's pageable copy
reads the source with ordinary loads, so the unprivileged kind is enough.

Build: `nvcc -O2 -arch=sm_86 -o uffd_check uffd_check.cu -lpthread`.
