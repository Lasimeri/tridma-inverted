# tridma_router.c

`libtridma.so`, the host side of the tridma DMA router, loaded into llama-server
with `LD_PRELOAD`. It makes the Xeon Phi's memory, not the disk, the backing store
of the weight rows the card keeps, so llama.cpp's own CUDA prefill can read every
weight while the host holds only its share. The full design and the measurements
are in [docs/router.md](../docs/router.md).

## Two jobs

1. **No populate.** For mappings of `.gguf` files, `MAP_POPULATE` is cleared and
   `MADV_WILLNEED` / `POSIX_MADV_WILLNEED` dropped (see
   [nopopulate.md](nopopulate.md)), so loading never reads the whole file.
2. **Card-backed pages.** `libggml_phi.so` calls `tridma_back(addr, len, id,
   a_off, window, threads)` (`backend.S` `back_or_drop`) right after it uploaded
   a dense matrix's card rows `[addr, addr + len)` as the card's resident tensor
   `id`, card byte `a_off`. The router:
   - puts an anonymous mapping (`MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS`,
     `MADV_NOHUGEPAGE`) over the whole pages of that range and registers it with
     userfaultfd (`UFFDIO_REGISTER_MODE_MISSING`), recording the range (sorted,
     for a binary search) with its card offset;
   - on a fault (its handler thread, `poll` + `read` on the userfaultfd), fetches
     the chunk containing the address (`TRIDMA_CHUNK`, 8 MiB) from the card:
     `K_DOWNLOAD` (32) of the card bytes, aligned down to a 4 KiB block (the
     worker's rule), into the window's staging area (`TRIDMA_STAGE`, 1 GiB), then
     `UFFDIO_COPY` from the staging area at the right byte into the pages, which
     wakes the faulting thread; it also fills the next `TRIDMA_AHEAD` (2) chunks;
   - keeps at most `TRIDMA_BUDGET` (512 MiB) of such pages: the oldest chunks go
     back with `MADV_DONTNEED` and fault again if read again.

A download that fails aborts the process: computing with missing weights would be
worse than stopping.

## The request channel

`K_DOWNLOAD` is issued on the worker's ordinary request line (descriptor at
`VPU_OFF_MATMUL`, request at `VPU_OFF_REQ`, the sequence number written last,
the reply's echoed sequence number polled), under the window's doorbell lock
(`VPU_OFF_LOCK`, 15360): the same test-and-set (`xchg` 1, store 0) the backend's
`doorbell` takes and its `wait_reply` frees, so the router's requests and the
backend's never interleave on the card's single request slot.

## Why userfaultfd works here unprivileged

On this host `vm.unprivileged_userfaultfd` is 0 and `/dev/userfaultfd` is root's,
so the router asks for `UFFD_USER_MODE_ONLY`: it handles faults raised by user-mode
loads only. CUDA's pageable host-to-device copy reads the source with ordinary
loads, so its faults qualify. Measured: a 64 MiB user-mode-only userfaultfd range
filled on fault and copied with `cudaMemcpy` to a 3090 Ti: 16,384 faults handled,
0 of 16,777,216 words wrong on the GPU.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `TRIDMA_CHUNK` | 8 MiB | bytes per fetch |
| `TRIDMA_AHEAD` | 2 | chunks read ahead after a fault |
| `TRIDMA_BUDGET` | 512 MiB | card-backed bytes resident at most |
| `TRIDMA_STAGE` | 1 GiB | window offset of the staging area (the backend uses at most 768 MiB of the window) |
| `TRIDMA_VERBOSE` | 0 | 1: log the populate changes, every backed range, and a fetch summary every 256 chunks |

Build: `gcc -O2 -Wall -Wextra -shared -fPIC -I<Intel-Phi-AVX-512-C>/card/vpu -o
libtridma.so tridma_router.c -ldl -lpthread`.
