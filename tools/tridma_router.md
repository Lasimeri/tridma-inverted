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
   - routes every fault through three threads, each with one job (2026-11-06, router
     v2; v1 did all of it in one thread, the card idle during each copy and the
     first chunk of every matrix a stall):
     - the **fault thread** (`poll` + `read` on the userfaultfd) puts a chunk
       (`TRIDMA_CHUNK`, 8 MiB) not yet asked for at the head of a demand queue and
       moves the read-ahead cursor there; it never waits for the card;
     - the **fetch thread** owns the card channel: demand chunks first, then the
       cursor's chunks, at most `TRIDMA_AHEAD` (32, 256 MiB) past the last fault,
       each a `K_DOWNLOAD` (32) of the card bytes aligned down to a 4 KiB block
       (the worker's rule) into one of four staging slots of the window
       (`TRIDMA_STAGE`, 1 GiB), back to back, so the card's link never waits for
       the host;
     - the **fill thread** `UFFDIO_COPY`s a finished slot at the right byte into
       the pages (which wakes every thread blocked on them) and gives the slot back.
   - carries the read-ahead from the end of one range into the range read after it
     the last time (`succ[]`, learned from the demand faults that open a range: the
     graph reads the weights in the same order on every prompt), so after the first
     prompt the stream runs from one matrix into the next without a stall;
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
| `TRIDMA_AHEAD` | 32 | chunks read ahead past the last fault (the budget is raised to at least `TRIDMA_AHEAD` + 6 chunks) |
| `TRIDMA_BUDGET` | 512 MiB | card-backed bytes resident at most |
| `TRIDMA_STAGE` | 1 GiB | window offset of the staging area (the backend uses at most 768 MiB of the window) |
| `TRIDMA_VERBOSE` | 0 | 1: log the populate changes, every backed range, a summary every 512 chunks, and one line per burst (a prompt's stream: bytes, seconds, GB/s, chunks on demand and read ahead) |
| `TRIDMA_VERIFY` | 0 | N: a test. Every Nth chunk filled is compared, in the pages themselves, with the GGUF file at the same offset (read for it and dropped from the page cache after); the burst line counts equal and differing chunks |

Build: `gcc -O2 -Wall -Wextra -shared -fPIC -I<Intel-Phi-AVX-512-C>/card/vpu -o
libtridma.so tridma_router.c -ldl -lpthread`.

## Measured (2026-11-06, the inverted layout, a 1400-token prompt, `cache_prompt` off)

| | v1 (one thread) | v2, first prompt (learning the order) | v2, after it |
|---|---|---|---|
| card -> host during the stream | 1.44-1.52 GB/s | 2.17-2.38 GB/s | **3.24-3.31 GB/s** (13.161 GB in 4.058 s) |
| chunks on demand / read ahead | | | 2 / 1655 |
| prefill | 111.33 tok/s (12.53 s) | 143.3-144.4 tok/s | **181.32-183.47 tok/s** (7.63 s) |

The card's own rate out (`vpu_dma -D`) is 3.32 GB/s on its PCIe Gen2 x8 link (5 GT/s,
4 GB/s raw), so v2 streams at the link's ceiling; more channels cannot add to it.
`TRIDMA_VERIFY=1` over a learning prompt and a read-ahead prompt: **3,286 of 3,286
chunks equal to the GGUF, 0 differ.** Two traps found on the way: llama.cpp unmaps
fragments of a mapping (`unmap_fragment`), the first of which can start at the
mapping's first byte, so `munmap` now forgets a GGUF mapping only when all of it
goes; and a later file can be mapped into such a hole (the MTP draft model inside
the main model's span), so a lookup by address takes the newest mapping covering it.
