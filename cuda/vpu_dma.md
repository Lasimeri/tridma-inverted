# vpu_dma.cu

The GPU as a client of the Xeon Phi VPU worker: CUDA kernels write the
activations into the worker's host window, post the request, wait for
the reply and read the results back into GPU memory. No host thread
copies data or rings the doorbell; the CPU only launches the kernels.

## How the two devices meet

The worker's whole data path is one host-memory window,
`/dev/shm/phi-hostmem` (4 GiB): the card pulls activations from it and
pushes results into it with its DMA channel 7 through the IOMMU
(Intel-Phi-AVX-512-C `card/vpu/cdma.md`), and reads the control words
through its uncached mapping of the same pages (`card/vpu/vpu_proto.h`).
The program maps the window's first 768 MiB and pins it for the GPU with
`cudaHostRegister(..., cudaHostRegisterMapped)`; `cudaHostGetDevicePointer`
gives the address kernels use. From then on:

- a GPU kernel's stores to that address are PCIe writes by the GPU into
  the window's pages, its loads PCIe reads;
- `cudaMemcpy` to or from it is the GPU's copy engine DMA, with no
  staging buffer, because the pages are pinned;
- the card moves the same pages with its own DMA engine.

So data crosses GPU -> host RAM -> card and back with one DMA on each
side and no CPU copy. A direct GPU-to-card transfer would need peer
mappings of the card's memory and is not attempted.

## The protocol, as the GPU speaks it

The headers are the worker's own (`vpu_proto.h`, `vpu_matmul.h`, included
with `-I .../card/vpu`), so their layout asserts run in this build too.

| Step | Who | What |
|---|---|---|
| 1 | GPU, any threads | activation rows into the window at `OFF_B` (640 MiB), stride `nb_b` (64-byte multiple) |
| 2 | each of those threads | `__threadfence_system()`, then `__syncthreads()` |
| 3 | GPU, one thread | the 128-byte `vpu_matmul` descriptor at 13312, the request's fields at 64 (not its `seq`) |
| 4 | that thread | `__threadfence_system()`, then `st.release.sys` of `seq + 1`: the doorbell, the only word the card polls |
| 5 | card | pulls the rows by DMA, multiplies against the resident weights, pushes `n x m` floats to `OFF_D` (704 MiB) by DMA, writes the reply's fields, then echoes `seq` |
| 6 | GPU, one thread | polls the reply's `seq` with `ld.acquire.sys` (gives up after 10 s, by `%globaltimer`) |
| 7 | GPU, any threads | after `__syncthreads()`, the results with `__ldcv` (fetched from memory, never a stale cache line) |

The same order as the host's Rust client (`host/crates/phi-vpu/src/matmul.rs`
`ring_and_wait`: fields, fence, sequence number, poll). The window ranges
(`OFF_A` 128 MiB for weights to upload, `OFF_B`, `OFF_D`) are those of
`phi-vpu matmul-check`, which is why the program refuses to run while any
other process maps the window (the "one program per card" rule of
`scripts/card-users.sh`; the stack's `phictl` and `phitop` are exempt).

## Kernels

| Kernel | Grid | Does |
|---|---|---|
| `k_call` | 1 thread | one request: descriptor, fields, doorbell, reply into a `call_out` |
| `k_put` | rows across blocks | float4 rows into the window, then a system fence per thread |
| `k_get` | rows across blocks | float4 rows out of the window with `__ldcv` |
| `k_fused` | 1 block, 512 threads | the whole round trip `reps` times inside one launch: put, fence, ring, wait, get; each round trip timed by `%globaltimer` |
| `k_ref` | one thread per result | the fp64 reference `d[c][r] = a[r] . b[c]` |
| `k_fill` | grid-stride | deterministic operands in [-1, 1) from an integer hash |

## The test it runs

`vpu_dma [-m rows] [-k width] [-n columns] [-r reps] [-t card threads]`,
default a 4096 x 2048 float32 matrix (a decode-sized projection), one
column, 2000 round trips, 120 card threads:

1. the weights into `OFF_A`, by the copy engine and again by kernel stores
   (both rates printed), then `VPU_K_UPLOAD` under id `0x4750555F444D41`
   ("GPU_DMA"), posted by `k_call`;
2. one multiply as `k_put`, `k_call`, `k_get`, checked against `k_ref`;
3. the GPU-driven loop (`k_fused`), checked, with latency percentiles and
   the card's own per-request pull / compute / push;
4. the host-orchestrated loop for comparison: copy engine into the window,
   the CPU's doorbell and poll, copy engine back, checked;
5. `VPU_K_FREE` of its own id only (other programs' resident weights stay);
6. the worker's request counter and its channel 7 byte and copy counters
   (`VPU_OFF_STATS`, `VPU_OFF_CDMA_STATS`) before and after, which show the
   card served the requests and moved the data by DMA.

Build: `nvcc -O3 -arch=sm_86 -std=c++17 -Xcompiler -Wall,-Wextra
-I<Intel-Phi-AVX-512-C>/card/vpu -o vpu_dma vpu_dma.cu`.

## The other modes

- `-F`: `VPU_K_FREE` with `a_id` 0 from the GPU: every resident tensor on the card
  dropped (what `scripts/serve.sh stop` runs).
- `-D N` (2026-11-06): after the upload, the resident matrix back to the window
  `N` times with `VPU_K_DOWNLOAD` (kernel 32, the card's DMA), the bytes compared
  with the uploaded matrix, then the window to the GPU by the copy engine `N`
  times. Measured on the 7120A, 32 MiB: identical bytes, 0.81 GB/s card to window
  (41.1 ms per download), 13.40 GB/s window to GPU. This is the transfer
  `tools/tridma_router.c` makes on a page fault.

Builds against the worker headers that have `VPU_K_DOWNLOAD` and `VPU_OFF_LOCK`
(the llama-phi patch).
