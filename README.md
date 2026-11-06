# tridma-inverted

A 27B model split between an NVIDIA GPU and an Intel Xeon Phi 7120A (Knights
Corner), with the KV cache and the MTP draft model in host RAM, served by an
unmodified [llama.cpp](https://github.com/ggml-org/llama.cpp). Every prompt is
prefilled on the GPU; the weight rows that live on the card reach the GPU from the
card by DMA, through the [tridma](https://github.com/Lasimeri/tridma) router, never
from the disk.

It runs on top of [llama-phi](https://github.com/Lasimeri/llama-phi) (the Xeon Phi
as a matrix co-processor for llama.cpp): its `scripts/setup.sh` builds llama.cpp,
the patched [Intel-Phi-AVX-512-C](https://github.com/Lasimeri/Intel-Phi-AVX-512-C)
backend and card worker, and its `scripts/worker.sh` starts the worker.

```
┌────────────────────────────┐ ┌──────────────────────────────┐ ┌────────────────────────────┐
│ RTX 3090 Ti                │ │ Ryzen 5800X RAM (32 GB)      │ │ Xeon Phi 7120A             │
│ layers 32-63 + output      │ │ KV 131072: 8,192 MiB + RS    │ │ layers 0-31, 93.8 % rows   │
│ 13,283.69 MiB weights      │ │ MTP draft 1.37 GB, locked    │ │ 13.1 GB, pool 6437 pages   │
│ prefill > 31 tokens        │ │ host rows + token_embd 2.17  │ │ generation of its layers   │
│ 16,378 MiB used            │ │ GB locked; 5.9-6.9 GB free   │ │ K_DOWNLOAD 3.32 GB/s out   │
└────────────────────────────┘ └──────────────────────────────┘ └────────────────────────────┘
     ▲ card rows for prefill: page fault -> K_DOWNLOAD -> window -> UFFDIO_COPY (libtridma.so)
 1400-token prompt: prefill 183.47 tok/s (router v2, 3.3 GB/s); generation 4.98-6.95 tok/s    
```

| File | What it is |
|---|---|
| [`scripts/start.sh`](scripts/start.sh) | the configuration: the pins, a memory guard, the server, in that order |
| [`scripts/stop.sh`](scripts/stop.sh) | undoes it: the server (the card's weights freed, its pool returned), the guard, the pins |
| [`scripts/serve.sh`](scripts/serve.sh) | the llama-server launcher both use (`SERVE_PLACE`, `SERVE_ARGS`, `SERVE_CTX`) |
| [`scripts/build.sh`](scripts/build.sh) | builds `pin`, `libtridma.so` and `vpu_dma` into the llama-phi work directory |
| [`tools/tridma_router.c`](tools/tridma_router.md) | the router (`LD_PRELOAD`): no full read of the GGUF at load, the card's rows backed by the card |
| [`tools/pin.c`](tools/pin.md) | the residency daemon: the host's bytes locked in RAM, everything else out of the page cache |
| [`cuda/vpu_dma.cu`](cuda/vpu_dma.md) | drives the card worker from the GPU; `-F` frees the card, `-D` tests the card's DMA out |
| [`cuda/uffd_check.cu`](cuda/uffd_check.md) | the router's premise, measured: CUDA pageable copies served by an unprivileged userfaultfd |
| [`docs/config.md`](docs/config.md) | where everything is, every setting and why, measurements, limits |

## Use

```
git clone https://github.com/Lasimeri/llama-phi && llama-phi/scripts/setup.sh WORK
PHI_VPU_HUGEPAGES=6437 llama-phi/scripts/worker.sh WORK      # the card worker, 13.5 GB pool
scripts/build.sh WORK
scripts/start.sh WORK MODEL_DIR        # Qwen3.8-27B-Q8_0.gguf and MTP/mtp-Qwen3.8-27B-Q4_0.gguf
scripts/stop.sh WORK
```

The model files are from
[unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).
`start.sh` serves an OpenAI-compatible API on 127.0.0.1:8095 with a 131,072-token
context (`CTX` changes it; 204,800 does not fit a 32 GB host with the KV cache in
host RAM). Needs `ulimit -l` unlimited (for the pins) and userfaultfd (the
unprivileged, user-mode-only kind is enough).

## License

MIT: [LICENSE](LICENSE).
