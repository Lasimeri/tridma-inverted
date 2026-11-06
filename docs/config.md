# The configuration

Qwen3.8-27B Q8_0 (29.05 GB) with its MTP draft head (Q4_0, 1.37 GB), served by
an unmodified llama.cpp on one host with an RTX 3090 Ti (24 GB, PCIe Gen4 x8), an
Intel Xeon Phi 7120A (16 GB, PCIe Gen2 x8) and a Ryzen 7 5800X with 32 GB:

- the weights are split between the GPU and the card; the host keeps only what
  the card has no room for;
- the KV cache, the recurrent state and the MTP draft model are in host RAM;
- every prompt of more than 31 tokens is prefilled on the GPU, the card's weight
  rows reaching it from the card by DMA (the tridma router), never from the disk.

## Where everything is (llama.cpp's own load lines, 131,072-token context)

| What | Where | Size |
|---|---|---|
| layers 32-63 + `output.weight` | GPU | 13,283.69 MiB |
| layers 0-31 | card: 93.8 % of every matrix's rows (its 13.1 GB budget); host: 6.2 %, pinned | 14,407.28 MiB mapped |
| `token_embd`, norms, the host's 6.2 % | host RAM, locked (`pin`, 2.17 GB) | |
| MTP draft model | host RAM, locked (`pin` share 0, 1.37 GB), run on the host (`-ngld 0`) | |
| KV cache + recurrent state | host RAM (`-nkvo`) | 8,192 + 598.5 MiB at 131,072 tokens |
| the card's rows while the GPU prefills | host RAM, at most 512 MiB at a time (`libtridma.so`) | |
| the GPU in all | | 16,378 MiB |

## The settings and why

| Setting | Effect |
|---|---|
| `-ngl 32 -ot ^output\.weight$=CUDA0` | the last 32 layers and the output matrix on the GPU; the anchors matter, `output\.weight` alone also matches every `attn_output.weight` |
| `-nkvo` | the KV cache (and attention) on the host |
| `-md ... --spec-type draft-mtp -ngld 0` | the MTP draft, on the host; without `--spec-type` the draft loads and nothing speculates |
| `PHI_GGML_ALL_ROWS=1` | the card may keep every row of a matrix (otherwise at most an equal split with the host) |
| `CARD_BYTES=13100000000`, `PHI_VPU_HUGEPAGES=6437` | 13.1 GB of rows in a 13.5 GB huge-page pool; 2.2 GB stay for the card's own system |
| `TG_ONLY=31` | multiplies of more than 31 tokens are declined by the card backend, so llama.cpp offloads them to the GPU |
| `LD_PRELOAD=libtridma.so` | the GGUF is not read whole at load (no `MAP_POPULATE`, no `WILLNEED`), and the card's rows are backed by the card: a fault on them fetches the chunk with `K_DOWNLOAD` |
| `PIN_LAYERS=0-31 pin MODEL 0.937` | the host's share of layers 0-31, `token_embd` and the norms locked in RAM; the GPU layers' pages dropped from the page cache |
| memory guard | kills the server if host available memory falls under 1 GiB (it never fired at 131,072 tokens) |

The card side: llama-phi's worker with `K_DOWNLOAD` (kernel 32), the doorbell lock
(`VPU_OFF_LOCK`), and the landing copy on the 120-thread pool; the card booted
with a 2 GiB host window and no swap in host memory.

## Measured (2026-11-06)

| | Value |
|---|---|
| prefill, 1400-token prompt | **183.47 tok/s** (7.63 s) with router v2 once it has learned the order (144.4 tok/s on the first prompt after a start; 108.15 with router v1) |
| generation | 4.98-6.95 tok/s, with the MTP draft accepting 19 of 34 to 22 of 27 |
| the card's DMA, card to host window | 3.32 GB/s (`vpu_dma -D`), the PCIe Gen2 x8 link's ceiling; router v2 streams 3.24-3.31 GB/s during a prefill (v1: 1.44-1.52) |
| disk reads | 22-28.6 MiB on the first request after a start (weights the backend uploads at their first use), 24-176 KiB on the next |
| host iowait | 0 |
| host memory available with the server up | 5.87 to 6.87 GB |

At a 204,800-token context the KV cache takes 12.8 GB of host RAM: the steady
state left about 1.1 GB available, and one load dipped under the guard's 1 GiB
(the guard killed it); hence the 131,072 default.

## Limits

- Generation is bound by the card reading its 12.9 GB share every token (about
  56 GB/s for Q8_0 in the worker's kernels): more layers on the GPU would speed it
  up, at the cost of the card's role.
- Every prompt over 31 tokens makes the card return its 12.9 GB of rows once per
  2048-token batch (about 4 s at 3.3 GB/s), so very short prompts pay that too;
  prompts of up to 31 tokens stay on the card.
- 176 of the model's matrices are never split by the card backend; the pin covers
  them in this layout because layers 0-31 are pinned by layer, not by split.
