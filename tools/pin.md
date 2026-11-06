# pin.c

`pin [-n] MODEL.gguf SHARE`: the residency half of the DMA router. For a model
served with `libggml_phi.so` under `PHI_GGML_OFFLOAD=1`, it keeps exactly the
host's bytes in host RAM and the card's bytes out of it, from a separate process.

## How

The page cache is shared between processes, so pages this process `mlock`s
through its own read-only shared mapping stay resident for the server's mapping of
the same file: the server never faults them in from the disk.

1. It parses the GGUF header (version, tensor count, key-value pairs for
   `general.alignment`, tensor infos: name, dimensions, ggml type, offset) and
   finds the data section.
2. For every matrix (2 dimensions or more, at least 2 rows, not `token_embd`) it
   reproduces the backend's split (`backend.S` `card_rows`, one card):
   `rows = round(m * SHARE)`, capped so one slice stays within `A_MAX` (512 MiB),
   `host = (m - rows) & ~63`. The host keeps rows `[0, host)`, the card the rest
   (taken from the top), so the host's part is one contiguous byte range at the
   start of the tensor's data.
3. The host part of every matrix, every tensor the card never takes (norms,
   `token_embd`) and the header are `mlock`ed (pages rounded outward). The card
   part is given `posix_fadvise(DONTNEED)` (pages rounded inward), out of the page
   cache.
4. The model's own MTP block (a `blk.N` with `nextn` tensors, unused by the main
   model) is dropped too.
5. It prints the plan and the bytes, then holds the locks until SIGINT or SIGTERM.

`SHARE` is the backend's own figure ("each keeps X% of every dense matrix's
rows"). Passing a slightly smaller value (0.468 for 46.9 %) pins at most 64 rows
per matrix more than the host keeps: a few card rows pinned by mistake cost a
little RAM, a host row left unpinned would cost disk reads. `SHARE 0` locks the
whole file (used for the MTP draft model, whose host-side tensors are 1.10 GB).

## Measured (Qwen3.8-27B Q8_0, 7120A keeping 46.9 %)

`pin -n MODEL 0.468`: 866 tensors, 545 split, 306 whole, the MTP block `blk.64.`
0.45 GB; host 15.82 GB locked, card 13.21 GB dropped. Run: 6 s, `Mlocked`
14.73 GiB, the model's page cache 14.9 GiB (the host rows only), host
`MemAvailable` 20.70 to 5.55 GB. Needs `ulimit -l` unlimited (it is on this
host).

Known gap: the backend splits only the matrices it multiplies on the card. With
this model 369 of the 545 matrices got card-backed; the other 176 stay whole on
the host, and `pin` locked only their host fraction, so about 131 MiB per prompt
still came from the disk. The fix is to take the backend's actual split set
instead of assuming every matrix is split.

Build: `gcc -O2 -Wall -Wextra -Wpedantic -std=gnu11 -o pin pin.c -lm`.

## Modes for other layouts (2026-11-06)

- `PIN_ONLY=prefix`: only the tensors whose names start with `prefix` are locked,
  whole (and the header); nothing is dropped. Used for `token_embd` when every
  matrix lives on the GPU or the card.
- `PIN_LAYERS=lo-hi`: only the matrices of layers `lo` to `hi` are split host/card
  at `SHARE` (host rows locked, card rows dropped); the matrices of every other
  layer and `output.weight` are dropped from the page cache whole (they are in
  VRAM); `token_embd` and the norms are locked whole. For the layout with the
  last layers on the GPU: `PIN_LAYERS=0-31 pin MODEL 0.937` locked 2.17 GB and
  dropped 26.87 GB, and a request's disk reads fell from 368 MiB to 140 KiB.
- `SHARE 0` locks a whole file, its `blk.N` with `nextn` tensors included: in an
  MTP draft file that block is the draft itself (1.37 GB, 18 tensors), needed in
  host RAM when the draft runs on the host (`-ngld 0`).
