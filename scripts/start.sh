#!/usr/bin/env bash
# start.sh WORK_DIR MODEL_DIR: the inverted layout, in the order a 32 GB host needs.
#
#   GPU   the last 32 layers and output.weight, every prompt of more than 31 tokens
#   card  the first 32 layers (as many rows as 13.1 GB holds: 93.8 % with Qwen3.8-27B Q8_0),
#         generation and prompts of up to 31 tokens
#   host  the KV cache and the recurrent state, the MTP draft model and its KV cache,
#         token_embd, the 6.2 % of the card layers' rows the card has no room for
#
# WORK_DIR is a llama-phi work directory (its scripts/setup.sh, with the patch that has
# K_DOWNLOAD, the doorbell lock and back_or_drop) whose card worker runs with a
# 6437-page pool (llama-phi scripts/worker.sh, PHI_VPU_HUGEPAGES=6437), and where
# scripts/build.sh of this repository has built runs/serve/pin, runs/serve/libtridma.so
# and runs/cuda/vpu_dma. MODEL_DIR holds Qwen3.8-27B-Q8_0.gguf and
# MTP/mtp-Qwen3.8-27B-Q4_0.gguf (unsloth/Qwen3.8-27B-GGUF).
#
# Steps: 1. pin the host's share of layers 0-31 and token_embd; 2. pin the draft file
# whole; 3. a memory guard that kills the server below MEMGUARD_MIB available (1024);
# 4. the server. CTX (131072) sets the context; SHARE (0.937) the pinned split, one tenth
# of a percent under the backend's "keeps X%" line. scripts/stop.sh undoes all of it.
set -euo pipefail
work=${1:?usage: start.sh WORK_DIR MODEL_DIR}
models=${2:?usage: start.sh WORK_DIR MODEL_DIR}
here=$(cd "$(dirname "$0")" && pwd)
runs=$work/runs/serve
model=$models/Qwen3.8-27B-Q8_0.gguf
mtp=$models/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
ctx=${CTX:-131072}   # 204800 puts 12.8 GB of KV cache in host RAM: on a 32 GB host the load dipped under 1 GiB available
share=${SHARE:-0.937}
guard_mib=${MEMGUARD_MIB:-1024}
mkdir -p "$runs"
for f in "$runs/pin" "$runs/libtridma.so" "$work/runs/cuda/vpu_dma" "$model" "$mtp"; do
    [ -e "$f" ] || { echo "start.sh: $f missing (scripts/build.sh, the model download)" >&2; exit 1; }
done

avail() { awk '/^MemAvailable/ {print int($2 / 1024)}' /proc/meminfo; }
wait_pin() {   # log pid
    for _ in $(seq 1 300); do
        grep -q 'holding' "$1" && { tail -1 "$1"; return 0; }
        kill -0 "$2" 2>/dev/null || { echo "start.sh: pin exited, see $1" >&2; return 1; }
        sleep 1
    done
    echo "start.sh: pin not holding after 300 s" >&2; return 1
}

echo "start.sh: $(avail) MiB available before the pins"
cd "$runs"   # the pins run from here; $! must be the pin itself, so no subshell around it
PIN_LAYERS=0-31 nohup ./pin "$model" "$share" > pin.log 2>&1 &
echo $! > "$runs/pin.pid"
wait_pin "$runs/pin.log" "$(cat "$runs/pin.pid")"
nohup ./pin "$mtp" 0 > pin-mtp.log 2>&1 &
echo $! > "$runs/pin-mtp.pid"
wait_pin "$runs/pin-mtp.log" "$(cat "$runs/pin-mtp.pid")"
echo "start.sh: $(avail) MiB available after the pins"

# the guard: the host has little margin with the KV cache in its RAM
(
    while true; do
        a=$(avail)
        echo "$(date +%T) avail $a MiB" >> "$runs/memguard.log"
        if [ "$a" -lt "$guard_mib" ]; then
            p=$(cat "$runs/server.pid" 2>/dev/null || true)
            if [ -n "$p" ] && [ "$(cat "/proc/$p/comm" 2>/dev/null)" = llama-server ]; then
                kill -9 "$p" && echo "$(date +%T) GUARD: killed the server at $a MiB" >> "$runs/memguard.log"
            fi
            break
        fi
        sleep 1
    done
) > /dev/null 2>&1 &
echo $! > "$runs/memguard.pid"

LD_PRELOAD=$runs/libtridma.so PHI_GGML_ALL_ROWS=1 TG_ONLY=31 PHI_VPU_HUGEPAGES=6437 \
CARD_BYTES=13100000000 SERVE_ALIAS=qwen3.8-27b-q8_0 SERVE_CTX=$ctx \
SERVE_PLACE='-ngl 32 -ot ^output\.weight$=CUDA0 -nkvo' \
SERVE_ARGS="-md $mtp --spec-type draft-mtp -ngld 0" \
    bash "$here/serve.sh" "$work" "$model" start
echo "start.sh: $(avail) MiB available with the server up"
