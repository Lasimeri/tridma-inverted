#!/usr/bin/env bash
# serve.sh WORK_DIR MODEL.gguf start|stop|status
#
# One llama-server (OpenAI API on 127.0.0.1:SERVE_PORT, default 8095) with the weights in
# host RAM and on the Xeon Phi, and the GPU for prefill and the KV cache:
#   -ngl 99 -ot ".*=CPU"   every layer is the GPU's (its KV cache and attention live in
#                          VRAM), every weight tensor stays in host RAM
#   libggml_phi.so         the card keeps a share of every weight matrix's rows (up to
#                          CARD_BYTES) and multiplies them for single tokens; the host's
#                          threads do the rest
#   CUDA op offload        batches of 32+ tokens (prefill) run their weight multiplies on
#                          the GPU: PHI_GGML_TG_ONLY=31 makes the phi backend decline them
#                          (ggml offers an op to CUDA only when its weight sits on the last,
#                          CPU backend: ggml-backend.cpp ggml_backend_sched_backend_id_from_cur);
#                          the GPU, the weights copied over per batch
# stop also frees every weight the card holds (VPU_K_FREE 0 through repo/cuda vpu_dma -F),
# so nothing stays resident on the card while no server uses it.
set -euo pipefail
work=${1:?usage: serve.sh WORK_DIR MODEL.gguf start|stop|status}
model=${2:?usage: serve.sh WORK_DIR MODEL.gguf start|stop|status}
cmd=${3:-status}
port=${SERVE_PORT:-8095}
ctx=${SERVE_CTX:-262144}
alias=${SERVE_ALIAS:-qwen3.8-35b-a3b-q8_0}
read -r -a extra <<< "${SERVE_ARGS:-}"   # extra llama-server arguments, e.g. a draft model
read -r -a place <<< "${SERVE_PLACE:--ngl 99 -ot .*=CPU}"   # where the weights go (default: every weight on the host side)
runs=$work/runs/serve
bin=$work/llama.cpp/build-native-cuda/bin/llama-server
vpu_dma=${VPU_DMA:-$work/runs/cuda/vpu_dma}
mkdir -p "$runs"

# The card worker takes weights from its 2 MiB huge-page pool (scripts/phi-vpu.sh start sets
# nr_hugepages to PHI_VPU_HUGEPAGES, 5600 = 10.9 GiB here); freed weights go back to the pool,
# not to the card. pool_release returns the pool's free pages to the card (nr_hugepages down
# to the pages still in use: the worker's own buffers); pool_reserve grows it back first.
pages=${PHI_VPU_HUGEPAGES:-5600}
card_sh() { timeout 60 phi -c "${PHI_GGML_CARDS:-0}" run sh -c "$1" < /dev/null; }
pool_reserve() {
    local have
    have=$(card_sh "echo $pages > /proc/sys/vm/nr_hugepages; cat /proc/sys/vm/nr_hugepages")
    echo "serve: card huge-page pool $have of $pages pages"
    [ "$have" = "$pages" ] || echo "serve: the card granted $have of $pages; larger uploads fall back to 4 KiB pages" >&2
}
pool_release() {
    card_sh 'set -- $(awk "/HugePages_Total/ {t=\$2} /HugePages_Free/ {f=\$2} END {print t, f}" /proc/meminfo); echo $(($1 - $2)) > /proc/sys/vm/nr_hugepages; grep -E "HugePages_Total|MemFree" /proc/meminfo' | tr '\n' ' '
    echo
}

server_pid() {   # the pid in the pid file, if it is still a llama-server
    [ -f "$runs/server.pid" ] || return 1
    local p
    p=$(cat "$runs/server.pid")
    [ "$(cat "/proc/$p/comm" 2>/dev/null)" = llama-server ] && echo "$p"
}

case "$cmd" in
    start)
        if p=$(server_pid); then echo "serve: already running (pid $p)"; exit 0; fi
        if ! bash "$work/Intel-Phi-AVX-512-C/scripts/card-users.sh" 0; then
            echo "serve: the card is in use; one program at a time" >&2; exit 1
        fi
        pool_reserve
        log=$runs/server-$(date +%Y%m%d-%H%M%S).log
        env GGML_BACKEND_PATH="$work/Intel-Phi-AVX-512-C/host/asm/out/libggml_phi.so" \
            PHI_GGML_CARDS=${PHI_GGML_CARDS:-0} PHI_GGML_THREADS=120 \
            PHI_GGML_CARD_BYTES=${CARD_BYTES:-10500000000} PHI_GGML_OFFLOAD=1 \
            PHI_GGML_HOST_THREADS=${HOST_THREADS:-14} PHI_GGML_TG_ONLY=${TG_ONLY:-31} \
            nohup "$bin" -m "$model" --alias "$alias" "${place[@]}" \
                -t "${HOST_THREADS:-14}" -c "$ctx" -np 1 -b 2048 -ub 2048 --fit off --jinja \
                --host 127.0.0.1 --port "$port" "${extra[@]}" > "$log" 2>&1 &
        echo $! > "$runs/server.pid"
        ln -sfn "$log" "$runs/server.log"
        for _ in $(seq 1 600); do
            grep -q 'listening on' "$log" && { echo "serve: up on 127.0.0.1:$port (pid $!), log $log"; exit 0; }
            kill -0 $! 2>/dev/null || { echo "serve: the server exited, see $log" >&2; exit 1; }
            sleep 1
        done
        echo "serve: not up after 600 s, see $log" >&2; exit 1 ;;
    stop)
        if p=$(server_pid); then
            kill "$p"
            # llama-server finishes its current batch before it acts on SIGTERM, which can take
            # minutes when the weights are in host RAM; after 10 s it is killed
            for _ in $(seq 1 10); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
            if kill -0 "$p" 2>/dev/null; then kill -9 "$p"; sleep 1; fi
            if kill -0 "$p" 2>/dev/null; then echo "serve: pid $p did not exit" >&2; exit 1; fi
            echo "serve: stopped (pid $p)"
        fi
        if bash "$work/Intel-Phi-AVX-512-C/scripts/card-users.sh" 0; then
            "$vpu_dma" -F
            pool_release
        else
            echo "serve: another program maps the card's window; its weights are left alone" >&2
        fi ;;
    status)
        if p=$(server_pid); then echo "serve: running (pid $p)"; else echo "serve: not running"; fi
        curl -sf "http://127.0.0.1:$port/health" > /dev/null && echo "serve: 127.0.0.1:$port healthy" || echo "serve: 127.0.0.1:$port not answering" ;;
    *)
        echo "serve.sh: start, stop or status" >&2; exit 2 ;;
esac
