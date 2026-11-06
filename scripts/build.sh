#!/usr/bin/env bash
# build.sh WORK_DIR: the programs start.sh needs, built into a llama-phi work directory
# against the card worker's own protocol headers (the llama-phi patch with VPU_K_DOWNLOAD
# and VPU_OFF_LOCK): runs/serve/pin, runs/serve/libtridma.so, runs/cuda/vpu_dma (which
# serve.sh stop runs to free the card). NVCC and ARCH (sm_86) as in llama-phi.
set -euo pipefail
work=${1:?usage: build.sh WORK_DIR}
here=$(cd "$(dirname "$0")/.." && pwd)
vpu=$work/Intel-Phi-AVX-512-C/card/vpu
grep -q VPU_K_DOWNLOAD "$vpu/vpu_matmul.h" 2> /dev/null && grep -q VPU_OFF_LOCK "$vpu/vpu_proto.h" \
    || { echo "build.sh: $vpu lacks VPU_K_DOWNLOAD / VPU_OFF_LOCK (the llama-phi patch of 2026-11-06)" >&2; exit 1; }
nvcc=${NVCC:-$(command -v nvcc || echo /opt/cuda/bin/nvcc)}
mkdir -p "$work/runs/serve" "$work/runs/cuda"
gcc -O2 -Wall -Wextra -Wpedantic -std=gnu11 -o "$work/runs/serve/pin" "$here/tools/pin.c" -lm
gcc -O2 -Wall -Wextra -shared -fPIC -I"$vpu" -o "$work/runs/serve/libtridma.so" "$here/tools/tridma_router.c" -ldl -lpthread
"$nvcc" -O3 -arch="${ARCH:-sm_86}" -std=c++17 -Xcompiler -Wall,-Wextra -I"$vpu" -o "$work/runs/cuda/vpu_dma" "$here/cuda/vpu_dma.cu"
for b in serve/pin serve/libtridma.so cuda/vpu_dma; do printf '%-24s %8s bytes\n' "runs/$b" "$(wc -c < "$work/runs/$b")"; done
