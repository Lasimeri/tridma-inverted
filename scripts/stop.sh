#!/usr/bin/env bash
# stop.sh WORK_DIR: undoes start.sh: the server (serve.sh stop: SIGTERM, 10 s, SIGKILL, then
# every weight freed on the card and its huge-page pool returned), the memory guard, and
# the two pins (the model's pages then leave host RAM when the kernel wants them).
set -euo pipefail
work=${1:?usage: stop.sh WORK_DIR}
here=$(cd "$(dirname "$0")" && pwd)
runs=$work/runs/serve
PHI_VPU_HUGEPAGES=6437 bash "$here/serve.sh" "$work" x stop || true
for f in memguard pin pin-mtp; do
    [ -f "$runs/$f.pid" ] || continue
    p=$(cat "$runs/$f.pid")
    case "$(cat "/proc/$p/comm" 2>/dev/null || true)" in
        pin*|bash) kill "$p" && echo "stop.sh: $f stopped (pid $p)" ;;
    esac
    mv "$runs/$f.pid" "$runs/$f.pid.old"
done
