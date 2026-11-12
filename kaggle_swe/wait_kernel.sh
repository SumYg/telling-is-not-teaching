#!/usr/bin/env bash
# Wait for a Kaggle kernel to finish, then download its outputs and log.
# Run inside WSL:  bash kaggle_swe/wait_kernel.sh sumyeung/swe-gonogo-experience runs/swe/gonogo/experience
set -o pipefail
source "$(dirname "$0")/../scripts/env.sh"
K=$1
OUT=$2
while true; do
    status="$(kaggle kernels status "$K" 2>&1 | tail -1)"
    echo "$(date +%H:%M) $status"
    case "$status" in
        *COMPLETE*|*complete*|*ERROR*|*error*|*CANCEL*|*cancel*) break ;;
    esac
    sleep ${POLL_SECONDS:-300}
done
mkdir -p "$OUT"
kaggle kernels output "$K" -p "$OUT" 2>&1 | tail -3
ls -la "$OUT"
