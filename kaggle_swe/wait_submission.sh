#!/usr/bin/env bash
# Poll the latest submission of a competition (default: Gemma) until it is scored or fails, then print it.
# Run inside WSL:  bash kaggle_swe/wait_submission.sh [COMPETITION]
source "$(dirname "$0")/../scripts/env.sh"
C=${1:-gemma-4-developer-agent}
while true; do
    latest="$(kaggle competitions submissions "$C" --csv 2>&1 | sed -n 2p)"
    echo "$(date +%H:%M) $latest"
    case "$latest" in
        *COMPLETE*|*complete*|*ERROR*|*error*|*FAIL*|*fail*) break ;;
    esac
    sleep 600
done
kaggle competitions submissions "$C" 2>&1 | head -5
