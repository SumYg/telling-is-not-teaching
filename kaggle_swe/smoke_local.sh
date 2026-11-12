#!/usr/bin/env bash
# Local smoke test: the sample agent on the 3 debug tasks with Gemma 4 E2B, no Docker.
# Starts the local model server, waits until it answers, runs the tasks, then stops it.
# Run inside WSL from the repo root:  bash kaggle_swe/smoke_local.sh
set -eo pipefail
source kaggle_swe/env.sh
mkdir -p runs/swe logs

bash kaggle_swe/serve_local.sh > logs/serve_local.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT
for i in $(seq 1 120); do  # up to 10 minutes for load and CUDA-graph capture
    curl -sf http://localhost:8000/v1/models > /dev/null && break
    kill -0 $SERVER 2>/dev/null || { echo "model server exited; see logs/serve_local.log"; exit 1; }
    sleep 5
done
echo "model server ready"

SUB=runs/swe/submissions/base_e2b
python kaggle_swe/make_submission.py "$SUB" --model gemma-4-e2b-it
swegemma eval \
    --tasks "$GEMMA4_DATA/tasks.jsonl" \
    --snapshots-dir "$GEMMA4_DATA/snapshots" \
    --submission-dir "$SUB" \
    --results-dir runs/swe/smoke_e2b \
    --sandbox subprocess \
    --task-ids fastapi_14492 rich_3278 requests_6589 \
    --max-time-minutes 6 --max-tool-calls 50 --max-turns 50 \
    --concurrency 1 --display quiet
cat runs/swe/smoke_e2b/summary.json
