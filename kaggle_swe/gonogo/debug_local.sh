#!/usr/bin/env bash
# Local dry run of the combined go/no-go kernel with Gemma 4 E2B on the debug tasks, no Kaggle
# quota. Run inside WSL from the repo root:  bash kaggle_swe/gonogo/debug_local.sh
set -eo pipefail
source kaggle_swe/env.sh
mkdir -p runs/swe logs
OUT=runs/swe/gonogo_local/full

bash kaggle_swe/serve_local.sh > logs/serve_local.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT
for i in $(seq 1 120); do  # up to 10 minutes for load and CUDA-graph capture
    curl -sf http://localhost:8000/v1/models > /dev/null && break
    kill -0 $SERVER 2>/dev/null || { echo "model server exited; see logs/serve_local.log"; exit 1; }
    sleep 5
done
echo "model server ready"

python kaggle_swe/gonogo/gonogo.py --local --stage full --out "$OUT" \
    --tasks fastapi_14492,requests_6589 --eval-tasks rich_3278 --minutes 2
cat "$OUT/experience/summary.json" "$OUT/evaluate/summary.json"
python kaggle_swe/gonogo/report.py "$OUT/evaluate"
