#!/usr/bin/env bash
# Serve Gemma 4 E2B locally under the alias the harness expects, with Gemma 4 tool-call and
# reasoning parsers as on Kaggle. Uses the research environment's vLLM 0.30; the harness
# environment does not include vLLM (see setup_harness.sh). Local runs check plumbing; Kaggle
# runs use Kaggle's own serving stack. Run inside WSL from the repo root:
#     bash kaggle_swe/serve_local.sh        (listens on http://localhost:8000/v1)
set -eo pipefail
source scripts/env.sh
exec vllm serve google/gemma-4-E2B-it \
    --served-model-name gemma-4-e2b-it \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
    --default-chat-template-kwargs '{"enable_thinking": true}'
