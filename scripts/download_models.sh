#!/usr/bin/env bash
# Downloads the model weights into the Hugging Face cache inside WSL (~/.cache/huggingface).
# Run inside WSL after setup_env.sh:  bash scripts/download_models.sh
set -euo pipefail

VENV="$HOME/venvs/research"
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
HF="$VENV/bin/hf"
[ -x "$HF" ] || HF="$VENV/bin/huggingface-cli"

"$HF" download Qwen/Qwen3-1.7B      # learner, bf16, about 4 GB
"$HF" download Qwen/Qwen3-8B-AWQ    # static comparison, 4-bit, about 6 GB
echo "downloads done"
