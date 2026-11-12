#!/usr/bin/env bash
# Creates the Python environment inside WSL (Linux filesystem, for speed).
# Run inside WSL:  bash scripts/setup_env.sh
set -euo pipefail

VENV="$HOME/venvs/research"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip uv

# One resolve for everything. --torch-backend=auto picks the PyTorch build that
# matches the installed NVIDIA driver (RTX 50-series needs CUDA 12.8 or newer).
"$VENV/bin/uv" pip install --python "$VENV/bin/python" --torch-backend=auto \
    vllm transformers peft accelerate datasets reasoning-gym

source "$REPO/scripts/env.sh"
"$VENV/bin/python" - <<'EOF'
import torch, vllm, transformers, peft, reasoning_gym
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gpu available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
print("vllm", vllm.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
EOF

# Exact versions, for reproducibility.
"$VENV/bin/uv" pip freeze --python "$VENV/bin/python" > "$REPO/requirements.lock"
echo "setup done"
