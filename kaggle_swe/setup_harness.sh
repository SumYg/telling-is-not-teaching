#!/usr/bin/env bash
# Local harness environment and debug model. Run inside WSL from the repo root:
#     bash kaggle_swe/setup_harness.sh
#
# The harness (swegemma 0.2.7) goes in its own environment, pinned to the competition
# Wheelhouse versions. The model server is a separate process reached over HTTP; locally it
# runs from the research environment's vLLM 0.30 (see serve_local.sh). The Wheelhouse's vLLM
# 0.19.1 cannot share an environment with the harness: its metrics package needs starlette>=1,
# while google-adk 1.36.1 needs starlette<1. It also fails on Gemma 4 E2B's per-layer config.
set -eo pipefail
VENV="$HOME/venvs/swegemma"
WHEELHOUSE="$HOME/data/gemma4/wheelhouse"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

python3 -m venv --clear "$VENV"
"$VENV/bin/pip" install -q --upgrade pip uv

# Pin every package the Wheelhouse contains to its Wheelhouse version. Without pins, uv takes
# newer releases from PyPI. Packages missing from the Wheelhouse come from PyPI.
CONSTRAINTS="$(mktemp)"
for wheel in "$WHEELHOUSE"/*.whl; do
    base="$(basename "$wheel")"
    name="$(echo "$base" | cut -d- -f1 | tr _ -)"
    [ "$name" = "vllm" ] || echo "$name==$(echo "$base" | cut -d- -f2)"
done > "$CONSTRAINTS"
"$VENV/bin/uv" pip install --python "$VENV/bin/python" --find-links "$WHEELHOUSE" \
    -c "$CONSTRAINTS" swegemma==0.2.7
"$VENV/bin/uv" pip freeze --python "$VENV/bin/python" > "$REPO/kaggle_swe/requirements.lock"

source "$REPO/kaggle_swe/env.sh"
python -c 'import swegemma, transformers; print("swegemma ok | transformers", transformers.__version__)'
swegemma eval --help > /dev/null && echo "swegemma CLI ok"

hf download google/gemma-4-E2B-it > /dev/null
echo "harness setup done"
