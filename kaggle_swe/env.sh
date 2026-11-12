# Load before running the harness, inside WSL from the repo root:  source kaggle_swe/env.sh
# Same CUDA library-path fix as scripts/env.sh, but activates the harness environment
# (swegemma 0.2.7 with vLLM 0.19.1, as on Kaggle's scoring machines).
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"
deactivate
source "$HOME/venvs/swegemma/bin/activate"
export GEMMA4_DATA="$HOME/data/gemma4"
