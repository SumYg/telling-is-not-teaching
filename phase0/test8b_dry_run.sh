#!/usr/bin/env bash
# Dry run of test 8b on tiny sizes (plumbing only; results discarded). Each run writes to a new folder; nothing is
# deleted. Run inside WSL from the repo root.
set -eo pipefail
source scripts/env.sh
R=runs/test8b_dry/$(date +%Y%m%d_%H%M%S)
A="--root $R --n-experience 4 --n-eval 4 --samples 1"
T="python phase0/test8b_gated.py"
$T train $A --epochs 1 --accum 2
for F in survo quantum_lock; do
    $T evaluate --eval-family $F $A
done
$T report --forgive $A
echo "DRY RUN OK: $R"
