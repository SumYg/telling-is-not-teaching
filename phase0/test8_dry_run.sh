#!/usr/bin/env bash
# Dry run of test 8 on tiny sizes (plumbing only; results discarded). Each run writes to a new folder; nothing is
# deleted. Run inside WSL from the repo root.
set -eo pipefail
source scripts/env.sh
R=runs/test8_dry/$(date +%Y%m%d_%H%M%S)
A="--root $R --n-experience 4 --n-eval 4 --samples 1"
T="python phase0/test8_union.py"
$T train $A --epochs 1 --accum 2
for F in survo course_schedule; do
    $T evaluate --eval-family $F $A
done
$T report --forgive $A
echo "DRY RUN OK: $R"
