#!/usr/bin/env bash
# Dry run of test 7 on tiny sizes (plumbing only; results discarded). Each run writes to a new folder; nothing is
# deleted. Run inside WSL from the repo root.
set -eo pipefail
source scripts/env.sh
R=runs/test7_dry/$(date +%Y%m%d_%H%M%S)
A="--root $R --n-experience 8 --per-problem 4 --n-eval 4 --samples 1"
S="--epochs 1 --accum 2 --min-examples 1 --min-probe 1 --probe-scored 3 --probe-limit 4"
T="python phase0/test7_families.py"
$T probe --family survo $A $S
$T practice --family course_schedule $A
$T base --family course_schedule $A
$T probe --family course_schedule $A $S
$T train --family course_schedule $A $S
$T evaluate --family course_schedule $A
$T evaluate --family course_schedule --eval-family survo $A
$T report $A
echo "DRY RUN OK: $R"
