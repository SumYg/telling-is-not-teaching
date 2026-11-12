#!/usr/bin/env bash
# Dry run of test 6 on tiny sizes (plumbing only; results discarded). Each run writes to a new folder; nothing is
# deleted. Run inside WSL from the repo root.
set -eo pipefail
source scripts/env.sh
R=runs/test6_dry/$(date +%Y%m%d_%H%M%S)
A="--root $R --n-experience 12 --per-problem 4 --n-eval 4 --samples 1"
S="--epochs 1 --accum 2"
T="python phase0/test6_iterate.py"
$T train --family survo --round 1 --draw s2 $A $S
$T evaluate --family survo --round 1 --draw s2 $A
$T practice --family survo --round 2 $A
$T train --family survo --round 2 --draw s1 $A $S
$T evaluate --family survo --round 2 --draw s1 $A
$T practice --family survo --round 3 $A
$T train --family survo --round 3 --draw s1 $A $S
$T evaluate --family survo --round 3 --draw s1 --eval-family quantum_lock $A
$T report --family survo $A
echo "DRY RUN OK: $R"
