#!/usr/bin/env bash
# Dry run of test 7b on tiny sizes (plumbing only; results discarded). Each run writes to a new folder; nothing is
# deleted. Run inside WSL from the repo root.
set -eo pipefail
source scripts/env.sh
R=runs/test7_check_dry/$(date +%Y%m%d_%H%M%S)
A="--root $R --n-problems 4 --per-problem 2 --cap 3 --min-attempts 1"
T="python phase0/test7_check.py"
for F in survo course_schedule; do
    $T validate --family $F $A
    $T base --family $F $A
done
$T score --family course_schedule $A
$T report $A --min-families 1
echo "DRY RUN OK: $R"
