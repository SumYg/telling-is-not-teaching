#!/usr/bin/env bash
# Test 8b (phase0/TEST8B_GATED.md). Run inside WSL from the repo root:
#   bash phase0/run_test8b.sh > logs/test8b.log 2>&1
# Every stage skips outputs that already exist, so the script can be restarted after an interruption.
set -eo pipefail
source scripts/env.sh
T="python phase0/test8b_gated.py"
$T train
for F in survo quantum_lock self_reference list_functions jugs course_schedule word_ladder mini_sudoku; do
    $T evaluate --eval-family $F
    echo "EVALUATED: $F $(date)"
done
$T report --forgive
echo "TEST 8B DONE"
