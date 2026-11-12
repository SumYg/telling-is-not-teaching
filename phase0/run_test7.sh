#!/usr/bin/env bash
# Test 7 (phase0/TEST7_FAMILIES.md). Run inside WSL from the repo root:
#   bash phase0/run_test7.sh > logs/test7.log 2>&1
# Every stage skips outputs that already exist, so the script can be restarted after an interruption.
set -eo pipefail
source scripts/env.sh
T="python phase0/test7_families.py"
for F in survo quantum_lock; do
    $T probe --family $F
done
for F in self_reference list_functions jugs course_schedule word_ladder mini_sudoku; do
    $T practice --family $F
    $T base --family $F
    $T probe --family $F
    $T train --family $F
    $T evaluate --family $F
    $T evaluate --family $F --eval-family survo
    echo "FAMILY DONE: $F $(date)"
done
$T report
echo "TEST 7 DONE"
