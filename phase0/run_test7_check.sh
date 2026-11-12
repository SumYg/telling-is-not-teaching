#!/usr/bin/env bash
# Test 7b (phase0/TEST7_CHECK.md). Run inside WSL from the repo root:
#   bash phase0/run_test7_check.sh > logs/test7_check.log 2>&1
# Every stage skips outputs that already exist, so the script can be restarted after an interruption.
set -eo pipefail
source scripts/env.sh
T="python phase0/test7_check.py"
FAMILIES="survo quantum_lock self_reference list_functions jugs course_schedule word_ladder mini_sudoku"
for F in $FAMILIES; do
    $T validate --family $F
    $T base --family $F
done
for F in $FAMILIES; do
    $T score --family $F
done
$T report
echo "TEST 7B DONE"
