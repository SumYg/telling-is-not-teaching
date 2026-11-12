#!/usr/bin/env bash
# Test 6 (phase0/TEST6_SELF_TRAINING.md) on one family, survo by default. Run inside WSL from the repo root:
#   bash phase0/run_test6.sh [survo|quantum_lock] > logs/test6[_quantum_lock].log 2>&1
set -eo pipefail
source scripts/env.sh
F=${1:-survo}
O=$([ "$F" = survo ] && echo quantum_lock || echo survo)  # the other family, for forgetting
T="python phase0/test6_iterate.py"
$T train --family $F --round 1 --draw s2
$T evaluate --family $F --round 1 --draw s2
$T practice --family $F --round 2
$T train --family $F --round 2 --draw s1
$T evaluate --family $F --round 2 --draw s1
$T practice --family $F --round 3
for d in s1 s2; do
    $T train --family $F --round 3 --draw $d
    $T evaluate --family $F --round 3 --draw $d
done
$T evaluate --family $F --round 3 --draw s1 --eval-family $O
$T report --family $F
echo "TEST 6 DONE"
