#!/usr/bin/env bash
# Test 4 end to end (PREREGISTRATION.md). Run inside WSL from the repo root:
#   bash phase0/run_test4.sh > logs/test4.log 2>&1
set -eo pipefail
source scripts/env.sh
for f in survo quantum_lock; do
    for stage in experience rationalize distill; do
        python phase0/test4_weights.py $stage --family $f
    done
    for arm in w2 w1; do
        python phase0/test4_weights.py train --family $f --arm $arm
        python phase0/test4_weights.py evaluate --family $f --arm $arm
    done
done
python phase0/test4_weights.py evaluate --family survo --arm w2 --eval-family quantum_lock
python phase0/test4_weights.py evaluate --family quantum_lock --arm w2 --eval-family survo
python phase0/test4_weights.py report
echo "TEST 4 DONE"
