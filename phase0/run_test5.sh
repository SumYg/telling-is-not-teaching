#!/usr/bin/env bash
# Test 5 end to end (PREREGISTRATION.md). Run inside WSL from the repo root:
#   bash phase0/run_test5.sh > logs/test5.log 2>&1
set -eo pipefail
source scripts/env.sh
for f in survo quantum_lock; do
    other=$([ "$f" = survo ] && echo quantum_lock || echo survo)
    python phase0/test5_gate.py experience --family $f
    for arm in all gate random; do
        python phase0/test5_gate.py train --family $f --arm $arm
        python phase0/test5_gate.py evaluate --family $f --arm $arm
        python phase0/test5_gate.py evaluate --family $f --arm $arm --eval-family $other
    done
done
python phase0/test5_gate.py report
echo "TEST 5 DONE"
