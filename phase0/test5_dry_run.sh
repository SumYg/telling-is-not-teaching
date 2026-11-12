#!/usr/bin/env bash
# Dry run of test 5 on tiny sizes (plumbing only). Run inside WSL from the repo root.
set -eo pipefail
source scripts/env.sh
R=runs/test5_dry
rm -rf "$R"
A="--root $R --n-experience 6 --per-problem 2 --n-eval 4 --samples 1"
python phase0/test5_gate.py experience --family survo $A
python phase0/test5_gate.py train --family survo --arm all $A --epochs 1 --accum 2
python phase0/test5_gate.py evaluate --family survo --arm all $A
python phase0/test5_gate.py evaluate --family survo --arm all --eval-family quantum_lock $A
echo "DRY RUN OK"
