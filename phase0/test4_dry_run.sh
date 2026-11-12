#!/usr/bin/env bash
# Dry run of test 4's pipeline on tiny sizes (plumbing only). Run inside WSL from the repo root.
set -eo pipefail
source scripts/env.sh
R=runs/test4_dry
rm -rf "$R"
A="--root $R --n-experience 8 --n-eval 8 --samples 1"
python phase0/test4_weights.py experience --family survo $A
python phase0/test4_weights.py rationalize --family survo $A
python phase0/test4_weights.py distill --family survo $A
python phase0/test4_weights.py train --family survo --arm w2 $A --epochs 1 --accum 2
python phase0/test4_weights.py evaluate --family survo --arm w2 $A
echo "DRY RUN OK"
