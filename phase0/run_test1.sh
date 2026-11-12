#!/usr/bin/env bash
# Early test 1, end to end: screen, screening report, then the full test one family at a time
# (learner, then comparison model), highest screen pass@1 first, with a report after each.
# Run inside WSL from the repo root:  bash phase0/run_test1.sh
# Finished families are skipped, so rerunning after an interruption resumes where it stopped.
set -eo pipefail
source scripts/env.sh
set -u

python phase0/test1_gap.py --model Qwen/Qwen3-1.7B --samples 2 --n 100 --out runs/test1_screen
python phase0/test1_report.py --screen

ORDER="$(cat runs/test1_screen/order.txt)"
if [ -z "$ORDER" ]; then
    echo "No family passed the screen."
    exit 0
fi
for family in $ORDER; do
    python phase0/test1_gap.py --model Qwen/Qwen3-1.7B --samples 8 --families "$family"
    python phase0/test1_gap.py --model Qwen/Qwen3-8B-AWQ --samples 4 --families "$family"
    python phase0/test1_report.py
done
