#!/usr/bin/env bash
# The gate signals diagnostic end to end (phase0/GATE_SIGNALS.md). Run inside WSL from the repo root:
#   bash phase0/run_gate_signals.sh > logs/gate_signals.log 2>&1
set -eo pipefail
source scripts/env.sh
python phase0/gate_signals.py signals
for arm in grad_top fam_top fam_bottom random_s1; do
    python phase0/gate_signals.py arm --arm $arm
done
python phase0/gate_signals.py report
echo "GATE SIGNALS DONE"
