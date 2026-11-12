#!/usr/bin/env bash
# Part 2 of the gate signals diagnostic (phase0/GATE_SIGNALS.md). Run inside WSL from the repo root:
#   bash phase0/run_gate_replication.sh > logs/gate_replication.log 2>&1
set -eo pipefail
source scripts/env.sh
for arm in fam_top_t1 random_s2 fam_top_t2 random_s3 random_s4; do
    python phase0/gate_signals.py arm --arm $arm
done
python phase0/gate_signals.py report
echo "GATE REPLICATION DONE"
