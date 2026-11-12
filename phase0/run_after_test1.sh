#!/usr/bin/env bash
# Runs the rest of Phase 0 once test 1 has stopped: test 2, then test 3, each with a report.
# Run inside WSL from the repo root, while test 1 is still running:
#     bash phase0/run_after_test1.sh
set -eo pipefail
while pgrep -f "phase0/run_test1.sh|phase0/test1_gap.py" > /dev/null; do
    sleep 60
done
sleep 30  # give the GPU memory time to be released
echo "Test 1 has stopped; starting test 2."
bash phase0/run_test2.sh
echo "Test 2 done; starting test 3."
source scripts/env.sh
python phase0/test3_controller.py run
python phase0/test3_controller.py report
echo "Phase 0 runs complete."
