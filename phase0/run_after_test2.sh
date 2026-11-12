#!/usr/bin/env bash
# Starts test 2b once the running test 2 has finished. Test 3 stays paused.
# Run inside WSL from the repo root:  bash phase0/run_after_test2.sh
set -eo pipefail
while pgrep -f "phase0/run_test2.sh" > /dev/null; do
    sleep 60
done
sleep 30  # give the GPU memory time to be released
echo "Test 2 has finished; starting test 2b."
bash phase0/run_test2b.sh
echo "Test 2b done."
