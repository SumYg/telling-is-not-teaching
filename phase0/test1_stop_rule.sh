#!/usr/bin/env bash
# Enforces the stage B stopping rule (PREREGISTRATION.md, 2026-09-28) on a running
# run_test1.sh: stop once 3 families qualify, or after the first 5 families.
# Run inside WSL from the repo root, alongside run_test1.sh:  bash phase0/test1_stop_rule.sh
REPORT=runs/test1/report.md
DONE_DIR=runs/test1/Qwen__Qwen3-8B-AWQ   # a family is done when the comparison model finishes

while pgrep -f "phase0/run_test1.sh" > /dev/null; do
    done_count=$(ls "$DONE_DIR"/*.jsonl 2>/dev/null | wc -l)
    rows=$(grep -c '^| [a-z_0-9]* | [0-9]' "$REPORT" 2>/dev/null || true)
    qualifying=$(sed -n 's/^Qualifying families: \([0-9]*\).*/\1/p' "$REPORT" 2>/dev/null)
    # Act only once the report covers every finished family.
    if [ "$done_count" -gt 0 ] && [ "${rows:-0}" -eq "$done_count" ]; then
        if [ "${qualifying:-0}" -ge 3 ] || [ "$done_count" -ge 5 ]; then
            echo "Stop rule met: ${qualifying:-0} qualifying after $done_count families."
            pkill -f "phase0/run_test1.sh"
            pkill -TERM -f "phase0/test1_gap.py"
            sleep 20
            pkill -KILL -f "phase0/test1_gap.py|EngineCore" 2>/dev/null
            echo "Stage B stopped."
            exit 0
        fi
    fi
    sleep 30
done
echo "run_test1.sh ended before the stop rule was met."
