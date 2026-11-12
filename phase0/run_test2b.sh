#!/usr/bin/env bash
# Test 2b (PREREGISTRATION.md): budget-aware lessons on survo. Reuses test 2's experience
# attempts and no-playbook evaluation (same problems, seeds and model); only the lesson
# writing changes. Finished stages are skipped, so a rerun resumes.
# Run inside WSL from the repo root:  bash phase0/run_test2b.sh
set -eo pipefail
source scripts/env.sh
set -u
F=survo
mkdir -p "runs/test2b/$F"
cp -n "runs/test2/$F/experience.jsonl" "runs/test2/$F/eval_none.jsonl" "runs/test2b/$F/"

[ -f "runs/test2b/$F/playbook_Qwen__Qwen3-1.7B.txt" ] ||
    python phase0/test2_lessons.py write --variant efficient --family "$F" --writer Qwen/Qwen3-1.7B
[ -f "runs/test2b/$F/playbook_Qwen__Qwen3-8B-AWQ.txt" ] ||
    python phase0/test2_lessons.py write --variant efficient --family "$F" --writer Qwen/Qwen3-8B-AWQ
python phase0/test2_lessons.py evaluate --variant efficient --family "$F"
python phase0/test2_lessons.py report --variant efficient
python phase0/test2_diagnose.py --family "$F" --root runs/test2b
