#!/usr/bin/env bash
# Early test 2 on the first 3 qualifying families from test 1 (in stage B order), with a
# report after each family. Finished stages are skipped, so a rerun resumes.
# Run inside WSL from the repo root, after test 1:  bash phase0/run_test2.sh
set -eo pipefail
source scripts/env.sh
set -u
SAMPLES="${SAMPLES:-2}"   # samples per problem in the evaluation (PREREGISTRATION.md: 2)

qualifies() {  # is family $1 marked "yes" in the test 1 report?
    awk -F'|' -v fam="$1" '{gsub(/ /, "", $2); gsub(/ /, "", $9)}
        $2 == fam && $9 == "yes" {found = 1} END {exit !found}' runs/test1/report.md
}
FAMILIES=""
for f in $(cat runs/test1_screen/order.txt); do
    qualifies "$f" && FAMILIES="$FAMILIES $f"
done
FAMILIES="$(echo $FAMILIES | tr ' ' '\n' | head -3 | tr '\n' ' ')"
echo "Test 2 families: $FAMILIES"

for f in $FAMILIES; do
    [ -f "runs/test2/$f/experience.jsonl" ] ||
        python phase0/test2_lessons.py experience --family "$f"
    [ -f "runs/test2/$f/playbook_Qwen__Qwen3-1.7B.txt" ] ||
        python phase0/test2_lessons.py write --family "$f" --writer Qwen/Qwen3-1.7B
    [ -f "runs/test2/$f/playbook_Qwen__Qwen3-8B-AWQ.txt" ] ||
        python phase0/test2_lessons.py write --family "$f" --writer Qwen/Qwen3-8B-AWQ
    python phase0/test2_lessons.py evaluate --family "$f" --samples "$SAMPLES"
    python phase0/test2_lessons.py report
done
