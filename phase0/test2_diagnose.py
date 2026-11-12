"""Test 2 diagnostics: why a playbook helps or hurts, per condition.

Splits failures into answers cut off at the token cap and finished-but-wrong answers,
and reports answer length, so a drop can be traced to longer reasoning or to bad advice.

    python phase0/test2_diagnose.py --family survo
"""
import argparse
import pathlib

from common import CORRECT, read_jsonl

ROOT = pathlib.Path("runs/test2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--root", default=str(ROOT), help="runs/test2b for test 2b")
    args = parser.parse_args()

    lines = ["| Condition | pass@1 | Cut off at cap | Finished but wrong | Mean tokens |",
             "| --- | --- | --- | --- | --- |"]
    for path in sorted((pathlib.Path(args.root) / args.family).glob("eval_*.jsonl")):
        samples = [(s, n) for r in read_jsonl(path) for s, n in zip(r["scores"], r["lengths"])]
        total = len(samples)
        correct = sum(s >= CORRECT for s, _ in samples)
        cut = sum(n >= args.max_tokens for s, n in samples if s < CORRECT)
        lines.append(f"| {path.stem.removeprefix('eval_')} | {100 * correct / total:.1f}% "
                     f"| {100 * cut / total:.1f}% | {100 * (total - correct - cut) / total:.1f}% "
                     f"| {sum(n for _, n in samples) / total:.0f} |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
