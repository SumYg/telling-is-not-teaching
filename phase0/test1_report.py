"""Early test 1 report: applies the pass lines in PREREGISTRATION.md.

Reads runs/test1/<model>/<family>.jsonl for the learner and the static comparison model and
prints one row per family: pass@1, pass@k, the gap with a paired bootstrap 95% interval,
and whether the family qualifies.

    python phase0/test1_report.py
"""
import argparse
import json
import math
import pathlib
import random

from common import CORRECT


def load(model_dir):
    families = {}
    for path in sorted(model_dir.glob("*.jsonl")):
        families[path.stem] = [json.loads(line) for line in path.read_text().splitlines() if line]
    return families


def pass_at_k(records, k):
    """Unbiased pass@k (Chen et al., 2021), averaged over problems."""
    total = 0.0
    for r in records:
        n = len(r["scores"])
        c = sum(s >= CORRECT for s in r["scores"])
        total += 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)
    return total / len(records)


def per_problem_accuracy(records):
    return [sum(s >= CORRECT for s in r["scores"]) / len(r["scores"]) for r in records]


def bootstrap_gap(small, large, rounds=2000, seed=0):
    """Paired over problems: the same problem index in both models."""
    rng = random.Random(seed)
    n = len(small)
    gaps = []
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        gaps.append(sum(large[i] - small[i] for i in idx) / n)
    gaps.sort()
    return gaps[int(0.025 * rounds)], gaps[int(0.975 * rounds) - 1]


def screen_report(args):
    """Stage A: keep families whose learner pass@1 lies in [1.5%, 45%]."""
    root = pathlib.Path("runs/test1_screen")
    small = load(root / args.small)
    lines = ["| Family | 1.7B pass@1 (screen) | Tokens/answer | Hit max tokens | Keep |",
             "| --- | --- | --- | --- | --- |"]
    keep = []
    for family, records in sorted(small.items()):
        acc = per_problem_accuracy(records)
        p1 = 100 * sum(acc) / len(acc)
        samples = sum(len(r["lengths"]) for r in records)
        tokens = sum(sum(r["lengths"]) for r in records) / samples
        truncated = 100 * sum(r["truncated"] for r in records) / samples
        kept = 1.5 <= p1 <= 45
        keep += [family] if kept else []
        lines.append(f"| {family} | {p1:.1f} | {tokens:.0f} | {truncated:.0f}% | "
                     f"{'yes' if kept else 'no'} |")
    lines += ["", f"Kept {len(keep)} of {len(small)}: {' '.join(keep)}"]
    (root / "keep.txt").write_text(" ".join(keep) + "\n")
    # Stage B runs the most promising families first, so complete gap results arrive early.
    screen_p1 = {f: sum(per_problem_accuracy(r)) / len(r) for f, r in small.items()}
    order = sorted(keep, key=lambda f: -screen_p1[f])
    (root / "order.txt").write_text(" ".join(order) + "\n")
    report = "\n".join(lines)
    (root / "report.md").write_text(report + "\n")
    print(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", default="Qwen__Qwen3-1.7B")
    parser.add_argument("--large", default="Qwen__Qwen3-8B-AWQ")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--screen", action="store_true", help="stage A screening report")
    args = parser.parse_args()
    if args.screen:
        return screen_report(args)

    root = pathlib.Path("runs/test1")
    small, large = load(root / args.small), load(root / args.large)

    rows = []
    for family in sorted(set(small) & set(large)):
        s_acc, l_acc = per_problem_accuracy(small[family]), per_problem_accuracy(large[family])
        s1, l1 = 100 * sum(s_acc) / len(s_acc), 100 * sum(l_acc) / len(l_acc)
        sk = 100 * pass_at_k(small[family], args.k)
        lo, hi = bootstrap_gap(s_acc, l_acc)
        mean_tokens = sum(sum(r["lengths"]) for r in small[family]) / sum(
            len(r["lengths"]) for r in small[family])
        qualifies = 5 <= s1 <= 30 and l1 - s1 >= 15
        rows.append((family, s1, sk, l1, l1 - s1, 100 * lo, 100 * hi, mean_tokens,
                     qualifies, sk < l1))

    header = ("| Family | 1.7B pass@1 | 1.7B pass@{k} | 8B pass@1 | Gap (points) | Gap 95% CI "
              "| 1.7B tokens/answer | Qualifies | Preferred |").format(k=args.k)
    lines = [header, "|" + " --- |" * 9]
    for f, s1, sk, l1, gap, lo, hi, tok, q, pref in rows:
        lines.append(f"| {f} | {s1:.1f} | {sk:.1f} | {l1:.1f} | {gap:+.1f} | [{lo:+.1f}, {hi:+.1f}] "
                     f"| {tok:.0f} | {'yes' if q else 'no'} | {'yes' if q and pref else ''} |")
    n_qualify = sum(r[8] for r in rows)
    lines.append("")
    lines.append(f"Qualifying families: {n_qualify}. Go needs at least 3.")
    report = "\n".join(lines)
    (root / "report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
