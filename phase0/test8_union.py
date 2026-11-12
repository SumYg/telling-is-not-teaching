"""Test 8 (phase0/TEST8_UNION.md): does one adapter trained on all eight families' own successes keep each family's
gain?

    python phase0/test8_union.py train                          # one LoRA on test 7's eight training sets together
    python phase0/test8_union.py evaluate --eval-family jugs    # test 2's evaluation (500 x 2) on one family
    python phase0/test8_union.py report

The training sets are test 7's round 1, unchanged (every correct, finished practice attempt of each family), and so are
the training settings (test 4's) and the evaluation (test 2's). Every stage skips work whose output already exists.
"""
import argparse
import json
import pathlib
import random

import test7_families as T
from common import read_jsonl
from test4_weights import evaluate_model, train_lora
from test5_gate import arm_examples, boot, per_problem

ROOT = pathlib.Path("runs/test8")
MARGIN = 1.5  # points: the union adapter "holds the gains" if the interval's lower end is above -MARGIN


def union_dir():
    return ROOT / "union"


def eval_file(family):
    return union_dir() / f"eval_on_{family}.jsonl"


def stage_train(args):
    out = union_dir()
    if T.skip(out / "train_log.json"):
        return
    examples, counts = [], {}
    for f in T.FAMILIES:
        ex = arm_examples(*T.training_set(f, args.n_experience), "all")
        counts[f] = len(ex)
        examples += ex
    out.mkdir(parents=True, exist_ok=True)
    (out / "counts.json").write_text(json.dumps(counts, indent=2))
    print(f"union: {len(examples)} examples {counts}", flush=True)
    train_lora(examples, out, args, "union", seed=1)


def stage_evaluate(args):
    out = eval_file(args.eval_family)
    if T.skip(out):
        return
    evaluate_model(union_dir() / "merged", args.eval_family, out, args, "union")


def stratified_boot(pairs, rounds=2000, seed=0):
    """Mean over families of mean(b - a) per family, with a 95% interval that resamples problems within each family."""
    rng = random.Random(seed)
    diffs = [[y - x for x, y in zip(a, b)] for a, b in pairs]
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    stats = sorted(mean([mean([d[rng.randrange(len(d))] for _ in d]) for d in diffs]) for _ in range(rounds))
    return mean([mean(d) for d in diffs]), stats[int(0.025 * rounds)], stats[int(0.975 * rounds) - 1]


def family_rates(family, forgive=False):
    """Per-problem pass rates of the base model, the family's own round-1 model(s) and the union adapter; None until the
    union adapter is evaluated on the family. With forgive, test 7's two layout slips are forgiven."""
    if not eval_file(family).exists():
        return None
    if forgive:
        import test7_format_check as F
        data, entries = T.dataset(family, 500, 3000)
        union = F.rates(data, entries, eval_file(family), True)
        base = F.rates(data, entries, T.base_file(family), True)
        own = F.mean_rates(data, entries, T.round1_files(family), True)
    else:
        union = per_problem(read_jsonl(eval_file(family)))
        base = per_problem(read_jsonl(T.base_file(family)))
        own = T.rates(T.round1_files(family))
    n = len(union)
    return base[:n], own[:n], union


def ci(t):
    return f"{100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}]"


def section(forgive):
    rates = {f: family_rates(f, forgive) for f in T.FAMILIES}
    lines = ["| Family | Base pass@1 | Own adapter (test 7) | Union adapter | Union minus base | Union minus own |",
             "| --- | --- | --- | --- | --- | --- |"]
    for f, r in rates.items():
        if r is None:
            lines.append(f"| {f} | not evaluated yet | | | | |")
            continue
        base, own, union = r
        m = lambda v: f"{100 * sum(v) / len(v):.1f}"  # noqa: E731
        lines.append(f"| {f} | {m(base)} | {m(own)} | {m(union)} | {ci(boot(base, union, [], []))} | "
                     f"{ci(boot(own, union, [], []))} |")
    done = [r for r in rates.values() if r is not None]
    lines.append("")
    if len(done) < len(T.FAMILIES):
        lines.append(f"**Not complete:** {len(done)} of {len(T.FAMILIES)} families evaluated.")
        return lines, None
    d = stratified_boot([(own, union) for _, own, union in done])
    g = stratified_boot([(base, union) for base, _, union in done])
    lines.append(f"- Mean over the 8 families, union minus own adapters: **{ci(d)}**")
    lines.append(f"- Mean over the 8 families, union minus base: {ci(g)}")
    return lines, d


def stage_report(args):
    main_lines, d = section(False)
    lines = ["# Test 8: does one adapter on all eight families' successes keep each family's gain?", "",
             "Plan: `phase0/TEST8_UNION.md`. Pass@1 points with paired bootstrap 95% intervals; the means over families "
             "resample problems within each family.", "", "## As scored (main)", ""] + main_lines
    if d is not None:
        holds, worse, better = d[1] > -MARGIN / 100, d[2] < 0, d[1] > 0
        lines += ["", f"**Main: one adapter {'HOLDS' if holds else 'does not hold'} the gains** (the interval's lower end "
                      f"must be above -{MARGIN} points)"
                      + ("; it is also clearly worse than the separate adapters" if worse else "")
                      + ("; it is clearly better than the separate adapters" if better else "")]
    if args.forgive:
        forgiven, _ = section(True)
        lines += ["", "## With test 7's layout slips forgiven (reported alongside)", ""] + forgiven
    report = "\n".join(lines)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "report.md").write_text(report + "\n")
    print(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["train", "evaluate", "report"])
    parser.add_argument("--eval-family", choices=T.FAMILIES)
    parser.add_argument("--n-experience", type=int, default=200)
    parser.add_argument("--n-eval", type=int, default=500)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--forgive", action="store_true", help="report: add the layout-forgiven section")
    parser.add_argument("--root", help="output root (for the dry run)")
    args = parser.parse_args()
    if args.root:
        global ROOT
        ROOT = pathlib.Path(args.root)
    if args.stage == "evaluate" and not args.eval_family:
        raise SystemExit("--eval-family is required")
    {"train": stage_train, "evaluate": stage_evaluate, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
