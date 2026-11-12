"""Test 8b (phase0/TEST8B_GATED.md): does leaving out the families whose own training did not help make one joint adapter
better?

    python phase0/test8b_gated.py train                          # one LoRA on the kept families' training sets
    python phase0/test8b_gated.py evaluate --eval-family jugs    # test 2's evaluation (500 x 2) on one family
    python phase0/test8b_gated.py report [--forgive]

The gate keeps a family if its own round-1 adapter gained at least 3 points with the interval above 0 in test 7.
Training (test 4's settings), evaluation (test 2's) and the full union adapter it is compared with (test 8) are reused
unchanged. Every stage skips work whose output already exists.
"""
import argparse
import json
import pathlib

import test7_families as T
import test8_union as U
from common import read_jsonl
from test4_weights import evaluate_model, train_lora
from test5_gate import arm_examples, boot, per_problem

ROOT = pathlib.Path("runs/test8b")
KEPT = ("survo", "self_reference", "jugs", "course_schedule", "mini_sudoku")  # test 7: gained >= 3, interval above 0


def gated_dir():
    return ROOT / "gated"


def eval_file(family):
    return gated_dir() / f"eval_on_{family}.jsonl"


def stage_train(args):
    out = gated_dir()
    if T.skip(out / "train_log.json"):
        return
    examples, counts = [], {}
    for f in KEPT:
        ex = arm_examples(*T.training_set(f, args.n_experience), "all")
        counts[f] = len(ex)
        examples += ex
    out.mkdir(parents=True, exist_ok=True)
    (out / "counts.json").write_text(json.dumps(counts, indent=2))
    print(f"gated: {len(examples)} examples {counts}", flush=True)
    train_lora(examples, out, args, "gated", seed=1)


def stage_evaluate(args):
    out = eval_file(args.eval_family)
    if T.skip(out):
        return
    evaluate_model(gated_dir() / "merged", args.eval_family, out, args, "gated")


def rates(family, forgive=False):
    """Per-problem pass rates: the base model, the reference (the family's own adapter if the gate kept it, else the
    base), test 8's full union adapter and the gated one. None until both joint adapters are evaluated on the family."""
    if not (eval_file(family).exists() and U.eval_file(family).exists()):
        return None
    if forgive:
        import test7_format_check as F
        data, entries = T.dataset(family, 500, 3000)
        gated, full = F.rates(data, entries, eval_file(family), True), F.rates(data, entries, U.eval_file(family), True)
        base = F.rates(data, entries, T.base_file(family), True)
        own = F.mean_rates(data, entries, T.round1_files(family), True)
    else:
        gated, full = per_problem(read_jsonl(eval_file(family))), per_problem(read_jsonl(U.eval_file(family)))
        base = per_problem(read_jsonl(T.base_file(family)))
        own = T.rates(T.round1_files(family))
    n = len(gated)
    ref = own if family in KEPT else base
    return base[:n], ref[:n], full[:n], gated


def section(forgive):
    rows = {f: rates(f, forgive) for f in T.FAMILIES}
    m = lambda v: f"{100 * sum(v) / len(v):.1f}"  # noqa: E731
    lines = ["| Family | Kept | Base | Reference | Full union (test 8) | Gated union | Gated minus full | Gated minus reference |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for f, r in rows.items():
        if r is None:
            lines.append(f"| {f} | {'yes' if f in KEPT else 'no'} | not evaluated yet | | | | | |")
            continue
        base, ref, full, gated = r
        lines.append(f"| {f} | {'yes' if f in KEPT else 'no'} | {m(base)} | {m(ref)} | {m(full)} | {m(gated)} | "
                     f"{U.ci(boot(full, gated, [], []))} | {U.ci(boot(ref, gated, [], []))} |")
    lines.append("")
    if any(r is None for r in rows.values()):
        lines.append(f"**Not complete:** {sum(r is not None for r in rows.values())} of {len(T.FAMILIES)} families evaluated.")
        return lines, None, None
    main = U.stratified_boot([(full, gated) for _, _, full, gated in rows.values()])
    ref = U.stratified_boot([(ref, gated) for _, ref, _, gated in rows.values()])
    kept = U.stratified_boot([(r[2], r[3]) for f, r in rows.items() if f in KEPT])
    out = U.stratified_boot([(r[2], r[3]) for f, r in rows.items() if f not in KEPT])
    lines += [f"- Mean over the 8 families, gated minus full union: **{U.ci(main)}**",
              f"- Mean over the 8 families, gated minus reference: {U.ci(ref)}",
              f"- Gated minus full union, mean over the 5 kept families: {U.ci(kept)}; over the 3 left out: {U.ci(out)}"]
    return lines, main, ref


def stage_report(args):
    main_lines, main, ref = section(False)
    lines = ["# Test 8b: does a gate on top of joint training make one adapter better?", "",
             f"Plan: `phase0/TEST8B_GATED.md`. Kept by the gate: {', '.join(KEPT)}. Pass@1 points with paired bootstrap 95% "
             "intervals; means over families resample problems within each family.", "", "## As scored (main)", ""]
    lines += main_lines
    if main is not None:
        lines += ["", f"**Main: the gate {'HELPS' if main[1] > 0 else 'is not shown to help'}** (gated minus full union, "
                      "the interval's lower end must be above 0)",
                  f"**Secondary: the gated adapter {'HOLDS' if ref[1] > -U.MARGIN / 100 else 'does not hold'} the gains of "
                  f"the best separate choice** (the interval's lower end must be above -{U.MARGIN} points)"]
    if args.forgive:
        forgiven, _, _ = section(True)
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
    parser.add_argument("--root", help="output root (for the dry run); test 8's union is still read from runs/test8")
    args = parser.parse_args()
    if args.root:
        global ROOT
        ROOT = pathlib.Path(args.root)
    if args.stage == "evaluate" and not args.eval_family:
        raise SystemExit("--eval-family is required")
    {"train": stage_train, "evaluate": stage_evaluate, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
