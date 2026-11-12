"""Test 6 (phase0/TEST6_SELF_TRAINING.md): does self-training compound over rounds?

    python phase0/test6_iterate.py practice --family survo --round 2      # the round-1 model plays the practice
    python phase0/test6_iterate.py train    --family survo --round 2 --draw s1
    python phase0/test6_iterate.py evaluate --family survo --round 2 --draw s1 [--eval-family quantum_lock]
    python phase0/test6_iterate.py report   --family survo

Round k trains a fresh LoRA on the base model with every correct, finished attempt of round k's practice, played by
the round k-1 model. Round 1 is test 5's R-all (its experience, model and evaluation). Practice, training and
evaluation are test 4's and test 5's functions, unchanged.
"""
import argparse
import json
import pathlib

from common import CORRECT, read_jsonl, save_config, score, write_jsonl
from test4_weights import dataset, evaluate_model, finished, generate, load_llm, summary, train_lora
from test5_gate import arm_examples, boot, per_problem

ROOT = pathlib.Path("runs/test6")
TEST5 = pathlib.Path("runs/test5")
TEST2 = pathlib.Path("runs/test2")
FAMILIES = ("survo", "quantum_lock")
DRAWS = {1: ("t5", "s2"), 2: ("s1",), 3: ("s1", "s2")}  # trainings per round; the first one plays the next practice


def model_dir(family, k, draw):
    return TEST5 / family / "all" if (k, draw) == (1, "t5") else ROOT / family / f"r{k}_{draw}"


def practice_file(family, k):
    return TEST5 / family / "experience.jsonl" if k == 1 else ROOT / family / f"r{k}" / "practice.jsonl"


def eval_file(family, k, draw, eval_family=None):
    name = "eval.jsonl" if eval_family in (None, family) else f"eval_on_{eval_family}.jsonl"
    return model_dir(family, k, draw) / name


def stage_practice(args):
    k = args.round
    if k < 2:
        raise SystemExit("round 1's practice is test 5's experience (played by the base model)")
    player = model_dir(args.family, k - 1, DRAWS[k - 1][0]) / "merged"
    out = practice_file(args.family, k)
    out.parent.mkdir(parents=True, exist_ok=True)
    data, entries = dataset(args.family, args.n_experience, 2000)
    outputs = generate(load_llm(player, args), [e["question"] for e in entries], 2000 + k, args, n=args.per_problem)
    records = []
    for i, (entry, samples) in enumerate(zip(entries, outputs)):
        attempts = []
        for text, length in samples:
            _, s = score(data, entry, text)
            attempts.append({"text": text, "length": length, "score": s,
                             "finished": finished(text, length, args.max_tokens)})
        records.append({"index": i, "attempts": attempts, "n_correct": sum(a["score"] >= CORRECT for a in attempts)})
    write_jsonl(out, records)
    save_config(out.parent, "config_practice.json", args)
    correct = sum(r["n_correct"] for r in records)
    print(f"{args.family} round {k} practice by {player}: {correct} correct attempts of "
          f"{len(records) * args.per_problem}; {sum(r['n_correct'] > 0 for r in records)} problems solved at least once")


def stage_train(args):
    if (args.round, args.draw) == (1, "t5"):
        raise SystemExit("test 5's R-all model is reused, not retrained")
    _, entries = dataset(args.family, args.n_experience, 2000)
    records = read_jsonl(practice_file(args.family, args.round))[:len(entries)]
    examples = arm_examples(records, entries, "all")
    if not examples:
        raise SystemExit("no correct, finished attempts to train on")
    train_lora(examples, model_dir(args.family, args.round, args.draw), args,
               f"{args.family}/r{args.round}_{args.draw}", seed=int(args.draw[1:]))


def stage_evaluate(args):
    if (args.round, args.draw) == (1, "t5"):
        raise SystemExit("test 5's R-all model is reused with its evaluations; evaluating it again would overwrite them")
    family = args.eval_family or args.family
    evaluate_model(model_dir(args.family, args.round, args.draw) / "merged", family,
                   eval_file(args.family, args.round, args.draw, family), args,
                   f"{args.family}/r{args.round}_{args.draw}")


def round_rates(family, k, eval_family=None):
    """Per-problem pass rates of round k, averaged over its trainings; None until all of them are evaluated."""
    files = [eval_file(family, k, d, eval_family) for d in DRAWS[k]]
    if not all(f.exists() for f in files):
        return None
    rates = [per_problem(read_jsonl(f)) for f in files]
    return [sum(col) / len(col) for col in zip(*rates)]


def stage_report(args):
    family = args.family
    other = [f for f in FAMILIES if f != family][0]
    base = per_problem(read_jsonl(TEST2 / family / "eval_none.jsonl"))
    lines = [f"Test 6 on {family} (phase0/TEST6_SELF_TRAINING.md); base pass@1 {100 * sum(base) / len(base):.1f}", "",
             "| Round | Training | Examples | Pass@1 | Cut off | Correct among finished |",
             "| --- | --- | --- | --- | --- | --- |"]
    for k, draws in DRAWS.items():
        for d in draws:
            f = eval_file(family, k, d)
            if not f.exists():
                continue
            s = summary(read_jsonl(f), args.max_tokens)
            log = model_dir(family, k, d) / "train_log.json"
            examples = json.loads(log.read_text())["examples"] if log.exists() else "?"
            finished = 100 - s["cut_off"]
            lines.append(f"| {k} | {d} | {examples} | {s['pass@1']:.1f} | {s['cut_off']:.1f}% | "
                         f"{100 * s['pass@1'] / finished if finished else 0:.1f}% |")
    r = {k: round_rates(family, k) for k in DRAWS}
    lines.append("")
    for k, rates in r.items():
        if rates:
            g, lo, hi = boot(base[:len(rates)], rates, [], [])
            lines.append(f"- Round {k} minus base: {100 * g:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    if r[3]:
        g, lo, hi = boot(base[:len(r[3])], r[3], [], [])
        main = 100 * g >= 5 and lo > 0
        lines.append(f"\n**Main, round 3 minus base: {100 * g:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]: self-training "
                     f"{'PASSES' if main else 'does not pass'}** (needs >= +5 with the interval above 0)")
        if r[1]:
            g, lo, hi = boot(r[1], r[3], [], [])
            adds = 100 * g >= 2 and lo > 0
            lines.append(f"**Compounding, round 3 minus round 1: {100 * g:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]: rounds "
                         f"{'ADD' if adds else 'do not add'}** (needs >= +2 with the interval above 0)")
        oth = eval_file(family, 3, "s1", other)
        if oth.exists():
            b_o = per_problem(read_jsonl(TEST2 / other / "eval_none.jsonl"))
            o = per_problem(read_jsonl(oth))
            g, lo, hi = boot(b_o[:len(o)], o, [], [])
            lines.append(f"- Round 3 (s1) on {other}, minus base: {100 * g:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    report = "\n".join(lines)
    (ROOT / family).mkdir(parents=True, exist_ok=True)
    (ROOT / family / "report.md").write_text(report + "\n")
    print(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["practice", "train", "evaluate", "report"])
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--round", type=int, choices=sorted(DRAWS))
    parser.add_argument("--draw")
    parser.add_argument("--eval-family", choices=FAMILIES)
    parser.add_argument("--n-experience", type=int, default=200)
    parser.add_argument("--per-problem", type=int, default=8)
    parser.add_argument("--n-eval", type=int, default=500)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--root", help="output root (for the dry run); round 1 and the base still come from tests 5 and 2")
    args = parser.parse_args()
    if args.root:
        global ROOT
        ROOT = pathlib.Path(args.root)
    if args.stage in ("train", "evaluate") and args.draw not in DRAWS[args.round]:
        raise SystemExit(f"round {args.round} has trainings {DRAWS[args.round]}")
    {"practice": stage_practice, "train": stage_train, "evaluate": stage_evaluate, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
