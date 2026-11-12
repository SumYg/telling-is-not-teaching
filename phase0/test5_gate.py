"""Test 5 (PREREGISTRATION.md): more experience, and a surprise gate on what to consolidate.

    python phase0/test5_gate.py experience --family survo        # 200 problems x 8 attempts
    python phase0/test5_gate.py train      --family survo --arm all|gate|random
    python phase0/test5_gate.py evaluate   --family survo --arm all [--eval-family quantum_lock]
    python phase0/test5_gate.py report

Training and evaluation are test 4's (LoRA settings, merge, test 2's evaluation); only the data
differs between arms.
"""
import argparse
import json
import pathlib
import random

from common import CORRECT, read_jsonl, save_config, score, write_jsonl
from test4_weights import (LEARNER, dataset, evaluate_model, finished, generate, load_llm, summary,
                           train_lora)

ROOT = pathlib.Path("runs/test5")
TEST2 = pathlib.Path("runs/test2")
FAMILIES = ("survo", "quantum_lock")
SURPRISE_MAX = 2  # the gate admits problems with at most this many correct attempts out of 8


def stage_experience(args):
    out = ROOT / args.family
    out.mkdir(parents=True, exist_ok=True)
    data, entries = dataset(args.family, args.n_experience, 2000)
    outputs = generate(load_llm(LEARNER, args), [e["question"] for e in entries], 2000, args,
                       n=args.per_problem)
    records = []
    for i, (entry, samples) in enumerate(zip(entries, outputs)):
        attempts = []
        for text, length in samples:
            _, s = score(data, entry, text)
            attempts.append({"text": text, "length": length, "score": s,
                             "finished": finished(text, length, args.max_tokens)})
        records.append({"index": i, "attempts": attempts,
                        "n_correct": sum(a["score"] >= CORRECT for a in attempts)})
    write_jsonl(out / "experience.jsonl", records)
    save_config(out, "config_experience.json", args)
    correct = sum(r["n_correct"] for r in records)
    print(f"{args.family}: {correct} correct attempts of {len(records) * args.per_problem}; "
          f"{sum(r['n_correct'] > 0 for r in records)} problems solved at least once")


def arm_examples(records, entries, arm, seed=0):
    """[(question, text)] for an arm. R-random matches R-gate's size, drawn from R-all's pool."""
    pool = [(entries[r["index"]]["question"], a["text"], r["n_correct"])
            for r in records for a in r["attempts"] if a["score"] >= CORRECT and a["finished"]]
    gated = [p for p in pool if p[2] <= SURPRISE_MAX]
    if arm == "all":
        chosen = pool
    elif arm == "gate":
        chosen = gated
    else:
        chosen = random.Random(seed).sample(pool, len(gated))
    return [(q, t) for q, t, _ in chosen]


def stage_train(args):
    out = ROOT / args.family
    _, entries = dataset(args.family, args.n_experience, 2000)
    records = read_jsonl(out / "experience.jsonl")
    train_lora(arm_examples(records, entries, args.arm), out / args.arm, args, f"{args.family}/{args.arm}")


def stage_evaluate(args):
    family = args.eval_family or args.family
    out = ROOT / args.family / args.arm
    name = "eval.jsonl" if family == args.family else f"eval_on_{family}.jsonl"
    evaluate_model(out / "merged", family, out / name, args, f"{args.family}/{args.arm}")


def per_problem(records):
    return [sum(s >= CORRECT for s in r["scores"]) / len(r["scores"]) for r in records]


def boot(own_a, own_b, other_a, other_b, rounds=2000, seed=0):
    """Mean and 95% interval of (own_b - own_a) + (other_b - other_a), resampling each set's
    problems separately. With other_* empty, only the own-family difference."""
    rng = random.Random(seed)
    d1 = [b - a for a, b in zip(own_a, own_b)]
    d2 = [b - a for a, b in zip(other_a, other_b)]
    mean = lambda d: sum(d) / len(d) if d else 0.0  # noqa: E731
    stats = sorted(mean([d1[rng.randrange(len(d1))] for _ in d1])
                   + (mean([d2[rng.randrange(len(d2))] for _ in d2]) if d2 else 0.0)
                   for _ in range(rounds))
    return mean(d1) + mean(d2), stats[int(0.025 * rounds)], stats[int(0.975 * rounds) - 1]


def stage_report(args):
    base = {f: per_problem(read_jsonl(TEST2 / f / "eval_none.jsonl")) for f in FAMILIES}
    lines = ["| Trained on | Arm | Examples | Own pass@1 | Own gain [95% CI] | Other-family change [95% CI] | Own cut off |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    results, learning_works = {}, []
    for family in FAMILIES:
        other = [f for f in FAMILIES if f != family][0]
        records = read_jsonl(ROOT / family / "experience.jsonl")
        admitted = sum(0 < r["n_correct"] <= SURPRISE_MAX for r in records)
        lines_info = f"{family}: {sum(r['n_correct'] for r in records)} correct attempts; gate admits {admitted} problems"
        for arm in ("all", "gate", "random"):
            d = ROOT / family / arm
            if not (d / "eval.jsonl").exists():
                continue
            own = read_jsonl(d / "eval.jsonl")
            oth = read_jsonl(d / f"eval_on_{other}.jsonl") if (d / f"eval_on_{other}.jsonl").exists() else None
            g, lo, hi = boot(base[family], per_problem(own), [], [])
            o = boot(base[other], per_problem(oth), [], []) if oth else None
            results[family, arm] = (per_problem(own), per_problem(oth) if oth else None)
            if arm == "all" and 100 * g >= 5 and lo > 0:
                learning_works.append(family)
            examples = json.loads((d / "train_log.json").read_text())["examples"]
            s = summary(own, args.max_tokens)
            other_text = f"{100 * o[0]:+.1f} [{100 * o[1]:+.1f}, {100 * o[2]:+.1f}]" if o else "n/a"
            lines.append(f"| {family} | R-{arm} | {examples} | {s['pass@1']:.1f} | {100 * g:+.1f} "
                         f"[{100 * lo:+.1f}, {100 * hi:+.1f}] | {other_text} | {s['cut_off']:.1f}% |")
        lines.append(f"\n{lines_info}\n")
    gate_go = []
    for family in learning_works:
        if (family, "gate") in results and (family, "random") in results:
            (g_own, g_oth), (r_own, r_oth) = results[family, "gate"], results[family, "random"]
            if g_oth is None or r_oth is None:
                continue
            nb, lo, hi = boot(r_own, g_own, r_oth, g_oth)
            lines.append(f"- {family}: net effect R-gate minus R-random {100 * nb:+.1f} points "
                         f"[{100 * lo:+.1f}, {100 * hi:+.1f}]")
            if 100 * nb >= 3 and lo > 0:
                gate_go.append(family)
    lines.append(f"\nPass line 1, learning works: **{'GO' if learning_works else 'NO-GO'}** {learning_works}")
    lines.append(f"Pass line 2, gate beats random: **{'GO' if gate_go else 'NO-GO'}** {gate_go}"
                 + ("" if learning_works else " (not testable: pass line 1 failed)"))
    report = "\n".join(lines)
    (ROOT / "report.md").write_text(report + "\n")
    print(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["experience", "train", "evaluate", "report"])
    parser.add_argument("--family")
    parser.add_argument("--arm", choices=["all", "gate", "random"])
    parser.add_argument("--eval-family")
    parser.add_argument("--n-experience", type=int, default=200)
    parser.add_argument("--per-problem", type=int, default=8)
    parser.add_argument("--n-eval", type=int, default=500)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--root")
    args = parser.parse_args()
    if args.root:
        global ROOT
        ROOT = pathlib.Path(args.root)
    {"experience": stage_experience, "train": stage_train, "evaluate": stage_evaluate,
     "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
