"""Test 7b (phase0/TEST7_CHECK.md): does a cheap likelihood check after training predict test 7's measured gains
and forgetting?

    python phase0/test7_check.py validate --family jugs   # the base model answers 100 unseen problems x 4 (seed 4000)
    python phase0/test7_check.py base     --family jugs   # the base model's log-probabilities of those answers
    python phase0/test7_check.py score    --family jugs   # the family's trained model(s), on its own and the anchor's
    python phase0/test7_check.py report

The check scores saved answers only, so it needs no generation once the validation answers exist. The models are test
7's round-1 models: test 5's R-all and test 6's seed 2 for survo and quantum_lock, seed 1 for the new families.
"""
import argparse
import json
import pathlib
import random

import test7_families as T
from common import CORRECT, read_jsonl, save_config, score, write_jsonl
from gate_signals import tokens
from test4_weights import LEARNER, chunked_loss, dataset, finished, generate, load_llm

ROOT = pathlib.Path("runs/test7_check")
SEED = 4000  # validation problems and sampling; practice used 2000 and evaluation 3000


def gain_models(family):
    """The models whose mean gain test 7 reports (round 1)."""
    if family in T.OLD:
        return [T.TEST5 / family / "all" / "merged", T.TEST6 / family / "r1_s2" / "merged"]
    return [T.model_dir(family) / "merged"]


def forgetting_models(family):
    """The models whose forgetting test 7 reports: test 5's R-all for the old families."""
    return gain_models(family)[:1]


def validation_file(family):
    return ROOT / family / "validation.jsonl"


def stage_validate(args):
    out = validation_file(args.family)
    if T.skip(out):
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    data, entries = dataset(args.family, args.n_problems, SEED)
    outputs = generate(load_llm(LEARNER, args), [e["question"] for e in entries], SEED, args, n=args.per_problem)
    records = []
    for i, (entry, samples) in enumerate(zip(entries, outputs)):
        attempts = []
        for text, length in samples:
            _, s = score(data, entry, text)
            attempts.append({"text": text, "length": length, "score": s,
                             "finished": finished(text, length, args.max_tokens)})
        records.append({"index": i, "attempts": attempts})
    write_jsonl(out, records)
    save_config(out.parent, "config_validate.json", args)
    right = sum(a["finished"] and a["score"] >= CORRECT for r in records for a in r["attempts"])
    print(f"{args.family}: {right} correct, finished validation attempts of {len(records) * args.per_problem}", flush=True)


def selection(family, n_problems, cap):
    """The right and wrong finished validation attempts that are scored (at most `cap` of each, random seed 0)."""
    _, entries = dataset(family, n_problems, SEED)
    right, wrong = [], []
    for r in read_jsonl(validation_file(family)):
        for a in r["attempts"]:
            if a["finished"]:
                (right if a["score"] >= CORRECT else wrong).append((entries[r["index"]]["question"], a["text"]))
    rng = random.Random(0)
    return [rng.sample(v, min(len(v), cap)) for v in (right, wrong)]


class Merged:
    """Lets test 4's chunked_loss score a merged (plain) model, as it scores a LoRA model during training."""
    def __init__(self, model):
        self.model = model

    def get_base_model(self):
        return self.model


def logprobs(model_path, sets):
    """Mean log-probability per completion token of each attempt in each set, under one model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(LEARNER)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    out = []
    with torch.no_grad():
        for examples in sets:
            vals = []
            for question, text in examples:
                ids, labels = tokens(tokenizer, question, text)
                vals.append(-chunked_loss(Merged(model), torch.tensor([ids]).cuda(), torch.tensor([labels]).cuda()).item())
            out.append(vals)
    del model
    torch.cuda.empty_cache()
    return out


def stage_base(args):
    out = ROOT / args.family / "base_logprobs.json"
    if T.skip(out):
        return
    right, wrong = logprobs(LEARNER, selection(args.family, args.n_problems, args.cap))
    out.write_text(json.dumps({"right": right, "wrong": wrong}))
    print(f"{args.family}: base scored {len(right)} right and {len(wrong)} wrong attempts", flush=True)


def stage_score(args):
    out = ROOT / args.family / "model_logprobs.json"
    if T.skip(out):
        return
    a = T.anchor(args.family)
    own, anchor = selection(args.family, args.n_problems, args.cap), selection(a, args.n_problems, args.cap)
    result = {"gain": [], "forgetting": []}
    for kind, models, sets in (("gain", gain_models(args.family), own), ("forgetting", forgetting_models(args.family), anchor)):
        for m in models:
            right, wrong = logprobs(m, sets)
            result[kind].append({"model": str(m), "right": right, "wrong": wrong})
    out.write_text(json.dumps(result))
    print(f"{args.family}: scored {len(gain_models(args.family))} model(s) on itself and on {a}", flush=True)


def contrast(base, trained):
    """Gain in mean log-probability on right attempts minus that on wrong attempts, in nats per token."""
    d = lambda k: sum(t - b for t, b in zip(trained[k], base[k])) / len(base[k])  # noqa: E731
    return d("right") - d("wrong")


def signals(family, min_attempts):
    """The gain check and the forgetting check for one family; None where too few attempts were scored."""
    path = ROOT / family / "model_logprobs.json"
    if not path.exists():
        return None, None
    scored = json.loads(path.read_text())
    out = []
    for kind, base_family in (("gain", family), ("forgetting", T.anchor(family))):
        base = json.loads((ROOT / base_family / "base_logprobs.json").read_text())
        if min(len(base["right"]), len(base["wrong"])) < min_attempts:
            out.append(None)
            continue
        out.append(sum(contrast(base, m) for m in scored[kind]) / len(scored[kind]))
    return tuple(out)


def stage_report(args):
    rows = []
    for f in T.FAMILIES:
        r = T.family_result(f)
        g, fo = signals(f, args.min_attempts)
        rows.append((f, g, fo, r.get("gain", [None])[0], r.get("forgetting", [None])[0]))
    lines = ["# Test 7b: does a likelihood check after training predict the measured gains and forgetting?", "",
             "Plan: `phase0/TEST7_CHECK.md`. Checks in nats per token (right minus wrong); outcomes in pass@1 points, as "
             "scored in test 7.", "",
             "| Family | Forgetting check | Forgetting measured (on) | Gain check | Gain measured |",
             "| --- | --- | --- | --- | --- |"]
    for f, g, fo, gain, forget in rows:
        fmt = lambda v, d: "n/a" if v is None else f"{v:+.{d}f}"  # noqa: E731
        lines.append(f"| {f} | {fmt(fo, 4)} | {fmt(forget, 1)} ({T.anchor(f)}) | {fmt(g, 4)} | {fmt(gain, 1)} |")
    lines.append("")
    for name, idx, out_idx, main in (("forgetting", 2, 4, True), ("gain", 1, 3, False)):
        pts = [(r[idx], r[out_idx]) for r in rows if r[idx] is not None and r[out_idx] is not None]
        if len(pts) < args.min_families:
            lines.append(f"**{name.capitalize()} check: not testable** ({len(pts)} families; needs {args.min_families})")
            continue
        rho, p = T.permutation_p([x for x, _ in pts], [y for _, y in pts])
        passes = rho >= 0.65 and p < 0.05
        lines.append(f"**{'Main' if main else 'Secondary'}, the {name} check {'PREDICTS' if passes else 'does not predict'} "
                     f"the measured {name}** (rho {rho:+.2f}, one-sided p {p:.4f}, {len(pts)} families; needs rho >= 0.65 "
                     f"and p < 0.05)")
    report = "\n".join(lines)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "report.md").write_text(report + "\n")
    print(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["validate", "base", "score", "report"])
    parser.add_argument("--family", choices=T.FAMILIES)
    parser.add_argument("--n-problems", type=int, default=100)
    parser.add_argument("--per-problem", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--cap", type=int, default=200, help="right and wrong attempts scored per family, each")
    parser.add_argument("--min-attempts", type=int, default=10)
    parser.add_argument("--min-families", type=int, default=6)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--root", help="output root (for the dry run)")
    args = parser.parse_args()
    if args.root:
        global ROOT
        ROOT = pathlib.Path(args.root)
    if args.stage != "report" and not args.family:
        raise SystemExit("--family is required")
    {"validate": stage_validate, "base": stage_base, "score": stage_score, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
