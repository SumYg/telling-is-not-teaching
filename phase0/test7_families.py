"""Test 7 (phase0/TEST7_FAMILIES.md): which families gain from self-training, and does a probe predict it?

    python phase0/test7_families.py practice --family jugs     # the base model plays 200 problems x 8 (test 5's)
    python phase0/test7_families.py base     --family jugs     # the base model on test 2's evaluation (500 x 2)
    python phase0/test7_families.py probe    --family jugs     # the signals, before any training on the family
    python phase0/test7_families.py train    --family jugs     # round 1: every correct, finished attempt, seed 1
    python phase0/test7_families.py evaluate --family jugs [--eval-family survo]
    python phase0/test7_families.py report

Survo and quantum_lock reuse tests 2, 5 and 6 (practice, base evaluation, round-1 models and their evaluations); only
their probe runs here. Practice, training and evaluation are tests 4 and 5's functions, unchanged. Every stage skips
work whose output already exists, so an interrupted run can be restarted.
"""
import argparse
import itertools
import json
import math
import pathlib
import random

import test5_gate
from common import CORRECT, read_jsonl, save_config
from gate_signals import TARGETS, tokens
from test4_weights import LEARNER, chunked_loss, dataset, evaluate_model, summary, train_lora
from test5_gate import arm_examples, boot, per_problem

ROOT = pathlib.Path("runs/test7")
TEST2, TEST5, TEST6 = pathlib.Path("runs/test2"), pathlib.Path("runs/test5"), pathlib.Path("runs/test6")
OLD = ("survo", "quantum_lock")
NEW = ("self_reference", "list_functions", "jugs", "course_schedule", "word_ladder", "mini_sudoku")
FAMILIES = OLD + NEW
SIGNALS = ("probe", "success_gain", "n_success", "finished_accuracy", "familiarity", "reliability", "base_pass1")


def anchor(family):
    """Where forgetting is measured: survo, or quantum_lock for survo itself (test 6's pairing)."""
    return "quantum_lock" if family == "survo" else "survo"


def practice_file(family):
    return (TEST5 if family in OLD else ROOT) / family / "experience.jsonl"


def base_file(family):
    return TEST2 / family / "eval_none.jsonl" if family in OLD else ROOT / family / "eval_base.jsonl"


def model_dir(family):
    return ROOT / family / "r1_s1"


def round1_files(family, eval_family=None):
    """Round-1 evaluations: test 6's two trainings (test 5's R-all and seed 2) for the old families, seed 1 for the new
    ones. Forgetting for the old families is test 5's R-all on the other family."""
    own = eval_family in (None, family)
    if family in OLD:
        if own:
            return [TEST5 / family / "all" / "eval.jsonl", TEST6 / family / "r1_s2" / "eval.jsonl"]
        return [TEST5 / family / "all" / f"eval_on_{eval_family}.jsonl"]
    return [model_dir(family) / ("eval.jsonl" if own else f"eval_on_{eval_family}.jsonl")]


def skip(path):
    if pathlib.Path(path).exists():
        print(f"{path} exists; skipping", flush=True)
        return True
    return False


def new_only(args, what):
    if args.family in OLD:
        raise SystemExit(f"{args.family}'s {what} is reused from tests 2, 5 and 6")


# ---- Practice, base evaluation, training, evaluation: tests 4 and 5's functions ----

def stage_practice(args):
    new_only(args, "practice")
    if skip(practice_file(args.family)):
        return
    test5_gate.ROOT = ROOT  # test 5's experience stage, written under runs/test7
    test5_gate.stage_experience(args)


def stage_base(args):
    new_only(args, "base evaluation")
    out = base_file(args.family)
    if skip(out):
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    save_config(out.parent, "config_base.json", args)
    evaluate_model(LEARNER, args.family, out, args, f"{args.family}/base")


def training_set(family, n_experience):
    _, entries = dataset(family, n_experience, 2000)
    records = read_jsonl(practice_file(family))[:len(entries)]
    return records, entries


def stage_train(args):
    new_only(args, "round 1")
    out = model_dir(args.family)
    if skip(out / "train_log.json") or skip(out / "too_few.json"):
        return
    examples = arm_examples(*training_set(args.family, args.n_experience), "all")
    if len(examples) < args.min_examples:
        out.mkdir(parents=True, exist_ok=True)
        (out / "too_few.json").write_text(json.dumps({"examples": len(examples)}))
        print(f"{args.family}: only {len(examples)} correct, finished attempts; not trained", flush=True)
        return
    train_lora(examples, out, args, f"{args.family}/r1_s1", seed=1)


def stage_evaluate(args):
    new_only(args, "round-1 evaluation")
    family = args.eval_family or args.family
    out = round1_files(args.family, family)[0]
    if skip(out) or skip(model_dir(args.family) / "too_few.json"):
        return
    evaluate_model(model_dir(args.family) / "merged", family, out, args, f"{args.family}/r1_s1")


# ---- Probe: the signals, computed from the base model and its practice only ----

def halves(records, entries):
    """Correct and wrong finished attempts, split by problem: even indices are half A, odd ones half B."""
    split = {h: {"right": [], "wrong": []} for h in "AB"}
    for r in records:
        h = "A" if r["index"] % 2 == 0 else "B"
        for a in r["attempts"]:
            if a["finished"]:
                split[h]["right" if a["score"] >= CORRECT else "wrong"].append((entries[r["index"]]["question"], a["text"]))
    return split


def logprobs(model, tokenizer, examples):
    """Mean log-probability per completion token of each (question, text), as train_lora builds them."""
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for question, text in examples:
            ids, labels = tokens(tokenizer, question, text)
            out.append(-chunked_loss(model, torch.tensor([ids]).cuda(), torch.tensor([labels]).cuda()).item())
    return out


def probe_direction(train, held_right, held_wrong, args, seed):
    """Train a LoRA on `train` with the real training's settings; return the base and trained log-probabilities of the
    held-out right and wrong attempts."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    tokenizer = AutoTokenizer.from_pretrained(LEARNER)
    model = AutoModelForCausalLM.from_pretrained(LEARNER, dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    torch.manual_seed(seed)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=TARGETS))
    with model.disable_adapter():
        before = [logprobs(model, tokenizer, s) for s in (held_right, held_wrong)]
    data = [tokens(tokenizer, q, t) for q, t in train]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(data) * args.epochs / args.accum)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(0.05 * steps)), steps)
    rng = random.Random(seed)
    for epoch in range(args.epochs):  # train_lora's loop, without saving
        model.train()
        order = list(range(len(data)))
        rng.shuffle(order)
        for i, idx in enumerate(order):
            ids, labels = data[idx]
            loss = chunked_loss(model, torch.tensor([ids]).cuda(), torch.tensor([labels]).cuda())
            (loss / args.accum).backward()
            if (i + 1) % args.accum == 0 or i + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        print(f"  probe epoch {epoch + 1} done", flush=True)
    after = [logprobs(model, tokenizer, s) for s in (held_right, held_wrong)]
    del model
    torch.cuda.empty_cache()
    return before, after


def mean(v):
    return sum(v) / len(v)


def stage_probe(args):
    out = ROOT / args.family / "signals.json"
    if skip(out):
        return
    records, entries = training_set(args.family, args.n_experience)
    split = halves(records, entries)
    n_right = sum(len(split[h]["right"]) for h in "AB")
    n_finished = n_right + sum(len(split[h]["wrong"]) for h in "AB")
    solved = sum(r["n_correct"] > 0 for r in records)
    base = per_problem(read_jsonl(base_file(args.family)))
    signals = {"family": args.family, "n_success": n_right, "finished_accuracy": n_right / n_finished if n_finished else None,
               "reliability": n_right / solved if solved else None, "base_pass1": mean(base),
               "practice_attempts": sum(len(r["attempts"]) for r in records)}
    rng = random.Random(0)
    cap = lambda v: rng.sample(v, min(len(v), args.probe_scored))  # noqa: E731
    small = [h for h in "AB" if min(len(split[h]["right"]), len(split[h]["wrong"])) < args.min_probe]
    if small:
        signals.update(probe=None, success_gain=None, familiarity=None,
                       note=f"half {small} has fewer than {args.min_probe} right or wrong finished attempts")
    else:
        directions, familiar = [], []
        for train_h, held_h in (("A", "B"), ("B", "A")):
            train = split[train_h]["right"][:args.probe_limit] if args.probe_limit else split[train_h]["right"]
            right, wrong = cap(split[held_h]["right"]), cap(split[held_h]["wrong"])
            print(f"{args.family}: probe trains on half {train_h} ({len(train)} attempts), scores half {held_h} "
                  f"({len(right)} right, {len(wrong)} wrong)", flush=True)
            (b_r, b_w), (a_r, a_w) = probe_direction(train, right, wrong, args, seed=1)
            d_r, d_w = mean([a - b for a, b in zip(a_r, b_r)]), mean([a - b for a, b in zip(a_w, b_w)])
            directions.append({"train": train_h, "n_train": len(train), "n_right": len(right), "n_wrong": len(wrong),
                               "gain_right": d_r, "gain_wrong": d_w, "base_right": mean(b_r), "base_wrong": mean(b_w)})
            familiar += b_r
        signals.update(probe=mean([d["gain_right"] - d["gain_wrong"] for d in directions]),
                       success_gain=mean([d["gain_right"] for d in directions]), familiarity=mean(familiar),
                       directions=directions)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(signals, indent=2))
    save_config(out.parent, "config_probe.json", args)
    print(json.dumps({k: v for k, v in signals.items() if k != "directions"}), flush=True)


# ---- Report ----

def ranks(v):
    """Ranks with ties given their average rank."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = mean(rx), mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx, sy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return cov / (sx * sy) ** 0.5 if sx and sy else 0.0


def permutation_p(x, y, one_sided=True):
    """Exact permutation p-value of Spearman's rho (all orderings of y): P(rho >= observed), or P(|rho| >= |observed|)."""
    rho = spearman(x, y)
    hits = total = 0
    for perm in itertools.permutations(y):
        r = spearman(x, list(perm))
        hits += (r >= rho - 1e-12) if one_sided else (abs(r) >= abs(rho) - 1e-12)
        total += 1
    return rho, hits / total


def rates(files):
    """Per-problem pass rates averaged over the files (one per training); None until all exist."""
    if not all(f.exists() for f in files):
        return None
    cols = zip(*[per_problem(read_jsonl(f)) for f in files])
    return [mean(c) for c in cols]


def family_result(family):
    base = per_problem(read_jsonl(base_file(family))) if base_file(family).exists() else None
    row = {"family": family, "base": base}
    if (model_dir(family) / "too_few.json").exists():
        row["too_few"] = json.loads((model_dir(family) / "too_few.json").read_text())["examples"]
    own = rates(round1_files(family))
    if base and own:
        row["pass1"] = 100 * mean(own)
        row["gain"] = [100 * x for x in boot(base[:len(own)], own, [], [])]
    a = anchor(family)
    other = rates(round1_files(family, a))
    if other and base_file(a).exists():
        b_a = per_problem(read_jsonl(base_file(a)))
        row["forgetting"] = [100 * x for x in boot(b_a[:len(other)], other, [], [])]
    sig = ROOT / family / "signals.json"
    row["signals"] = json.loads(sig.read_text()) if sig.exists() else {}
    return row


def fmt(v, digits=1, sign=False):
    if v is None:
        return "n/a"
    return f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"


def ci(t):
    return f"{t[0]:+.1f} [{t[1]:+.1f}, {t[2]:+.1f}]" if t else "n/a"


def stage_report(args):
    rows = [family_result(f) for f in FAMILIES]
    lines = ["# Test 7: which families gain from self-training, and does the probe predict it?", "",
             "Plan: `phase0/TEST7_FAMILIES.md`. Gains and forgetting in pass@1 points with paired bootstrap 95% intervals.",
             "", "| Family | Base pass@1 | Round 1 pass@1 | Gain | Forgetting (on) | Successes | Probe (nats/token) | "
             "Success gain | Accuracy finished | Familiarity | Reliability |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        s = r["signals"]
        gain = ci(r.get("gain")) if "gain" in r else ("not trained (" + str(r["too_few"]) + " examples)" if "too_few" in r else "n/a")
        lines.append(f"| {r['family']} | {fmt(100 * mean(r['base']) if r['base'] else None)} | {fmt(r.get('pass1'))} | "
                     f"{gain} | {ci(r.get('forgetting'))} ({anchor(r['family'])}) | {s.get('n_success', 'n/a')} | "
                     f"{fmt(s.get('probe'), 4, True)} | {fmt(s.get('success_gain'), 4, True)} | "
                     f"{fmt(100 * s['finished_accuracy']) + '%' if s.get('finished_accuracy') is not None else 'n/a'} | "
                     f"{fmt(s.get('familiarity'), 3)} | {fmt(s.get('reliability'), 2)} |")
    done = [r for r in rows if "gain" in r]
    gains = [r for r in done if r["gain"][0] >= 3 and r["gain"][1] > 0]
    losses = [r for r in done if r["gain"][2] < 0]
    forget = [r for r in rows if r.get("forgetting") and r["forgetting"][2] < 0]
    lines += ["", f"- **Families evaluated:** {len(done)} of {len(FAMILIES)}.",
              f"- **Gain (at least +3, interval above 0):** {len(gains)}: {', '.join(r['family'] for r in gains) or 'none'}.",
              f"- **Loss (interval below 0):** {len(losses)}: {', '.join(r['family'] for r in losses) or 'none'}.",
              f"- **Forgetting (interval below 0):** {len(forget)}: {', '.join(r['family'] for r in forget) or 'none'}.",
              "", "## Do the signals predict the gain? (Spearman over families, exact permutation p)", ""]
    main_line = None
    for name in SIGNALS:
        pts = [(r["signals"][name], r["gain"][0]) for r in done if r["signals"].get(name) is not None]
        if len(pts) < 3:
            lines.append(f"- {name}: fewer than 3 families")
            continue
        x, y = [p[0] for p in pts], [p[1] for p in pts]
        rho, p = permutation_p(x, y, one_sided=(name == "probe"))
        lines.append(f"- {name}: rho {rho:+.2f}, p {p:.4f} ({'one' if name == 'probe' else 'two'}-sided), "
                     f"{len(pts)} families")
        if name == "probe":
            main_line = (rho, p, len(pts))
    lines.append("")
    if main_line is None or main_line[2] < args.min_families:
        lines.append(f"**Main: not testable** (needs the probe and the gain on at least {args.min_families} families)")
    else:
        rho, p, n = main_line
        passes = rho >= 0.65 and p < 0.05
        lines.append(f"**Main: the probe {'PREDICTS' if passes else 'does not predict'} which families gain** "
                     f"(rho {rho:+.2f}, one-sided p {p:.4f}, {n} families; needs rho >= 0.65 and p < 0.05)")
        kept = [r for r in done if r["signals"].get("probe") is not None and r["signals"]["probe"] > 0]
        total = lambda rs: sum(r["gain"][0] for r in rs)  # noqa: E731
        lines.append(f"- As a gate (keep a family when the probe is above 0): keeps {len(kept)} families, total gain "
                     f"{total(kept):+.1f}; keeping all {total(done):+.1f}; keeping only those that gained "
                     f"{total([r for r in done if r['gain'][0] > 0]):+.1f}")
    report = "\n".join(lines)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "report.md").write_text(report + "\n")
    print(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["practice", "base", "probe", "train", "evaluate", "report"])
    parser.add_argument("--family", choices=FAMILIES)
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
    parser.add_argument("--min-examples", type=int, default=10, help="fewer correct, finished attempts: not trained")
    parser.add_argument("--min-probe", type=int, default=10, help="per half, right and wrong finished attempts")
    parser.add_argument("--probe-scored", type=int, default=200, help="held-out attempts scored per kind and direction")
    parser.add_argument("--probe-limit", type=int, help="dry run only: train the probe on at most N attempts")
    parser.add_argument("--min-families", type=int, default=6)
    parser.add_argument("--root", help="output root (for the dry run); tests 2, 5 and 6 are still read from runs/")
    args = parser.parse_args()
    if args.root:
        global ROOT
        ROOT = pathlib.Path(args.root)
    if args.stage != "report" and not args.family:
        raise SystemExit("--family is required")
    {"practice": stage_practice, "base": stage_base, "probe": stage_probe, "train": stage_train,
     "evaluate": stage_evaluate, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
