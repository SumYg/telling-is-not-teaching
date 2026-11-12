"""Diagnostic (phase0/GATE_SIGNALS.md): can the model's own neural signals pick which experiences to consolidate?

    python phase0/gate_signals.py signals [--limit N] [--root DIR]   # per-example signals for test 5's survo pool
    python phase0/gate_signals.py arm --arm grad_top|fam_top|fam_bottom|random_s1
    python phase0/gate_signals.py report

Training and evaluation are test 5's (test 4's LoRA settings, test 2's evaluation); only the chosen 62 examples differ.
"""
import argparse
import json
import pathlib
import random
import re

from common import CORRECT, chat, read_jsonl, system_prompt, write_jsonl
from test4_weights import LEARNER, MAX_LEN, chunked_loss, dataset, evaluate_model, summary, train_lora
from test5_gate import TEST2, boot, per_problem

ROOT = pathlib.Path("runs/gate_signals")
T5 = pathlib.Path("runs/test5/survo")
FAMILY = "survo"
K = 62  # the size of test 5's gate and random arms
ARMS = ("grad_top", "fam_top", "fam_bottom", "random_s1",
        "random_s2", "random_s3", "random_s4", "fam_top_t1", "fam_top_t2")  # the second line: part 2, the replication
RANDOM_SEED = re.compile(r"random_s(\d+)")
RETRAIN_SEED = re.compile(r"fam_top_t(\d+)")
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # test 4's LoRA


def pool():
    """Test 5's R-all pool in its order: [(question, text, n_correct)] (test5_gate.arm_examples)."""
    _, entries = dataset(FAMILY, 200, 2000)
    records = read_jsonl(T5 / "experience.jsonl")
    return [(entries[r["index"]]["question"], a["text"], r["n_correct"])
            for r in records for a in r["attempts"] if a["score"] >= CORRECT and a["finished"]]


def tokens(tokenizer, question, text):
    """The exact ids and labels train_lora builds for one example."""
    prompt = chat(tokenizer, system_prompt(), question, True)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(text + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
    return (p_ids + c_ids)[:MAX_LEN], ([-100] * len(p_ids) + c_ids)[:MAX_LEN]


def stage_signals(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    examples = pool()[:args.limit] if args.limit else pool()
    tokenizer = AutoTokenizer.from_pretrained(LEARNER)
    model = AutoModelForCausalLM.from_pretrained(LEARNER, dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    torch.manual_seed(0)  # fixes LoRA's random A, so every example's gradient uses the same projection
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                                             target_modules=TARGETS))
    model.train()  # gradient checkpointing needs train mode; with no dropout the pass is deterministic
    b_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n]
    print(f"{len(examples)} examples; gradient sketch of {sum(p.numel() for p in b_params):,} LoRA-B values", flush=True)
    rows, grads = [], []
    for i, (question, text, n_correct) in enumerate(examples):
        ids, labels = tokens(tokenizer, question, text)
        model.zero_grad(set_to_none=True)
        loss = chunked_loss(model, torch.tensor([ids]).cuda(), torch.tensor([labels]).cuda())
        loss.backward()
        g = torch.cat([p.grad.detach().float().flatten() for p in b_params])
        norm = g.norm().item()
        if not norm > 0:
            raise RuntimeError(f"example {i}: zero gradient on LoRA B")
        grads.append((g / norm).half().cpu())
        rows.append({"i": i, "n_correct": n_correct, "trained_tokens": sum(t != -100 for t in labels),
                     "mean_logprob": -loss.item(), "grad_norm": norm})
        if i % 25 == 0:
            print(f"  {i}: mean log-prob {-loss.item():.4f}, gradient norm {norm:.4g}", flush=True)
    total = torch.zeros(grads[0].shape, dtype=torch.float32)
    for g in grads:
        total += g.float()
    for row, g in zip(rows, grads):
        g = g.float()
        rest = total - g
        row["agreement"] = float(torch.dot(g, rest) / (g.norm() * rest.norm()).clamp(min=1e-12))
    out = pathlib.Path(args.root)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "signals.jsonl", rows)
    print(f"wrote {out / 'signals.jsonl'}", flush=True)


def chosen(arm, root=ROOT):
    """The 62 pool indices an arm trains on."""
    rows = read_jsonl(pathlib.Path(root) / "signals.jsonl")
    if RANDOM_SEED.fullmatch(arm):
        return sorted(random.Random(int(RANDOM_SEED.fullmatch(arm).group(1))).sample(range(len(rows)), K))
    if RETRAIN_SEED.fullmatch(arm):
        arm = "fam_top"  # the same examples, retrained with another seed
    key = {"grad_top": lambda r: -r["agreement"], "fam_top": lambda r: -r["mean_logprob"],
           "fam_bottom": lambda r: r["mean_logprob"]}[arm]
    return sorted(r["i"] for r in sorted(rows, key=key)[:K])


def stage_arm(args):
    examples = pool()
    idx = chosen(args.arm, args.root)
    out = pathlib.Path(args.root) / args.arm
    out.mkdir(parents=True, exist_ok=True)
    (out / "chosen.json").write_text(json.dumps(idx))
    seed = RETRAIN_SEED.fullmatch(args.arm)
    train_lora([(examples[i][0], examples[i][1]) for i in idx], out, args, f"{FAMILY}/{args.arm}",
               seed=int(seed.group(1)) if seed else None)
    evaluate_model(out / "merged", FAMILY, out / "eval.jsonl", args, f"{FAMILY}/{args.arm}")


def test5_random_indices():
    """Pool indices of test 5's R-random (test5_gate.arm_examples: random.Random(0).sample(pool, len(gated)))."""
    examples = pool()
    gated = [e for e in examples if e[2] <= 2]
    picked = random.Random(0).sample(examples, len(gated))
    return sorted(examples.index(p) for p in picked)


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / ((sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5)


def stage_report(args):
    root = pathlib.Path(args.root)
    rows = read_jsonl(root / "signals.jsonl")
    base = per_problem(read_jsonl(TEST2 / FAMILY / "eval_none.jsonl"))
    ref = per_problem(read_jsonl(T5 / "random" / "eval.jsonl"))
    ref_idx = set(test5_random_indices())
    nc = [r["n_correct"] for r in rows]
    lines = ["# Gate signals diagnostic: report", "",
             f"Signals on {len(rows)} examples. Spearman with the problem's successes out of 8 (how easy it was): "
             f"agreement {spearman([r['agreement'] for r in rows], nc):+.2f}, familiarity "
             f"{spearman([r['mean_logprob'] for r in rows], nc):+.2f}; agreement vs familiarity "
             f"{spearman([r['agreement'] for r in rows], [r['mean_logprob'] for r in rows]):+.2f}.", "",
             "| Arm | Pass@1 | Gain over base [95% CI] | Minus test 5's R-random [95% CI] | Mean successes of chosen problems (of 8) | Shared with R-random |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name, path in (("test 5 R-all (266)", T5 / "all" / "eval.jsonl"), ("test 5 R-random (62)", T5 / "random" / "eval.jsonl"),
                       ("test 5 R-gate (62)", T5 / "gate" / "eval.jsonl")):
        own = per_problem(read_jsonl(path))
        g = boot(base, own, [], [])
        lines.append(f"| {name} | {summary(read_jsonl(path), args.max_tokens)['pass@1']:.1f} | {100 * g[0]:+.1f} "
                     f"[{100 * g[1]:+.1f}, {100 * g[2]:+.1f}] | | | |")
    results = {}
    for arm in ARMS:
        f = root / arm / "eval.jsonl"
        if not f.exists():
            lines.append(f"| {arm} | not run | | | | |")
            continue
        own = per_problem(read_jsonl(f))
        idx = json.loads((root / arm / "chosen.json").read_text())
        g, d = boot(base, own, [], []), boot(ref, own, [], [])
        results[arm] = d
        lines.append(f"| {arm} | {summary(read_jsonl(f), args.max_tokens)['pass@1']:.1f} | {100 * g[0]:+.1f} "
                     f"[{100 * g[1]:+.1f}, {100 * g[2]:+.1f}] | **{100 * d[0]:+.1f} [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}]** | "
                     f"{sum(rows[i]['n_correct'] for i in idx) / len(idx):.2f} | {len(ref_idx & set(idx))} |")
    noisy = "random_s1" in results and abs(100 * results["random_s1"][0]) >= 3
    lines += ["", "## Decision rule (phase0/GATE_SIGNALS.md)", ""]
    for arm in ("grad_top", "fam_top", "fam_bottom"):
        if arm in results:
            d = results[arm]
            passes = 100 * d[0] >= 3 and d[1] > 0
            lines.append(f"- {arm}: {'selects better than random' if passes else 'does not pass'} "
                         f"({100 * d[0]:+.1f} points [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}])"
                         + (" -- but not established: the random reference itself moved by 3 points or more" if passes and noisy else ""))
    if "random_s1" in results:
        lines.append(f"- Noise reference: random_s1 minus test 5's R-random {100 * results['random_s1'][0]:+.1f} points")
    lines += replication_lines(root, args)
    report = "\n".join(lines)
    (root / "report.md").write_text(report + "\n")
    print(report)


def replication_lines(root, args):
    """Part 2's rule (phase0/GATE_SIGNALS.md): both new fam_top runs above all five random picks, and their mean at
    least 3 points above the random picks' mean. The original fam_top run is not counted."""
    new_fam = ["fam_top_t1", "fam_top_t2"]
    rand = ["random_s1", "random_s2", "random_s3", "random_s4"]
    if not all((root / a / "eval.jsonl").exists() for a in new_fam + rand):
        return ["", "## Part 2 (replication): not complete"]
    pass1 = lambda path: summary(read_jsonl(path), args.max_tokens)["pass@1"]  # noqa: E731
    r = [pass1(T5 / "random" / "eval.jsonl")] + [pass1(root / a / "eval.jsonl") for a in rand]
    f = [pass1(root / a / "eval.jsonl") for a in new_fam]
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    sd = (sum((x - mean(r)) ** 2 for x in r) / (len(r) - 1)) ** 0.5
    above, gap = min(f) > max(r), mean(f) - mean(r)
    return ["", "## Part 2 (replication; phase0/GATE_SIGNALS.md)", "",
            f"- Random picks (test 5's, random_s1..s4): {', '.join(f'{x:.1f}' for x in r)}; mean {mean(r):.1f}, SD {sd:.1f}",
            f"- New fam_top runs (t1, t2): {', '.join(f'{x:.1f}' for x in f)}; mean {mean(f):.1f} "
            f"(the original fam_top run, 23.0, is not counted)",
            f"- Both above all five random picks: {'yes' if above else 'no'}; mean gap {gap:+.1f} points (needs +3)",
            f"- **Outcome: familiarity {'selects better than random' if above and gap >= 3 else 'not shown to select better than random'}**"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["signals", "arm", "report"])
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--limit", type=int, help="signals: only the first N pool examples (smoke test)")
    parser.add_argument("--root", default=str(ROOT))
    # test 5's settings (phase0/test5_gate.py)
    parser.add_argument("--n-eval", type=int, default=500)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()
    {"signals": stage_signals, "arm": stage_arm, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
