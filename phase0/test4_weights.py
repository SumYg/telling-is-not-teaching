"""Test 4 (PREREGISTRATION.md): does the same experience help when it goes into the weights?

    python phase0/test4_weights.py experience  --family survo
    python phase0/test4_weights.py rationalize --family survo   # W2 data from failures
    python phase0/test4_weights.py distill     --family survo   # W1 data (playbook in the prompt)
    python phase0/test4_weights.py train       --family survo --arm w2
    python phase0/test4_weights.py evaluate    --family survo --arm w2 [--eval-family quantum_lock]
    python phase0/test4_weights.py report

Experience, evaluation sets, sampling and solving code are test 2's. Each arm is a LoRA adapter on
Qwen3-1.7B, merged into the base for evaluation with vLLM.
"""
import argparse
import json
import math
import pathlib
import random
import re

from common import CORRECT, SAMPLING, chat, read_jsonl, save_config, score, system_prompt, write_jsonl

LEARNER = "Qwen/Qwen3-1.7B"
ROOT = pathlib.Path("runs/test4")
TEST2 = pathlib.Path("runs/test2")
MAX_LEN = 6656
REFERS_TO_HINT = re.compile(r"hint|given answer|provided answer|answer (?:is )?(?:given|provided)|stated answer", re.I)
HINT = ("\n\nHint: the correct answer is:\n{answer}\n"
        "Work through the problem to arrive at this answer, then give it in the required format.")


def finished(text, length, max_tokens):
    return "</think>" in text and length < max_tokens


def generate(llm, users, seed, args, system=None, n=1):
    from vllm import SamplingParams
    tokenizer = llm.get_tokenizer()
    prompts = [chat(tokenizer, system or system_prompt(), u, True) for u in users]
    params = SamplingParams(n=n, max_tokens=args.max_tokens, seed=seed, **SAMPLING[True])
    return [[(o.text, len(o.token_ids)) for o in out.outputs] for out in llm.generate(prompts, params)]


def load_llm(model, args):
    from vllm import LLM
    return LLM(model=str(model), gpu_memory_utilization=args.gpu_memory_utilization,
               max_model_len=args.max_tokens + 2048, seed=0)


def dataset(family, size, seed):
    import reasoning_gym
    data = reasoning_gym.create_dataset(family, size=size, seed=seed)
    return data, list(data)


# ---- Data stages ----

def stage_experience(args):
    out = ROOT / args.family
    out.mkdir(parents=True, exist_ok=True)
    data, entries = dataset(args.family, args.n_experience, 2000)
    outputs = generate(load_llm(LEARNER, args), [e["question"] for e in entries], 2000, args)
    records = []
    for i, (entry, [(text, length)]) in enumerate(zip(entries, outputs)):
        answer, s = score(data, entry, text)
        records.append({"index": i, "text": text, "length": length, "answer": answer, "score": s,
                        "finished": finished(text, length, args.max_tokens)})
    write_jsonl(out / "experience.jsonl", records)
    save_config(out, "config_experience.json", args)
    print(f"{args.family}: {sum(r['score'] >= CORRECT for r in records)} of {len(records)} correct")


def stage_rationalize(args):
    out = ROOT / args.family
    data, entries = dataset(args.family, args.n_experience, 2000)
    failed = [r for r in read_jsonl(out / "experience.jsonl") if r["score"] < CORRECT]
    users = [entries[r["index"]]["question"] + HINT.format(answer=entries[r["index"]]["answer"]) for r in failed]
    outputs = generate(load_llm(LEARNER, args), users, 2001, args)
    records = []
    for r, [(text, length)] in zip(failed, outputs):
        answer, s = score(data, entries[r["index"]], text)
        mentions = bool(REFERS_TO_HINT.search(text))
        keep = finished(text, length, args.max_tokens) and s >= CORRECT and not mentions
        records.append({"index": r["index"], "text": text, "length": length, "score": s, "keep": keep,
                        "mentions_hint": mentions})
    write_jsonl(out / "rationalized.jsonl", records)
    save_config(out, "config_rationalize.json", args)
    print(f"{args.family}: {sum(r['keep'] for r in records)} of {len(records)} rationalizations kept "
          f"({sum(r['mentions_hint'] for r in records)} mention the hint)")


def stage_distill(args):
    out = ROOT / args.family
    data, entries = dataset(args.family, args.n_experience, 2000)
    playbook = (TEST2 / args.family / "playbook_Qwen__Qwen3-1.7B.txt").read_text().strip()
    outputs = generate(load_llm(LEARNER, args), [e["question"] for e in entries], 2002, args,
                       system=system_prompt(playbook))
    records = []
    for i, (entry, [(text, length)]) in enumerate(zip(entries, outputs)):
        answer, s = score(data, entry, text)
        records.append({"index": i, "text": text, "length": length, "score": s,
                        "keep": finished(text, length, args.max_tokens)})
    write_jsonl(out / "distill.jsonl", records)
    save_config(out, "config_distill.json", args)
    print(f"{args.family}: {sum(r['keep'] for r in records)} of {len(records)} finished with the playbook")


def training_examples(family, arm, n_experience):
    """[(question, completion text)] for an arm, always paired with the plain prompt."""
    _, entries = dataset(family, n_experience, 2000)
    out = ROOT / family
    if arm == "w2":
        rows = [r for r in read_jsonl(out / "experience.jsonl") if r["score"] >= CORRECT and r["finished"]]
        rows += [r for r in read_jsonl(out / "rationalized.jsonl") if r["keep"]]
    else:
        rows = [r for r in read_jsonl(out / "distill.jsonl") if r["keep"]]
    return [(entries[r["index"]]["question"], r["text"]) for r in rows]


# ---- Training ----

def chunked_loss(model, input_ids, labels, chunk=1024):
    """Causal LM loss without materialising the full-vocabulary logits at once."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    base = model.get_base_model()
    hidden = base.model(input_ids=input_ids).last_hidden_state[:, :-1]
    targets = labels[:, 1:]
    count = (targets != -100).sum()

    def piece(h, y):
        logits = base.lm_head(h).float()
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                               ignore_index=-100, reduction="sum")

    total = torch.zeros((), device=input_ids.device)
    for s in range(0, hidden.shape[1], chunk):
        y = targets[:, s:s + chunk]
        if (y != -100).any():
            total = total + checkpoint(piece, hidden[:, s:s + chunk], y, use_reentrant=False)
    return total / count.clamp(min=1)


def stage_train(args):
    out = ROOT / args.family / args.arm
    train_lora(training_examples(args.family, args.arm, args.n_experience), out, args,
               f"{args.family}/{args.arm}")


def train_lora(examples, out, args, label, seed=None):
    """Train one LoRA on [(question, completion)] with the pre-registered settings; save
    adapter, merged model and training log under `out`. `seed` (default None: unchanged behaviour, data
    order seed 0 and an unseeded LoRA init) seeds both the LoRA init and the data order, for retraining
    the same examples (phase0/GATE_SIGNALS.md, replication)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(LEARNER)
    data = []
    for question, completion in examples:
        prompt = chat(tokenizer, system_prompt(), question, True)
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(completion + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
        ids = (p_ids + c_ids)[:MAX_LEN]
        labels = ([-100] * len(p_ids) + c_ids)[:MAX_LEN]
        data.append((torch.tensor([ids]), torch.tensor([labels])))
    print(f"{label}: {len(data)} training examples, "
          f"{sum(d[0].shape[1] for d in data):,} tokens", flush=True)

    model = AutoModelForCausalLM.from_pretrained(LEARNER, dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if seed is not None:
        torch.manual_seed(seed)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(data) * args.epochs / args.accum)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(0.05 * steps)), steps)
    rng = random.Random(0 if seed is None else seed)
    model.train()
    log = []
    for epoch in range(args.epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        running = 0.0
        for i, idx in enumerate(order):
            ids, labels = (t.cuda() for t in data[idx])
            loss = chunked_loss(model, ids, labels)
            (loss / args.accum).backward()
            running += loss.item()
            if (i + 1) % args.accum == 0 or i + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        log.append({"epoch": epoch + 1, "mean_loss": running / len(order)})
        print(f"epoch {epoch + 1}: mean loss {running / len(order):.4f}", flush=True)
    model.save_pretrained(out / "adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(out / "merged")
    tokenizer.save_pretrained(out / "merged")
    (out / "train_log.json").write_text(json.dumps({"examples": len(data), "log": log}, indent=2))
    save_config(out, "config_train.json", args)


# ---- Evaluation and report ----

def stage_evaluate(args):
    family = args.eval_family or args.family
    out = ROOT / args.family / args.arm
    name = "eval.jsonl" if family == args.family else f"eval_on_{family}.jsonl"
    evaluate_model(out / "merged", family, out / name, args, f"{args.family}/{args.arm}")


def evaluate_model(model_dir, family, out_file, args, label):
    """Test 2's evaluation (500 problems x 2 samples, seed 3000) of a merged model."""
    from test2_lessons import solve
    data, entries = dataset(family, args.n_eval, 3000)
    llm = load_llm(model_dir, args)
    records = solve(llm, data, entries, args.samples, True, args.max_tokens, seed=3000)
    write_jsonl(out_file, records)
    acc = sum(sum(s >= CORRECT for s in r["scores"]) / len(r["scores"]) for r in records) / len(records)
    print(f"{label} on {family}: pass@1 {100 * acc:.1f}")


def summary(records, max_tokens):
    samples = [(s, a, n) for r in records for s, a, n in zip(r["scores"], r["answers"], r["lengths"])]
    cut = sum(a is None and n >= max_tokens for _, a, n in samples)
    wrong = sum(s < CORRECT and not (a is None and n >= max_tokens) for s, a, n in samples)
    return {"pass@1": 100 * sum(s >= CORRECT for s, _, _ in samples) / len(samples),
            "cut_off": 100 * cut / len(samples), "finished_wrong": 100 * wrong / len(samples),
            "mean_tokens": sum(n for _, _, n in samples) / len(samples)}


def stage_report(args):
    from test2_lessons import paired_gain
    acc = lambda recs: [sum(s >= CORRECT for s in r["scores"]) / len(r["scores"]) for r in recs]  # noqa: E731
    lines = ["| Family | Condition | pass@1 | Gain vs base | 95% CI | Cut off | Finished wrong | Mean tokens |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    go = False
    for family in ("survo", "quantum_lock"):
        base = read_jsonl(TEST2 / family / "eval_none.jsonl")
        conditions = [("base (test 2)", base),
                      ("playbook (test 2)", read_jsonl(TEST2 / family / "eval_Qwen__Qwen3-1.7B.jsonl"))]
        for arm in ("w1", "w2"):
            path = ROOT / family / arm / "eval.jsonl"
            if path.exists():
                conditions.append((arm.upper(), read_jsonl(path)))
        for name, recs in conditions:
            s = summary(recs, args.max_tokens)
            gain, lo, hi = paired_gain(acc(base), acc(recs)) if name != "base (test 2)" else (0, 0, 0)
            if name == "W2" and 100 * gain >= 5 and lo > 0:
                go = True
            lines.append(f"| {family} | {name} | {s['pass@1']:.1f} | {100 * gain:+.1f} | "
                         f"[{100 * lo:+.1f}, {100 * hi:+.1f}] | {s['cut_off']:.1f}% | "
                         f"{s['finished_wrong']:.1f}% | {s['mean_tokens']:,.0f} |")
    lines.append("")
    for family, other in (("survo", "quantum_lock"), ("quantum_lock", "survo")):
        path = ROOT / family / "w2" / f"eval_on_{other}.jsonl"
        if path.exists():
            base = read_jsonl(TEST2 / other / "eval_none.jsonl")
            gain, lo, hi = paired_gain(acc(base), acc(read_jsonl(path)))
            lines.append(f"- Forgetting: W2 trained on {family}, evaluated on {other}: "
                         f"{100 * gain:+.1f} points [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    for family in ("survo", "quantum_lock"):
        for arm in ("w1", "w2"):
            log = ROOT / family / arm / "train_log.json"
            if log.exists():
                info = json.loads(log.read_text())
                lines.append(f"- {family}/{arm.upper()}: {info['examples']} training examples, "
                             f"loss by epoch {[round(e['mean_loss'], 3) for e in info['log']]}")
    lines.append(f"\nOutcome: **{'GO' if go else 'NO-GO'}** (W2 >= +5 points with CI above 0 on a family)")
    report = "\n".join(lines)
    (ROOT / "report.md").write_text(report + "\n")
    print(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["experience", "rationalize", "distill", "train", "evaluate", "report"])
    parser.add_argument("--family")
    parser.add_argument("--arm", choices=["w1", "w2"])
    parser.add_argument("--eval-family", help="evaluate on another family (forgetting check)")
    parser.add_argument("--n-experience", type=int, default=200)
    parser.add_argument("--n-eval", type=int, default=500)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--root", help="output root (default runs/test4); for dry runs")
    args = parser.parse_args()
    if args.root:
        global ROOT
        ROOT = pathlib.Path(args.root)
    {"experience": stage_experience, "rationalize": stage_rationalize, "distill": stage_distill,
     "train": stage_train, "evaluate": stage_evaluate, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
