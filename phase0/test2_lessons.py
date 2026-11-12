"""Early test 2: do the learner's own lessons help it on OTHER problems? (PREREGISTRATION.md)

Four stages, each its own process so only one model is on the GPU at a time:
    experience  the learner attempts the experience problems (seed 2000), one sample each
    write       a writer model turns each failure into a lesson, then condenses them
    evaluate    the learner solves new problems (seed 3000) with and without each playbook
    report      pass@1 per condition, with a paired bootstrap interval for each gain

Run inside WSL from the repo root, after `source scripts/env.sh`:
    python phase0/test2_lessons.py experience --family countdown
    python phase0/test2_lessons.py write --family countdown --writer Qwen/Qwen3-1.7B
    python phase0/test2_lessons.py write --family countdown --writer Qwen/Qwen3-8B-AWQ
    python phase0/test2_lessons.py evaluate --family countdown
    python phase0/test2_lessons.py report
"""
import argparse
import json
import pathlib
import random

import reasoning_gym
from vllm import LLM, SamplingParams

from common import (CORRECT, SAMPLING, chat, model_slug, read_jsonl, save_config, score,
                    system_prompt, write_jsonl)

LEARNER = "Qwen/Qwen3-1.7B"
ROOT = pathlib.Path("runs/test2")
MAX_LESSON_CHARS = 1500  # a lesson is at most 80 words; this only guards against runaway text

LESSON_PROMPT = """You tried to solve a problem and got it wrong.

Problem:
{question}

Your final answer: {answer}
Correct answer: {correct}

Write ONE general lesson that would help solve OTHER problems of this type, not just this one.
At most 80 words. Do not mention specific numbers, names or values from this problem.
Output only the lesson."""

CONDENSE_PROMPT = """Below are lessons written after mistakes on problems of the same type.

{lessons}

Merge them into at most 10 general, non-overlapping lessons, most useful first.
Remove anything specific to a single problem. Output only a numbered list."""

# Test 2b (PREREGISTRATION.md): the writer learns whether an attempt ran out of budget, and is
# asked for lessons that shorten the path to a correct answer.
EFFICIENT_LESSON_PROMPT = """You tried to solve a problem and got it wrong.

Problem:
{question}

What happened: {outcome}
Correct answer: {correct}

You have about {budget} tokens of reasoning per problem, and most failed attempts run out of it.
Write ONE general lesson that would help reach the correct answer on OTHER problems of this type
in FEWER reasoning steps: a shortcut, a better order of work, or something safe to skip.
Do not recommend extra checking or re-verification. At most 80 words.
Do not mention specific numbers, names or values from this problem. Output only the lesson."""

EFFICIENT_CONDENSE_PROMPT = """Below are lessons for solving problems of one type within a tight reasoning budget.

{lessons}

Merge them into at most 10 general, non-overlapping lessons that shorten the path to a correct
answer, most useful first. Drop any lesson that only adds checking or re-verification.
Remove anything specific to a single problem. Output only a numbered list."""

VARIANT_ROOTS = {"thorough": pathlib.Path("runs/test2"), "efficient": pathlib.Path("runs/test2b")}


def outcome_text(record, max_tokens):
    """What happened on a failed attempt, as the efficient-variant writer sees it."""
    answer, length = record["answers"][0], record["lengths"][0]
    if answer is None and length >= max_tokens:
        return "your reasoning ran out of the token budget before you gave an answer"
    if answer is None:
        return "you finished without giving an answer in the required format"
    return f"you answered {answer}, which is wrong"


def load_llm(model, args):
    return LLM(model=model, gpu_memory_utilization=args.gpu_memory_utilization,
               max_model_len=args.max_tokens + 2048, seed=0)


def solve(llm, dataset, entries, samples, thinking, max_tokens, seed, playbook=None):
    tokenizer = llm.get_tokenizer()
    system = system_prompt(playbook)
    prompts = [chat(tokenizer, system, e["question"], thinking) for e in entries]
    params = SamplingParams(n=samples, max_tokens=max_tokens, seed=seed, **SAMPLING[thinking])
    records = []
    for index, (entry, out) in enumerate(zip(entries, llm.generate(prompts, params))):
        answers, scores = zip(*(score(dataset, entry, s.text) for s in out.outputs))
        records.append({"index": index, "answers": list(answers), "scores": list(scores),
                        "lengths": [len(s.token_ids) for s in out.outputs]})
    return records


def stage_experience(args):
    dataset = reasoning_gym.create_dataset(args.family, size=args.n_experience, seed=2000)
    llm = load_llm(LEARNER, args)
    records = solve(llm, dataset, list(dataset), 1, args.thinking, args.max_tokens, seed=2000)
    out_dir = ROOT / args.family
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(out_dir, "config_experience.json", args)
    write_jsonl(out_dir / "experience.jsonl", records)
    failures = sum(r["scores"][0] < CORRECT for r in records)
    print(f"{args.family}: {failures} failures out of {len(records)}")


def generate_texts(llm, users, thinking, max_tokens):
    tokenizer = llm.get_tokenizer()
    prompts = [chat(tokenizer, "You are a careful problem solver.", u, thinking) for u in users]
    params = SamplingParams(n=1, max_tokens=max_tokens, seed=0, **SAMPLING[thinking])
    texts = []
    for out in llm.generate(prompts, params):
        text = out.outputs[0].text
        if thinking and "</think>" not in text:
            texts.append(None)  # thinking hit the token cap: no usable answer
            continue
        texts.append(text.split("</think>")[-1].strip())
    return texts


def condense(llm, lessons, thinking, max_tokens, chunk=30, prompt=CONDENSE_PROMPT):
    """Two levels: chunks of up to 30 lessons become at most 10 each, then one final merge."""
    def merge(group):
        numbered = "\n".join(f"{i + 1}. {lesson}" for i, lesson in enumerate(group))
        merged = generate_texts(llm, [prompt.format(lessons=numbered)], thinking, max_tokens)[0]
        if merged is None:  # condensing ran out of tokens: keep the first 10 lessons as they are
            print("condense hit the token cap; falling back to the first 10 lessons")
            merged = "\n".join(f"{i + 1}. {lesson}" for i, lesson in enumerate(group[:10]))
        return merged
    if len(lessons) <= chunk:
        return merge(lessons)
    partial = [merge(lessons[i:i + chunk]) for i in range(0, len(lessons), chunk)]
    return merge([line for block in partial for line in block.splitlines() if line.strip()])


def stage_write(args):
    out_dir = ROOT / args.family
    dataset = reasoning_gym.create_dataset(args.family, size=args.n_experience, seed=2000)
    entries = list(dataset)
    failures = [r for r in read_jsonl(out_dir / "experience.jsonl") if r["scores"][0] < CORRECT]
    if any(entries[r["index"]]["answer"] is None for r in failures):
        raise SystemExit(f"{args.family} stores no correct answer; lessons need a revealed answer")

    llm = load_llm(args.writer, args)
    efficient = getattr(args, "variant", "thorough") == "efficient"
    if efficient:
        users = [EFFICIENT_LESSON_PROMPT.format(
                     question=entries[r["index"]]["question"], outcome=outcome_text(r, args.max_tokens),
                     correct=entries[r["index"]]["answer"], budget=f"{args.max_tokens:,}")
                 for r in failures]
    else:
        users = [LESSON_PROMPT.format(question=entries[r["index"]]["question"],
                                      answer=r["answers"][0], correct=entries[r["index"]]["answer"])
                 for r in failures]
    written = generate_texts(llm, users, args.thinking, args.max_tokens)
    lessons = [lesson[:MAX_LESSON_CHARS] for lesson in written if lesson]  # cap lessons, not the playbook
    print(f"{args.family}: {len(written) - len(lessons)} of {len(written)} lessons dropped "
          "(thinking hit the token cap)")
    random.Random(0).shuffle(lessons)
    playbook = condense(llm, lessons, args.thinking, args.max_tokens,
                        prompt=EFFICIENT_CONDENSE_PROMPT if efficient else CONDENSE_PROMPT)

    slug = model_slug(args.writer)
    save_config(out_dir, f"config_write_{slug}.json", args)
    (out_dir / f"lessons_{slug}.json").write_text(json.dumps(lessons, indent=2))
    (out_dir / f"playbook_{slug}.txt").write_text(playbook + "\n")
    print(f"{args.family}: {len(lessons)} lessons -> playbook\n{playbook}")


def stage_evaluate(args):
    out_dir = ROOT / args.family
    dataset = reasoning_gym.create_dataset(args.family, size=args.n_eval, seed=3000)
    entries = list(dataset)
    llm = load_llm(LEARNER, args)
    save_config(out_dir, "config_evaluate.json", args)
    conditions = {"none": None}
    for path in sorted(out_dir.glob("playbook_*.txt")):
        conditions[path.stem.removeprefix("playbook_")] = path.read_text().strip()
    for name, playbook in conditions.items():
        out_file = out_dir / f"eval_{name}.jsonl"
        if out_file.exists():
            print(f"skip {name} (done)")
            continue
        # Same problems and the same sampling seed in every condition: a paired comparison.
        records = solve(llm, dataset, entries, args.samples, args.thinking, args.max_tokens,
                        seed=3000, playbook=playbook)
        write_jsonl(out_file, records)
        acc = sum(sum(s >= CORRECT for s in r["scores"]) / len(r["scores"]) for r in records)
        print(f"{args.family} / {name}: pass@1 {100 * acc / len(records):.1f}")


def paired_gain(base, treated, rounds=2000, seed=0):
    rng = random.Random(seed)
    n = len(base)
    diffs = [t - b for b, t in zip(base, treated)]
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds))
    return sum(diffs) / n, boots[int(0.025 * rounds)], boots[int(0.975 * rounds) - 1]


def stage_report(args):
    lines = ["| Family | Playbook by | Without (pass@1) | With (pass@1) | Gain (points) | 95% CI |",
             "| --- | --- | --- | --- | --- | --- |"]
    for family_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        base_file = family_dir / "eval_none.jsonl"
        if not base_file.exists():
            continue
        acc = lambda recs: [sum(s >= CORRECT for s in r["scores"]) / len(r["scores"]) for r in recs]
        base = acc(read_jsonl(base_file))
        for path in sorted(family_dir.glob("eval_*.jsonl")):
            if path.stem == "eval_none":
                continue
            treated = acc(read_jsonl(path))
            gain, lo, hi = paired_gain(base, treated)
            lines.append(f"| {family_dir.name} | {path.stem.removeprefix('eval_')} "
                         f"| {100 * sum(base) / len(base):.1f} | {100 * sum(treated) / len(treated):.1f} "
                         f"| {100 * gain:+.1f} | [{100 * lo:+.1f}, {100 * hi:+.1f}] |")
    report = "\n".join(lines)
    (ROOT / "report.md").write_text(report + "\n")
    print(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["experience", "write", "evaluate", "report"])
    parser.add_argument("--family")
    parser.add_argument("--writer", default=LEARNER)
    parser.add_argument("--n-experience", type=int, default=200)
    parser.add_argument("--n-eval", type=int, default=500)
    parser.add_argument("--samples", type=int, default=2)  # PREREGISTRATION.md, 2026-09-28
    parser.add_argument("--thinking", type=lambda s: s.lower() in ("1", "true", "on"), default=True)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--variant", choices=sorted(VARIANT_ROOTS), default="thorough",
                        help="efficient = test 2b's budget-aware lesson prompts, in runs/test2b")
    args = parser.parse_args()
    global ROOT
    ROOT = VARIANT_ROOTS[args.variant]
    {"experience": stage_experience, "write": stage_write,
     "evaluate": stage_evaluate, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
