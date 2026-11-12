"""Early test 3: is there room for a learned controller? Text-only proxy (PREREGISTRATION.md).

Runs the awake/sleep loop (lifetime.py) on invented task families. It sweeps the promotion
threshold tau across lifetime types, then asks whether the best tau depends on the type.

    python phase0/test3_controller.py run      # defaults: the reduced pre-registered design
    python phase0/test3_controller.py report

Run inside WSL from the repo root, after `source scripts/env.sh`. Finished lifetimes are
skipped on restart.
"""
import argparse
import json
import pathlib
import random

from vllm import LLM, SamplingParams

from common import SAMPLING, chat, save_config, system_prompt
from invented import drifting_stream, score_answer
from lifetime import run_lifetime
from reasoning_gym.utils import extract_answer

LEARNER = "Qwen/Qwen3-1.7B"
ROOT = pathlib.Path("runs/test3")
FAMILIES = [chr(ord("A") + i) for i in range(12)]

WRITER_PROMPT = """Here are examples of secret transformation {family}, applied to lists:

{examples}

State the rule of transformation {family} in one or two sentences, precise enough to apply to
any new list. Start with "Transformation {family}:". Output only the rule."""


def lifetime_stream(kind, n, seed):
    """The four pre-registered lifetime types."""
    if kind == "stable_few":
        return drifting_stream("A", n, seed=seed)
    if kind == "drift":
        return drifting_stream("A", n, drift_at=n // 2, seed=seed)
    if kind not in ("stable_many", "mixed"):
        raise ValueError(f"unknown lifetime type: {kind}")
    per = n // len(FAMILIES)
    streams = [drifting_stream(f, per, seed=seed,
                               drift_at=per // 2 if kind == "mixed" and i % 2 else None)
               for i, f in enumerate(FAMILIES)]  # in "mixed", every other family drifts
    return [e for group in zip(*streams) for e in group]  # interleave the families


def make_solver(llm, args):
    tokenizer = llm.get_tokenizer()

    def solver(entries, playbook, seed):
        lessons = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(playbook)) or None
        system = system_prompt(lessons)
        prompts = [chat(tokenizer, system, e["question"], args.thinking) for e in entries]
        params = SamplingParams(n=1, max_tokens=args.max_tokens, seed=seed,
                                **SAMPLING[args.thinking])
        out = []
        for entry, result in zip(entries, llm.generate(prompts, params, use_tqdm=False)):
            answer = extract_answer(result.outputs[0].text)
            out.append((answer, score_answer(answer, entry)))
        return out
    return solver


def make_writer(llm, args):
    tokenizer = llm.get_tokenizer()

    def writer(failures):
        by_family = {}
        for entry, _ in failures:
            by_family.setdefault(entry["metadata"]["family"], []).append(entry)
        prompts, families = [], []
        for family, entries in by_family.items():
            examples = "\n".join(f"{e['metadata']['input']} -> {e['answer']}"
                                 for e in entries[-5:])  # the most recent revealed answers
            prompts.append(chat(tokenizer, "You are a careful problem solver.",
                                WRITER_PROMPT.format(family=family, examples=examples),
                                args.thinking))
            families.append(family)
        params = SamplingParams(n=1, max_tokens=args.max_tokens, seed=0,
                                **SAMPLING[args.thinking])
        outputs = llm.generate(prompts, params, use_tqdm=False)
        lessons = []
        for f, o in zip(families, outputs):
            text = o.outputs[0].text
            if args.thinking and "</think>" not in text:
                continue  # thinking hit the token cap: no usable lesson
            lessons.append((f, text.split("</think>")[-1].strip()[:1500]))
        return lessons
    return writer


def stage_run(args):
    llm = LLM(model=LEARNER, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_tokens + 2048, seed=0)
    solver, writer = make_solver(llm, args), make_writer(llm, args)
    for kind in args.types:
        out_dir = ROOT / kind
        out_dir.mkdir(parents=True, exist_ok=True)
        save_config(out_dir, "config.json", args)
        for seed in args.seeds:
            stream = lifetime_stream(kind, args.n, seed)
            for tau in args.taus:
                out_file = out_dir / f"tau{tau:g}_seed{seed}.json"
                if out_file.exists():
                    print(f"skip {out_file} (done)")
                    continue
                result = run_lifetime(stream, solver, writer, tau=tau,
                                      sleep_every=args.sleep_every, replay_n=args.replay_n,
                                      capacity=args.capacity, seed=seed)
                out_file.write_text(json.dumps({
                    "kind": kind, "tau": tau, "seed": seed, "scores": result.scores,
                    "events": result.events, "replays": result.replays}))
                acc = 100 * sum(result.scores) / len(result.scores)
                print(f"{kind} tau={tau} seed={seed}: accuracy {acc:.1f}, "
                      f"{result.replays} replays", flush=True)


def paired_cost(best_scores, other_scores, rounds=2000, seed=0):
    """Accuracy lost by using another tau, paired by (seed, problem); points and 95% CI."""
    if len(best_scores) != len(other_scores):
        raise ValueError("paired runs must cover the same seeds and problems")
    diffs = [b - o for b, o in zip(best_scores, other_scores)]
    rng = random.Random(seed)
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds))
    return (100 * sum(diffs) / n, 100 * boots[int(0.025 * rounds)],
            100 * boots[int(0.975 * rounds) - 1])


def stage_report(args, root=None):
    root = root or ROOT
    runs = {}
    for path in root.glob("*/tau*_seed*.json"):
        r = json.loads(path.read_text())
        runs.setdefault((r["kind"], r["tau"]), {})[r["seed"]] = r["scores"]
    kinds = sorted({k for k, _ in runs})
    taus = sorted({t for _, t in runs})

    def pooled(kind, tau, seeds=None):  # scores over seeds, in a fixed order for pairing
        per_seed = runs[(kind, tau)]
        return [s for seed in sorted(seeds or per_seed) for s in per_seed[seed]]

    def shared_seeds(kind, a, b):
        return sorted(set(runs[(kind, a)]) & set(runs[(kind, b)]))

    lines = ["| Lifetime type | " + " | ".join(f"tau={t}" for t in taus) + " | Best tau |",
             "| --- |" + " --- |" * (len(taus) + 1)]
    best = {}
    for kind in kinds:
        accs = {t: 100 * sum(pooled(kind, t)) / len(pooled(kind, t))
                for t in taus if (kind, t) in runs}
        best[kind] = max(accs, key=accs.get)
        lines.append(f"| {kind} | " + " | ".join(
            f"{accs[t]:.1f}" if t in accs else "" for t in taus) + f" | {best[kind]} |")

    lines += ["", "| Lifetime type | Using the best tau of | Cost (points) | 95% CI |",
              "| --- | --- | --- | --- |"]
    room = False
    for kind in kinds:
        for other in kinds:
            if best[other] == best[kind] or (kind, best[other]) not in runs:
                continue
            seeds = shared_seeds(kind, best[kind], best[other])
            if not seeds:
                continue
            cost, lo, hi = paired_cost(pooled(kind, best[kind], seeds),
                                       pooled(kind, best[other], seeds))
            room |= cost >= 3 and lo > 0
            lines.append(f"| {kind} | {other} (tau={best[other]}) | {cost:+.1f} "
                         f"| [{lo:+.1f}, {hi:+.1f}] |")
    lines += ["", f"Room for a learned controller: {'yes' if room else 'no'} "
                  "(the best tau differs, and using another type's best costs >= 3 points "
                  "with the interval above 0)"]
    report = "\n".join(lines)
    (root / "report.md").write_text(report + "\n")
    print(report)
    return room


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["run", "report"])
    # Defaults are the reduced design in PREREGISTRATION.md (2026-09-28).
    parser.add_argument("--types", nargs="*", default=["stable_few", "drift"])
    parser.add_argument("--taus", nargs="*", type=float, default=[0, 5, 10])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--sleep-every", type=int, default=25)
    parser.add_argument("--replay-n", type=int, default=12)
    parser.add_argument("--capacity", type=int, default=10)
    parser.add_argument("--thinking", type=lambda s: s.lower() in ("1", "true", "on"), default=True)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()
    {"run": stage_run, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
