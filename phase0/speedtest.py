"""Speed test: generation throughput and a first accuracy check, thinking on vs off.

Turns the GPU-hour budget from a guess into a measurement, and informs the thinking
on/off/capped choice recorded in PREREGISTRATION.md before test 1.

Run inside WSL from the repo root, one model per process:
    ~/venvs/research/bin/python phase0/speedtest.py --model Qwen/Qwen3-1.7B
    ~/venvs/research/bin/python phase0/speedtest.py --model Qwen/Qwen3-8B-AWQ
"""
import argparse
import json
import pathlib
import time

import reasoning_gym
from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer
from vllm import LLM, SamplingParams

FAMILIES = ["basic_arithmetic", "leg_counting", "letter_counting", "number_sorting"]

# Qwen3 model-card sampling settings.
SAMPLING = {
    True: dict(temperature=0.6, top_p=0.95, top_k=20),   # thinking on
    False: dict(temperature=0.7, top_p=0.8, top_k=20),   # thinking off
}


def build_prompts(tokenizer, thinking, n_per_family, seed, families):
    prompts, items = [], []
    for family in families:
        dataset = reasoning_gym.create_dataset(family, size=n_per_family, seed=seed)
        for entry in dataset:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPTS["default"]},
                {"role": "user", "content": entry["question"]},
            ]
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking))
            items.append((family, dataset, entry))
    return prompts, items


def run(llm, thinking, args):
    prompts, items = build_prompts(llm.get_tokenizer(), thinking, args.n_per_family, args.seed,
                                   args.families)
    params = SamplingParams(max_tokens=args.max_tokens, seed=args.seed, **SAMPLING[thinking])

    start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    seconds = time.perf_counter() - start

    lengths = [len(o.outputs[0].token_ids) for o in outputs]
    per_family = {}
    for (family, dataset, entry), out in zip(items, outputs):
        answer = extract_answer(out.outputs[0].text)
        score = dataset.score_answer(answer=answer, entry=entry)
        per_family.setdefault(family, []).append(score)

    return {
        "thinking": thinking,
        "prompts": len(prompts),
        "seconds": round(seconds, 1),
        "generated_tokens": sum(lengths),
        "tokens_per_second": round(sum(lengths) / seconds, 1),
        "mean_output_tokens": round(sum(lengths) / len(lengths), 1),
        "hit_max_tokens": sum(o.outputs[0].finish_reason == "length" for o in outputs),
        "accuracy": {f: round(sum(s) / len(s), 3) for f, s in per_family.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-per-family", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--families", nargs="*", default=FAMILIES)
    parser.add_argument("--tag", default="", help="suffix for the output file name")
    args = parser.parse_args()

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_tokens + 1024, seed=args.seed)
    results = {"model": args.model, "families": args.families,
               "runs": [run(llm, thinking, args) for thinking in (False, True)]}

    out_dir = pathlib.Path("runs/speedtest")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (args.model.replace("/", "__") + args.tag + ".json")
    out_file.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
