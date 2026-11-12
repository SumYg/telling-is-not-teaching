"""Early test 1: gap and headroom (see PREREGISTRATION.md).

Samples each candidate Reasoning Gym family and stores per-problem results, so the
report step can compute pass@1, pass@k and bootstrap intervals. Each family is saved as
soon as it finishes, and finished families are skipped on restart.

Run the whole test with `bash phase0/run_test1.sh`: the stage A screen, then stage B one
family at a time. One model per process.
"""
import argparse
import pathlib
import time

import reasoning_gym
from vllm import LLM, SamplingParams

from common import SAMPLING, chat, model_slug, save_config, score, system_prompt, write_jsonl
from families import CANDIDATES


def run_family(llm, family, args):
    dataset = reasoning_gym.create_dataset(family, size=args.n, seed=args.seed)
    tokenizer = llm.get_tokenizer()
    prompts = [chat(tokenizer, system_prompt(), e["question"], args.thinking) for e in dataset]
    params = SamplingParams(n=args.samples, max_tokens=args.max_tokens, seed=args.seed,
                            **SAMPLING[args.thinking])

    start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    seconds = time.perf_counter() - start

    records = []
    for index, (entry, out) in enumerate(zip(dataset, outputs)):
        scores = [score(dataset, entry, s.text)[1] for s in out.outputs]
        records.append({"index": index, "scores": scores,
                        "lengths": [len(s.token_ids) for s in out.outputs],
                        "truncated": sum(s.finish_reason == "length" for s in out.outputs)})
    return records, seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--thinking", type=lambda s: s.lower() in ("1", "true", "on"), default=True)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--families", nargs="*", default=CANDIDATES)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--out", default="runs/test1", help="runs/test1_screen for stage A")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out) / model_slug(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(out_dir, "config.json", args)

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_tokens + 1024, seed=args.seed)

    for family in args.families:
        out_file = out_dir / f"{family}.jsonl"
        if out_file.exists():
            print(f"skip {family} (done)")
            continue
        try:
            records, seconds = run_family(llm, family, args)
        except Exception as error:  # a broken generator should not stop the other families
            print(f"{family}: FAILED ({type(error).__name__}: {error})", flush=True)
            continue
        write_jsonl(out_file, records)
        tokens = sum(sum(r["lengths"]) for r in records)
        print(f"{family}: {seconds:.0f}s, {tokens / seconds:.0f} tok/s", flush=True)


if __name__ == "__main__":
    main()
