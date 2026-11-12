"""SWE go/no-go (kaggle_swe/PREREGISTRATION.md): do in-context lessons transfer across repositories?

Stage "experience" (kernel 1): the base agent attempts the 80 fastapi + requests tasks once, the
model writes one lesson per failure and condenses them to at most 10, then a second server checks
whether a LoRA adapter with random weights changes the output at all.
Stage "evaluate" (kernel 2): the base agent and the base agent plus lessons each attempt the 48
rich tasks twice. The four runs go at once, one task each, so both conditions share the GPU load.
Stage "full": both in one session (one server start, one wait in the L4x4 queue), then the LoRA
check. Evaluation is skipped if lesson writing produced no numbered list.
Stage "lora": only the LoRA check, on the wheelhouse mounted now (the re-check of 2026-10, after the host
reported the reset bug fixed in wheelhouse v23; our first check ran on 2026-09-30).

On Kaggle this runs as a script kernel built by build_kernels.py, which sets STAGE (and LESSONS for
kernel 2). Locally, for debugging with a small Gemma against kaggle_swe/serve_local.sh:
    python kaggle_swe/gonogo/gonogo.py --local --stage full --tasks fastapi_14492 \
        --eval-tasks rich_3278 --minutes 2
"""
import argparse
import asyncio
import concurrent.futures
import glob
import importlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

STAGE = 'lora'  # set by build_kernels.py for Kaggle; --stage locally
LESSONS = None  # kernel 2's playbook, embedded by build_kernels.py; --lessons locally

ON_KAGGLE = Path("/kaggle/input").exists()
TARGET = "gemma-4-31b-it-qat-w4a16-ct"
SOURCE_REPOS = ("fastapi/fastapi", "psf/requests")
EVAL_REPO = "Textualize/rich"
MAX_TIME_MINUTES, MAX_TOOL_CALLS, MAX_TURNS = 6, 100, 100
CONCURRENCY = 4
SAMPLES = 2
SAMPLING_YAML = ("temperature: 0.2\ntop_p: 0.95\nmax_output_tokens: 8192\n"
                 "thinking_config:\n  thinking_budget: 4096\n  include_thoughts: true\n")
MAX_LESSON_CHARS = 1500
LESSONS_HEADER = "## Lessons From Past Tasks"

LESSON_PROMPT = """You are a software engineering agent. You tried to fix an issue in a Python repository and failed.

Issue:
{problem}

What happened: {outcome}

Your patch:
{patch}

End of the test output for your patch:
{tests}

A fix that passes the tests:
{reference}

You get {minutes} minutes and at most {tool_calls} tool calls per issue, and most failed attempts run out of time.
Write ONE general lesson that would help you fix issues in OTHER Python repositories in FEWER steps:
a better order of work, a faster way to find the cause, or something safe to skip.
Do not recommend extra checking or re-verification. Do not mention names of files, functions,
classes or libraries from this repository. At most 80 words. Output only the lesson."""

CONDENSE_PROMPT = """Below are lessons for fixing issues in Python repositories within a tight time budget.

{lessons}

Merge them into at most 10 general, non-overlapping lessons that shorten the path to a correct fix,
most useful first. Drop any lesson that only adds checking or re-verification.
Remove anything specific to a single repository or issue. Output only a numbered list."""


# ---- Pure helpers (unit-tested in tests/test_gonogo.py) ----

def ended_by(row):
    """How an attempt ended: context, time, tool_calls, turns, other_error or finished."""
    error = row.get("error") or ""
    if "ContextWindowExceeded" in error:
        return "context"
    if "timeout" in error.lower():
        return "time"
    if (row.get("tool_calls") or 0) >= MAX_TOOL_CALLS:
        return "tool_calls"
    if (row.get("total_llm_calls") or 0) >= MAX_TURNS:
        return "turns"
    return "other_error" if error else "finished"


def outcome_text(row, has_patch):
    how = {
        "context": "your conversation grew past the model's context window and the run crashed",
        "time": f"you ran out of the {MAX_TIME_MINUTES}-minute time limit",
        "tool_calls": f"you used up all {MAX_TOOL_CALLS} tool calls",
        "turns": f"you used up all {MAX_TURNS} turns",
        "other_error": "the run stopped with an error",
        "finished": "you finished",
    }[ended_by(row)]
    result = "your patch failed the tests" if has_patch else "you produced no patch"
    return f"{how} after {row.get('tool_calls') or 0} tool calls, and {result}."


def clip(text, limit, tail=False):
    text = text.strip()
    if len(text) <= limit:
        return text
    return "[...]\n" + text[-limit:] if tail else text[:limit] + "\n[...]"


def lesson_prompt(task, row, patch, tests):
    return LESSON_PROMPT.format(
        problem=clip(task["problem_statement"], 4000),
        outcome=outcome_text(row, bool(patch.strip())),
        patch=clip(patch, 3000) if patch.strip() else "(no patch)",
        tests=clip(tests, 3000, tail=True) if tests.strip() else "(tests not run: no patch)",
        reference=clip(task["patch"], 4000),
        minutes=MAX_TIME_MINUTES, tool_calls=MAX_TOOL_CALLS)


def numbered(lessons):
    return "\n".join(f"{i + 1}. {lesson}" for i, lesson in enumerate(lessons))


def add_lessons(system_md, lessons):
    return system_md.rstrip() + f"\n\n{LESSONS_HEADER}\n{lessons.strip()}\n"


def pick_tasks(tasks, repos):
    return sorted(t["instance_id"] for t in tasks if t["repo"] in repos)


def build_agent(sample_dir, dest, model_alias=TARGET, lessons=None, minutes=MAX_TIME_MINUTES):
    """The official sample without its placeholder adapters, with our sampling and budget."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(sample_dir, dest, ignore=shutil.ignore_patterns("adapters"))
    for yaml_file in [dest / "agent.yaml", *dest.glob("sub_agents/*.yaml")]:
        text = re.sub(r"(?m)^adapter:.*\n", "", yaml_file.read_text())
        yaml_file.write_text(text.replace(TARGET, model_alias))
    (dest / "configs" / "sampling.yaml").write_text(SAMPLING_YAML)
    (dest / "eval_config.yaml").write_text(
        "evaluation:\n  timeout_seconds: 300\n"
        f"  max_tool_calls: {MAX_TOOL_CALLS}\n  max_time_minutes: {minutes:g}\n"
        f"  max_turns: {MAX_TURNS}\n")
    if lessons:
        system = dest / "prompts" / "system.md"
        system.write_text(add_lessons(system.read_text(), lessons))
    return dest


def usable_playbook(text):
    return bool(text) and re.search(r"(?m)^\s*1[.)]\s+\S", text) is not None


# ---- Harness plumbing ----

def install_wheelhouse():
    """As in the official "Getting Started - Gemma 4 Developer Agent" notebook."""
    os.environ.update({
        "LITELLM_LOCAL_MODEL_COST_MAP": "True", "TRANSFORMERS_NO_TF": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
        "VLLM_ENGINE_READY_TIMEOUT_S": "1200", "VLLM_NO_USAGE_STATS": "1",
        "OTEL_SDK_DISABLED": "true", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    wheelhouse = Path("/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse")
    for pattern in ("/usr/local/lib/python*/dist-packages/*cutlass*.pth",
                    "/usr/local/lib/python*/site-packages/*cutlass*.pth"):
        for pth in glob.glob(pattern):
            try:
                os.unlink(pth)
            except OSError:
                pass
    tmp_whl = Path("/tmp/wheelhouse")
    tmp_whl.mkdir(parents=True, exist_ok=True)
    for w in wheelhouse.glob("*.whl"):
        if "cutlass" in w.name.lower():
            continue
        name = w.name.replace("cu128", "+cu128") if ("cu128" in w.name and "+" not in w.name) else w.name
        if not (tmp_whl / name).exists():
            os.symlink(w, tmp_whl / name)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall",
                    *sorted(str(w) for w in tmp_whl.glob("*.whl"))], check=True)
    importlib.invalidate_caches()


class Setup:
    """Paths, model server and model registry, on Kaggle or locally."""

    def __init__(self, local):
        if local:
            self.data = Path(os.environ.get("GEMMA4_DATA", Path.home() / "data" / "gemma4"))
            self.model_alias = "gemma-4-e2b-it"
        else:
            self.data = Path("/kaggle/input/competitions/gemma-4-developer-agent")
            self.model_alias = TARGET
        self.local = local
        self.server = None
        self.models = None
        self.base_url = "http://127.0.0.1:8000/v1"

    def server_config(self, **overrides):
        import torch
        from adk_submission import VllmConfig
        gpus = torch.cuda.device_count()
        kwargs = dict(
            model="/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2",
            port=8000, host="127.0.0.1", tool_call_parser="gemma4", reasoning_parser="gemma4",
            max_model_len=32768, dtype="bfloat16", gpu_memory_utilization=0.90,
            enable_auto_tool_choice=True, enable_lora=True, max_loras=8, max_lora_rank=128,
            tensor_parallel_size=4 if gpus >= 4 else max(1, gpus), startup_timeout=60 * 20)
        kwargs.update(overrides)
        return VllmConfig(**kwargs)

    def start(self, agent_dir):
        """Start the server (Kaggle) or connect to a running one (local); no-op if started."""
        if self.models is not None:
            return 0.0
        from swegemma.models import setup_gemma_model_registry
        from swegemma.models.discovery import validate_single_declared_model
        declared = validate_single_declared_model(agent_dir)
        if self.local:
            self.models = setup_gemma_model_registry()
            return 0.0
        from adk_submission import VllmServer
        t0 = time.time()
        # No adapters are mounted, so vLLM runs without LoRA, as in the smoke test.
        self.server = VllmServer(self.server_config())
        self.server.start()
        self.models = self.server.create_model_registry(
            aliases=[declared, TARGET], model_prefix="openai/", api_key="EMPTY")
        seconds = time.time() - t0
        print(f"server ready in {seconds:.0f}s", flush=True)
        return seconds

    def stop(self):
        if self.server is not None:
            self.server.stop()
            self.server = None
        self.models = None

    def build_agent(self, dest, lessons=None):
        return build_agent(self.data / "sample_submission", dest, self.model_alias, lessons)

    async def evaluate(self, agent_dir, results_dir, task_ids, concurrency, minutes):
        from adk_submission import discover_adapters
        from google.adk.agents.context_cache_config import ContextCacheConfig
        from google.adk.apps._configs import EventsCompactionConfig
        from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
        from swegemma.evaluate import Evaluator
        limits, constraints = build_submission_limits()
        config = EvalConfig(
            tasks_path=self.data / "tasks.jsonl", snapshots_dir=self.data / "snapshots",
            results_dir=Path(results_dir), submission_dir=Path(agent_dir), models=self.models,
            sandbox="subprocess", timeout_seconds=300, max_time_minutes=minutes,
            max_tool_calls=MAX_TOOL_CALLS, max_turns=MAX_TURNS, task_ids=list(task_ids),
            limits=limits, generation_constraints=constraints,
            adapter_manifest=discover_adapters(str(agent_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS),
            context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
            events_compaction_config=EventsCompactionConfig(
                compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5),
            graph_dir=str(self.data / "graphs"), embeddings_dir=str(self.data / "embeddings"),
            wheels_dir=self.data / "wheels", concurrency=concurrency, display_mode="quiet")
        start = time.time()
        await Evaluator(config).run()
        return time.time() - start

    def client(self):
        from openai import OpenAI
        client = OpenAI(base_url=self.base_url, api_key="EMPTY", timeout=900)
        return client, client.models.list().data[0].id


def load_tasks_jsonl(data):
    return [json.loads(line) for line in (data / "tasks.jsonl").read_text().splitlines() if line.strip()]


def read_rows(results_dir):
    path = Path(results_dir) / "task_results.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_summary(rows, wall):
    counts = {}
    for row in rows:
        counts[ended_by(row)] = counts.get(ended_by(row), 0) + 1
    return {"tasks": len(rows), "resolved": sum(bool(r["resolved"]) for r in rows),
            "ended_by": counts, "wall_seconds": round(wall),
            "mean_tool_calls": round(sum(r.get("tool_calls") or 0 for r in rows) / max(1, len(rows)), 1)}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


# ---- Lessons ----

def generate(setup, prompts, max_tokens=8192, workers=16):
    """One thinking-mode answer per prompt; None where the answer was cut off by max_tokens."""
    client, model = setup.client()

    def one(prompt):
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0.2,
            top_p=0.95, max_tokens=max_tokens, seed=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}})
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        return text if choice.finish_reason == "stop" and text else None

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, prompts))


def condense(setup, lessons, chunk=30):
    """Two levels, as in phase0/test2_lessons.py: chunks of up to 30 become at most 10 each."""
    def merge(groups):
        merged = generate(setup, [CONDENSE_PROMPT.format(lessons=numbered(g)) for g in groups])
        if None in merged:
            print("condense hit the token cap; falling back to the first 10 lessons", flush=True)
        return [m if m is not None else numbered(g[:10]) for m, g in zip(merged, groups)]
    if len(lessons) <= chunk:
        return merge([lessons])[0]
    partial = merge([lessons[i:i + chunk] for i in range(0, len(lessons), chunk)])
    lines = [re.sub(r"^\s*\d+[.)]\s*", "", line) for block in partial for line in block.splitlines()]
    return merge([[line for line in lines if line.strip()]])[0]


def write_lessons(setup, tasks, results_dir, out):
    by_id = {t["instance_id"]: t for t in tasks}
    failures = [r for r in read_rows(results_dir) if not r["resolved"]]
    prompts = []
    for row in failures:
        task_id = row["instance_id"]
        patch_file = Path(results_dir) / "patches" / f"{task_id}.patch"
        tests_file = Path(results_dir) / "test_outputs" / f"{task_id}.log"
        patch = patch_file.read_text(errors="replace") if patch_file.exists() else ""
        tests = tests_file.read_text(errors="replace") if tests_file.exists() else ""
        prompts.append(lesson_prompt(by_id[task_id], row, patch, tests))
    written = generate(setup, prompts)
    records = [{"instance_id": r["instance_id"], "ended_by": ended_by(r), "lesson": w}
               for r, w in zip(failures, written)]
    lessons = [w[:MAX_LESSON_CHARS] for w in written if w]
    print(f"{len(written) - len(lessons)} of {len(written)} lessons dropped (cut off)", flush=True)
    random.Random(0).shuffle(lessons)
    playbook = condense(setup, lessons) if lessons else ""
    write_json(out / "lessons.json", records)
    (out / "lessons.md").write_text(playbook.strip() + "\n")
    (out / "lesson_prompt_example.txt").write_text(prompts[0] if prompts else "")
    print(f"{len(lessons)} lessons -> playbook:\n{playbook}", flush=True)
    return {"failures": len(failures), "lessons": len(lessons), "dropped": len(written) - len(lessons)}


# ---- LoRA check ----

def lora_check(setup, out):
    """Serve an adapter with random lora_A and lora_B; identical temperature-0 logprobs mean
    LoRA has no effect on this vLLM build (forum discussion 743508)."""
    import requests
    import torch
    from adk_submission import VllmServer, discover_adapters
    from safetensors.torch import load_file, save_file
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

    source = setup.data / "sample_submission" / "adapters" / "main_lora"
    sub = out / "lora_check"
    adapter = sub / "adapters" / "random_lora"
    adapter.mkdir(parents=True, exist_ok=True)
    shutil.copy(source / "adapter_config.json", adapter / "adapter_config.json")
    tensors = load_file(str(source / "adapter_model.safetensors"))
    result = {"sample_adapter_max_abs": {k: float(t.abs().max()) for k, t in tensors.items()}}
    generator = torch.Generator().manual_seed(0)
    randomized = {}
    for key, t in tensors.items():
        noise = torch.randn(t.shape, generator=generator)
        randomized[key] = (noise / t.shape[1] ** 0.5 if "lora_A" in key else noise).to(t.dtype)
    save_file(randomized, str(adapter / "adapter_model.safetensors"))

    manifest = discover_adapters(str(sub), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    server = VllmServer(setup.server_config(), adapter_manifest=manifest)
    try:
        server.start()
        result["served_models"] = [m["id"] for m in requests.get(f"{setup.base_url}/models").json()["data"]]
        base = next(m for m in result["served_models"] if m != "random_lora")
        outputs = {}
        for name, model in (("base", base), ("random_lora", "random_lora")):
            response = requests.post(f"{setup.base_url}/chat/completions", json={
                "model": model, "temperature": 0, "max_tokens": 24, "logprobs": True, "top_logprobs": 1,
                "messages": [{"role": "user", "content": "Write a Python function that reverses a string."}],
                "chat_template_kwargs": {"enable_thinking": False}}, timeout=300)
            body = response.json()
            if response.status_code != 200:
                outputs[name] = {"status": response.status_code, "body": body}
                continue
            content = body["choices"][0]["logprobs"]["content"]
            outputs[name] = {"text": body["choices"][0]["message"]["content"],
                             "tokens": [c["token"] for c in content],
                             "logprobs": [c["logprob"] for c in content]}
        result["outputs"] = outputs
        if all("logprobs" in o for o in outputs.values()):
            a, b = outputs["base"], outputs["random_lora"]
            result["identical_tokens"] = a["tokens"] == b["tokens"]
            result["max_abs_logprob_diff"] = max(abs(x - y) for x, y in zip(a["logprobs"], b["logprobs"]))
            result["lora_has_effect"] = not (result["identical_tokens"] and result["max_abs_logprob_diff"] < 1e-6)
    except Exception as error:  # a failed check must not lose the experience results
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        server.stop()
    write_json(out / "lora_check.json", result)
    print("lora check:", json.dumps({k: v for k, v in result.items() if k != "outputs"}), flush=True)
    return result


# ---- Stages ----

def stage_experience(setup, out, task_ids, minutes):
    """The base agent on the source tasks, then lessons. Leaves the server running."""
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks_jsonl(setup.data)
    task_ids = task_ids or pick_tasks(tasks, SOURCE_REPOS)
    summary = {"stage": "experience", "tasks": task_ids}
    agent = setup.build_agent(out / "base_agent")
    summary["server_seconds"] = round(setup.start(agent))
    results = out / "results" / "base"
    wall = asyncio.run(setup.evaluate(agent, results, task_ids, CONCURRENCY, minutes))
    summary["run"] = run_summary(read_rows(results), wall)
    write_json(out / "summary.json", summary)
    print("experience:", json.dumps(summary["run"]), flush=True)

    t0 = time.time()
    summary["write"] = write_lessons(setup, tasks, results, out)
    summary["write"]["seconds"] = round(time.time() - t0)
    write_json(out / "summary.json", summary)


def stage_evaluate(setup, out, task_ids, minutes, lessons):
    """Base vs base plus lessons on the evaluation tasks. Leaves the server running."""
    if not usable_playbook(lessons):
        raise SystemExit("the evaluate stage needs a numbered list of lessons")
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks_jsonl(setup.data)
    task_ids = task_ids or pick_tasks(tasks, (EVAL_REPO,))
    summary = {"stage": "evaluate", "tasks": task_ids, "lessons": lessons}
    agents = {"base": setup.build_agent(out / "base_agent"),
              "lessons": setup.build_agent(out / "lessons_agent", lessons)}
    summary["server_seconds"] = round(setup.start(agents["base"]))
    runs = [(condition, sample) for sample in range(1, SAMPLES + 1) for condition in agents]

    async def all_runs():
        # One task at a time per run and all runs at once: 2 base + 2 lessons tasks in flight.
        return await asyncio.gather(*(
            setup.evaluate(agents[c], out / "results" / f"{c}_s{s}", task_ids, 1, minutes)
            for c, s in runs))

    walls = asyncio.run(all_runs())
    summary["runs"] = {f"{c}_s{s}": run_summary(read_rows(out / "results" / f"{c}_s{s}"), wall)
                       for (c, s), wall in zip(runs, walls)}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary["runs"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["experience", "evaluate", "full", "lora"], default=STAGE)
    parser.add_argument("--local", action="store_true", help="use a running local server")
    parser.add_argument("--tasks", help="comma-separated task ids for the stage (full: experience)")
    parser.add_argument("--eval-tasks", help="comma-separated evaluation task ids (full only)")
    parser.add_argument("--minutes", type=float, default=MAX_TIME_MINUTES)
    parser.add_argument("--lessons", help="playbook file for --stage evaluate (local)")
    parser.add_argument("--out", help="output directory")
    args, _ = parser.parse_known_args()
    if args.stage is None:
        raise SystemExit("--stage is required")
    if ON_KAGGLE and not args.local:
        install_wheelhouse()
    import litellm
    litellm.drop_params = True

    root = Path(args.out or ("/kaggle/working" if ON_KAGGLE else f"runs/swe/gonogo_local/{args.stage}"))
    root.mkdir(parents=True, exist_ok=True)
    setup = Setup(args.local)
    task_ids = args.tasks.split(",") if args.tasks else None
    eval_ids = args.eval_tasks.split(",") if args.eval_tasks else None
    try:
        if args.stage == "lora":
            from importlib.metadata import version
            wheels = sorted(p.name for p in Path("/kaggle/input").glob("**/gemma-4-developer-agent-wheelhouse/*.whl"))
            write_json(root / "versions.json", {"vllm": version("vllm"), "wheels": wheels})
            print("vllm", version("vllm"), "| wheels:", wheels, flush=True)
            lora_check(setup, root)
            return
        if args.stage == "evaluate":
            lessons = Path(args.lessons).read_text() if args.lessons else LESSONS
            stage_evaluate(setup, root, task_ids, args.minutes, lessons)
            return
        experience = root if args.stage == "experience" else root / "experience"
        stage_experience(setup, experience, task_ids, args.minutes)
        if args.stage == "full":
            lessons = (experience / "lessons.md").read_text()
            if usable_playbook(lessons):
                stage_evaluate(setup, root / "evaluate", eval_ids, args.minutes, lessons)
            else:
                print("no usable playbook; evaluation skipped", flush=True)
                write_json(root / "evaluate_skipped.json", {"lessons": lessons})
        setup.stop()
        if not setup.local:
            lora_check(setup, root)
    finally:
        setup.stop()


if __name__ == "__main__":
    main()
