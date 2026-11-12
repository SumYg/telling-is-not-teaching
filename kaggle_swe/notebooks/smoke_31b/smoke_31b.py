"""Kaggle smoke run: the official sample agent with Gemma 4 31B on 4xL4.

Measures time per task, tool calls and context-window errors at two levels of parallelism,
to set the budgets in kaggle_swe/PREREGISTRATION.md. Setup follows the official
"Getting Started - Gemma 4 Developer Agent" notebook (wheelhouse install, vLLM server,
Evaluator with event compaction and the subprocess sandbox).

Run A: 8 tasks, 4 in parallel. Run B: 8 other tasks, 8 in parallel. 6 minutes per task.
"""
import glob
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.update({
    "LITELLM_LOCAL_MODEL_COST_MAP": "True", "TRANSFORMERS_NO_TF": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
    "VLLM_ENGINE_READY_TIMEOUT_S": "1200", "VLLM_NO_USAGE_STATS": "1",
    "OTEL_SDK_DISABLED": "true", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
})

# ---- 1. Install the harness from the wheelhouse (as in the official notebook) ----
WHEELHOUSE_DIR = Path("/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse")
for pattern in ("/usr/local/lib/python*/dist-packages/*cutlass*.pth",
                "/usr/local/lib/python*/site-packages/*cutlass*.pth"):
    for pth in glob.glob(pattern):
        try:
            os.unlink(pth)
        except OSError:
            pass
tmp_whl = Path("/tmp/wheelhouse")
tmp_whl.mkdir(parents=True, exist_ok=True)
for w in WHEELHOUSE_DIR.glob("*.whl"):
    if "cutlass" in w.name.lower():
        continue
    name = w.name.replace("cu128", "+cu128") if ("cu128" in w.name and "+" not in w.name) else w.name
    if not (tmp_whl / name).exists():
        os.symlink(w, tmp_whl / name)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall",
                *sorted(str(w) for w in tmp_whl.glob("*.whl"))], check=True)
importlib.invalidate_caches()

import asyncio  # noqa: E402

import litellm  # noqa: E402
import torch  # noqa: E402
from adk_submission import VllmConfig, VllmServer, discover_adapters  # noqa: E402
from google.adk.agents.context_cache_config import ContextCacheConfig  # noqa: E402
from google.adk.apps._configs import EventsCompactionConfig  # noqa: E402
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits  # noqa: E402
from swegemma.evaluate import Evaluator  # noqa: E402
from swegemma.models import load_tasks  # noqa: E402
from swegemma.models.discovery import validate_single_declared_model  # noqa: E402

litellm.drop_params = True
DATA_DIR = Path("/kaggle/input/competitions/gemma-4-developer-agent")
WORKING = Path("/kaggle/working")
MODEL_PATH = Path("/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2")
TARGET = "gemma-4-31b-it-qat-w4a16-ct"

# ---- 2. Base agent: the official sample without its placeholder adapters ----
AGENT_DIR = WORKING / "base_agent"
if AGENT_DIR.exists():
    shutil.rmtree(AGENT_DIR)
shutil.copytree(DATA_DIR / "sample_submission", AGENT_DIR, ignore=shutil.ignore_patterns("adapters"))
for yaml_file in [AGENT_DIR / "agent.yaml", *AGENT_DIR.glob("sub_agents/*.yaml")]:
    yaml_file.write_text(re.sub(r"(?m)^adapter:.*\n", "", yaml_file.read_text()))
(AGENT_DIR / "configs" / "sampling.yaml").write_text(
    "temperature: 0.2\ntop_p: 0.95\nmax_output_tokens: 16384\n"
    "thinking_config:\n  thinking_budget: 4096\n  include_thoughts: true\n")

# ---- 3. vLLM server on 4xL4 ----
t0 = time.time()
declared = validate_single_declared_model(AGENT_DIR)
adapters = discover_adapters(str(AGENT_DIR), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
gpus = torch.cuda.device_count()
server = VllmServer(VllmConfig(
    model=str(MODEL_PATH), port=8000, host="127.0.0.1", tool_call_parser="gemma4",
    reasoning_parser="gemma4", max_model_len=32768, dtype="bfloat16",
    gpu_memory_utilization=0.90, enable_auto_tool_choice=True, enable_lora=True, max_loras=8,
    max_lora_rank=128, tensor_parallel_size=4 if gpus >= 4 else max(1, gpus),
    startup_timeout=60 * 20), adapter_manifest=adapters)
server.start()
models = server.create_model_registry(aliases=[declared, TARGET], model_prefix="openai/",
                                      api_key="EMPTY")
server_seconds = time.time() - t0
print(f"server ready in {server_seconds:.0f}s on {gpus} GPUs", flush=True)

# ---- 4. Two runs at different parallelism ----
tasks = load_tasks(DATA_DIR / "tasks.jsonl")
by_repo = {}
for t in sorted(tasks, key=lambda t: t.instance_id):
    by_repo.setdefault(t.repo, []).append(t.instance_id)
pick = lambda repo, a, b: by_repo[repo][a:b]  # noqa: E731
RUNS = {
    "A_conc4": (4, pick("fastapi/fastapi", 0, 3) + pick("Textualize/rich", 0, 3) + pick("psf/requests", 0, 2)),
    "B_conc8": (8, pick("fastapi/fastapi", 3, 6) + pick("Textualize/rich", 3, 6) + pick("psf/requests", 2, 4)),
}
limits, constraints = build_submission_limits()
summary = {"server_seconds": server_seconds, "runs": {}}
for name, (concurrency, ids) in RUNS.items():
    config = EvalConfig(
        tasks_path=DATA_DIR / "tasks.jsonl", snapshots_dir=DATA_DIR / "snapshots",
        results_dir=WORKING / "results" / name, submission_dir=AGENT_DIR, models=models,
        sandbox="subprocess", timeout_seconds=300, max_time_minutes=6, max_tool_calls=100,
        max_turns=100, task_ids=ids, limits=limits, generation_constraints=constraints,
        adapter_manifest=adapters,
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5),
        graph_dir=str(DATA_DIR / "graphs"), embeddings_dir=str(DATA_DIR / "embeddings"),
        wheels_dir=DATA_DIR / "wheels", concurrency=concurrency, display_mode="quiet")
    start = time.time()
    asyncio.run(Evaluator(config).run())
    wall = time.time() - start
    rows = [json.loads(l) for l in (WORKING / "results" / name / "task_results.jsonl").read_text().splitlines() if l]
    summary["runs"][name] = {
        "concurrency": concurrency, "tasks": len(rows), "wall_seconds": round(wall),
        "resolved": sum(r["resolved"] for r in rows),
        "context_errors": sum("ContextWindowExceeded" in (r.get("error") or "") for r in rows),
        "other_errors": sum(bool(r.get("error")) and "ContextWindowExceeded" not in r["error"] for r in rows),
        "mean_task_seconds": round(sum(r["duration_seconds"] for r in rows) / max(1, len(rows))),
        "mean_tool_calls": round(sum(r.get("tool_calls") or 0 for r in rows) / max(1, len(rows)), 1),
    }
    print(name, json.dumps(summary["runs"][name]), flush=True)

(WORKING / "smoke_summary.json").write_text(json.dumps(summary, indent=2))
server.stop()
print(json.dumps(summary, indent=2))
