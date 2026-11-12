"""Build the Kaggle script kernels of the SWE go/no-go from gonogo.py.

    python kaggle_swe/gonogo/build_kernels.py full        # both stages in one session
    python kaggle_swe/gonogo/build_kernels.py experience
    python kaggle_swe/gonogo/build_kernels.py evaluate --lessons runs/swe/gonogo/experience/lessons.md
    python kaggle_swe/gonogo/build_kernels.py lora        # only the random-adapter LoRA check (sumyeung/swe-lora-recheck)
    python kaggle_swe/gonogo/build_kernels.py compare     # kaggle_swe/PREREGISTRATION_SKILL.md: base, skill, advice
                                                          # (sumyeung/swe-skill-compare; run in WSL, it needs the sample)
    python kaggle_swe/gonogo/build_kernels.py time --shard 0   # kaggle_swe/OFFLINE_TIME.md: the default zip at 4 and
    python kaggle_swe/gonogo/build_kernels.py time --shard 1   # 5.5 minutes, one task at a time, half the tasks each
    python kaggle_swe/gonogo/build_kernels.py time --smoke     # the same on 3 tasks (sumyeung/swe-time-smoke)

Each kernel folder gets a copy of gonogo.py with STAGE (and LESSONS or COMPARE_ARMS) filled in, plus
kernel-metadata.json. Push with:  kaggle kernels push -p kaggle_swe/gonogo/kernels/<stage>
"""
import argparse
import json
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
METADATA = {
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse"],
    "kernel_sources": [],
    "competition_sources": ["gemma-4-developer-agent"],
    "model_sources": ["google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2"],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
    "machine_shape": "NvidiaL4",
}
# kaggle_swe/PREREGISTRATION_SKILL.md: three arms, each the budget-fit agent plus at most one change
BUDGET_FIT = ["--minutes", "5.5", "--max-tool-calls", "28", "--no-graph-tools", "--no-subagent"]
COMPARE = {"base": [], "skill": ["--skill", "code-context"], "advice": ["--advice"]}
# kaggle_swe/OFFLINE_TIME.md: the two submitted zips of leaderboard test 2, byte for byte
TIME_ZIPS = {"default": "runs/swe/submissions/base_4min/submission.zip",
             "more_time": "runs/swe/submissions/base_5p5min/submission.zip"}


def fill(source, stage, lessons=None, arms=None, lanes=None, shard=None):
    """gonogo.py with STAGE, LESSONS, COMPARE_ARMS, COMPARE_LANES and TASK_SHARD set; fails loudly if the placeholders
    moved."""
    for placeholder in ("STAGE = None", "LESSONS = None", "COMPARE_ARMS = None", "COMPARE_LANES = None",
                        "TASK_SHARD = None"):
        if source.count(f"\n{placeholder}  #") != 1:
            raise ValueError(f"placeholder {placeholder!r} not found exactly once")
    source = source.replace("\nSTAGE = None  #", f"\nSTAGE = {stage!r}  #")
    if lessons is not None:
        source = source.replace("\nLESSONS = None  #", f"\nLESSONS = {lessons.strip()!r}  #")
    if arms is not None:
        source = source.replace("\nCOMPARE_ARMS = None  #", f"\nCOMPARE_ARMS = json.loads({json.dumps(arms)!r})  #")
    if lanes is not None:
        source = source.replace("\nCOMPARE_LANES = None  #", f"\nCOMPARE_LANES = {lanes!r}  #")
    if shard is not None:
        source = source.replace("\nTASK_SHARD = None  #", f"\nTASK_SHARD = {tuple(shard)!r}  #")
    return source


def zip_arms():
    """{arm: {relative path: text}} read straight from the submitted zips."""
    import zipfile
    arms = {}
    for arm, rel in TIME_ZIPS.items():
        with zipfile.ZipFile(HERE.parent.parent / rel) as z:
            arms[arm] = {n: z.read(n).decode() for n in sorted(z.namelist()) if not n.endswith("/")}
    return arms


def agent_files(agent):
    return {str(f.relative_to(agent)).replace("\\", "/"): f.read_text()
            for f in sorted(pathlib.Path(agent).rglob("*")) if f.is_file() and "__pycache__" not in f.parts}


def compare_arms():
    """{arm: {relative path: text}}, built by build_submission.py exactly as a submission would be."""
    sys.path.insert(0, str(HERE.parent))
    import build_submission
    arms = {}
    with tempfile.TemporaryDirectory() as tmp:
        for arm, extra in COMPARE.items():
            dest = pathlib.Path(tmp) / arm
            args = build_submission.parser().parse_args([str(dest), *BUDGET_FIT, *extra])
            arms[arm] = agent_files(build_submission.build(args, dest / "agent"))
    return arms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["experience", "evaluate", "full", "lora", "compare", "time"])
    parser.add_argument("--lessons", help="playbook file (required for evaluate)")
    parser.add_argument("--shard", type=int, choices=[0, 1], help="time: which half of the tasks")
    parser.add_argument("--smoke", action="store_true", help="time: 3 tasks only")
    args = parser.parse_args()
    if args.stage == "evaluate" and not args.lessons:
        parser.error("evaluate needs --lessons")
    if args.stage == "time":
        if args.smoke == (args.shard is not None):
            parser.error("time needs exactly one of --shard and --smoke")
        name = "time_smoke" if args.smoke else f"time_{'ab'[args.shard]}"
        out = HERE / "kernels" / name
        out.mkdir(parents=True, exist_ok=True)
        code = out / f"gonogo_{name}.py"
        shard = (0, 64) if args.smoke else (args.shard, 2)  # (0, 64): tasks 0, 64 and 128 of the sorted 129
        code.write_text(fill((HERE / "gonogo.py").read_text(), "compare", arms=zip_arms(), lanes=1, shard=shard))
        slug = "swe-time-smoke" if args.smoke else f"swe-time-compare-{'ab'[args.shard]}"
        metadata = {"id": f"sumyeung/{slug}", "title": "SWE " + slug[4:].replace("-", " "),
                    "code_file": code.name, **METADATA}
        (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"kernel written to {out} (shard {shard}, one task at a time, arms {list(TIME_ZIPS)})")
        return

    lessons = pathlib.Path(args.lessons).read_text() if args.lessons else None
    arms = compare_arms() if args.stage == "compare" else None
    out = HERE / "kernels" / args.stage
    out.mkdir(parents=True, exist_ok=True)
    code = out / f"gonogo_{args.stage}.py"
    code.write_text(fill((HERE / "gonogo.py").read_text(), args.stage, lessons, arms))
    slug, title = {"lora": ("swe-lora-recheck", "SWE LoRA recheck"),
                   "compare": ("swe-skill-compare", "SWE skill compare")}.get(
        args.stage, (f"swe-gonogo-{args.stage}", f"SWE gonogo {args.stage}"))
    metadata = {"id": f"sumyeung/{slug}", "title": title,
                "code_file": code.name, **METADATA}
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"kernel written to {out}" + (f" with arms {list(arms)}" if arms else ""))


if __name__ == "__main__":
    main()
