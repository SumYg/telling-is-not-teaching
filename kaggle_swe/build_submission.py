"""Build a leaderboard submission.zip from the official sample, the same way the go/no-go builds
its agents (kaggle_swe/gonogo/gonogo.py: no placeholder adapters, max_output_tokens 8192).

    python kaggle_swe/build_submission.py runs/swe/submissions/base_4min --minutes 4
    python kaggle_swe/build_submission.py runs/swe/submissions/lessons_4min --minutes 4 --lessons FILE
    python kaggle_swe/build_submission.py runs/swe/submissions/time_4min --minutes 4 \
        --addendum kaggle_swe/prompts/time_awareness.md
    python kaggle_swe/build_submission.py runs/swe/submissions/budget_5p5min --minutes 5.5 \
        --max-tool-calls 28 --no-graph-tools --no-subagent
    python kaggle_swe/build_submission.py runs/swe/submissions/random_adapter --minutes 5.5 \
        --max-tool-calls 28 --no-graph-tools --no-subagent --random-adapter

Scoring runs the hidden tasks one after another within 12 h, sandbox setup included and patch
validation excluded, so the per-task time limit must leave room for ~120 tasks.

--no-graph-tools drops get_code_neighbors, search_similar_code and get_code_subgraph, and rewords the
one prompt line that sends the agent to search_similar_code. Community audits found these tools match
only exact fully-qualified ids and usually return nothing. --no-subagent drops the code_analyzer
agent tool.

--random-adapter tests whether adapters take effect in scoring. It adds the sample's main_lora with lora_A and lora_B
replaced by random values (the go/no-go LoRA check's recipe) as adapters/random_lora, and routes the root agent to it.
If adapters are applied, the score collapses; if not, it matches the same agent without the adapter.
"""
import argparse
import pathlib
import re
import shutil
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "gonogo"))

from gonogo import build_agent  # noqa: E402

SAMPLE = pathlib.Path.home() / "data" / "gemma4" / "sample_submission"
ALLOWED = {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}
MAX_BYTES = 3 * 1024 ** 3
GRAPH_TOOLS = ("get_code_neighbors", "search_similar_code", "get_code_subgraph")
GRAPH_PROMPT_LINE = ("- If the problem statement does not provide explicit file paths, use `search_similar_code` with keywords "
                     "from the error message to locate relevant files efficiently, rather than running `find` or `grep` "
                     "across the entire repo.")
GREP_PROMPT_LINE = ("- If the problem statement does not provide explicit file paths, search for its most specific "
                    "identifier or error message with `git grep -n \"text\" -- '*.py' | head -20` to locate the relevant "
                    "files, rather than running `find` or `grep` across the entire repo.")
# kaggle_swe/PREREGISTRATION_SKILL.md: the skill arm's and the advice arm's one added prompt line
SKILLS = pathlib.Path(__file__).resolve().parent / "skills"
SKILL_PROMPT_LINES = {
    "code-context": ("- To find code, first call `run_skill_script(skill_name=\"code-context\", file_path=\"scripts/context.py\", "
                     "args=[\"Name1\", \"Name2\"])`. `file_path` is always the skill's own script, `scripts/context.py`, never "
                     "a repository file; put the identifiers or error text from the problem statement in `args`. It returns "
                     "whole definitions with line numbers, their call sites and related tests in one call; use it instead of "
                     "several `grep` and small `read_file` calls."),
}
ADVICE_PROMPT_LINE = ("- Read whole definitions in one call: use `read_file` over 100 or more lines, or the whole file, and do "
                      "not re-read lines you have already seen. Search once with `git grep -n` for the identifier instead of "
                      "repeating searches.")


def replace_once(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise SystemExit(f"{path.name}: expected one {old[:50]!r}, found {text.count(old)}")
    path.write_text(text.replace(old, new))


def add_random_adapter(agent):
    import torch
    from safetensors.torch import load_file, save_file
    source = SAMPLE / "adapters" / "main_lora"
    target = agent / "adapters" / "random_lora"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy(source / "adapter_config.json", target / "adapter_config.json")
    generator = torch.Generator().manual_seed(0)
    tensors = {}
    for key, t in load_file(str(source / "adapter_model.safetensors")).items():
        noise = torch.randn(t.shape, generator=generator)
        tensors[key] = (noise / t.shape[1] ** 0.5 if "lora_A" in key else noise).to(t.dtype)
    save_file(tensors, str(target / "adapter_model.safetensors"))
    replace_once(agent / "agent.yaml", "model: gemma-4-31b-it-qat-w4a16-ct\n",
                 "model: gemma-4-31b-it-qat-w4a16-ct\nadapter: random_lora\n")


def add_skill(agent, name):
    """Copy kaggle_swe/skills/<name> into the submission, register it on the root agent and add its prompt line."""
    shutil.copytree(SKILLS / name, agent / "skills" / name, ignore=shutil.ignore_patterns("__pycache__"))
    replace_once(agent / "agent.yaml", "instruction: !include prompts/system.md\n",
                 f"instruction: !include prompts/system.md\nskills:\n  - skills/{name}\n")
    replace_once(agent / "prompts" / "system.md", GREP_PROMPT_LINE, GREP_PROMPT_LINE + "\n" + SKILL_PROMPT_LINES[name])


def add_advice(agent):
    replace_once(agent / "prompts" / "system.md", GREP_PROMPT_LINE, GREP_PROMPT_LINE + "\n" + ADVICE_PROMPT_LINE)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    p.add_argument("--minutes", type=float, required=True)
    p.add_argument("--lessons", help="numbered list appended to the root agent's system prompt")
    p.add_argument("--addendum", help="markdown appended verbatim to the root agent's system prompt")
    p.add_argument("--max-tool-calls", type=int, help="override the tool-call cap per task")
    p.add_argument("--no-graph-tools", action="store_true", help="drop the three code-graph tools")
    p.add_argument("--no-subagent", action="store_true", help="drop the code_analyzer agent tool")
    p.add_argument("--random-adapter", action="store_true", help="route the root agent to a random-weight adapter")
    p.add_argument("--skill", choices=sorted(SKILL_PROMPT_LINES), help="add a skill from kaggle_swe/skills (needs --no-graph-tools)")
    p.add_argument("--advice", action="store_true", help="add the read-whole-definitions advice line (needs --no-graph-tools)")
    return p


def main():
    args = parser().parse_args()
    out = pathlib.Path(args.out_dir)
    agent = build(args, out / "agent")
    package(out, agent)


def build(args, dest):
    """The agent directory for these options (also used by kaggle_swe/gonogo/build_kernels.py compare)."""
    lessons = pathlib.Path(args.lessons).read_text() if args.lessons else None
    if (args.skill or args.advice) and not args.no_graph_tools:
        raise SystemExit("--skill and --advice add their line after the git grep line, which needs --no-graph-tools")
    agent = build_agent(SAMPLE, dest, lessons=lessons, minutes=args.minutes)
    if args.addendum:
        system = agent / "prompts" / "system.md"
        system.write_text(system.read_text().rstrip() + "\n" + pathlib.Path(args.addendum).read_text())
    if args.max_tool_calls:
        eval_config = agent / "eval_config.yaml"
        eval_config.write_text(re.sub(r"(?m)^(\s*max_tool_calls:) \d+$", rf"\1 {args.max_tool_calls}", eval_config.read_text()))
    if args.no_graph_tools:
        for tool in GRAPH_TOOLS:
            replace_once(agent / "agent.yaml", f"  - {tool}\n", "")
        replace_once(agent / "prompts" / "system.md", GRAPH_PROMPT_LINE, GREP_PROMPT_LINE)
    if args.no_subagent:
        replace_once(agent / "agent.yaml",
                     "  - agent_tool:\n      config_path: sub_agents/code_analyzer.yaml\n      skip_summarization: true\n", "")
        shutil.rmtree(agent / "sub_agents")
        (agent / "prompts" / "analyzer.md").unlink()
    if args.random_adapter:
        add_random_adapter(agent)
    if args.skill:
        add_skill(agent, args.skill)
    if args.advice:
        add_advice(agent)
    return agent


def package(out, agent):
    zip_path = pathlib.Path(shutil.make_archive(str(out / "submission"), "zip", root_dir=agent))
    with zipfile.ZipFile(zip_path) as archive:
        names = [i.filename for i in archive.infolist() if not i.is_dir()]
        size = sum(i.file_size for i in archive.infolist())
    bad = [n for n in names if pathlib.Path(n).suffix.lower() not in ALLOWED]
    if bad or size > MAX_BYTES or "agent.yaml" not in names:
        raise SystemExit(f"invalid archive: disallowed {bad}, size {size}, names {names}")
    print(f"{zip_path}: {len(names)} files, {size:,} bytes")
    for name in sorted(names):
        print("  " + name)


if __name__ == "__main__":
    main()
