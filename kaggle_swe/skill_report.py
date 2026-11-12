"""Report the Gemma skill test (kaggle_swe/PREREGISTRATION_SKILL.md).

    python kaggle_swe/skill_report.py runs/swe/skill_compare

Reads <dir>/results/<arm>/<task>/task_results.jsonl and the traces next to them, for the arms base, skill and advice.
Only tasks with a result in all three arms are compared.
- Main: skill minus base, the share of tasks solved, paired by task, with a bootstrap over tasks (2,000 rounds, seed 0).
  Go if the gain is at least 5 points and the 95% interval is above 0.
- Tool or advice: skill minus advice, the same statistic and line.
- Reported alongside: advice minus base; the share of runs finished in time; tool calls per run and calls before the first
  edit; skill use and errors; per-repository results.
"""
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "gonogo"))
from gonogo import ended_by, skill_call_failed  # noqa: E402

ARMS = ("base", "skill", "advice")
EDITS = ("edit_file", "write_file")


def load(root, arm):
    """{task: (row, trace steps or None)} for one arm."""
    out = {}
    for path in sorted(Path(root, "results", arm).glob("*/task_results.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        task = path.parent.name
        trace = next(path.parent.glob("traces/trace_*.json"), None)
        steps = json.loads(trace.read_text()).get("steps", []) if trace else None
        out[task] = (rows[0], steps)
    return out


def calls_before_first_edit(steps):
    if not steps:
        return None
    n = 0
    for step in steps:
        for call in step.get("tool_calls") or []:
            if call.get("function_name") in EDITS:
                return n
            n += 1
    return None


def skill_calls(steps):
    calls = errors = 0
    for step in steps or []:
        for call in step.get("tool_calls") or []:
            if call.get("function_name") == "run_skill_script":
                calls += 1
                errors += skill_call_failed(step.get("observation"))
    return calls, errors


def paired(a, b, rounds=2000, seed=0):
    """Mean of b - a in points, with a bootstrap 95% interval over the paired items."""
    d = [100 * (y - x) for x, y in zip(a, b)]
    rng = random.Random(seed)
    stats = sorted(st.mean(d[rng.randrange(len(d))] for _ in d) for _ in range(rounds))
    return st.mean(d), stats[int(0.025 * rounds)], stats[int(0.975 * rounds) - 1]


def verdict(gain):
    return "**GO**" if gain[0] >= 5 and gain[1] > 0 else "NO-GO"


def report(root):
    arms = {arm: load(root, arm) for arm in ARMS}
    tasks = sorted(set.intersection(*(set(a) for a in arms.values())))
    missing = {arm: len(a) - len(tasks) for arm, a in arms.items()}
    solved = {arm: [int(bool(arms[arm][t][0]["resolved"])) for t in tasks] for arm in ARMS}
    lines = [f"Tasks with a result in all three arms: {len(tasks)} (extra results not compared: {missing})", "",
             "| Arm | Solved | Finished in time | Mean tool calls | Calls before the first edit (median) | Runs that never edit |",
             "| --- | --- | --- | --- | --- | --- |"]
    for arm in ARMS:
        rows = [arms[arm][t][0] for t in tasks]
        firsts = [calls_before_first_edit(arms[arm][t][1]) for t in tasks]
        known = [f for f in firsts if f is not None]
        never = sum(1 for t, f in zip(tasks, firsts) if f is None and arms[arm][t][1] is not None)
        lines.append(f"| {arm} | {sum(solved[arm])} ({100 * st.mean(solved[arm]):.1f}%) | "
                     f"{sum(ended_by(r) == 'finished' for r in rows)} | {st.mean(r.get('tool_calls') or 0 for r in rows):.1f} | "
                     f"{st.median(known) if known else 'n/a'} | {never} |")
    main, tool, advice = (paired(solved["base"], solved["skill"]), paired(solved["advice"], solved["skill"]),
                          paired(solved["base"], solved["advice"]))
    both = lambda a, b: sum(x == 1 and y == 0 for x, y in zip(solved[a], solved[b]))  # noqa: E731
    lines += ["",
              f"**Main (skill minus base): {main[0]:+.1f} points, 95% CI [{main[1]:+.1f}, {main[2]:+.1f}]: {verdict(main)}**",
              f"Tool or advice (skill minus advice): {tool[0]:+.1f} points, 95% CI [{tool[1]:+.1f}, {tool[2]:+.1f}]: "
              f"{'the tool matters' if verdict(tool) == '**GO**' else 'not shown to matter'}",
              f"Advice minus base: {advice[0]:+.1f} points, 95% CI [{advice[1]:+.1f}, {advice[2]:+.1f}]",
              f"Discordant pairs: skill-only {both('skill', 'base')}, base-only {both('base', 'skill')}; "
              f"skill-only vs advice {both('skill', 'advice')}, advice-only {both('advice', 'skill')}"]
    uses = [skill_calls(arms["skill"][t][1]) for t in tasks]
    used = sum(c > 0 for c, _ in uses)
    lines.append(f"Skill use: called in {used} of {len(tasks)} runs, {sum(c for c, _ in uses)} calls, "
                 f"{sum(e for _, e in uses)} returned errors")
    repos = sorted({arms["base"][t][0].get("repo", "?") for t in tasks})
    lines += ["", "| Repository | Tasks | Base | Skill | Advice |", "| --- | --- | --- | --- | --- |"]
    for repo in repos:
        idx = [i for i, t in enumerate(tasks) if arms["base"][t][0].get("repo", "?") == repo]
        lines.append(f"| {repo} | {len(idx)} | " + " | ".join(str(sum(solved[a][i] for i in idx)) for a in ARMS) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(sys.argv[1]))
