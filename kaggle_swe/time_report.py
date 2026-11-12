"""Report the offline time test (kaggle_swe/OFFLINE_TIME.md).

    python kaggle_swe/time_report.py runs/swe/time_compare/a runs/swe/time_compare/b

Reads <dir>/results/<arm>/<task>/task_results.jsonl from each session's output, for the arms default and more_time.
Only tasks with a result in both arms are compared.
- Main: more_time minus default, the share of tasks solved, paired by task, with a bootstrap over tasks (2,000 rounds,
  seed 0). "More time solves more" if the gain is at least 5 points and the 95% interval is above 0.
- Reported alongside: the share of runs that end by the time limit, tool calls per run, discordant pairs, and results
  per repository.
"""
import statistics as st
import sys

from skill_report import load, paired
from gonogo import ended_by

ARMS = ("default", "more_time")


def report(roots):
    arms = {arm: {} for arm in ARMS}
    for root in roots:
        for arm in ARMS:
            arms[arm].update(load(root, arm))
    tasks = sorted(set(arms["default"]) & set(arms["more_time"]))
    solved = {arm: [int(bool(arms[arm][t][0]["resolved"])) for t in tasks] for arm in ARMS}
    lines = [f"Tasks with a result in both arms: {len(tasks)} (default {len(arms['default'])}, more_time "
             f"{len(arms['more_time'])})", "",
             "| Arm | Solved | Ended by the time limit | Mean tool calls |", "| --- | --- | --- | --- |"]
    for arm in ARMS:
        rows = [arms[arm][t][0] for t in tasks]
        lines.append(f"| {arm} | {sum(solved[arm])} ({100 * st.mean(solved[arm]):.1f}%) | "
                     f"{sum(ended_by(r) == 'time' for r in rows)} | {st.mean(r.get('tool_calls') or 0 for r in rows):.1f} |")
    gain = paired(solved["default"], solved["more_time"])
    passes = gain[0] >= 5 and gain[1] > 0
    only = lambda a, b: sum(x and not y for x, y in zip(solved[a], solved[b]))  # noqa: E731
    lines += ["", f"**Main (more_time minus default): {gain[0]:+.1f} points, 95% CI [{gain[1]:+.1f}, {gain[2]:+.1f}]: "
                  f"{'more time solves more' if passes else 'not shown'}** (needs at least +5 with the interval above 0)",
              f"Discordant pairs: solved only with more time {only('more_time', 'default')}, only with the default "
              f"{only('default', 'more_time')}", "", "| Repository | Tasks | Default | More time |", "| --- | --- | --- | --- |"]
    for repo in sorted({arms["default"][t][0].get("repo", "?") for t in tasks}):
        idx = [i for i, t in enumerate(tasks) if arms["default"][t][0].get("repo", "?") == repo]
        lines.append(f"| {repo} | {len(idx)} | {sum(solved['default'][i] for i in idx)} | "
                     f"{sum(solved['more_time'][i] for i in idx)} |")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.path.insert(0, "kaggle_swe/gonogo")
    print(report(sys.argv[1:]))
