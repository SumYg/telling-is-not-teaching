"""Report the SWE go/no-go from the evaluate kernel's outputs (kaggle_swe/PREREGISTRATION.md).

    python kaggle_swe/gonogo/report.py runs/swe/gonogo/evaluate

Go if lessons raise rich's resolution rate by at least 5 points and the paired bootstrap 95%
interval is above 0. The bootstrap resamples tasks; each task's gain is the mean over its samples,
as in phase0/test2_lessons.py.
"""
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from gonogo import SAMPLES, ended_by  # noqa: E402

CONDITIONS = ("base", "lessons")
ENDINGS = ("finished", "time", "tool_calls", "turns", "context", "other_error")


def paired_gain(base, treated, rounds=2000, seed=0):
    rng = random.Random(seed)
    n = len(base)
    diffs = [t - b for b, t in zip(base, treated)]
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds))
    return sum(diffs) / n, boots[int(0.025 * rounds)], boots[int(0.975 * rounds) - 1]


def load(root):
    rows = {}
    for condition in CONDITIONS:
        for sample in range(1, SAMPLES + 1):
            path = root / "results" / f"{condition}_s{sample}" / "task_results.jsonl"
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    rows[condition, sample, row["instance_id"]] = row
    return rows


def prompt_tokens_per_call(root, condition):
    tokens = calls = 0
    for sample in range(1, SAMPLES + 1):
        for trace in (root / "results" / f"{condition}_s{sample}" / "traces").glob("*.json"):
            for step in json.loads(trace.read_text()).get("steps") or []:
                metrics = step.get("metrics") or {}
                if step.get("source") == "agent" and metrics.get("prompt_tokens"):
                    tokens += metrics["prompt_tokens"]
                    calls += 1
    return tokens / calls if calls else float("nan")


def report(root):
    root = pathlib.Path(root)
    rows = load(root)
    tasks = sorted({task for _, _, task in rows})
    missing = [(c, s, t) for c in CONDITIONS for s in range(1, SAMPLES + 1) for t in tasks
               if (c, s, t) not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} runs missing, e.g. {missing[:3]}")

    rate = {c: [sum(bool(rows[c, s, t]["resolved"]) for s in range(1, SAMPLES + 1)) / SAMPLES
                for t in tasks] for c in CONDITIONS}
    gain, lo, hi = paired_gain(rate["base"], rate["lessons"])
    go = gain >= 0.05 and lo > 0
    lines = [
        f"Tasks: {len(tasks)}, samples: {SAMPLES}, runs per condition: {len(tasks) * SAMPLES}",
        "",
        "| Condition | Resolved | " + " | ".join(f"Ended: {e}" for e in ENDINGS)
        + " | Mean tool calls | Prompt tokens per model call |",
        "| --- " * (4 + len(ENDINGS)) + "|",
    ]
    for c in CONDITIONS:
        runs = [rows[c, s, t] for s in range(1, SAMPLES + 1) for t in tasks]
        endings = [ended_by(r) for r in runs]
        lines.append(
            f"| {c} | {sum(bool(r['resolved']) for r in runs)} / {len(runs)} "
            f"({100 * sum(rate[c]) / len(tasks):.1f}%) | "
            + " | ".join(str(endings.count(e)) for e in ENDINGS)
            + f" | {sum(r.get('tool_calls') or 0 for r in runs) / len(runs):.1f}"
            f" | {prompt_tokens_per_call(root, c):,.0f} |")
    lines += ["", f"Gain: {100 * gain:+.1f} points, 95% CI [{100 * lo:+.1f}, {100 * hi:+.1f}] "
                  f"(paired bootstrap over tasks, 2000 rounds)",
              f"Outcome: **{'GO' if go else 'NO-GO'}** (needs gain >= 5 points and CI above 0)"]
    return "\n".join(lines), go


if __name__ == "__main__":
    text, _ = report(sys.argv[1])
    print(text)
