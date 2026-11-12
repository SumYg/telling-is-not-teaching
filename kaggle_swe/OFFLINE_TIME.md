# Offline time test: does the default agent solve more training tasks with 5.5 minutes per task than with 4?

**Drafted 2026-10-29, before anything in it ran. The author asked to use the week's Kaggle GPU quota rather than let it
lapse ("Yeah, I don't want to waste"), after this test was named as its least-bad use. Binding from then on; changes go
in the change log.**

## Why

- **Leaderboard test 2 asks the same question on the hidden tasks** (`kaggle_swe/LEADERBOARD_TIME.md`): the default
  agent with 5.5 minutes per task against 4. It needs 10 scorings of 58 tasks each, one per day.
- **This test asks it on the 129 training tasks,** under the scoring's own conditions, in one night of otherwise unused
  quota. It is a second, independent measurement, not a replacement.
- **Earlier offline runs** ran 4 tasks at once, which slowed each model call to about 18 seconds, a different regime
  from scoring. This one runs one task at a time, as the scorer does.

## Arms

- **Default:** the files of `runs/swe/submissions/base_4min/submission.zip`, as submitted.
- **More time:** the files of `runs/swe/submissions/base_5p5min/submission.zip`, as submitted: the same agent with
  `max_time_minutes: 5.5`.
- **Each arm runs with its own `eval_config.yaml` limits,** as the scorer applies them.

## Design

- **Harness:** the skill test's offline runner (`kaggle_swe/gonogo/gonogo.py`, compare stage), unchanged except for two
  settings: one task at a time (`COMPARE_LANES = 1`), and a half of the tasks per session (`TASK_SHARD`).
- **Tasks:** the 129 training tasks, sorted by id and split into two interleaved halves (even and odd positions). Each
  half runs in its own Kaggle session on 4×L4 (`sumyeung/swe-time-compare-a` and `-b`), and the two sessions run at
  the same time.
- **Within a session,** the two arms alternate task by task (each runs first on half the tasks), so drift over the
  session hits both alike. No new task starts after 10.75 hours, inside Kaggle's 12-hour limit; tasks not reached are
  left out of both arms.
- **Smoke run first:** `sumyeung/swe-time-smoke`, the same code on 3 tasks, checks that the harness, the wheelhouse and
  both arms still run. Its results are discarded.
- **Cost:** about 23 GPU-hours of the week's quota (two sessions of about 11 hours, plus the smoke run). The quota
  resets on Friday.

## Reading, fixed before the runs

- **Main:** more time minus default, the share of tasks solved, paired by task, on the tasks with a result in both
  arms, with a bootstrap 95% interval over tasks (2,000 rounds, seed 0).
  - **"More time solves more"** if the gain is at least 5 points and the interval is above 0: the skill test's line.
- **Reported alongside:** the share of runs that end by the time limit, tool calls per run, discordant pairs, per
  repository, and the tasks not reached.
- **Report:** `python kaggle_swe/time_report.py runs/swe/time_compare/a runs/swe/time_compare/b`.
- **A session that fails** is rerun once and noted.

## Limitations, stated before the results

- **Training tasks, not the hidden ones:** 129 tasks from 4 public repositories (fastapi, rich, requests, httpx).
- **One sample per arm per task.** The go/no-go found outcomes nearly fixed per task (93 of 96 pairs agreed), so the
  comparison rests on the discordant pairs.
- **Not all tasks may be reached** in 10.75 hours: at about 5 and 6.5 minutes per run, a session fits about 56 of its
  65 tasks in both arms.

## Change log

- **2026-10-29:** plan drafted; the three kernels built (`python kaggle_swe/gonogo/build_kernels.py time --smoke`,
  `--shard 0`, `--shard 1`), each with the two zips' files byte for byte.
- **Smoke run passed** (`sumyeung/swe-time-smoke` v1, output in `runs/swe/time_compare/smoke`): the server was ready in
  395 s, and all 6 runs (3 tasks × 2 arms) finished with no harness error, each within its own limit (default about
  250 s, more time about 334 s). One task at a time gave 23 to 54 tool calls per run. Its scores are discarded.
- **Both sessions pushed 2026-10-29** (`sumyeung/swe-time-compare-a` and `-b`, version 1), from commit
  4c50e5e.
- **Both sessions completed**, after 9.2 and 9.0 hours of runs, with all 129 tasks reached in
  both arms and no session failure. Outputs in `runs/swe/time_compare/a` and `b`.
- **Result: +1.6 points, 95% CI [−1.6, +4.7]: not shown** (29 against 27 of 129). Write-up:
  `results/swe_offline_time.md`.

