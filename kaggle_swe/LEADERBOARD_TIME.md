# Leaderboard test 2: does more time per task raise the default agent's hidden score?

**A measurement on the hidden set. Drafted 2026-10-25, before any submission of the new arm, after the author asked to keep
using the daily Gemma submissions. Each submission needs the author's OK, given by running the command.**

## Why

- **The first A/B found no difference** between the default agent and budget-fit: +0.005, Welch 95% interval [−0.027,
  +0.037] (`kaggle_swe/LEADERBOARD_AB.md`).
- **Budget-fit changed two things that may pull in opposite directions:** more time per task (5.5 minutes against 4),
  and far fewer tool calls (28 against 100), along with dropping the graph tools and the sub-agent.
  - **Time binds these agents.** Offline, our Gemma agent timed out on 73% of attempts, and on ARC-AGI-3 doubling the
    time per game raised the score from 27 to 46 (`results/arc3_budget_elasticity.md`).
  - **Tool calls may bind too.** A participant's local runs (forum discussion 746250, single runs, unverified) lost 14 of
    129 tasks when the tool budget was halved.
- **So this test changes only the time:** the default agent, with its 100 tool calls and all its tools, given 5.5
  minutes per task instead of 4.

## Arms

- **Default:** `runs/swe/submissions/base_4min/submission.zip` (sha256 starts 7ea3f3a3), as in the first A/B.
- **More time:** `runs/swe/submissions/base_5p5min/submission.zip` (sha256 starts a6f2051f). It is the default zip with
  one line changed in `eval_config.yaml`: `max_time_minutes: 5.5` instead of 4. Every other file is byte-identical.
- **Run-time limit:** 5.5 minutes for about 120 hidden tasks is about 11 hours, under the 12-hour limit. Budget-fit ran at
  5.5 minutes in four scorings without exceeding it.

## Design

- **Ten submissions, alternating, the new arm first:** more time, default, more time, default, and so on, 5 of each.
- **Pace:** one pending submission at a time; recent scorings took under a day, so about ten days.

## Reading, fixed before the scores

- **Primary:** the more-time mean minus the default mean over the 10 new scores, with a Welch 95% interval
  (`scripts/welch.py`).
  - **"More time raises the hidden score"** if the gain is at least 0.03 (about 2 of 58 tasks) and the interval is
    above 0. Otherwise: not shown.
- **Secondary:** the more-time scores against all of the default zip's scores (the 5 earlier ones plus the 5 new).
- **Failures:** a submission that errors on Kaggle's side, or gets no score, is not counted and is resubmitted until
  scored (the rule of 2026-10-22). If a more-time submission fails because the run exceeds the time limit, the series
  stops and that is the result.
- **The fixed length avoids optional stopping:** the reading is reported after all 10, whatever the interim scores.

## Change log

- **2026-10-25:** plan drafted and the zip built and checked (only `eval_config.yaml` differs).
- **Submission 1, more time:** ref 57622587, 2026-10-25, PENDING. Taken as the author's OK
  for the series.
- **Submission 1 (more time) scored 0.12** (about 7 of 58 tasks). No submissions were made 2026-10-26 to 10-29.
- **2026-10-29, the offline time test** (`results/swe_offline_time.md`), the same question on the 129 training tasks:
  +1.6 points, 95% CI [−1.6, +4.7], not shown. This test continues as fixed.
- **Submission 2, default:** ref 57794957, 2026-10-30, PENDING.
- **Submission 2 (default) scored 0.10** (about 6 of 58 tasks). Next: more time.
- **Submission 3, more time:** ref 57892486, 2026-11-02, PENDING.
- **Submission 3 (more time) scored 0.15** (about 9 of 58 tasks). Next: default.
- **Submission 4, default:** ref 57974840, 2026-11-04, PENDING.
- **Submission 4 (default) scored 0.13** (about 8 of 58 tasks). Next: more time.
- **Submission 5, more time:** ref 58015587, 2026-11-05, PENDING.
- **Submission 5 (more time) scored 0.17** (about 10 of 58 tasks). Next: default.
- **Submission 6, default:** ref 58047538, 2026-11-06, PENDING.
- **Submission 6 (default) scored 0.13** (about 8 of 58 tasks). Next: more time.
