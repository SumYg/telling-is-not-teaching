# Leaderboard A/B: the budget-fit agent against the default agent

**A measurement for the Gemma paper. Drafted 2026-10-13, before any new score. Each submission needs the author's OK.**

## Why

- **Budget-fit is our one change that may have helped:** 0.15 and 0.10 against the default's 0.08
  (`results/gemma_submissions.md`). Two scorings of one zip differed by 0.05, so these scores cannot show it.
- **Scoring is the paper's only measurement on hidden tasks.** Adapters are not applied there (the random-adapter check),
  so only harness changes can be tested.
- **Our agent already thinks** (`thinking_budget` 4096; every model step in the skill test's traces is a thinking event,
  with a median of 105 output tokens). The forum's thinking-on gain is not a lever for us.

## Arms (byte-identical re-uploads)

- **Default:** `runs/swe/submissions/base_4min/submission.zip` (ref 56699594). 4 minutes and 100 tool calls per task, the
  graph tools and the code-analyzer sub-agent.
- **Budget-fit:** `runs/swe/submissions/budget_5p5min/submission.zip` (refs 56894770, 56932398). 5.5 minutes and 28 tool
  calls, no graph tools or sub-agent, its own system prompt.
- **Checked 2026-10-13:** every tool the default agent names, and `agent_tool`, still exist in the scorer's current
  packages (swegemma 0.2.11, adk_submission 0.2.13).

## Design

- **Eight submissions, alternating, default first** (it has fewer scores): D, B, D, B, D, B, D, B.
- **Pace:** Kaggle allows one pending submission per team, and scoring takes about 14–33 hours (host, topic 747519), so
  about one every one to two days. Done by early November; the paper is due 2026-11-12.

## Reading, fixed before the scores

- **Primary:** the budget-fit mean minus the default mean over the 8 new scores, with a Welch 95% interval.
- **"Budget-fit raises the hidden score"** if the gain is at least 0.03 (about 2 of 58 tasks) and the interval is above 0.
  Otherwise: not shown.
- **Secondary:** the same over all scores of the two zips (adds 0.08; 0.15 and 0.10). The random-adapter score (0.13) is
  reported but not counted, since its zip differs.
- **A scoring error** is resubmitted once and noted.

## Quota check (optional)

- **This week Kaggle counted about 4 GPU-hours more than our sessions,** while our random-adapter submission was being
  scored. If scoring is charged to the submitter, the first submission after the Friday reset will show it: read the quota
  just before submitting and again once it is scored, with no session running in between.

## Change log

- **2026-10-13: approved by the author** ("Approve both").
- **Submission 1, default:** ref 57160511, 2026-10-14, PENDING. The shell cut the
  description to "AB" (cosmetic).
- **Submission 1 scored 0.13** (about 8 of 58 tasks; complete by 2026-10-15).
- **Submission 2, budget-fit:** ref 57202303, 2026-10-15, PENDING.
- **Submission 2 scored 0.15** (about 9 of 58 tasks).
- **Submission 3, default:** ref 57296710, 2026-10-17, PENDING.
- **Quota check skipped this week** (the author's choice, 2026-10-16), so whether scoring is charged stays unknown.
- **Submission 3 scored 0.13** (about 8 of 58 tasks).
- **Submission 4, budget-fit:** ref 57386361, 2026-10-19, PENDING.
- **Submission 4 scored 0.13** (about 8 of 58 tasks). So far: default 0.13, 0.13; budget-fit 0.15, 0.13.
- **Submission 5, default:** ref 57423276, 2026-10-20, PENDING.
- **Submission 5 scored 0.12** (about 7 of 58 tasks). So far: default 0.13, 0.13, 0.12; budget-fit 0.15, 0.13.
- **Submission 6, budget-fit:** ref 57470193, 2026-10-21: **"Notebook Threw Exception", no score.** Other teams
  reported the same on Oct 21 for working zips (forum topics 751339, 751264), the day our two ARC trainings were also
  stopped. Resubmitted once, as planned: **ref 57514302, 2026-10-22, PENDING.**
- **The quota question is answered:** on 2026-10-22 Kaggle showed 26h 18m of GPU used this week. Our completed notebook
  sessions add up to 26.26 hours. So Gemma scoring is not charged to the submitter's GPU quota: four Gemma submissions
  were scored or attempted this week and added nothing. Sessions Kaggle stopped were not charged either.
- **Rule for failures, fixed 2026-10-22 before submission 6b is scored:** Kaggle's machines were failing again that day
  (ARC notebooks that had scored before scored 0.00 while model loading was slow). A submission that errors or gets no
  score is not counted, and the same arm is resubmitted, as often as needed, instead of only once. The series still ends
  after 8 scores, or when the last score can no longer arrive by about Nov 10 for the paper; the reading then covers the
  scores obtained.
- **Submission 6b scored 0.10** (about 6 of 58 tasks). So far: default 0.13, 0.13, 0.12; budget-fit 0.15, 0.13, 0.10.
- **Submission 7, default:** ref 57543571, 2026-10-23, PENDING.
- **Submission 7 scored 0.10** (about 6 of 58 tasks). So far: default 0.13, 0.13, 0.12, 0.10; budget-fit 0.15, 0.13, 0.10.
  Submission 8, budget-fit, is the last.
- **Submission 8, budget-fit (the last):** ref 57587066, 2026-10-24, PENDING. The
  reading follows once it is scored.
- **Submission 8 scored 0.12** (about 7 of 58 tasks). **Series complete.**
- **Reading: budget-fit is not shown to raise the hidden score.** Budget-fit 0.125 (0.15, 0.13, 0.10, 0.12) against the
  default 0.120 (0.13, 0.13, 0.12, 0.10): **+0.005, Welch 95% interval [−0.027, +0.037]**, df 5.3. The line was a gain
  of at least 0.03 with the interval above 0. Over all scores of the two zips (adding the default's 0.08 and
  budget-fit's 0.15 and 0.10): +0.013 [−0.017, +0.043]. So the hidden set rules out an effect beyond about 2 of the 58
  tasks either way. Computed with `scripts/welch.py`, which reproduces the lessons reading and table t quantiles.
