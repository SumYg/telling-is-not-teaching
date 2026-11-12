# Gemma SWE agent: 5.5 minutes per task against 4, offline: not shown

**Final (2026-10-29).** Plan: `kaggle_swe/OFFLINE_TIME.md` (drafted 2026-10-29 before anything ran, change log included).
Notebooks `sumyeung/swe-time-compare-a` and `-b` version 1, run at the same time on 4×L4, one task at a time: 258 task
runs (2 arms × 129 training tasks) in 9.2 and 9.0 hours. Every task was reached, and no session failed.
Report: `python kaggle_swe/time_report.py runs/swe/time_compare/a runs/swe/time_compare/b` (saved in
`runs/swe/time_compare/report.txt`).

## Registered outcome

| Arm | Solved (of 129) | Ended by the time limit | Mean tool calls | Runs with no patch |
|---|---|---|---|---|
| Default (4 minutes, the submitted zip) | 27 (20.9%) | 79 | 21.5 | 25 |
| More time (5.5 minutes, the submitted zip) | 29 (22.5%) | 69 | 25.0 | 18 |

- **Main, more time minus default: +1.6 points, 95% CI [−1.6, +4.7]: not shown** (the line was at least +5 with the
  interval above 0). 3 tasks were solved only with more time (fastapi_11194, fastapi_9753, rich_3278) and 1 only with
  the default (rich_3052).
- **Per repository:** rich 13 / 13, fastapi 13 / 15, requests 1 / 1, httpx 0 / 0 (default / more time).
- **Per session:** 13 / 15 on the even half and 14 / 14 on the odd half.
- **No harness errors:** the only errors recorded are runs stopped by their time limit.

## What the extra time changed (descriptive)

- **More runs finished on their own,** 60 against 50. On 17 tasks the default hit its limit while more time finished,
  and on 7 the reverse. 62 tasks hit the limit in both arms.
- **Few of those runs became solves:** 2 tasks net. The 3 tasks solved only with more time had all hit the 4-minute
  limit in the default arm. The one solved only by the default finished in both arms.
- **Runs did more work:** 25.0 tool calls against 21.5, and 18 runs ended with no patch at all, against 25.
- **Outcomes repeat but not perfectly:** the smoke run had already run 3 of these tasks in both arms, and 5 of those 6
  outcomes came out the same here. More time solved rich_4079 in the smoke run but not in the full run.
- **For scale:** the skill test's base arm (budget-fit, 4 tasks at once) solved 28 of the same 129 tasks.

## What it means

- **1.5 more minutes per task does not buy 5 points on these tasks.** The interval's upper end, +4.7, falls short of the
  line. A run that is still going at 4 minutes is rarely about to finish a correct fix.
- **This agrees with the two earlier results.** The leaderboard A/B found budget-fit, which also had 5.5 minutes, no
  better than the default (+0.005 [−0.027, +0.037]). The skill test found that handing the agent code did not help
  either. What limits these runs looks like reaching a correct fix at all, not the time limit at this margin.
- **Leaderboard test 2 continues as fixed** (`kaggle_swe/LEADERBOARD_TIME.md`). It asks the same question on the hidden
  tasks, and this result suggests a small difference there at most: +1.6 points of 58 tasks is about 1 task.
- **Leaderboard:** nothing here suggests a new submission.

## Limitations

- **Training tasks, not the hidden ones:** 129 tasks from 4 public repositories, most of them from fastapi and rich.
- **One sample per arm per task,** so the comparison rests on 4 discordant pairs.
- **The quota:** about 19 GPU-hours of the week's 30, the two sessions plus the smoke run, before the Friday reset.
