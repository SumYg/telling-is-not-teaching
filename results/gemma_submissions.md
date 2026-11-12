# Gemma 4 Developer Agent: leaderboard submissions

Public leaderboard = 58 tasks, so one task is about 0.017. Scores as reported by `kaggle competitions submissions
gemma-4-developer-agent` on 2026-10-10.

| Ref | Submitted (UTC) | Configuration | Public score | About |
|---|---|---|---|---|
| 56699594 | 2026-09-30 | Base agent, 4 minutes per task | 0.08 | 5 tasks |
| 56754971 | 2026-10-01 | Base + time-awareness prompt section, 4 min, 100 tool calls | 0.08 | 5 tasks |
| 56894770 | 2026-10-07 | Budget-fit: 5.5 min, 28 tool calls, 300 s test timeout, no graph tools or sub-agent | 0.15 | 9 tasks |
| 56932398 | 2026-10-08 | The same budget-fit zip, resubmitted | 0.10 | 6 tasks |
| 57064938 | 2026-10-11 | The budget-fit zip routed to a random-weight adapter (a check, `results/paper_tracks.md`) | 0.13 | 8 tasks |
| 57160511 | 2026-10-14 | The default zip of 56699594 again: A/B submission 1 (`kaggle_swe/LEADERBOARD_AB.md`) | 0.13 | 8 tasks |
| 57202303 | 2026-10-15 | The budget-fit zip again: A/B submission 2 | 0.15 | 9 tasks |
| 57296710 | 2026-10-17 | The default zip again: A/B submission 3 | 0.13 | 8 tasks |
| 57386361 | 2026-10-19 | The budget-fit zip again: A/B submission 4 | 0.13 | 8 tasks |
| 57423276 | 2026-10-20 | The default zip again: A/B submission 5 | 0.12 | 7 tasks |
| 57470193 | 2026-10-21 | The budget-fit zip again: A/B submission 6 | Notebook Threw Exception (Kaggle-side, Oct 21) | |
| 57514302 | 2026-10-22 | The budget-fit zip again: A/B submission 6, resubmitted | 0.10 | 6 tasks |
| 57543571 | 2026-10-23 | The default zip again: A/B submission 7 | 0.10 | 6 tasks |
| 57587066 | 2026-10-24 | The budget-fit zip again: A/B submission 8, the last | 0.12 | 7 tasks |
| 57622587 | 2026-10-25 | The default zip with 5.5 minutes per task: leaderboard test 2, submission 1 (`kaggle_swe/LEADERBOARD_TIME.md`) | 0.12 | 7 tasks |
| 57794957 | 2026-10-30 | The default zip again: leaderboard test 2, submission 2 | 0.10 | 6 tasks |
| 57892486 | 2026-11-02 | The default zip with 5.5 minutes again: leaderboard test 2, submission 3 | 0.15 | 9 tasks |
| 57974840 | 2026-11-04 | The default zip again: leaderboard test 2, submission 4 | 0.13 | 8 tasks |
| 58015587 | 2026-11-05 | The default zip with 5.5 minutes again: leaderboard test 2, submission 5 | 0.17 | 10 tasks |
| 58047538 | 2026-11-06 | The default zip again: leaderboard test 2, submission 6 | 0.13 | 8 tasks |

- **56894770 was first reported as an error** ("A system error. Please try resubmitting", 2026-10-07). By 2026-10-10 Kaggle
  showed it complete with 0.15, so it was rescored after we resubmitted.
- **Two scorings of the identical zip differ by 0.05** (about 3 tasks). That is a direct measure of scoring noise on the
  public set: a difference of one or two tasks between single submissions means nothing.
- **Budget-fit vs base (descriptive):** 0.15 and 0.10 (mean 0.125, about 7 tasks) against 0.08 and 0.08 (about 5 tasks).
  It may help by a task or two, but with this noise four submissions cannot show it. Not pre-registered.
- **Leaderboard:** Kaggle shows the best score, so the team now stands at 0.17 (2026-11-05, leaderboard test 2, submission 5). On 2026-10-06 the pack was at 0.17–0.18
  and the top at 0.24.
- **The random-adapter check scored like the plain agent,** so adapters are not applied in scoring.
- **Configs:** `kaggle_swe/build_submission.py` (budget-fit flags in commit a2af280); zip in
  `runs/swe/submissions/budget_5p5min/`.
- **Scores are truncated, not rounded,** to two decimals: 5, 6, 8, 9 and 10 of 58 tasks show as 0.08, 0.10, 0.13, 0.15 and 0.17.
- **A/B in progress** (`kaggle_swe/LEADERBOARD_AB.md`, approved 2026-10-13): the default and budget-fit zips alternate,
  8 new scores, read only at the end. The default zip's second score (0.13) is 3 tasks above its first (0.08).

- **The A/B reading (`kaggle_swe/LEADERBOARD_AB.md`): no detectable difference.** Budget-fit minus default over the 8
  new scores +0.005, Welch 95% interval [−0.027, +0.037]; over all scores of the two zips +0.013 [−0.017, +0.043].
