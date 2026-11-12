# Gemma SWE agent: a code-context skill against advice and base: NO-GO

**Final (2026-10-12).** Pre-registration: `kaggle_swe/PREREGISTRATION_SKILL.md` (approved 2026-10-11, change log included).
Notebook `sumyeung/swe-skill-compare` version 2: 387 task runs (3 arms × 129 training tasks), 8.1 hours, no cutoff or stop.
Report: `python kaggle_swe/skill_report.py runs/swe/skill_compare` (saved in `runs/swe/skill_compare/report.txt`).

## Registered outcome

| Arm | Solved (of 129) | Finished in time | Mean tool calls | Calls before the first edit (median) | Runs that never edit |
|---|---|---|---|---|---|
| Base (the budget-fit submission) | 28 (21.7%) | 62 | 18.4 | 6 | 19 |
| Skill (`code-context`) | 28 (21.7%) | 59 | 18.7 | 7 | 25 |
| Advice (one prompt line) | 27 (20.9%) | 68 | 19.0 | 6 | 20 |

- **Main, skill minus base: +0.0 points, 95% CI [−5.4, +5.4]: NO-GO.** 6 tasks were solved only with the skill and 6 only
  without.
- **Tool or advice, skill minus advice: +0.8 [−3.9, +5.4]:** not shown to matter.
- **Advice minus base:** −0.8 [−5.4, +3.9].
- **Per repository:** no arm stands out (rich 13 / 12 / 11, fastapi 14 / 15 / 15, requests 1 / 1 / 1 for base / skill / advice).

## Why the skill did not help (descriptive)

- **Half its calls were malformed.** The model called the skill in 68 of 129 runs, 134 calls in all, and 68 of those calls
  were rejected as `INVALID_ARGUMENTS`. The model mixed the backticks of the prompt's example into the arguments (for
  example `file_path: "scripts/context.py`,skill_name:"`), which swallowed the `skill_name` field. One more call hit the
  time budget. The 65 calls that went through all worked, returning about 2,400 characters of code each (median).
- **Where the skill did work, it still did not help.** On the 52 tasks where at least one call succeeded, the skill arm
  solved 4, base 5 and advice 3. On the 61 tasks where the model never called it, 20 / 18 / 20. The model reached for the
  skill on hard tasks (base solved only 5 of those 52), and the code it returned did not turn them into solved ones.
- **The exploration measures did not move either.** The first edit came at call 7 with the skill against 6 without, and more
  skill runs never edited at all (25 against 19).

## What it means

- **Call efficiency is not the lever it looked like.** The go/no-go analysis showed runs time out while exploring. But
  handing the agent whole definitions, as a tool or as advice, did not raise the solved share. What limits these runs is
  more likely the model's ability to reach a correct fix in about 20 calls, not how it reads code.
- **Tool against advice, the research question this test was built for:** neither moved the score, so this test cannot
  separate them. The tool arm was also delivered only half the time.
- **A repeat with a fixed instruction is not recommended on this evidence.** Even the delivered calls showed no gain (4
  against 5 on the same tasks).
- **Leaderboard:** nothing here beats the budget-fit submission, so no new submission is suggested.
