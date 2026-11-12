# Gemma SWE agent: does a code-context skill help a time-bound agent, and is it the tool or the advice?

**Approved by the author on 2026-10-11** ("Yes"), before anything in it ran. Binding from then on; changes go in the
change log.

## Question

The go/no-go's base agent (48 rich tasks × 2 samples, `results/swe_gonogo.md`) fails mostly by running out of time:
- **Outcomes:** 70 of 96 runs hit the 6-minute limit, and none of them was solved, though 39 left a patch. Of the 26 runs
  that finished in time, 19 were solved.
- **Pace:** each tool call takes about 15 s (median), so a run gets about 24 calls.
- **Where the calls go:** 84% explore. grep was used 354 times and throwaway Python 246 times; files are read in windows of
  a median 30 lines, each file about 4 times. The first edit comes at call 7 (median); 33 of 96 runs never edit, and only
  18 ever run a test.

If the agent reached the code it needs in fewer calls, more runs would finish, and finished runs are mostly solved.
- **Main question:** does a skill that returns whole definitions and their call sites in one call raise the share of tasks
  solved?
- **Tool or advice:** a third arm gives the same guidance as plain advice, with no tool. Three earlier tests found that
  advice alone changed little (Reasoning Gym, SWE lessons, ARC lessons). This arm separates the two.

## Setup

- **Base:** the budget-fit agent, our best leaderboard configuration (0.15 and 0.10): 5.5 minutes and 28 tool calls per
  task, a 300 s command timeout, no graph tools or sub-agent (`kaggle_swe/build_submission.py`, commit a2af280).
- **Arms**, identical except as stated:
  1. **Base.**
  2. **Skill:** adds a skill `code_context`, run with `run_skill_script`, plus one prompt line saying when to use it.
     - Its script takes identifiers or error text and searches the repository (`/workspace`) with `git grep`.
     - For each identifier it prints the defining file and line, and the whole definition (up to 120 lines).
     - It also prints up to 10 call sites (file:line and code) and the test files that mention the identifier.
     - The output is capped at about 12,000 characters.
  3. **Advice only:** adds one prompt line with the same guidance and no tool: read whole definitions in one call, with
     `read_file` over 100+ lines or the whole file; do not re-read lines already seen; search once with `git grep` for the
     identifier.
- **Tasks:** all 129 training tasks (fastapi 67, rich 48, requests 13, httpx 1), one sample per arm. As in the go/no-go,
  4 tasks run at once on Kaggle's 4×L4, with the same harness and wheelhouse. The arms are interleaved by task, so drift
  over the session hits all three alike.
- **Checks before the main run:**
  - **Locally,** with a small Gemma on the local harness: the skill compiles, `run_skill_script` runs the script in the task
    sandbox, and it returns code from `/workspace`.
  - **On Kaggle:** the first task of each arm is inspected in the log for the same three points. The run stops if the skill
    errors.

## Pass line

- **Primary metric:** the share of the 129 tasks solved, per arm.
- **Main: skill minus base,** paired by task, with a bootstrap 95% interval over tasks (2,000 rounds, seed 0). **Go** if the
  gain is at least **5 points** and the interval is above 0, the line of every earlier test.
- **Tool or advice: skill minus advice-only,** the same statistic and line. The tool "matters" if it passes.
- **Reported alongside:**
  - advice-only minus base;
  - the share of runs that finish in time;
  - tool calls per run and calls before the first edit;
  - how often the skill is used and errors;
  - per-repository results.
- **Power:** outcomes are nearly fixed per task (the go/no-go had 93 of 96 pairs agree). A 5-point gain is about 6–7 more
  tasks solved out of 129, so only a clear effect can pass.

## Compute

| Step | Estimate (Kaggle 4×L4, the same weekly quota) |
| --- | --- |
| 387 task runs (3 arms × 129), 4 at once, about 6 minutes each | about 9.7 h |
| Server start and setup | about 0.5 h |
| **Total** | **about 10 h**, within the 12.5 h left this week |

## What each outcome means

- **Skill passes, and beats advice:** a tool that changes what the agent can do in a call helps where advice did not. That
  supports moving experience into the harness as tools. The next step is the stage the research plan names: the agent
  writes such tools from its own runs, with a gate that keeps only verified ones.
- **Skill passes, advice does as well:** the guidance is what matters; the tool is not needed.
- **Nothing passes:** call efficiency is not the bottleneck it looks like, or the model does not use the help. The analysis
  of tool use will say which.
- **Leaderboard:** if the skill passes, submitting it is the author's call. One submission resolves only about 3 of 58 public
  tasks (`results/gemma_submissions.md`).

## Change log

- **2026-10-11, after approval and before the Kaggle run: implementation details, not design changes.**
  - **Code:**
    - The skill: `kaggle_swe/skills/code-context/` (ADK requires a lowercase kebab-case name).
    - The arms: `kaggle_swe/build_submission.py --skill code-context` and `--advice`, both on top of the budget-fit flags.
    - The stage: `kaggle_swe/gonogo/gonogo.py --stage compare`; the notebook is `sumyeung/swe-skill-compare`, built by
      `kaggle_swe/gonogo/build_kernels.py compare`.
  - **The base arm is byte-identical to the scored budget-fit submission** (0.15 and 0.10).
  - **Each arm runs under its own `eval_config.yaml`:** 300 s command timeout, 28 tool calls, 5.5 minutes, as the scorer
    applies them. The go/no-go stage had passed fixed limits.
  - **Interleaving:** the arms' order rotates from task to task, so each arm runs first, second and third equally often.
  - **Finding the repository:** the skill runtime starts the script in a temporary folder. The script finds the task
    workspace from `PWD`, then `PYTHONPATH` (which the local sandbox sets to the workspace), then the parent shell's working
    directory, then `/workspace` (the scorer's docker sandbox).
  - **Local check, partly passed:** the small local Gemma (E2B) is too weak at tool calling for a full run. In all three
    arms it called the prompt's example `git grep` command as a tool name, and it gave `run_skill_script` a repository file
    as the script path. Each link was therefore checked separately:
    1. The skill is registered and offered to the model.
    2. `run_skill_script` routes to it (the local run).
    3. Run through ADK's own wrapper from a temporary folder, the script finds the repository and prints whole
       definitions. This was checked with the local sandbox's environment and with a bare one.
  - **Stop rule, as registered ("the run stops if the skill errors"):** once 8 skill-arm tasks are done, the session stops
    if the skill was called at least 3 times and at least 80% of the calls returned errors.
  - **Time limit:** Kaggle ends a session at 12 hours and its outputs with it, so no new task starts after 10.75 hours.
    Tasks not run are reported, and the comparison then uses the tasks run in all three arms.
- **2026-10-11, the Kaggle run and the analysis.**
  - **The run:** `sumyeung/swe-skill-compare` version 1 was pushed and started at once.
  - **Analysis script, written before any result:** `kaggle_swe/skill_report.py`, with 3 tests in
    `tests/test_skill_report.py`. It computes the main line, the tool-or-advice line and the reported extras as registered,
    over the tasks run in all three arms.
- **2026-10-11: version 1 failed after 11 minutes, before any task ran. A harness version change, not a design change.**
  - **The failure:** `ImportError: cannot import name 'setup_gemma_model_registry'`. The current wheelhouse ships swegemma
    0.2.11 and adk_submission 0.2.13; the code was written against 0.2.7 and 0.2.11, the versions installed locally.
    Friday's LoRA re-check did not use this code path.
  - **The probe:** a CPU notebook (`sumyeung/swe-api-probe`, no GPU quota) installed the three harness packages from the
    current wheelhouse into its output (`runs/swe/api_probe/pkgs`).
  - **The fix:** in 0.2.11 the model registry is built with `adk_submission`'s `create_model_registry` on Kaggle, and as the
    harness's own CLI builds it locally (declared alias → `openai/<alias>`). `EvalConfig` still accepts the same flat
    settings.
  - **Before the next push:** the local check is rerun against those exact 0.2.11 packages.
- **2026-10-11, after the fix and before version 2: the local check passed on the 0.2.11 harness.**
  - **The run:** the probe's exact packages, with the small local Gemma. All 6 task runs (2 tasks × 3 arms) finished
    normally, and results and traces sit where the analysis expects them.
  - **One instruction clarified (wording, not design):** the small model twice passed a repository file as `file_path`, and
    the skill answered "script not found". The skill arm's one prompt line, and its SKILL.md, now say that `file_path` is
    always `scripts/context.py`, never a repository file, and that the identifiers go in `args`.
  - **After the clarification, the full chain worked in a real task sandbox:** the agent called the skill correctly, and
    the script found the repository and returned `super_len`'s whole definition with line numbers.
  - **Unchanged:** the base arm is still byte-identical to the scored budget-fit submission. The advice arm is unchanged.
- **2026-10-12, the result: NO-GO** (`results/gemma_skill.md`).
  - **The run:** version 2 ran on all 387 task runs, with no cutoff and no stop.
  - **Registered statistics:** skill minus base +0.0 [−5.4, +5.4]; skill minus advice +0.8 [−3.9, +5.4]; advice minus base
    −0.8 [−5.4, +3.9].
  - **The skill reached the model only half the time:** 68 of its 134 calls were malformed (`INVALID_ARGUMENTS`). The model
    mixed the example's backticks into the arguments, so the stop rule's 80% was not reached.
  - **Where the skill worked, no gain either:** on those 52 tasks the arms solved 4, 5 and 3 (skill, base, advice).
