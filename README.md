# Telling Is Not Teaching

Code, pre-registered plans and results for *Telling Is Not Teaching: Self-Written Lessons Do Not Improve Coding and
Reasoning Agents*, a paper in the [Gemma 4 Developer Agent paper track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)
on Kaggle (2026).

The paper asks whether a language-model agent improves from its own experience by writing lessons for itself. It tests
two agents against pass lines fixed in advance: Gemma 4 31B on the competition's software-engineering (SWE) tasks, and
Qwen3-1.7B on Reasoning Gym puzzles, where training is cheap enough to compare lessons in the prompt with the same
experience trained into LoRA weights.

## Layout

| Path | Contents |
| --- | --- |
| `PREREGISTRATION.md` | Plans for the Reasoning Gym tests 1, 2, 2b, 3, 4 and 5, with their change log |
| `phase0/` | Reasoning Gym code, and the plans for tests 6, 7, 7b, 8 and 8b and the gate-signals diagnostic |
| `kaggle_swe/` | Gemma 4 SWE code and plans: the go/no-go, the tool test, the leaderboard tests and the offline time test |
| `results/` | One report per test |
| `scripts/` | Environment setup, model download, and the Welch interval used for the leaderboard series |
| `tests/` | Unit tests; each file runs on its own, for example `python tests/test_gonogo.py` |

## Where each result comes from

| Paper | Plan | Code | Report |
| --- | --- | --- | --- |
| Tests 1, 2 and 2b: lessons in the prompt (Reasoning Gym) | `PREREGISTRATION.md` | `phase0/test1_gap.py`, `phase0/test2_lessons.py`, `phase0/test2_diagnose.py` | `results/test1_screen.md`, `results/test1.md`, `results/test2.md` |
| Test 4: W1 and W2, experience into weights | `PREREGISTRATION.md` | `phase0/test4_weights.py` | `results/test4_weights.md` |
| Test 5: more successes, and the surprise gate | `PREREGISTRATION.md` | `phase0/test5_gate.py` | `results/test5_gate.md` |
| Test 6: three rounds of self-training | `phase0/TEST6_SELF_TRAINING.md` | `phase0/test6_iterate.py` | `results/test6_self_training.md` |
| Test 7: eight families | `phase0/TEST7_FAMILIES.md` | `phase0/test7_families.py`, `phase0/test7_format_check.py` | `results/test7_families.md` |
| Test 7b: likelihood checks after training | `phase0/TEST7_CHECK.md` | `phase0/test7_check.py` | `results/test7_check.md` |
| Test 8: one adapter for all families | `phase0/TEST8_UNION.md` | `phase0/test8_union.py` | `results/test8_union.md` |
| Test 8b: only the families that gained | `phase0/TEST8B_GATED.md` | `phase0/test8b_gated.py`, `phase0/test8b_posthoc.py` | `results/test8b_gated.md` |
| Gate signals (exploratory, not pre-registered) | `phase0/GATE_SIGNALS.md` | `phase0/gate_signals.py` | `results/gate_signals.md` |
| SWE go/no-go, and the LoRA check | `kaggle_swe/PREREGISTRATION.md` | `kaggle_swe/gonogo/` | `results/swe_gonogo.md` |
| Tool test: the code-context skill | `kaggle_swe/PREREGISTRATION_SKILL.md` | `kaggle_swe/skills/code-context/`, `kaggle_swe/skill_report.py` | `results/gemma_skill.md` |
| Leaderboard test 1: budget-fit against default | `kaggle_swe/LEADERBOARD_AB.md` | `kaggle_swe/build_submission.py`, `scripts/welch.py` | `results/gemma_submissions.md` |
| Leaderboard test 2: more time per task | `kaggle_swe/LEADERBOARD_TIME.md` | `kaggle_swe/build_submission.py`, `scripts/welch.py` | `results/gemma_submissions.md` |
| Offline time test: 5.5 against 4 minutes per task | `kaggle_swe/OFFLINE_TIME.md` | `kaggle_swe/time_report.py` | `results/swe_offline_time.md` |
| Practical findings for Gemma 4 agents (section 5) | | | `results/swe_gonogo.md`, `results/swe_smoke_kaggle.md`, `results/gemma_submissions.md`, `results/time_use.md` |

Each Reasoning Gym test runs from its `phase0/run_*.sh`. Test 3 in `PREREGISTRATION.md` was paused before it ran, so it
has no report.

## Running it

Reasoning Gym tests (written for Ubuntu under WSL2, with one 16 GB GPU):

```bash
bash scripts/setup_env.sh          # environment in ~/venvs/research; versions as in requirements.lock
bash scripts/download_models.sh    # Qwen3-1.7B and Qwen3-8B-AWQ
source scripts/env.sh
bash phase0/run_test6.sh           # each test has its run script; its plan gives the expected run time
```

SWE tests (Kaggle notebooks on 4×L4). Accept the competition's rules first, then:

```bash
python kaggle_swe/download.py small          # docs, task list and the sample agent, into ~/data/gemma4
python kaggle_swe/download.py wheelhouse     # the harness, for local debugging
bash kaggle_swe/setup_harness.sh             # harness environment; versions as in kaggle_swe/requirements.lock
python kaggle_swe/gonogo/build_kernels.py full     # or experience, lora, compare, time: see its docstring
kaggle kernels push -p kaggle_swe/gonogo/kernels/full
```

The notebooks are named for our Kaggle account (`sumyeung/...` in `build_kernels.py`); change that to yours.
Leaderboard submissions are built with `kaggle_swe/build_submission.py`, whose docstring lists the variants we submitted.

## Not included

- **Competition data.** The competition's rules do not allow redistributing it. That also leaves out four generated
  notebooks that embed the competition's sample agent (`kaggle_swe/gonogo/kernels/compare`, `time_a`, `time_b` and
  `time_smoke`); `build_kernels.py` rebuilds them from your own copy of the data.
- **Run outputs and logs** (`runs/`, `logs/`), which the reports cite by path. They are large, and the SWE ones
  contain competition data.
- **The companion ARC-AGI-3 study,** a separate paper in the ARC Prize 2026 paper track. A few records cite its files
  (`arc3/...`, `results/arc3_...`), which are not in this repository.

## About the records

- Each pre-registered plan was written and approved before its test ran, with its pass line, and later changes were
  logged with their dates. Post-hoc and exploratory analyses say so in their files. This repository is a single
  snapshot of the working repository, so these dates are as the files record them, and the commit hashes the records
  cite belong to that repository's history, which is not published.
- This copy differs from the working records in two ways: clock times are removed (dates are kept, and run-folder
  names show `hhmmss` in place of their time), and where the records said "the user" they say "the author".
- The plans, code and reports were drafted with an AI assistant (Claude) working under the author's direction. The
  author approved each pre-registered plan before its test ran; the quoted approvals in the plans are the author's
  replies.
- "Project 1" and "Project 2" in the plans are the wider research programme: Project 1 asks whether an agent's own
  experience helps it, as text or as weights; Project 2 is a learned gate that decides which experience to consolidate.

## License

Apache License 2.0, in `LICENSE`.
