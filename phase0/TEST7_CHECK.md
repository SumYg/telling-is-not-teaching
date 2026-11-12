# Test 7b: can a cheap likelihood check after training predict test 7's measured gains and forgetting?

**Drafted 2026-10-25, before anything in it ran, from the proposal the author approved that day ("approve both").
Binding from then on; changes go in the change log. A diagnostic: test 7's outcomes were already known when it was
defined, so a pass would need confirming on new families.**

## Why

- **Test 7 measured each family's gain and its forgetting on survo by generation:** about 2.5 GPU-hours of evaluation
  per trained model (`results/test7_families.md`).
- **There, the gate's real job was forgetting, not the gain:** training on everything was nearly the best choice for
  each family's own score, but 2 of 8 families cost survo 3 to 4 points.
- **A gate that keeps an adapter on probation** (the research plan's accept-only-if-no-regression check) needs a cheap
  regression test. If a likelihood check on saved answers tracks what generation measures, every new adapter can be
  checked in minutes and rolled back when it would forget.

## Design

- **Validation answers, generated once per family:** the base model answers 100 problems it has seen in neither practice
  nor evaluation (seed 4000), 4 times each, with test 7's settings (thinking on, 6,144-token cap, sampling seed 4000).
  The evaluation records keep only extracted answers, not texts, which is why new answers are needed.
- **Scored attempts:** each family's correct and wrong finished validation attempts, at most 200 of each (random,
  seed 0).
- **The check:** the mean log-probability per completion token of each scored attempt, under the base model and under
  the trained model. A check's value is the gain on the correct attempts minus the gain on the wrong ones, in nats per
  token (test 7's probe contrast, now after the real training and on unseen problems).
  - **Forgetting check (main):** each family's model, scored on its forgetting family's validation attempts (survo, or
    quantum_lock for survo itself). For survo and quantum_lock the model is test 5's R-all, the model whose forgetting
    test 7 reports.
  - **Gain check (secondary):** each family's model on its own validation attempts. For survo and quantum_lock it is the
    mean over test 7's two round-1 models (test 5's R-all and test 6's seed 2), matching the reported gain.
- **Models:** test 7's round-1 models, unchanged. No training.
- **Code:** `phase0/test7_check.py` (stages validate, base, score, report), run by `phase0/run_test7_check.sh`. The
  log-probabilities use test 4's training loss function, as test 7's probe did.
- **Cost:** local GPU only, about 4 hours, nearly all of it generating the 8 validation sets.

## Reading, fixed before the runs

- **Outcomes:** test 7's measured round-1 gain and forgetting, as scored (`runs/test7/report.md`).
- **Main:** Spearman's rank correlation between the forgetting check and the measured forgetting over the 8 families,
  with an exact one-sided permutation p-value (a higher check should mean less forgetting).
  - **"The forgetting check predicts forgetting"** if rho is at least 0.65 and p is below 0.05.
- **Secondary:** the same for the gain check against the measured gain.
- **Not testable** for either with fewer than 6 families, or for a family with fewer than 10 correct or 10 wrong scored
  attempts (it is left out).
- **Reported alongside:** both checks against test 7's gains and forgetting with the layout slips forgiven
  (`phase0/test7_format_check.py`).
- **Report:** `python phase0/test7_check.py report` (written to `runs/test7_check/report.md`).
- **A stage that fails** is rerun once and noted.

## Checks before the main run

- **A unit test** of the contrast and the report on synthetic records (`tests/test_test7_check.py`).
- **A dry run** on tiny sizes with `phase0/test7_check_dry_run.sh`: plumbing only, results discarded.

## Change log

- **2026-10-25: approved by the author** ("approve both"), on a proposal of about 2 GPU-hours that re-scored saved
  answers. Building it showed that the evaluation records keep no texts, so the check generates validation answers first:
  about 4 hours. The author was told before the run.
- **Unit tests passed** (`tests/test_test7_check.py`, 3 tests), and test 7's still pass.
- **Dry run passed** (`runs/test7_check_dry/20261025_hhmmss`, `logs/test7_check_dry.log`): every stage ran, including the
  exclusion of a family with too few scored attempts. Its numbers are discarded.
- **Full run 2026-10-25** (3.7 hours) from commit b4ef76d (`logs/test7_check.log`). **Result: neither
  check predicts.** Forgetting check rho +0.48 (one-sided p 0.12); gain check rho +0.38 (p 0.18). Write-up:
  `results/test7_check.md`.

