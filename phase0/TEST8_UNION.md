# Test 8: does one adapter trained on all eight families' own successes keep each family's gain?

**Drafted 2026-10-25, before anything in it ran, from the proposal the author approved that day ("Approve test 8").
Binding from then on; changes go in the change log.**

## Why

- **Per family, self-training helps or does nothing** (test 7, `results/test7_families.md`): round 1 gained in 5 of 8
  families as scored, 3 to 7 points of reasoning in three with the layout slips forgiven, and no family got worse on
  itself.
- **The harm was forgetting:** a model trained on one family lost 3.7 to 4.6 points on survo in 2 of 8 cases.
- **Neither detecting nor predicting forgetting is cheap:** two likelihood signals failed (tests 7 and 7b), and a
  generation check needs about the whole evaluation to catch a 4-point loss (`results/test7_check.md`).
- **So this test tries to prevent it instead:** one adapter trained on every family's successes at once, the way one
  model would hold many skills. If it keeps each family's gain, a gate need not pick families at all; if it loses them,
  the families interfere, and avoiding that is the gate's job.

## Design

- **Training:** one fresh LoRA on the union of test 7's eight round-1 training sets (every correct, finished practice
  attempt: 266, 334, 216, 814, 134, 674, 53 and 931, together 3,422), with test 4's settings (3 epochs, learning rate
  1e-4, accumulation 8), seed 1. Examples are shuffled across families.
- **Evaluation:** test 2's, 500 problems × 2 samples (seed 3000), on each of the 8 families.
- **Comparison:** each family's own round-1 adapter from test 7 (for survo and quantum_lock the mean of test 6's two
  round-1 trainings, as test 7 reported) and the base model, on the same problems.
- **Code:** `phase0/test8_union.py` (stages train, evaluate, report), run by `phase0/run_test8.sh`. Training and
  evaluation are tests 4 and 5's functions, unchanged.
- **Cost:** local GPU only, about 13 hours: the training about 4.5 (3,422 examples, about 13 times survo's), the eight
  evaluations about 8.5.

## Reading, fixed before the runs

- **Main:** the union adapter's pass@1 minus each family's own adapter's, averaged over the 8 families, with a 95%
  interval from a bootstrap that resamples the problems within each family (2,000 rounds, seed 0).
  - **"One adapter holds the gains"** if the interval's lower end is above −1.5 points.
  - It is also called **clearly worse** if the interval lies below 0, or **clearly better** if it lies above 0.
- **Reported alongside:**
  - per family, the union adapter minus the base and minus the family's own adapter, with paired bootstrap intervals;
  - survo, where other families' adapters caused forgetting;
  - the same with test 7's layout slips forgiven (`phase0/test7_format_check.py`).
- **Report:** `python phase0/test8_union.py report --forgive` (written to `runs/test8/report.md`).
- **A stage that fails** is rerun once and noted.

## Limitations, stated before the results

- **The union adapter trains for many more steps** than any single-family one (3,422 examples against 53 to 931), with
  the same learning rate and epochs. Differences can come from that as well as from mixing.
- **One training,** with no second seed. Test 6 measured training noise at 0.3 to 1.1 points per family.

## Checks before the main run

- **A unit test** of the stratified interval and the report on synthetic records (`tests/test_test8_union.py`).
- **A dry run** on tiny sizes with `phase0/test8_dry_run.sh`: plumbing only, results discarded.

## Change log

- **2026-10-25: approved by the author** ("Approve test 8"), before anything in it ran.
- **Unit tests passed** (`tests/test_test8_union.py`, 2 tests), and tests 7 and 7b's still pass.
- **Dry run passed** (`runs/test8_dry/20261025_hhmmss`, `logs/test8_dry.log`): training on a tiny union, two evaluations
  and both report sections ran. Its numbers are discarded.
- **Full run started 2026-10-25** from commit 65aa015 (`logs/test8.log`).
- **Finished 2026-10-25 (11.2 hours); read on 2026-10-29, when the run's notice arrived. Result: one adapter
  is not shown to hold the gains** (union minus own adapters −0.7 [−1.6, +0.2]; the line was a lower end above −1.5).
  Write-up: `results/test8_union.md`.
