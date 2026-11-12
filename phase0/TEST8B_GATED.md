# Test 8b: does a gate on top of joint training make one adapter better?

**Drafted 2026-10-29, before anything in it ran, from the proposal the author approved that day ("Approve the gated
test"). Binding from then on; changes go in the change log.**

## Why

- **One adapter trained on all eight families' successes kept most gains** but cost 0.7 points on average against
  the separate adapters, −0.7 [−1.6, +0.2], narrowly missing its line (test 8, `results/test8_union.md`).
- **The clearest loss was a family whose own training had not helped:** quantum_lock lost 2.6 points against its own
  adapter and 2.9 against the base model.
- **So a gate may help even with joint training:** consolidate only the families whose own training helped, and leave
  the rest to the base model's behaviour.

## Design

- **The gate:** keep a family if its own round-1 adapter gained at least 3 points with the interval above 0 in test 7.
  That is test 7's own pre-registered count of families that gained: survo, self_reference, jugs, course_schedule and
  mini_sudoku. It leaves out quantum_lock, list_functions and word_ladder.
- **Training:** one fresh LoRA on the kept families' round-1 training sets from test 7 (266 + 216 + 134 + 674 + 931 =
  2,221 examples), with test 4's settings, seed 1, as in test 8.
- **Evaluation:** test 2's, 500 problems × 2 samples, on all 8 families, since one adapter serves every family.
- **Compared with:**
  - test 8's full union adapter, on the same problems;
  - the best separate choice, the "reference": the family's own adapter where the gate keeps it, the base model where
    it does not.
- **Code:** `phase0/test8b_gated.py` (stages train, evaluate, report), run by `phase0/run_test8b.sh`. Training,
  evaluation and the statistics are tests 4, 5 and 8's, unchanged.
- **Cost:** local GPU only, about 9.5 hours (training about 2.5, eight evaluations about 7).

## Reading, fixed before the runs

- **Main:** the gated adapter's pass@1 minus the full union adapter's, averaged over the 8 families, with test 8's
  95% interval (a bootstrap that resamples problems within each family; 2,000 rounds, seed 0).
  - **"The gate helps"** if the interval's lower end is above 0.
- **Secondary:** the gated adapter minus the reference, averaged over the 8 families. **"It holds the gains of the best
  separate choice"** if the interval's lower end is above −1.5 points, test 8's margin.
- **Reported alongside:** per family, both differences with paired intervals; the main difference over the kept and
  the left-out families separately; everything again with test 7's layout slips forgiven.
- **Report:** `python phase0/test8b_gated.py report --forgive` (written to `runs/test8b/report.md`).
- **A stage that fails** is rerun once and noted.

## Limitations, stated before the results

- **The gated adapter trains on fewer examples** (2,221 against 3,422), so fewer steps as well as different data.
- **The gate uses test 7's outcomes,** measured on the same 500 evaluation problems as this test. It is a gate that
  pays for a full evaluation per family; a cheaper gate is still missing (tests 7 and 7b).
- **One training,** as in test 8.

## Checks before the main run

- **A unit test** of the report on synthetic records (`tests/test_test8b_gated.py`).
- **A dry run** on tiny sizes with `phase0/test8b_dry_run.sh`: plumbing only, results discarded.

## Change log

- **2026-10-29: approved by the author** ("Approve the gated test"), before anything in it ran.
- **Unit tests passed** (`tests/test_test8b_gated.py`, 2 tests), and test 8's still pass.
- **Dry run passed** (`runs/test8b_dry/20261029_hhmmss`, `logs/test8b_dry.log`): training on the five kept families only,
  evaluations on a kept and a left-out family, and both report sections ran. Its numbers are discarded.
- **Full run started 2026-10-29** from commit d59389e (`logs/test8b.log`).
- **Full run completed 2026-10-29** (10.7 hours), with no stage rerun.
- **Result: the gate is not shown to help,** +0.1 [−0.9, +1.1] against the full union. The secondary line holds, −0.4
  [−1.3, +0.6], but test 8's full union holds against the same reference too (post hoc). Write-up:
  `results/test8b_gated.md`.

