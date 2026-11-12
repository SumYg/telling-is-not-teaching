# Test 7: which families gain from self-training, and can a probe tell before training?

**Drafted 2026-10-22, before anything in it ran, from the proposal the author approved that day ("Approve"). Binding from
then on; changes go in the change log.**

## Why

- **Self-training is the first carrier of experience that passed a pre-registered line,** but only on one of two
  families (test 6, `results/test6_self_training.md`):
  - on survo it added +5.4 [+2.9, +7.9] after three rounds, +4.2 after one;
  - on quantum_lock it added nothing (−0.2 after one round) and cost survo 3.7 points.
- **A gate that learns *when* to consolidate has to tell such families apart before training** (the research plan's
  Project 2). With two families, no signal can be tested: any signal that differs between them "predicts" them.
- **So this test widens the set to 8 families** and asks two things:
  - how common gains and forgetting are;
  - whether one signal, measured before the real training, ranks the families by their gain.

## Design

- **Model and settings: test 6's round 1, unchanged.**
  - Qwen3-1.7B, thinking on, a 6,144-token cap.
  - **Practice:** the base model plays 200 problems (seed 2000) 8 times each (test 5's experience stage).
  - **Training:** a fresh LoRA on every correct, finished attempt (test 4's settings: 3 epochs, learning rate 1e-4,
    accumulation 8), seed 1.
  - **Evaluation:** test 2's, 500 problems (seed 3000) × 2 samples, for the base model and the trained one.
- **Families:**
  - **Reused, survo and quantum_lock:** tests 2, 5 and 6 already hold their practice, base evaluation and round-1
    models. Their round-1 gain is the mean of test 6's two round-1 trainings, as test 6 reported it (+4.2, −0.2). Only
    their probe runs here.
  - **New, 6 families from test 1's screen** (`results/test1_screen.md`):
    - **self_reference, jugs and word_ladder:** the screen's other kept families with a pass@1 of at least 3.5%. The 4
      kept families below 3% would give at most about 40 successes in 1,600 attempts, too few to train on.
    - **list_functions, course_schedule and mini_sudoku:** the screen left out families above 45%, but these three are
      the nearest to 50%. They give many successes and still have room to gain.
    - Some families' answers can be guessed (course_schedule answers true or false; self_reference a small count), so
      their successes may hold much luck. That is the kind of difference the gate has to see.
  - **Order:** self_reference, list_functions, jugs, course_schedule, word_ladder, mini_sudoku. Low and high pass rates
    alternate, so a run cut short still covers both.
- **Forgetting:** each new family's trained model is also evaluated on survo's 500 problems, test 6's check. For the
  reused families it is test 5's round-1 model on the other family.
- **Too few successes:** a family with fewer than 10 correct, finished attempts is not trained. Its gain is then not
  measured, and it is left out of the correlation.

### The probe (main signal), computed before the family's real training

1. Split the practice problems into two halves by index (even and odd), so the held-out attempts are on other
   problems.
2. Train a LoRA on one half's correct, finished attempts, with the real training's settings and seed.
3. On the other half, score up to 200 correct and 200 wrong finished attempts (random, seed 0). Each score is the mean
   log-probability per completion token, before and after the probe's training.
4. The probe's value for that direction is the gain on the correct attempts minus the gain on the wrong ones, in nats
   per token.
5. Repeat with the halves swapped, and average the two directions.

- **Meaning:** a positive value says that learning from some problems' successes makes other problems' *correct*
  reasoning more likely than their *wrong* reasoning, so the successes share a method. Near zero says the training only
  learned what right and wrong answers have in common, such as format, which is what lucky successes would give.
- **Needs** at least 10 correct and 10 wrong finished attempts in each half. Otherwise it is undefined, and the family
  is left out of the correlation.

### Other signals, reported alongside (exploratory)

- the probe's gain on the correct attempts alone;
- the number of correct, finished attempts (the training set's size);
- accuracy among finished practice attempts;
- familiarity: the base model's mean log-probability per token of its own correct attempts;
- reliability: correct attempts per problem solved at least once (of 8);
- the base pass@1.

### Code and cost

- **Code:** `phase0/test7_families.py` (stages practice, base, probe, train, evaluate, report), run by
  `phase0/run_test7.sh`. Practice, training and evaluation are tests 4 and 5's functions, unchanged. Every stage skips
  outputs that already exist, so an interrupted run restarts where it stopped.
- **Cost:** local GPU only, about 40 hours: per new family, practice 1.3–2 hours, two evaluations of about 1 hour each,
  the survo check about 1.25, the probe and the training 0.1–1.4 each (they grow with the number of successes); the two
  reused families' probes about 1 hour together. The PC stays on throughout.

## Reading, fixed before the runs

- **Per family:**
  - the round-1 gain over the base model, with a paired bootstrap 95% interval over the 500 problems (2,000 rounds,
    seed 0);
  - the change on the forgetting family, with the same statistic.
- **Counts:** families with a gain of at least 3 points and the interval above 0; families with the interval below 0;
  families whose forgetting interval is below 0.
- **Main:** Spearman's rank correlation between the probe and the gain, over the families where both are defined, with
  an exact one-sided permutation p-value (all orderings).
  - **"The probe predicts which families gain"** if rho is at least 0.65 and p is below 0.05.
  - **Not testable** with fewer than 6 such families.
- **Reported alongside, with no pass line:**
  - the other signals' rank correlations with the gain, with exact two-sided p-values. Six exploratory tests, so a
    single p below 0.05 among them is weak evidence;
  - the probe as a gate: the total gain of the families it keeps (probe above 0), against keeping all of them and against
    keeping only those that gained.
- **Report:** `python phase0/test7_families.py report` (written to `runs/test7/report.md`).
- **A stage that fails** is rerun once and noted.

## Limitations, stated before the results

- **Eight families allow only strong correlations to show:** with 8, rho must reach about 0.64 to give p below 0.05.
  A weaker link would read as "does not predict".
- **One training per new family,** against two for the reused ones. Test 6 measured training noise at 0.3–1.1 points.
- **The reused families' base evaluations ran in September** with the same code. The new families' base and trained
  evaluations run in the same week.
- **Forgetting is measured on one family,** survo (quantum_lock for survo itself), not on all of them.

## Checks before the main run

- **A unit test** of the split, the statistics and the report on synthetic records (`tests/test_test7_families.py`).
- **A dry run** on tiny sizes with `phase0/test7_dry_run.sh`: plumbing only, results discarded.

## Change log

- **2026-10-22: approved by the author** ("Approve"), on the proposal: 6 more families, the probe, the line rho ≥ 0.65
  with p below 0.05, about 36 GPU-hours. Working out the stages put the cost at about 40.
- **Unit tests passed** (`tests/test_test7_families.py`, 5 tests), and test 6's tests still pass.
- **Dry run passed** (`runs/test7_dry/20261022_hhmmss`, `logs/test7_dry.log`). Every stage ran: the
  survo probe from test 5's practice, then course_schedule's practice, base evaluation, probe, training, both
  evaluations and the report. Its numbers are discarded. Only the report's formatting changed after it (no "n/a%").
- **Full run started 2026-10-22** from commit 76fa08e (`logs/test7.log`). A first, detached launch
  died at once with an empty log; it was restarted attached to its WSL call, as test 6 was.
- **Finished 2026-10-24 (33.8 hours, against the estimated 40). Result: the probe does not predict which
  families gain (rho +0.52, one-sided p 0.098).** 5 of 8 families gained, none lost, 2 caused forgetting on survo.
- **Post-hoc, not pre-registered:** the two largest gains (course_schedule, mini_sudoku) were the answers' layout, which
  the models learned from their own successes (`phase0/test7_format_check.py`). Write-up: `results/test7_families.md`.
