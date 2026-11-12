# Test 6: does self-training compound over rounds?

**Drafted 2026-10-19, before anything in it ran. Needs the author's approval; binding from then on, with changes in the
change log.**

## Why

- **The only weights result with a positive interval came from the model's own experience.** In test 5
  (`results/test5_gate.md`), Qwen3-1.7B trained on its 266 correct, finished survo attempts gained +4.4 points
  [+1.8, +7.1]. That missed the +5 line, and quantum_lock did not gain at all.
- **Imitating a stronger model's play hurt on ARC** (`results/arc3_weights.md`), as it did for weaker models in
  arXiv 2609.09134. Self-generated data may be what works.
- **The budget binds here too:** 58–68% of attempts hit the 6,144-token cap. Training on finished successes raised
  correctness among finished answers from 53.9% to 62.5%.
- **Each round's successes would come from a better model,** so rounds could compound. STaR and ReST-EM retrain the base
  model on each round's own successes.
- **A carrier that passes is what a learned gate needs first** (`results/gate_signals.md`): so far none has passed.

## Design

- **Model, family, settings:** test 5's.
  - Qwen3-1.7B with LoRA, test 4's settings (3 epochs, learning rate 1e-4, accumulation 8);
  - thinking on, a 6,144-token cap;
  - survo first.
- **Practice:** test 5's 200 survo problems (seed 2000), 8 attempts each per round.
- **Rounds:**
  - **Round k** trains a fresh LoRA on the base model with every correct, finished attempt from round k's practice. That
    practice is played by the round k−1 model, and round 0 is the base model.
  - **Round 1 is test 5's R-all,** reused: its experience, its model (21.5) and its evaluation.
  - **Round 2:** the round-1 model plays the practice problems, and round 2 trains on its successes.
  - **Round 3:** the round-2 model plays them, and round 3 trains on its successes.
  - Practice sampling uses seed 2000 + k.
- **Training seeds,** to measure training noise (two random 62-example picks differed by 4 points in the gate study):
  - **Round 1:** test 5's model plus seed 2.
  - **Round 2:** seed 1.
  - **Round 3:** seeds 1 and 2.
  - The next round's practice is played by seed 1's model (in round 1, test 5's).
- **Evaluation:** test 2's: 500 survo problems × 2 samples (seed 3000). These are the same problems as test 5's base
  (17.1) and R-all (21.5). Round 3's seed-1 model is also evaluated on quantum_lock, for forgetting.
- **Code:** `phase0/test6_iterate.py`, run by `phase0/run_test6.sh`, reusing test 4's and test 5's functions unchanged.
- **Cost:** local GPU only, about 12 hours:
  - 2 practice rounds (about 2 hours each);
  - 4 trainings (about 30 minutes each);
  - 5 evaluations (about 75 minutes each).

## Reading, fixed before the runs

- **Per-problem pass rate:** for a round with 2 trainings, the mean of the two models' per-problem pass rates.
- **Main: round 3 minus base,** with a paired bootstrap 95% interval over the 500 problems (2,000 rounds, seed 0).
  - **Self-training passes** if the gain is at least 5 points and the interval is above 0: the line of tests 4 and 5.
- **Compounding: round 3 minus round 1,** the same statistic.
  - **Rounds add** if the gain is at least 2 points and the interval is above 0.
- **Reported alongside:**
  - round 2's pass rate;
  - examples per round;
  - the cut-off rate and correctness among finished answers, per round;
  - the difference between the two trainings in rounds 1 and 3;
  - round 3's change on quantum_lock.
- **If survo passes,** quantum_lock gets the same three rounds as a second test (about 12 hours more, with its own
  approval). If not, there is no second family.

## Checks before the main run

- **A unit test of the report** on synthetic records.
- **A dry run** on a few problems with tiny settings: plumbing only, results discarded.

## Change log

- **2026-10-19: approved by the author** ("Approve test 6"), before anything in it ran.
- **Report test passed** (`tests/test_test6_iterate.py`, 3 tests), and test 5's tests still pass.
- **Dry run passed** (`runs/test6_dry/20261019_hhmmss`, `logs/test6_dry.log`, about 60 minutes). Every stage ran: a round-1
  retrain, practice by test 5's model and by a round-2 model, training, evaluation and the report. Its numbers are
  discarded.
- **Full run started 2026-10-19** from commit de713a9 (`logs/test6.log`).
- **Finished (16.7 hours, against the estimated 12). Result: self-training passes (round 3 minus base
  +5.4 [+2.9, +7.9]); rounds do not add (round 3 minus round 1 +1.1 [−0.8, +3.1]).** Write-up:
  `results/test6_self_training.md`. Quantum_lock is next only with the author's approval.
- **Quantum_lock approved by the author on 2026-10-19** ("approve test 6 on quantum lock"), as the plan's second test with the
  same design and reading. `phase0/run_test6.sh` now takes the family as an argument (survo by default, unchanged for it);
  the forgetting check uses the other family. No new dry run: the code is the same, and test 5's quantum_lock round-1
  files and test 2's base evaluation are all in place.
- **Quantum_lock run started 2026-10-19** from commit 9a44090 (`logs/test6_quantum_lock.log`).
- **Quantum_lock finished 2026-10-20 (11.5 hours). Result: self-training does not pass (round 3 minus base
  −0.4 [−3.1, +2.5]); rounds do not add (−0.1 [−2.1, +2.1]); forgetting on survo −3.7 [−6.1, −1.1].** WSL stopped before
  the run's exit line was written; every stage had finished. Write-up: `results/test6_self_training.md`.

