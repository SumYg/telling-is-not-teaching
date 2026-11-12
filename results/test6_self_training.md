# Test 6: self-training on own successes, over three rounds (survo, then quantum_lock)

**Final, 2026-10-19.** Plan: `phase0/TEST6_SELF_TRAINING.md` (approved by the author before anything ran). Run:
`bash phase0/run_test6.sh` from commit de713a9 (16.7 hours, against the estimated 12). Log
`logs/test6.log`; report `python phase0/test6_iterate.py report --family survo` (saved in `runs/test6/survo/report.md`).

**Post-hoc note (2026-10-24, from test 7, `results/test7_families.md`):** the base model sometimes wrote survo's grid
with a literal backslash-n between rows, which the scorer rejects, and training on its own successes taught it the
layout. With that slip forgiven, round 3 minus base is **+4.7 [+2.2, +7.2]**, just under the +5 line, and round 1 is
+3.5 instead of +4.2. The registered PASS stands under the scorer the plan fixed, but about 0.7 points of it was layout.

## Pre-registered outcome: self-training passes; rounds do not add

| Round | Training | Examples | Pass@1 | Cut off | Correct among finished |
|---|---|---|---|---|---|
| Base | | | 17.1 | 68.3% | 53.9% |
| 1 | test 5's | 266 | 21.5 | 65.6% | 62.5% |
| 1 | seed 2 | 266 | 21.2 | 66.0% | 62.4% |
| 2 | seed 1 | 330 | 23.0 | 64.8% | 65.3% |
| 3 | seed 1 | 328 | 22.3 | 65.2% | 64.1% |
| 3 | seed 2 | 328 | 22.7 | 64.1% | 63.2% |

- **Main: round 3 minus base +5.4 points, 95% interval [+2.9, +7.9].** The line was a gain of at least 5 points with the
  interval above 0, so **self-training passes**. It is the first carrier of experience in this project to pass its
  pre-registered line.
- **Compounding: round 3 minus round 1 +1.1 [−0.8, +3.1].** The line was at least +2 with the interval above 0, so
  **rounds do not add**.

## Reported alongside (fixed in the plan)

- **Per round, minus base:** round 1 +4.2 [+2.1, +6.6] (two trainings), round 2 +5.9 [+3.3, +8.5] (one), round 3 +5.4
  [+2.9, +7.9] (two).
- **Practice successes stopped growing after round 1:**

  | Practice played by | Correct, finished attempts (of 1,600) | Problems solved at least once (of 200) |
  |---|---|---|
  | The base model (test 5) | 266 | 89 |
  | The round-1 model | 331 | 97 |
  | The round-2 model | 328 | 107 |
- **Training noise is small once all successes are used:** the two trainings differ by 0.3 points in round 1 and 0.4 in
  round 3. Two random picks of 62 successes differed by 4 points (`results/gate_signals.md`), so that spread came from
  which examples were picked, not from training.
- **No forgetting:** round 3 (seed 1) on quantum_lock, minus base, +2.3 [−0.6, +5.5]. One round had given −0.5 in test 5.
- **Where the gain is:** answer quality. Finished answers are right 63–65% of the time against 54%, while cut-offs fall
  only from 68% to 64–66%.

## What it means

- **Training on the model's own verified successes works,** by about 4 to 6 points on survo. By the registered rule it
  passes after three rounds, but most of the gain comes in the first round.
- **Rounds plateau at this scale.** The round-1 and round-2 models found about as many successes in practice (331 and
  328), so later rounds trained on similar data and gained little.
- **The budget still binds.** About 65% of answers still hit the 6,144-token cap, as before training. Self-training made
  answers better, not shorter. A carrier that also made the model finish within budget could add more.
- **For the gate (Project 2), this is the first carrier with a reliable effect,** and its training noise is small (±0.4).
  A gate test can now compare selections of these successes against several random selections of the same size.
- **Survo was the first family;** quantum_lock, the second, follows below.

## Second family: quantum_lock — self-training does not pass, and it costs survo

**Final, 2026-10-20.** Approved by the author 2026-10-19; the same design and reading. Run:
`bash phase0/run_test6.sh quantum_lock` from commit 9a44090 (11.5 hours). Log
`logs/test6_quantum_lock.log`; report `python phase0/test6_iterate.py report --family quantum_lock` (saved in
`runs/test6/quantum_lock/report.md`). The run's final exit line was never written, because WSL stopped soon after the
script ended. The script had printed "TEST 6 DONE" after every stage.

| Round | Training | Examples | Pass@1 | Cut off | Correct among finished |
|---|---|---|---|---|---|
| Base | | | 20.9 | | |
| 1 | test 5's | 334 | 20.1 | 60.5% | 50.9% |
| 1 | seed 2 | 334 | 21.2 | 57.7% | 50.1% |
| 2 | seed 1 | 307 | 22.5 | 57.2% | 52.6% |
| 3 | seed 1 | 317 | 20.9 | 57.8% | 49.5% |
| 3 | seed 2 | 317 | 20.2 | 57.8% | 47.9% |

- **Main: round 3 minus base −0.4 [−3.1, +2.5]: self-training does not pass.**
- **Compounding: round 3 minus round 1 −0.1 [−2.1, +2.1]: rounds do not add.**
- **Per round, minus base:** round 1 −0.2 [−2.6, +2.3], round 2 +1.6 [−1.7, +5.0], round 3 −0.4 [−3.1, +2.5].
- **Forgetting:** round 3 (seed 1) on survo, minus base, **−3.7 [−6.1, −1.1]**. Trained on quantum_lock, the model got
  worse at survo.
- **The trained models did not even practise better:** 307 and 317 correct, finished attempts of 1,600, against 334 for
  the base model; 104 and 111 problems solved at least once, against 117.
- **Training noise is small here too:** the two trainings differ by 1.1 points in round 1 and 0.7 in round 3.

## Both families together

- **Self-training on own successes is family-dependent:**
  - **On survo** it adds about 5 points and does no harm elsewhere.
  - **On quantum_lock** it adds nothing and costs survo 3.7 points.

  Test 5 saw the same split after one round (+4.4 against −0.8).
- **This is exactly the choice a gate has to make.** Consolidating survo's experience helps; consolidating
  quantum_lock's helps nothing and causes forgetting. A gate that learns *when* to consolidate should keep the first and
  refuse the second.
- **These two families are now a ready test bed for the gate (Project 2).** Each has measured outcomes, small training
  noise and a known sign. The question becomes whether any signal available before training, from the model or from its
  successes, predicts which family's experience to keep.
