# Test 5: more experience, and a surprise gate on what to consolidate. NO-GO (2026-10-03)

Pre-registration: `PREREGISTRATION.md`, "Test 5". Script: `phase0/test5_gate.py` (`run_test5.sh`).
Outputs: `runs/test5/`, report `runs/test5/report.md`.

## Setup

- **Model and training:** Qwen3-1.7B self-trains with LoRA, as in test 4.
- **Experience:** 8 attempts at each of 200 problems per family, with a 6,144-token cap. That
  gave 266 correct, finished attempts on survo and 334 on quantum_lock.
- **Arms:**
  - **R-all:** every correct, finished attempt.
  - **R-gate:** only attempts from problems solved at most 2 times in 8 ("surprising"
    successes); 62 examples on survo, 87 on quantum_lock.
  - **R-random:** a random sample of R-all, the same size as R-gate.
- **Evaluation:** 500 problems × 2 samples on the trained family, and on the other family for
  forgetting.

## Result

| Trained on | Arm | Examples | Own pass@1 | Own gain [95% CI] | Other family [95% CI] |
| --- | --- | --- | --- | --- | --- |
| survo | R-all | 266 | 21.5 | **+4.4 [+1.8, +7.1]** | −0.5 [−3.5, +2.7] |
| survo | R-gate | 62 | 18.1 | +1.0 [−1.4, +3.5] | +1.1 [−1.8, +4.0] |
| survo | R-random | 62 | 21.3 | **+4.2 [+1.4, +7.1]** | +0.6 [−2.3, +3.6] |
| quantum_lock | R-all | 334 | 20.1 | −0.8 [−3.7, +2.2] | −2.3 [−4.8, +0.3] |
| quantum_lock | R-gate | 87 | 19.7 | −1.2 [−4.1, +1.8] | −2.6 [−5.1, +0.1] |
| quantum_lock | R-random | 87 | 19.4 | −1.5 [−4.5, +1.6] | −2.8 [−5.5, −0.1] |

- **Pass line 1 (learning works): NO-GO.** Survo's R-all gain of +4.4 has its interval above 0,
  but it misses the 5-point line. Quantum_lock does not gain at all.
- **Pass line 2 (gate beats random): not testable.** Descriptively, the gate does worse than
  random on survo: +1.0 against +4.2 with the same number of examples.

## Where survo's gain comes from

| survo | Cut off | Correct among finished | Overall |
| --- | --- | --- | --- |
| base | 68.3% | 53.9% | 17.1% |
| R-all | 65.6% | 62.5% | 21.5% |
| R-gate | 65.2% | 52.0% | 18.1% |
| R-random | 62.9% | 57.4% | 21.3% |

- **Mostly better answers, not just shorter ones.** R-all's finished answers are right 62.5% of
  the time against 53.9%, while the cut-off rate falls only 2.7 points.
- **Training on rare successes does not improve answer quality** (52.0%). Successes on problems the
  model usually fails look more like luck than skill. Surprise selects noise, not lessons.
- **The budget is binding everywhere:** 58–68% of all attempts hit the 6,144-token cap.

## What it means

- **First real gain from self-experience in weights:** +4.4 on survo with a clean interval, small
  and on one of two families. Test 4, with about 30 examples, saw none.
- **More is not better here.** 62 random successes gave about the same gain as 266.
- **"Consolidate what surprised you" is the wrong gate.** It is worse than random selection of
  the same size. A gate on what to learn needs a different signal from rarity of success.

## Later note (2026-10-10): the random reference was noisy

The gate signals diagnostic (`results/gate_signals.md`) drew a second random set of 62 survo successes with the same
training and evaluation. It scored +0.2 [−2.4, +2.9] over base, against +4.2 for this test's random set: two random picks
differ by 4.0 points [−6.8, −1.1]. So "the surprise gate (+1.0) is worse than random" is **not established**. The surprise
gate is within the spread of random 62-example picks.
