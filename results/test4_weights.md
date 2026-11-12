# Test 4: experience into weights. NO-GO (2026-10-01)

Pre-registration: `PREREGISTRATION.md`, test 4, approved in chat 2026-10-01, with logged
changes. Code: `phase0/test4_weights.py`, run by `phase0/run_test4.sh` on the local RTX 5070 Ti
in about 9 h. Outputs are in `runs/test4/`.

## Result

pass@1 on 500 new problems × 2 samples (seed 3000); the base and playbook rows are test 2's.

| Family | Condition | pass@1 | Gain vs base | 95% CI | Cut off | Finished wrong |
| --- | --- | --- | --- | --- | --- | --- |
| survo | base | 17.1 | | | 68.3% | 14.6% |
| survo | text playbook (test 2) | 10.7 | −6.4 | [−9.0, −3.7] | 76.7% | 12.6% |
| survo | **W1** playbook compiled into weights | 8.3 | **−8.8** | [−11.3, −6.3] | 67.1% | 24.6% |
| survo | **W2** own experience into weights | 18.3 | **+1.2** | [−1.4, +3.9] | 68.7% | 13.0% |
| quantum_lock | base | 20.9 | | | 57.8% | 21.3% |
| quantum_lock | text playbook (test 2) | 22.3 | +1.4 | [−1.6, +4.4] | 54.5% | 23.2% |
| quantum_lock | **W1** | 22.1 | **+1.2** | [−1.7, +4.2] | 52.0% | 25.9% |
| quantum_lock | **W2** | 19.2 | **−1.7** | [−4.6, +1.4] | 61.1% | 19.7% |

**NO-GO:** W2 does not reach +5 points with the CI above 0 on either family.

**Forgetting.** W2 trained on one family loses a little on the other:

| Trained on | Evaluated on | Change | 95% CI |
| --- | --- | --- | --- |
| survo | quantum_lock | −2.3 | [−5.3, +0.8] |
| quantum_lock | survo | −3.2 | [−5.8, −0.7] |

## Training data

| Arm | survo | quantum_lock |
| --- | --- | --- |
| W2: correct experience attempts | 27 | 35 |
| W2: rationalizations kept | 4 of 173 (160 referred to the hint) | 0 of 165 (all 165 referred to the hint) |
| W1: playbook-conditioned outputs that finished | 36 | 101 |

So W2 was in effect rejection-sampling fine-tuning on the model's own ~30 successes. Training
loss fell steadily (e.g. survo W2: 0.130 → 0.110 over 3 epochs).

## What it shows

1. **Compiling a playbook into the weights reproduces the playbook's effect.** On survo, the text
   playbook gave −6.4 and W1 −8.8. On quantum_lock, +1.4 and +1.2.
   - So yesterday's question is answered: the text lessons did not hurt because of their length
     in the prompt. They hurt because of the behaviour they cause, and that behaviour carries over
     when trained in.
   - Text vs weights is not the deciding variable. What the lesson does to behaviour is.
2. **About 30 of the model's own successes are too little to improve it** (W2: +1.2 and −1.7).
   They also cost a little on the other family (−2.3 and −3.2), a small sign of the interference
   the continual-learning plan worries about.
3. **Hint-based rationalization (STaR) does not work cleanly with a thinking model.** In 325 of
   338 cases the reasoning referred to the hint, so it could not be used as training data
   without the hint.

## Where this leaves Project 1

Every route to self-improvement from a small amount of its own experience has now failed:

| Route | Setting | Effect |
| --- | --- | --- |
| text lessons | Reasoning Gym, 1.7B | hurt |
| text lessons | SWE, 31B | null |
| text lessons | ARC-AGI-3, Franzen harness | null |
| lessons compiled into weights | Reasoning Gym, 1.7B | same as the text |
| own successes into weights | Reasoning Gym, 1.7B | null, mild forgetting |

The common factor is the learning signal: a few dozen successes, or lessons the model writes
for itself. Neither carries much the model does not already have.
