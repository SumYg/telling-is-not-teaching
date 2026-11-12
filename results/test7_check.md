# Test 7b: a likelihood check after training predicts neither forgetting nor gains

**Final, 2026-10-25.** Plan: `phase0/TEST7_CHECK.md` (approved by the author that day; a diagnostic, since test 7's
outcomes were known when it was defined). Run: `bash phase0/run_test7_check.sh` from commit b4ef76d (3.7 hours). Log `logs/test7_check.log`; report `python phase0/test7_check.py report` (saved in
`runs/test7_check/report.md`).

## Pre-registered outcome: neither check predicts

- **Main: the forgetting check does not predict the measured forgetting.** Rho +0.48, one-sided p 0.12, over 8
  families. The line was rho of at least 0.65 with p below 0.05.
- **Secondary: the gain check does not predict the measured gain.** Rho +0.38, p 0.18.

| Family | Forgetting check (nats/token) | Forgetting measured (on) | Gain check (nats/token) | Gain measured |
|---|---|---|---|---|
| survo | +0.0007 | −0.5 (quantum_lock) | −0.0023 | +4.2 |
| quantum_lock | −0.0030 | −2.3 (survo) | +0.0002 | −0.2 |
| self_reference | −0.0028 | −3.7 (survo) | −0.0008 | +7.0 |
| list_functions | −0.0026 | +0.8 (survo) | −0.0063 | +2.0 |
| jugs | −0.0031 | −1.8 (survo) | −0.0022 | +3.0 |
| course_schedule | −0.0030 | −4.6 (survo) | +0.0031 | +24.2 |
| word_ladder | −0.0028 | +1.0 (survo) | +0.0001 | +0.6 |
| mini_sudoku | −0.0025 | −0.8 (survo) | +0.0026 | +17.7 |

## Reported alongside (fixed in the plan)

- **Against test 7's outcomes with the layout slips forgiven** (`phase0/test7_format_check.py`), at full precision:
  - the forgetting check, rho +0.57 (one-sided p 0.076);
  - the gain check, rho −0.48 (p 0.89).
- **Validation attempts scored** (correct and wrong, at most 200 each, from 400 per family): survo 63 and 49,
  quantum_lock 84 and 72, self_reference 66 and 165, list_functions 200 and 52, jugs 49 and 50, course_schedule 165 and
  133, word_ladder 23 and 200, mini_sudoku 200 and 71. Every family had enough.

## What it means

- **Training on any other family moved survo's likelihoods by nearly the same amount.** All 7 models trained on another
  family lowered survo's correct answers relative to its wrong ones by 0.0025 to 0.0031 nats per token. That held
  whether the model then cost survo 4.6 points or gained it 1.0. The check sees that training moved the model, not which
  training hurts. Only survo's own model, checked on quantum_lock, stands apart (+0.0007).
- **So a likelihood check on saved answers cannot stand in for generation** as the gate's probation test, at least in
  this form. With the slips forgiven the forgetting check comes closer (+0.57), but it still misses the line.
- **Together with test 7's probe, that makes two likelihood signals, before and after training, that fail to rank
  families** by what training does to them. The gate's decisions still need measured outcomes: here about 2.5 GPU-hours
  of generation per candidate adapter.

## Caveats

- **A diagnostic:** the outcomes were known before the check was defined. A pass would have needed confirming; a
  failure needs nothing more.
- **Eight families,** and the forgetting check uses one family (survo) for 7 of them.
- **One validation set per family** (100 problems × 4 attempts, seed 4000).

## Afterwards (not pre-registered): a cheaper generation check is not cheap enough

`python phase0/test7_probation_power.py` resamples test 7's saved survo evaluations to ask how many answers a
generation-based probation check would need.

| Model on survo (test 7's change) | 100 answers | 250 | 500 | 1,000 (the full evaluation) |
|---|---|---|---|---|
| course_schedule (−4.6) | flagged 30% | 49% | 79% | 96% |
| self_reference (−3.7) | 24% | 37% | 62% | 88% |
| word_ladder (+1.0), a false alarm | 2% | 2% | 2% | 1% |

- **A check of 100 to 250 answers misses most of the real forgetting;** catching a 4-point loss about 90% of the time
  takes about 1,000 answers, the whole evaluation (about 75 minutes per checked family).
- **So the gate cannot detect forgetting cheaply, by likelihood or by a small test.** Preventing it, for example by
  training one adapter on several families' successes together, may be the better route.

