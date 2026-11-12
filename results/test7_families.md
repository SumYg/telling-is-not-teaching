# Test 7: self-training on own successes, across 8 families: no signal predicts the gain, and the two largest gains were answer layout

**Final, 2026-10-24.** Plan: `phase0/TEST7_FAMILIES.md` (approved by the author on 2026-10-22, before anything ran). Run:
`bash phase0/run_test7.sh` from commit 76fa08e, 2026-10-22 to 2026-10-24 (33.8 hours, against the
estimated 40). Log `logs/test7.log`; report `python phase0/test7_families.py report` (saved in `runs/test7/report.md`).
Post-hoc check: `python phase0/test7_format_check.py` (output in `runs/test7/format_check.md`).

## Pre-registered outcome: the probe does not predict which families gain

- **Main: rho +0.52, one-sided p 0.098, over 8 families.** The line was rho of at least 0.65 with p below 0.05.
- **Counts, as scored:**
  - **5 of 8 families gained** at least 3 points with the interval above 0: survo, self_reference, jugs,
    course_schedule and mini_sudoku;
  - **none lost** on its own family;
  - **2 caused forgetting** on survo, with the interval below 0: self_reference and course_schedule.

| Family | Base pass@1 | Round 1 | Gain [95% CI] | Forgetting [95% CI] (on) | Successes | Probe (nats/token) |
|---|---|---|---|---|---|---|
| survo | 17.1 | 21.3 | +4.2 [+2.1, +6.6] | −0.5 [−3.5, +2.7] (quantum_lock) | 266 | −0.0018 |
| quantum_lock | 20.9 | 20.6 | −0.2 [−2.6, +2.3] | −2.3 [−4.8, +0.3] (survo) | 334 | +0.0007 |
| self_reference | 9.9 | 16.9 | +7.0 [+4.4, +9.7] | −3.7 [−6.2, −1.1] (survo) | 216 | +0.0004 |
| list_functions | 54.2 | 56.2 | +2.0 [−0.1, +4.1] | +0.8 [−1.6, +3.2] (survo) | 814 | −0.0043 |
| jugs | 10.5 | 13.5 | +3.0 [+0.6, +5.4] | −1.8 [−4.1, +0.6] (survo) | 134 | −0.0002 |
| course_schedule | 38.6 | 62.8 | +24.2 [+20.1, +28.2] | −4.6 [−7.2, −2.0] (survo) | 674 | +0.0037 |
| word_ladder | 5.4 | 6.0 | +0.6 [−1.1, +2.3] | +1.0 [−1.7, +3.6] (survo) | 53 | −0.0004 |
| mini_sudoku | 59.8 | 77.5 | +17.7 [+13.8, +21.6] | −0.8 [−3.4, +1.9] (survo) | 931 | +0.0021 |

## Reported alongside (fixed in the plan)

- **No other signal predicts the gain either** (Spearman, exact two-sided p):
  - the probe's gain on correct attempts alone: −0.14 (p 0.75);
  - the number of successes: +0.38 (0.36);
  - accuracy among finished practice attempts: +0.26 (0.54);
  - familiarity: +0.38 (0.36);
  - reliability: +0.33 (0.43);
  - the base pass@1: +0.33 (0.43).
- **As a gate,** keeping the families whose probe is above 0 keeps 4 of them, for a total gain of +48.6. Keeping all 8
  gives +58.5, and keeping only those that gained +58.8: on its own family's score, training on everything is nearly
  the best choice.

## Post-hoc check (not pre-registered): two of the five gains were the answers' layout

Inspecting the two largest gains showed that the trained models had mostly learned the scorer's answer layout from
their own successes. `phase0/test7_format_check.py` rescores every saved answer with the family's own scorer after
fixing only two layout slips: a literal backslash-n written where a line break belongs, and true or false in lower case.

- **course_schedule:** the base model wrote "false" in lower case in 270 of its 1,000 answers, and the scorer accepts
  only "False". Its own successes all used "True" and "False", and after training it never wrote "false". With lower case
  forgiven, the base model scores 60.3, and the gain is **+2.5 [−1.0, +6.0]** instead of +24.2.
- **mini_sudoku:** 199 of the base model's 203 finished wrong grids had exactly the right digits, written with a
  literal backslash-n between rows. With that forgiven, the base model scores 79.7, and the gain is **−2.0 [−4.9, +0.9]**
  instead of +17.7.
- **Survo's grids had the same slip, rarely.** Forgiven, survo's round-1 gain is +3.5 instead of +4.2.
- **Test 6's survo pass shrinks to just under its line.** Round 3 minus base is **+4.7 [+2.2, +7.2]** with the slip
  forgiven, against the registered +5.4 [+2.9, +7.9]. Its pre-registered result stands as registered, under the scorer
  the plan fixed, but about 0.7 points of it was layout.

| Family | Gain as scored | Gain, slips forgiven | Forgetting as scored | Forgetting, slips forgiven |
|---|---|---|---|---|
| survo | +4.2 [+2.1, +6.6] | +3.5 [+1.3, +5.8] | −0.5 [−3.5, +2.7] | −0.5 [−3.5, +2.7] |
| quantum_lock | −0.2 [−2.6, +2.3] | −0.2 [−2.6, +2.3] | −2.3 [−4.8, +0.3] | −2.0 [−4.4, +0.5] |
| self_reference | +7.0 [+4.4, +9.7] | +7.0 [+4.4, +9.7] | −3.7 [−6.2, −1.1] | −3.3 [−5.7, −0.7] |
| list_functions | +2.0 [−0.1, +4.1] | +2.0 [−0.1, +4.1] | +0.8 [−1.6, +3.2] | +0.7 [−1.7, +3.1] |
| jugs | +3.0 [+0.6, +5.4] | +3.0 [+0.6, +5.4] | −1.8 [−4.1, +0.6] | −2.4 [−4.7, +0.0] |
| course_schedule | +24.2 [+20.1, +28.2] | +2.5 [−1.0, +6.0] | −4.6 [−7.2, −2.0] | −4.1 [−6.7, −1.4] |
| word_ladder | +0.6 [−1.1, +2.3] | +0.6 [−1.1, +2.3] | +1.0 [−1.7, +3.6] | +0.4 [−2.3, +3.0] |
| mini_sudoku | +17.7 [+13.8, +21.6] | −2.0 [−4.9, +0.9] | −0.8 [−3.4, +1.9] | −1.5 [−4.1, +1.2] |

- **Against the forgiven gains, no signal predicts anything:** the probe −0.29 (p 0.50); none of the others has |rho|
  above 0.50 or p below 0.2.
- **The probe's +0.52 came from the two layout families.** Its contrast between correct and wrong attempts picks up
  layout there, because the "wrong" attempts included right answers in the wrong layout.
- **Where the reasoning gain is:** in the families that still gain, accuracy among finished answers rose (survo 54% to
  62%, self_reference 18% to 28%, jugs 48% to 58%). The share cut off at the token cap moved less: survo 68% to 66%,
  jugs 78% to 77%, self_reference 45% to 39%.

## What it means

- **Training on its own successes improves reasoning in some families, by 3 to 7 points** (survo, self_reference,
  jugs), and in no family did it clearly hurt the family trained on.
- **Its largest effects were layout.** The model learns the scorer's answer format from its own successes. A test of
  learning from experience needs scoring that forgives layout, or a base model that already uses it; otherwise format
  gains pass for skill. Our earlier survo results are only slightly affected (0.7 points).
- **At the level of families, the gate's real job is forgetting, not the gain:**
  - training on self_reference or course_schedule cost survo 3 to 4 points, and quantum_lock or jugs about 2;
  - consolidating everything is nearly the best choice for each family's own score.
  None of this test's signals was built to predict forgetting.
- **No signal available before training ranked the families by their gain,** neither the probe nor the simple
  counts. The family-level gate stays unsolved, and the next test should target forgetting directly.

## Caveats

- **One training per new family,** against two for survo and quantum_lock. Test 6 measured training noise at 0.3 to 1.1
  points.
- **Forgetting is measured on one family,** survo (quantum_lock for survo itself).
- **The post-hoc check fixes only the two slips found** by inspecting the two largest gains. Other families may hide
  others.
- **Eight families** let only strong correlations show; with 8, rho must reach about 0.64 for p below 0.05.
