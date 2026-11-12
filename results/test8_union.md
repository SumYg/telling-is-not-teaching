# Test 8: one adapter on all eight families' successes keeps most of each family's gain, but narrowly misses its line

**Final, 2026-10-29** (the run ended on 2026-10-25; its result was read on 2026-10-29). Plan:
`phase0/TEST8_UNION.md` (approved by the author on 2026-10-25, before anything ran). Run: `bash phase0/run_test8.sh` from
commit 65aa015 (11.2 hours, against the estimated 13). Log `logs/test8.log`; report
`python phase0/test8_union.py report --forgive` (saved in `runs/test8/report.md`).

## Pre-registered outcome: not shown to hold the gains, by 0.1 point

- **Main: the union adapter minus each family's own adapter, averaged over the 8 families: −0.7 [−1.6, +0.2].** The
  line was the interval's lower end above −1.5, so one adapter is **not shown to hold the gains**. It is not clearly
  worse either, since the interval includes 0.
- **Training:** 3,422 examples (test 7's eight training sets, unchanged); mean loss by epoch 0.100, 0.093, 0.087.

| Family | Base pass@1 | Own adapter (test 7) | Union adapter | Union minus base | Union minus own |
|---|---|---|---|---|---|
| survo | 17.1 | 21.4 | 20.0 | +2.9 [+0.3, +5.6] | −1.4 [−3.7, +1.1] |
| quantum_lock | 20.9 | 20.6 | 18.0 | −2.9 [−6.1, +0.2] | −2.6 [−5.2, −0.2] |
| self_reference | 9.9 | 16.9 | 16.7 | +6.8 [+4.1, +9.7] | −0.2 [−3.2, +2.7] |
| list_functions | 54.2 | 56.2 | 55.1 | +0.9 [−1.4, +3.1] | −1.1 [−3.4, +1.1] |
| jugs | 10.5 | 13.5 | 12.2 | +1.7 [−0.7, +4.0] | −1.3 [−3.7, +1.2] |
| course_schedule | 38.6 | 62.8 | 64.5 | +25.9 [+21.5, +30.2] | +1.7 [−1.9, +5.2] |
| word_ladder | 5.4 | 6.0 | 4.5 | −0.9 [−2.7, +0.8] | −1.5 [−3.1, +0.0] |
| mini_sudoku | 59.8 | 77.5 | 78.2 | +18.4 [+14.5, +22.3] | +0.7 [−2.7, +3.8] |

## Reported alongside (fixed in the plan)

- **Against the base model,** the union adapter gains +6.6 [+5.6, +7.6] on average as scored, and +1.4 [+0.4, +2.3] with
  test 7's layout slips forgiven.
- **Survo does not suffer:** the union adapter gains +2.9 [+0.3, +5.6] there, where the self_reference and
  course_schedule adapters alone had cost it 3.7 and 4.6 points. Survo's own successes were in the union, and its own
  adapter gained +4.2.
- **quantum_lock is the one family clearly hurt by the mixing:** −2.6 [−5.2, −0.2] against its own adapter, −2.9
  against the base. word_ladder is borderline (−1.5 [−3.1, +0.0]).
- **With the layout slips forgiven,** the main comparison is the same: −0.7 [−1.6, +0.3].

## What it means

- **Training on everything at once keeps most of each family's gain.** The large gains survive in full
  (self_reference, course_schedule, mini_sudoku). The smaller reasoning gains survive in part: survo 2.9 of 4.2, jugs
  1.7 of 3.0, list_functions 0.9 of 2.0. The average cost is 0.7 points, and the forgetting that single adapters caused
  on survo is gone.
- **But it misses the pre-registered line by 0.1 point,** so "one adapter holds the gains within 1.5 points" is not
  established, and one family (quantum_lock) is clearly worse off.
- **For the gate, the family-level choice is now clearer:**
  - separate adapters keep every gain but need a router to pick one per problem;
  - one adapter per family in turn causes forgetting that is expensive to detect (tests 7 and 7b);
  - one joint adapter costs about 0.7 points on average, spread over the families, with no forgetting.
  The open question is whether a gate that leaves out families whose own training did not help (here quantum_lock and
  word_ladder) would remove most of that cost.

## Caveats

- **The union adapter trained for about 13 times as many steps** as survo's own, with the same learning rate and
  epochs. The difference can come from that as well as from the mixing.
- **One training,** with no second seed; test 6 measured training noise at 0.3 to 1.1 points per family.
- **The miss is narrow:** −1.6 against a line of −1.5, so a second seed could land on either side.
