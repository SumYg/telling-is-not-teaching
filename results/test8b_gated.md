# Test 8b: a gate on top of joint training moves the cost between families but does not lower it

**Final, 2026-10-29.** Plan: `phase0/TEST8B_GATED.md` (approved by the author on 2026-10-29, before anything ran). Run:
`bash phase0/run_test8b.sh` from commit d59389e (10.7 hours, against the estimated 9.5). Log
`logs/test8b.log`; report `python phase0/test8b_gated.py report --forgive` (saved in `runs/test8b/report.md`).

## Pre-registered outcome: the gate is not shown to help

- **Main: the gated adapter minus test 8's full union adapter, averaged over the 8 families: +0.1 [−0.9, +1.1].** The
  line was the interval's lower end above 0, so **the gate is not shown to help.**
- **Secondary: the gated adapter minus the reference: −0.4 [−1.3, +0.6].** The lower end is above −1.5, so it **holds
  the gains of the best separate choice.** That is not a gain from the gate: see the post-hoc check below.
- **Training:** 2,221 examples from the five kept families (survo, self_reference, jugs, course_schedule, mini_sudoku);
  mean loss by epoch 0.103, 0.097, 0.092.

| Family | Kept | Base | Reference | Full union (test 8) | Gated union | Gated minus full | Gated minus reference |
|---|---|---|---|---|---|---|---|
| survo | yes | 17.1 | 21.4 | 20.0 | 18.9 | −1.1 [−4.0, +1.6] | −2.5 [−4.8, −0.1] |
| quantum_lock | no | 20.9 | 20.9 | 18.0 | 22.0 | +4.0 [+0.8, +7.1] | +1.1 [−1.8, +4.0] |
| self_reference | yes | 9.9 | 16.9 | 16.7 | 15.5 | −1.2 [−3.9, +1.5] | −1.4 [−4.2, +1.4] |
| list_functions | no | 54.2 | 54.2 | 55.1 | 52.9 | −2.2 [−4.6, +0.3] | −1.3 [−3.6, +1.1] |
| jugs | yes | 10.5 | 13.5 | 12.2 | 14.1 | +1.9 [−0.7, +4.6] | +0.6 [−2.0, +3.2] |
| course_schedule | yes | 38.6 | 62.8 | 64.5 | 63.6 | −0.9 [−4.6, +2.9] | +0.8 [−2.7, +4.3] |
| word_ladder | no | 5.4 | 5.4 | 4.5 | 4.1 | −0.4 [−2.1, +1.3] | −1.3 [−3.0, +0.4] |
| mini_sudoku | yes | 59.8 | 77.5 | 78.2 | 78.6 | +0.4 [−2.6, +3.4] | +1.1 [−1.8, +4.0] |

The reference is the family's own adapter (test 7) where the gate keeps it, and the base model where it does not.

## Reported alongside (fixed in the plan)

- **Kept against left-out families,** gated minus full union: −0.2 [−1.5, +1.1] over the 5 kept, +0.5 [−1.0, +1.8]
  over the 3 left out.
- **With test 7's layout slips forgiven,** the readings are the same: main +0.0 [−0.9, +1.0], secondary −0.4 [−1.3,
  +0.6].

## Post-hoc checks (not in the plan; `python phase0/test8b_posthoc.py`)

- **Test 8's full union adapter also holds against this test's reference:** −0.4 [−1.4, +0.5]. The two references
  differ. Test 8 compared with each family's own adapter everywhere. This test uses the base model for the left-out
  families: a lower bar on list_functions (54.2 against its own adapter's 56.2) and word_ladder (5.4 against 6.0), a
  slightly higher one on quantum_lock (20.9 against 20.6), 0.3 points lower on average. So test 8's "narrowly not held"
  and this test's "holds" come from the change of reference, not from the gate.
- **Against the base model,** both adapters gain the same: gated +6.7 [+5.6, +7.8], full union +6.6 [+5.6, +7.6] as
  scored, and +1.4 for both with the slips forgiven.

## What it means

- **Leaving families out fixes the family it was aimed at:** quantum_lock goes from 18.0 with the full union to 22.0,
  +4.0 [+0.8, +7.1], back to about the base model's 20.9.
- **But the cost moves rather than shrinks.** list_functions loses the partial gain it kept in the full union (52.9
  against 55.1), and survo, a kept family, now keeps less of its gain: 18.9 against its own adapter's 21.4, −2.5 [−4.8,
  −0.1]. On average the two joint adapters are the same.
- **For the gate:** dropping the families whose own training did not gain is not enough to make one joint adapter
  better. Joint training on all eight families' successes did as well, and per-family differences of 1 to 2 points in
  either direction could come from training noise alone. A gate decision that matters would have to pick out smaller
  units than families, or weigh the data rather than drop it.

## Caveats

- **One training per adapter,** with no second seed; test 6 measured training noise at 0.3 to 1.1 points per family.
- **The gated adapter trained on fewer examples** (2,221 against 3,422), so fewer steps as well as different data.
- **The gate used test 7's outcomes,** measured on the same 500 evaluation problems. A cheaper gate is still missing
  (tests 7 and 7b).
