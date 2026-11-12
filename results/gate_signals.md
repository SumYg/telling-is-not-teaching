# Gate signals diagnostic: can the model's own signals pick which experiences to consolidate?

**Exploratory, rule fixed before running (`phase0/GATE_SIGNALS.md`); run overnight 2026-10-10 on the local RTX 5070 Ti.**
Report: `runs/gate_signals/report.md`. Setup: test 5's survo pool (Qwen3-1.7B, 266 correct finished attempts); each arm
trains on 62 of them with test 5's LoRA settings and is scored on 500 problems × 2 samples.

## Result by the fixed rule: no signal passes, and the random reference is unstable

| Arm (62 examples) | Pass@1 | Gain over the untrained model [95% CI] | Minus test 5's random pick [95% CI] |
|---|---|---|---|
| test 5 random pick (seed 0) | 21.3 | +4.2 [+1.4, +7.1] | |
| **random_s1** (a second random pick, seed 1) | 17.3 | +0.2 [−2.4, +2.9] | −4.0 [−6.8, −1.1] |
| grad_top (gradient agreement) | 15.7 | −1.4 [−3.9, +1.2] | −5.6 [−8.3, −2.9] |
| fam_top (the solutions the model finds most likely) | **23.0** | **+5.9 [+3.3, +8.4]** | +1.7 [−1.1, +4.5] |
| fam_bottom (least likely) | 18.6 | +1.5 [−1.0, +4.0] | −2.7 [−5.2, −0.2] |
| for reference: test 5, all 266 | 21.5 | +4.4 [+1.8, +7.1] | |
| for reference: test 5, surprise gate | 18.1 | +1.0 [−1.4, +3.5] | |

- **No signal beats test 5's random pick by the required 3 points.**
- **The noise reference is triggered.** Two random picks of 62 differ by 4.0 points [−6.8, −1.1]. By the fixed rule,
  differences of that size are within selection and training noise.

## Descriptive (not covered by the fixed rule)

- **Familiarity is the most promising signal.** Against same-day arms, fam_top scores +5.7 [+3.0, +8.3] over the new
  random pick, +4.4 [+1.9, +7.0] over fam_bottom and +7.3 [+4.7, +9.8] over grad_top. Its 23.0 is the best of every arm in
  tests 4, 5 and this one, above training on all 266 (21.5).
- **Its gain is better answers, not fewer cut-offs:**

  | Arm | Cut off | Correct among finished | Pass@1 |
  |---|---|---|---|
  | untrained | 68.3% | 54% | 17.1 |
  | random_s1 | 67.6% | 53% | 17.3 |
  | fam_top | 65.5% | **67%** | 23.0 |
  | test 5, all 266 | 65.6% | 63% | 21.5 |

- **fam_top also picks shorter, easier attempts** (3,418 trained tokens against 4,244 in the pool; 4.69 successes in 8
  against 4.23). Training on the model's most confident correct solutions looks like consolidating what it nearly has.
- **Gradient agreement mostly measures length** (Spearman +0.77 with trained tokens), so it picked long solutions (5,270
  tokens). It is not the measure of generality intended. A length-controlled version (for example, per-token gradients)
  would be needed before judging the idea.
- **Signal correlations:** agreement vs familiarity −0.58; with problem easiness (successes out of 8), agreement −0.22 and
  familiarity +0.19.

## Correction to test 5

Test 5 read its surprise gate (+1.0) as worse than random (+4.2). The second random pick scores +0.2, so test 5's random
pick may have been a high draw. **"The surprise gate is worse than random" is not established**; a note is added to
`results/test5_gate.md`.

## What it means

- **On the question asked:** no neural signal was shown to pick better experiences than random, by the fixed rule. This
  setup cannot separate gates with single 62-example picks: random picks alone vary by about 4 points.
- **One lead worth a proper test:** the model's own confidence in a correct solution (familiarity). A replication with
  several random picks (to measure the null spread) and several training seeds of fam_top could establish or refute it
  locally in about one night (5 arms × about 1.6 hours).
- **For a learned gate:** labels (measured benefit) must come from many subsets, since single subsets are this noisy.

## Part 2: replication of the familiarity lead (2026-10-11): not shown, by the fixed rule

Rule fixed before running (`phase0/GATE_SIGNALS.md`, part 2): both new `fam_top` retrains above all five random picks, and
their mean at least 3 points above the random mean. Only new runs are judged.

| Arm (62 examples) | Pass@1 | Gain over the untrained model [95% CI] |
|---|---|---|
| random picks: test 5's, s1, s2, s3, s4 | 21.3, 17.3, 18.0, 19.7, 21.3 | mean +2.4 (range +0.2 to +4.2) |
| fam_top_t1 (retrain, seed 1) | 21.0 | +3.9 [+1.3, +6.6] |
| fam_top_t2 (retrain, seed 2) | 23.8 | +6.7 [+4.0, +9.2] |
| fam_top (original, not counted) | 23.0 | +5.9 [+3.3, +8.4] |

- **Both conditions fail, narrowly:**
  1. `fam_top_t1` (21.0) is not above the two best random picks (21.3).
  2. The gap between means is +2.9 points (22.4 against 19.5), against the +3 required.
- **Outcome: familiarity is not shown to select better than random.** By the plan, the original 23.0 is read as a high draw,
  and no neural signal has been shown to pick better experiences in this setup.
- **Descriptively:** all three `fam_top` trainings (23.0, 21.0, 23.8) are at the top of, or above, the random range (17.3 to
  21.3). A real effect of about 3 points is consistent with these runs. With random picks varying by SD 1.8, though,
  establishing it would take more runs per arm (roughly 5 or more) than this diagnostic had.
- **Noise lesson:** single 62-example picks vary over 4 points (17.3 to 21.3) with identical training and evaluation. Any
  gate test at this scale needs several random picks as its reference, as this part did.
