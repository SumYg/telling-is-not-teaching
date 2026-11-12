# Diagnostic: can the model's own neural signals pick which experiences to consolidate?

**An exploratory diagnostic, not a pre-registered test.** The author asked to "just test" it (2026-10-10). This
plan and its decision rule were written before anything ran.

## Why

Every gate so far was a hand-written rule. Test 5's surprise gate (learn only from problems solved at most 2 times in 8)
did worse than a random pick of the same size: +1.0 against +4.2 (`results/test5_gate.md`). The research design calls for a
gate learned from the model's own signals. This diagnostic asks the first question: do such signals carry any information
about which experiences are worth consolidating?

## Setup

- **Everything is test 5's survo setup except which examples are trained:**
  - Qwen3-1.7B, and the same pool of 266 correct, finished attempts.
  - The same LoRA training: rank 16, 3 epochs, learning rate 1e-4.
  - The same evaluation: 500 problems × 2 samples, seed 3000.
- **Each arm trains on 62 examples**, the size of test 5's gate and random arms.
- **Signals**, computed with the untrained model on exactly the tokens training uses. LoRA has test 4's configuration, with A
  fixed by seed 0 and no dropout.
  - **Gradient agreement (primary):** the cosine between an example's gradient and the sum of all other examples'
    unit-length gradients. The gradient is taken on LoRA's B matrices, which is a random projection of the full-weight
    gradient. High agreement means the example pushes the weights the same way as the rest of the experience. This is the
    design's "consolidate what agrees across experiences".
  - **Familiarity:** the mean log-probability of the attempt's tokens under the model. Both ends are run.
- **Arms, in this order:**
  1. `grad_top`: the 62 with the highest gradient agreement.
  2. `fam_top`: the 62 the model finds most likely.
  3. `fam_bottom`: the 62 it finds least likely.
  4. `random_s1`: a second random draw of 62 (seed 1), as a noise reference.

## Decision rule

- **Comparison:** own-family pass@1 against test 5's random arm (62 examples, +4.2 over base). Paired bootstrap over the 500
  problems, 2,000 rounds, seed 0.
- **A signal "selects better than random"** if its arm beats test 5's random arm by at least 3 points with the 95% interval
  above 0. This is test 5's gate line, on the trained family only.
- **If gradient agreement passes:** pre-register a learned gate, a small model on per-example neural features trained on
  measured benefit.
- **If no signal passes:** these signals do not pick better experiences in this setup. A learned gate would then need
  per-example benefit measured by training, not these proxies.
- **The noise reference:** if `random_s1` differs from test 5's random arm by 3 points or more, differences of that size are
  within selection and training noise. A pass would then be read as not established.
- **Also reported, descriptive:** each arm's gain over the untrained model, the selections' overlap, and how easy the chosen
  problems were (their successes out of 8).

## Not measured, and cost

- **Not measured:** forgetting on quantum_lock, to save time.
- **Compute:** the local RTX 5070 Ti. Signals take about 15 minutes; each arm about 1.3 hours (training, then 1,000
  generations). About 5.5 hours in all.
- **Code:** `phase0/gate_signals.py`; run with `bash phase0/run_gate_signals.sh > logs/gate_signals.log 2>&1`.

## Result (2026-10-10)

No signal passes, and the noise reference is triggered: the second random pick scored 4.0 points below test 5's
(`results/gate_signals.md`, `runs/gate_signals/report.md`). Descriptively, familiarity (fam_top, 23.0) beat every same-day
arm, including the new random pick by +5.7 [+3.0, +8.3]. Gradient agreement mostly measured solution length.

## Part 2: replicating the familiarity lead (planned 2026-10-10, before running)

The author approved running it tonight. `fam_top` scored 23.0, the best arm so far, but two random picks differed by 4 points,
so one run cannot establish it. It was chosen for replication after its score was seen, so **only new runs are judged**.

- **Arms**, with the same pool, training and evaluation as part 1:
  - `random_s2`, `random_s3`, `random_s4`: three more random picks of 62 (seeds 2, 3, 4). With test 5's random pick and
    `random_s1` that makes five random picks.
  - `fam_top_t1`, `fam_top_t2`: `fam_top`'s 62 examples retrained with training seeds 1 and 2. The seed sets the LoRA
    initialisation and the data order.
  - **Order:** `fam_top_t1`, `random_s2`, `fam_top_t2`, `random_s3`, `random_s4`, alternating so that drift over the night
    affects both kinds.
- **Decision rule.** Familiarity "selects better than random" only if both hold:
  1. Both new `fam_top` runs score above all five random picks. If the seven runs were interchangeable, the chance of that
     would be 1 in 21 (p ≈ 0.048).
  2. The mean of the two new `fam_top` runs is at least 3 points above the mean of the five random picks.
- **If it passes:** a learned gate is worth pre-registering: per-example features from the model, labelled by training on
  many subsets.
- **If not:** the 23.0 is read as a high draw. No neural signal has then been shown to pick better experiences in this setup.
- **Reported alongside:** each run's pass@1 and cut-off share, and the spread of the five random picks. The original
  `fam_top` run is shown but not counted.
- **Compute:** 5 arms of about 1.6 hours, about 8 hours in all. A delayed start
  (`bash phase0/run_gate_replication.sh`).

## Part 2 result (2026-10-11)

Not shown, by the fixed rule: `fam_top_t1` 21.0 is not above all five random picks (two scored 21.3), and the mean gap is
+2.9 points against the +3 required (22.4 against 19.5). Descriptively, all three `fam_top` trainings sit at or above the
top of the random range. A small effect is possible but not established (`results/gate_signals.md`).
