# Phase 0 pre-registration

Written 2026-09-27, before any test runs. A pass line may change only with a dated note
below, written before the affected test runs.

## Common settings

- Learner: `Qwen/Qwen3-1.7B`. Static comparison: `Qwen/Qwen3-8B-AWQ`.
- Sampling follows the Qwen3 model card: thinking on uses temperature 0.6 and top-p 0.95;
  thinking off uses temperature 0.7 and top-p 0.8.
- Thinking on, off or capped is decided from the speed test and recorded here before test 1.
- Tasks: Reasoning Gym families at default difficulty, with fixed seeds. After each attempt
  the environment reveals the correct answer, which is the feedback lessons are written from.
- "Points" means percentage points of pass@1, the average success over samples.

## Test 1: gap and headroom

- Candidates: about 20 Reasoning Gym families, 300 problems each (seed 1000).
- The 1.7B draws 8 samples per problem, giving pass@1 and pass@8. The 8B draws 4 samples per
  problem, giving pass@1.
- A family **qualifies** if both hold:
  - the 1.7B's pass@1 is between 5% and 30%;
  - the 8B's pass@1 is at least 15 points higher.
- Among qualifying families, prefer those where the 1.7B's pass@8 is below the 8B's pass@1.
  There, sampling alone does not close the gap.
- **Go** if at least 3 families qualify.
- **If not:** adjust difficulty once and re-test, and rely more on invented families.

## Test 2: lessons help other problems, in context

For up to 3 qualifying families:

1. **Experience:** the 1.7B attempts 200 problems (seed 2000). After each failure it sees
   its answer and the correct one. It writes one general lesson of at most 80 words, with no
   numbers specific to that problem.
2. **Playbook:** the 1.7B condenses its lessons into at most 10.
3. **Evaluation:** 500 new problems (seed 3000), 4 samples each, with and without the
   playbook in the prompt. Same problems and seeds in both conditions.

- **Go** if, on at least 2 families, the playbook raises pass@1 by at least 5 points and the
  paired bootstrap 95% confidence interval of the gain excludes 0.
- **Diagnostic, run either way:** repeat step 3 with a playbook written by the 8B. If the
  8B's lessons help the 1.7B but its own do not, the bottleneck is lesson quality, not
  transfer.
- **If no-go:** Project 1 as designed stops. Next steps are to train reflection first, or
  report the negative result.

## Test 3: room for a learned controller (text-only proxy)

- Lifetimes of about 500 problems, of four types:
  1. stable rules, few lessons;
  2. stable rules, many lessons (mixed families);
  3. rules that drift mid-lifetime (an invented family);
  4. a mix of 2 and 3.
- The hand rule adds a lesson to the playbook when its paired replay gain is at least
  τ points. Sweep τ over {0, 2, 5, 10}, with 2 seeds per cell.
- **Room** if the best τ differs between at least two lifetime types, and using one type's
  best τ on another costs at least 3 points, with the 95% confidence interval excluding 0.
- This proxy can show that the best setting varies. It cannot show that the setting never
  varies for moving lessons into weights. A null result here is re-checked against the
  Phase 2 map before Project 2 is dropped.
- **Either way, Project 1 continues.** This test only decides whether Project 2 is worth doing.

## Test 2b: budget-aware lessons (added 2026-09-29, before it runs)

Added after test 2's survo result. Both playbooks lowered pass@1 by 6–9 points, almost
entirely through more answers cut off at the token cap. About four in five failures were cut
off, but the lesson writer saw only "your final answer: None". Test 2b changes only the
lesson writing:

- The writer is told what happened on each failure: ran out of the token budget, finished
  without a formatted answer, or answered wrongly. It is also told the budget.
- It is asked for lessons that reach the answer in fewer reasoning steps, and not to
  recommend extra checking. Condensing drops lessons that only add checks.
- Everything else is as in test 2: survo, the same 200 experience attempts (reused), writers
  1.7B and 8B, and 500 evaluation problems (seed 3000) with 2 samples each. The no-playbook
  baseline is reused from test 2: same problems, seed, model and solving code.
- **Go** if the 1.7B's own playbook raises survo's pass@1 by at least 5 points, with the
  paired bootstrap 95% interval above 0. The 8B-written playbook is a diagnostic, as before.
- Reported alongside: the share of answers cut off at the cap and finished but wrong, per
  condition.
- **If go:** Project 1 continues with budget-aware lesson writing; test 1's families are
  re-scoped. **If no-go:** report the negative result. In-context lessons did not help this
  model under this budget. Next is training reflection first, or a larger model.

## Test 4: experience into weights (added 2026-10-01, before it runs)

The author approved the design in chat on 2026-10-01 ("Go ahead with the weights test"). This
section fixes its settings.

- **Background:** text lessons failed three times. Test 2 and 2b on survo and quantum_lock hurt
  or did nothing, as did the SWE go/no-go and the ARC cross-game test. The question now is
  whether the same experience, put into the weights, helps where text did not.
- **Model and families:** Qwen3-1.7B; survo and quantum_lock, the two families that passed
  test 1 and were used in test 2.
- **Experience:** the 1.7B attempts test 2's 200 experience problems (seed 2000), 1 sample each,
  with thinking on and a 6,144-token cap. It is regenerated with the same settings because test
  2 did not store the text, so individual outcomes may differ slightly from test 2's.
- **Arms:** each is one LoRA adapter on the 1.7B, trained separately per family and merged for
  evaluation.
  - **W2, experience into weights (primary; STaR with rationalization).** Training data:
    - every correct experience attempt, as generated (thinking plus answer)
    - for each failed problem, one attempt in which the 1.7B is given the correct answer as a
      hint, kept only if it finishes within the cap and its final answer is correct; it is
      trained on the original prompt, without the hint
  - **W1, lessons into weights (a diagnostic).** The 1.7B solves the 200 experience problems
    with test 2's 1.7B-written playbook in its system prompt. Every output that finishes within
    the cap, correct or not, is trained on the prompt without the playbook. W1 shows whether the
    playbook hurt because of its length in the context or because of the behaviour it caused.
- **Training:**
  - LoRA rank 16, alpha 32, dropout 0.05, on all attention and MLP projections
  - loss on the completion tokens only
  - AdamW, learning rate 1e-4 with a cosine schedule and 5% warmup, 3 epochs, effective batch 8
  - bf16 with gradient checkpointing; sequences of at most 6,656 tokens
- **Evaluation:** identical to test 2: 500 new problems (seed 3000), 2 samples each, thinking
  on, a 6,144-token cap and the same solving code. The no-lesson baseline is test 2's
  `eval_none`: same problems, seeds, model and code.
- **Go** if W2 raises pass@1 by at least 5 points, with the paired bootstrap 95% interval
  (over problems, as in test 2) above 0, on at least one of the two families.
- **Reported alongside:**
  - W1 against the baseline
  - W2 against test 2's playbook
  - cut-off and finished-but-wrong rates, mean tokens, and training examples per arm
  - **forgetting:** each family's W2 model evaluated on the other family's 500 × 2 problems,
    against the baseline
- **If go:** the first evidence that consolidating experience into weights helps where text
  did not. Project 1 then maps where each works, and Project 2's learned gate follows. A
  plain STaR gain is expected in-domain; the comparison with text at equal experience is the
  point.
- **If no-go:** at this scale, with 200 attempts of experience, consolidation does not help
  either. Next is more experience or a larger model.
- **Compute:** the local GPU only, about 7 h.

## Test 5: more experience, and a gate on what to consolidate (added 2026-10-01, before it runs)

The author approved the direction in chat on 2026-10-01 ("Ok" to option 1: make learning work
with more experience, then test the gate).

- **Background:** test 4's ~30 self-successes did not improve the 1.7B and caused mild
  forgetting.
- **Questions:**
  - With 8× more experience, does self-training help?
  - If it does, does a gate on what to consolidate do better than consolidating the same amount
    of experience chosen at random?
- **Model, families and training:** as in test 4: Qwen3-1.7B, survo and quantum_lock, the same
  LoRA settings, 3 epochs.
- **Experience:** the 1.7B attempts the 200 experience problems (seed 2000) **8 times each**,
  with thinking on and a 6,144-token cap.
- **Arms** (each one LoRA per family; every example uses its plain prompt):
  - **R-all:** every correct attempt that finishes within the cap.
  - **R-gate (surprise gate):** correct, finished attempts only from problems the model usually
    fails, at most 2 of 8 correct. The rule is "consolidate when the experience is surprising".
  - **R-random:** a uniform random sample (seed 0) of R-all's examples, the same size as R-gate.
- **Evaluation:** as in test 4: 500 problems × 2 samples (seed 3000) on the trained family, and
  the same on the other family for forgetting. The baselines are test 2's `eval_none` for both
  families.
- **Pass lines:**
  1. **Learning works** if R-all raises the trained family's pass@1 by at least 5 points, with
     the paired bootstrap 95% interval above 0, on at least one family.
  2. **The gate helps:** on a family that passes (1), compare R-gate with R-random on the **net
     effect**: the gain on the trained family plus the change on the other family. Go if R-gate's
     net effect beats R-random's by at least 3 points, with the bootstrap 95% interval above 0.
     The bootstrap resamples each family's problems separately; 2,000 rounds, seed 0.
- **Reported alongside:**
  - every arm's gain on its own family and its change on the other family
  - training examples per arm, and how many problems the gate admits
  - cut-off rates and mean tokens
- **If (1) fails:** even 8× experience does not improve this model at this scale, and the gate
  cannot be tested here. Next is a larger model or a different domain.
- **If (1) passes and (2) fails:** consolidation works, but a surprise gate is no better than
  random selection. A learned gate (Project 2) is the next candidate.
- **Compute:** the local GPU only, about 24 h: 2 × 2 h of sampling, 6 trainings, and 12
  evaluations of about 75 min each.

## Change log

- 2026-09-27: first version.
- 2026-09-27, before test 1: **thinking is on** for all tests, with at most 6,144 new
  tokens per answer. In the speed test on four easy families, thinking raised the 1.7B's
  average accuracy from 21% to 77% for about 25% more time.
- 2026-09-27, before test 1: **test 1 runs in two stages**, to save about 15 GPU-hours.
  Pass lines are unchanged.
  - Stage A screens all candidates with the 1.7B only: 100 problems (seed 1000), 2 samples
    each. It drops families with pass@1 below 1.5% or above 45%. With this sample size, a
    family whose true rate is inside 5–30% is dropped less than 4% of the time.
  - Stage B applies the full protocol above to the remaining families.
- 2026-09-28, after the screen and before any stage B result: stage B runs **one family at
  a time** (learner, then comparison model), highest screen pass@1 first. Only the order
  changes; each family is measured exactly as above. The screen kept 9 of 20 families.
- 2026-09-28, after the first stage B result (survo) and before any other: **stage B stops**
  once 3 families qualify, or after the first 5 families in the order above, whichever
  comes first. This saves compute: each family costs about 5 GPU-hours.
- 2026-09-28, before test 3 runs: **test 3 is reduced** from about 30 to about 7 GPU-hours.
  - Two lifetime types instead of four: stable rules, and one rule change at the midpoint.
  - τ swept over {0, 5, 10}, with 2 seeds and lifetimes of 300 problems.
  - The pass line is unchanged, applied to these two types.
- 2026-09-28, before test 2 runs: **test 2's evaluation uses 2 samples per problem**
  instead of 4, to save about 12 GPU-hours. Answers in the qualifying families run about
  5,700 tokens. The 95% interval on the gain widens from about ±2 to ±3 points. The pass
  line is unchanged.
- 2026-10-01, before test 4 runs: W2's rationalizations that refer to the hint ("hint", "given/provided
  answer", "stated answer") are dropped too.
  Otherwise the model would be trained to refer to a hint it will not have at test time. Their
  number is reported.
