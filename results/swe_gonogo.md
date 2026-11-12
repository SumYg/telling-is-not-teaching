# SWE go/no-go: do model-written lessons transfer across repositories? NO-GO

Pre-registration: `kaggle_swe/PREREGISTRATION.md`, approved 2026-09-29. Kernel
`sumyeung/swe-gonogo-full` v1 (code `308fb3b`), both stages in one Kaggle 4×L4 session, run
2026-09-30. Outputs are in `runs/swe/gonogo/full/`.

## Pass line result

The run covered the 48 rich tasks, 2 samples each, with lessons written from fastapi + requests
failures.

| Condition | Resolved | Finished | Out of time | Context error | Mean tool calls | Prompt tokens per model call |
| --- | --- | --- | --- | --- | --- | --- |
| base | 19 / 96 (19.8%) | 25 | 70 | 1 | 18.6 | 8,853 |
| base + lessons | 20 / 96 (20.8%) | 26 | 70 | 0 | 23.0 | 9,160 |

**Gain: +1.0 point, 95% CI [−2.1, +4.2]** (paired bootstrap over tasks, 2,000 rounds).
**NO-GO**: the pass line needs a gain of at least 5 points with the CI above 0.

## What happened

- **The lessons changed almost nothing.**
  - Over the 96 paired runs: 18 solved in both conditions, 75 in neither, 1 only by base, 2
    only with lessons.
  - The same 11 of 48 tasks were solvable either way. Only three tasks' totals differ:
    rich_3006 1→2, rich_3718 1→2, rich_3454 2→1.
  - Which tasks get solved is nearly fixed per task at this budget.
- **The time limit dominates.** 70 of 96 runs in each condition ran out of the 6 minutes
  (4 tasks in parallel).
- **The lessons made the agent do more, not less.** It averaged 23.0 tool calls against 18.6,
  with about the same time per task (353 vs 357 s). The prompt grew only 3.5% per call, so the
  context-cost failure mode of test 2 was small here.
- **The experience stage** (80 tasks) resolved 13 (fastapi 12/67, requests 1/13). All 67
  failures produced a lesson (none cut off), condensed to 7:
  - most are generic engineering advice (search tracebacks, trace the call chain, mirror
    conventions)
  - one adds work ("update existing tests and documentation")
  - none is specific to the time budget, despite the prompt asking for that
- **The context-window fix** (`max_output_tokens` 8192) held: 1 context error in 272 runs,
  against 1 in 16 in the smoke test.
- **Timing:**
  - server start 469 s
  - experience 1.9 h
  - lesson writing 8 min
  - evaluation 4.8 h (the four runs in parallel)
  - LoRA check a few minutes
  - total about 7.3 h, as estimated

## LoRA check (a precondition for a weights arm, not part of the pass line)

- **The setup:**
  - An adapter was built from the sample's `main_lora` template (layer 0, `q_proj` and
    `o_proj`, rank 4).
  - Both `lora_A` (N(0, 1/√fan_in)) and `lora_B` (N(0, 1)) were randomised.
  - It was served by the wheelhouse vLLM 0.19.1 and listed in `/v1/models` as `random_lora`.
- **The result:** at temperature 0, 24 tokens, the text was **identical** to the base model's,
  and the largest logprob difference was 0.031.
- **Reading it:**
  - My pre-registered wording was "identical logprobs mean LoRA is broken". The logprobs are
    not bit-identical, so the script's flag says `lora_has_effect: true`.
  - But an adapter of this size, if applied, changes layer 0's attention by more than the base
    weights themselves (ΔW ≈ 0.05 per entry), which would change the text.
  - Identical text with differences at bf16 noise level fits the adapter not being applied,
    as reported in forum discussion 743508.
  - The check lacked a base-vs-base noise control, so this is strong evidence, not proof.
- **Consequence:** LoRA adapters should be treated as having no effect in this scoring build,
  unless Kaggle announces a fix.

## First leaderboard submission (same agent, 4-minute budget)

- Submitted 2026-09-30 and scored within about 10 h.
- **Public score 0.08**, about 5 of the 58 public tasks. That is in line with public
  prompt-only notebooks (0.08–0.12).

## What this means

- **Text lessons now fail at two scales.**
  - Reasoning Gym, Qwen3 1.7B (tests 2 and 2b): lessons hurt.
  - SWE, Gemma 4 31B (here): no measurable effect.
- In this setting the model's own lessons are generic advice it already follows. They do not
  change which tasks it can solve.
- On Gemma, the planned text-vs-weights comparison does not go ahead in its pre-registered form.
  Its text arm showed nothing to compare against, and the weights arm appears unusable in the
  scoring build.

## Did the agent know it should be efficient? (added 2026-09-30, from the traces)

- **It was told the goal and the budget.**
  - The sample's system prompt says to fix and submit "in the minimum number of tool calls
    (under 8–10 turns)".
  - The harness's task message states the budget: 6 minutes, 100 tool calls, 100 turns.
- **It had no clock during the run.**
  - No message during the run reported the remaining time. There were only 12 generic "please
    continue" nudges across 384 runs.
  - The agent could have checked with `get_status`, but did so 0 times in the 96 base runs
    and once in the 96 lessons runs.
  - Each model call took about 18 s (thinking plus re-reading about 9,000 tokens, with 4 tasks
    sharing the GPU), so 6 minutes allowed about 20 calls. It used 18.6 on average and ran out
    of time in 73% of runs, well past its own "8–10 turns" target.
- **The lesson writer was told too.**
  - Its prompt gave the budget, said most failures ran out of time, and asked for lessons that
    need fewer steps. Each failure's description said "you ran out of the 6-minute time limit
    after N tool calls".
  - Condensing washed this out: none of the 7 final lessons mentions time.
  - Two lessons contradict the system prompt:
    - Lesson 1, "use global search", against "don't grep across the repo". Shell commands rose
      from 727 to 1,241.
    - Lesson 7, "update existing tests", against "never modify test files". Test-file edits did
      not rise: 11 base patches and 7 lessons patches touch test files.
- **Two separate problems follow:**
  - The agent cannot perceive time. That is a harness or prompt issue, for example
    instructing it to call `get_status` regularly.
  - The model's own lesson pipeline turns specific budget experience into generic advice. That
    matches test 2, where the lessons were likewise generic advice.
