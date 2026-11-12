# SWE go/no-go: do in-context lessons transfer across repositories?

Approved by the author on 2026-09-29, before any go/no-go run. Binding from then on; changes go
in the change log at the end.
Budget numbers come from the Kaggle smoke test (`results/swe_smoke_kaggle.md`).

## Question

Do lessons that Gemma 4 31B writes from its own failures on some repositories help it, in the
prompt only, on a repository it has never seen? No training is involved. If yes, the
text-vs-weights comparison (standard LoRA vs compiled lessons) follows on the same split.

## Setup

- **Model and harness:** `gemma-4-31b-it-qat-w4a16-ct` on Kaggle 4×L4 (vLLM 0.19.1,
  swegemma 0.2.7, subprocess sandbox), with the scoring notebook's event compaction.
- **Agent:** the official sample submission's agent and prompts, with its placeholder LoRA
  adapters removed. Its system prompt already holds hand-written efficiency advice; lessons are
  appended after it.
- **One change to the sample, in both conditions:** `max_output_tokens: 8192` instead of
  16,384 in `configs/sampling.yaml`, which the root agent and its `code_analyzer` sub-agent
  share. With 16,384, one smoke task crashed when the sub-agent's history passed 16,384 tokens;
  compaction does not cover the sub-agent. Thinking budget stays 4,096.
- **Per-task budget:** `max_time_minutes: 6`, `max_tool_calls: 100`, `max_turns: 100`, with
  4 tasks in parallel. In the smoke test, 8 in parallel gave each task about a third of the
  model calls, and 4 in parallel gave about 93 s of wall time per task. Scoring runs one task
  at a time (host, forum discussion 743063). Running all 272 task-runs that way would take
  about 29 h, the whole weekly quota. So our absolute rates will be lower than on the
  leaderboard, and time-outs more common. The comparison is paired at a fixed setting.
- **Weights arm precondition (not part of the pass line):** participants report that LoRA
  adapters have no effect in the wheelhouse's patched vLLM 0.19.1 (forum discussion 743508).
  Before any weights arm, the kernel ends by comparing temperature-0 logprobs for the base model and
  for an adapter with both `lora_A` and `lora_B` randomized. Identical logprobs mean LoRA is
  broken on this build.
- **Split:**
  - lessons come from **fastapi + requests** (80 tasks)
  - evaluation is on **rich** (48 tasks)
  - httpx (1 task) is left out
  - the reverse direction (lessons from rich + requests, tested on fastapi) runs only if quota
    allows

## Procedure

1. **Experience:** the base agent attempts each of the 80 tasks once. We record resolved or
   not, how the attempt ended (finished, out of time, out of tool calls, or context error), and
   the agent's patch.
2. **Lessons:** for each failure, the same model sees:
   - the problem statement
   - how the attempt ended, and its patch (or "no patch")
   - the failing test output
   - the reference fix

   It writes one general lesson for OTHER repositories: no repository-specific names,
   budget-aware as in test 2b. The lessons are condensed to at most 10.
3. **Evaluation:** the base agent and the base agent plus lessons each attempt all 48 rich
   tasks, 2 samples each (paired by task and sample). Both conditions run in the same kernel,
   interleaved so that each batch of 4 holds both conditions. This way both see the same GPU
   load, which matters because the time limit is wall-clock. Concretely, four runs go at
   once (base and lessons, samples 1 and 2), each taking one task at a time in the same
   order.

## Pass line

- **Go** if lessons raise rich's resolution rate by at least 5 points, with the paired
  bootstrap 95% interval above 0. The bootstrap resamples the 48 tasks (2,000 rounds, seed 0).
  Each task's gain is the mean over its 2 samples, as in test 2.
- With 96 paired runs of pass/fail outcomes, that needs an observed gain of roughly 8 points
  or more.
- Reported alongside, per condition:
  - how attempts ended (finished, out of time, out of tool calls, context error)
  - mean tool calls per task
  - mean prompt tokens per model call

  Lessons lengthen every prompt, and prompts were 98% of the tokens in the smoke test. So a
  slowdown from the lessons themselves has to be visible.

## Quota

| Step | Task-runs | Estimate |
| --- | --- | --- |
| Experience | 80 | about 2.1 h |
| Lesson writing | none | about 0.3 h |
| Evaluation | 192 | about 5.0 h |
| LoRA check (own server start) | none | about 0.1 h |
| Server start | none | about 0.1 h |
| **Total** | 272 | **about 7.6 h** |

This is about a quarter of the 30 h/week of 4×L4, run as one kernel. Even with every task using
its full 6 minutes, the kernel stays under Kaggle's 12 h session limit (about 8.6 h). The reverse
direction would cost about the same again.

## Change log

- 2026-09-29, at approval, before any run: spelled out that the bootstrap resamples tasks, not
  single runs, and how the four runs are interleaved. Code: `kaggle_swe/gonogo/`.
- 2026-09-29, before any data: both stages now run in one kernel (`--stage full`), because
  4×L4 sessions wait hours in Kaggle's queue (forum discussions 743901 and 744054). Two kernels
  would mean waiting twice. The first kernel (experience only) was cancelled while still
  queued, so it produced nothing. The lessons are still fixed before evaluation starts.
  Evaluation is skipped if lesson writing yields no numbered list.
