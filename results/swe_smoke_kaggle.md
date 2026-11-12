# SWE harness smoke test, Kaggle 4×L4 with Gemma 4 31B (2026-09-29, code `bce1724`)

Kernel `sumyeung/swe-smoke-gemma4-31b` (private), from
`kaggle_swe/notebooks/smoke_31b/smoke_31b.py`. The official sample agent ran with its placeholder
adapters removed. Its sampling config (`max_output_tokens` 16,384, thinking budget 4,096) and the
Kaggle notebook's event compaction (threshold 14,336 tokens) were used unchanged. vLLM ran with
tensor parallel 4, subprocess sandbox. Budget: 6 min, 100 tool calls and 100 turns per task.
Outputs: `runs/swe/kaggle_smoke_31b/`.

Server start: 381 s. Kernel total: about 28 min.

## Per run

| Run | Parallel | Tasks | Wall time | Resolved | Out of time | Context error | Mean tool calls | Model calls/s | Tokens/s (prompt + output) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 4 | 8 | 743 s | 2 | 4 | 1 | 33.4 | 0.37 | 3,446 |
| B | 8 | 8 | 429 s | 1 | 6 | 0 | 12.8 | 0.25 | 2,200 |

"Model calls/s" and "Tokens/s" are totals over the whole run divided by its wall time. Tokens
come from the traces' `final_metrics`. Prompt tokens are 98% of the total, since every step
re-sends the full history (traces report 0 cached tokens).

## Per task

| Run | Task | Resolved | Ended by | Patch | Time | Tool calls |
| --- | --- | --- | --- | --- | --- | --- |
| A | fastapi_12942 | no | context error (in `code_analyzer` sub-agent) | none | 331 s | 18 |
| A | fastapi_11355 | no | time | 1,874 B, tests fail | 369 s | 49 |
| A | fastapi_11194 | no | time | 3,375 B, tests fail | 369 s | 33 |
| A | requests_6589 | no | time | 262 B, tests fail | 429 s | 14 |
| A | requests_6592 | no | finished | 1,039 B, tests fail | 412 s | 24 |
| A | rich_2943 | **yes** | finished | 400 B | 354 s | 96 |
| A | rich_2725 | no | time | none | 364 s | 18 |
| A | rich_3006 | **yes** | finished | 930 B | 165 s | 15 |
| B | fastapi_13207 | no | time | 823 B, tests fail | 369 s | 14 |
| B | fastapi_13713 | no | time | none | 365 s | 19 |
| B | fastapi_13537 | no | time | 1,025 B, tests fail | 369 s | 11 |
| B | requests_6629 | no | time | 1,418 B, tests fail | 429 s | 9 |
| B | requests_6644 | **yes** | finished | 1,391 B | 291 s | 13 |
| B | rich_3052 | no | finished | 5,298 B, tests fail | 360 s | 11 |
| B | rich_3043 | no | time | none | 363 s | 9 |
| B | rich_3061 | no | time | none | 363 s | 16 |

Time includes patch verification, so requests tasks exceed 360 s.

## Findings

- **The time limit is what binds.** 10 of 16 attempts ran out of time. None hit the 100
  tool-call limit; the most was rich_2943 with 96. Timed-out attempts still have their partial patch extracted and
  tested.
- **4 in parallel beats 8.** At 8, each task got about a third of the model calls it got at 4
  (12.8 vs 33.4 tool calls), and total throughput fell (0.25 vs 0.37 model calls/s). The
  likely cause is prefill: prompts are about 98% of tokens and are not reported as cached.
  With only 8 tasks per run this is indicative, not precise.
- **Throughput at 4 in parallel:** about 93 s of wall time per task with a 6-minute budget,
  so about 39 tasks per hour.
- **Context errors happen on Kaggle too, and the notebook's compaction does not prevent
  them.** The one error (fastapi_12942) came from inside the `code_analyzer` sub-agent
  (`agent_tool.py` in the traceback). Compaction is configured on the root app, and the
  sub-agent's history grew past 16,384 tokens while 16,384 output tokens were reserved. Fix
  for the experiments: `max_output_tokens: 8192` in `configs/sampling.yaml`, which both agents
  share. Thinking (4,096) still fits, and the input can then grow to about 24,500 tokens.
- **Resolution:** 3 of 16 overall. By repository: rich 2/6, requests 1/4, fastapi 0/6.
- **Compute per task differs from scoring.** `EvalConfig` defaults to 1 task at a time, and
  a submission's `eval_config.yaml` can set only the time, tool-call, turn and command
  budgets. The host has said scoring runs the hidden tasks one at a time (forum discussion
  743063, per the paper-track session), so each task gets the whole server. The same 6 minutes at 4 in
  parallel gives less compute per task, so absolute rates here will be lower than on the
  leaderboard. Paired comparisons at a fixed setting are unaffected.
