# SWE harness smoke test, local (2026-09-29, code `70bc4de`)

The official sample agent ran with Gemma 4 E2B on the local server (research env, vLLM 0.30)
and the swegemma 0.2.7 harness (subprocess sandbox), on 3 debug tasks. Budget: 6 min, 50 tool
calls, 50 turns.

| Task | Resolved | Patch | Time | Tool calls | Error |
| --- | --- | --- | --- | --- | --- |
| fastapi_14492 | no | 1,292 bytes, tests fail | 92 s | 12 | |
| requests_6589 | no | none | 180 s | 1 | context window exceeded |
| rich_3278 | no | none | 228 s | 17 | context window exceeded |

**The pipeline works end to end:** server, agent, sandbox, patch extraction and verification.
0 of 3 resolved is expected from a 2B-effective model.

**Context-window errors:** the sample config asks for `max_output_tokens: 16384`. So once
the prompt passes 16,384 tokens, every request exceeds the 32,768-token window. Kaggle's
scoring script adds ADK event compaction (threshold 14,336 tokens), which the local CLI does
not appear to use. Any lessons appended to the prompt use this same window. Options:
lower `max_output_tokens` (e.g. 8,192) in every condition, or measure on Kaggle whether
compaction prevents the error.
