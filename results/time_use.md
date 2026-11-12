# Where the agents' time goes (descriptive checks, 2026-10-19, no GPU)

Three checks for a cheap time lever, after the ARC budget result (`results/arc3_budget_elasticity.md`) showed the agent is
budget-bound. None found a lever; each is a descriptive finding for the papers.

## ARC: does running many games at once waste time on re-reading?

- **Question:** the hidden run plays 110 games on one server, against our 25. The server's cache holds about ten games'
  histories. A game that falls out of the cache must re-read its whole history, which could explain part of the gap
  between the hidden score (about 26) and the offline score at a similar share per game (46).
- **Data:** each request's `cached_tokens` in the 6 base runs (3 at 62 minutes, 3 at 124).
- **Result:**
  - 92.5–93.9% of prompt tokens came from the cache in every run.
  - Re-reads of more than half of a prompt over 30k tokens: 1.4–1.5 per game at 62 minutes and 3.8–4.6 at 124. That
    matches the history trims (1.5 and 3.6 per game), so games rarely fall out of the cache with 25 games.
  - Uncached prompt tokens were 2.2–2.9 times the generated tokens.
- **Reading:** with 110 games, rotation could add re-reads, but even doubling them would cost a modest share of
  throughput: a few points, not the gap. Not worth a Kaggle run.

## Gemma: does the agent's time go to reading its conversation or to writing?

- **Data:** 2,597 model steps of the base arm in the skill test (`runs/swe/skill_compare`, 129 tasks, 4 lanes on 4×L4).
- **Fit:** step time ≈ 3.3 s + 41.7 s per 1,000 output tokens, with no cost for prompt size (−0.2 s per 1,000 prompt
  tokens); R² 0.76.
- **Result:**
  - Output tokens explain 88% of model time, and the prompt nothing. The median prompt is 8.4k tokens and the median
    step writes 105.
  - **Long outputs are spread across all step types:**

    | Step type | Share of output tokens | Median output |
    |---|---|---|
    | run_command | 34% | 80 |
    | read_file | 26% | 72 |
    | edit_file | 22% | 310 |
    | write_file | 15% | 459 |

    They are mostly the reasoning before each action.
- **Reading:** time is generation, spread over every step. No single habit (re-reading, whole-file writes) is worth
  cutting. To fit more work into the 5.5 minutes, the agent must take fewer or shorter steps.
- **Caveat:** these timings come from 4 tasks sharing the server. The scoring run plays one task at a time, so its steps
  are faster.

## ARC: how much of the budget goes to games that are stuck?

- **Data:** each action's `generated_tokens` in the 6 base runs.

| Runs | Tokens in games that completed no level | Tokens after the last level-up, in unfinished games |
|---|---|---|
| 62 minutes (3 runs) | 9.0%, 5.7%, 12.0% | 33.3%, 32.2%, 39.5% |
| 124 minutes (3 runs) | 3.9%, 1.3%, 2.8% | 30.6%, 22.3%, 42.4% |

- **Reading:** little goes to hopeless games. A third goes to levels still in progress when time runs out. That is not
  waste, since more time converts part of it (the budget result). Franzen's scheduler already shifts the last 20% of the
  time toward games close to a level. No cheap reallocation lever.
