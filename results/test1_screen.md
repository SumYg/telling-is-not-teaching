# Test 1, stage A: screen

Run 2026-09-27/28 with code `da05e2d`: Qwen3-1.7B, thinking on, at most 6,144 new tokens,
100 problems per family (seed 1000) with 2 samples each. The screen keeps families with
pass@1 between 1.5% and 45%.

| Family | 1.7B pass@1 (screen) | Tokens/answer | Hit max tokens | Keep |
| --- | --- | --- | --- | --- |
| arc_1d | 2.5 | 5907 | 88% | yes |
| countdown | 60.0 | 3264 | 30% | no |
| course_schedule | 48.0 | 3856 | 28% | no |
| cryptarithm | 1.0 | 6143 | 98% | no |
| futoshiki | 0.0 | 5894 | 82% | no |
| game_of_life | 2.5 | 5927 | 78% | yes |
| graph_color | 87.0 | 2364 | 0% | no |
| jugs | 10.5 | 5698 | 78% | yes |
| kakurasu | 1.5 | 6080 | 96% | yes |
| knights_knaves | 86.0 | 1161 | 0% | no |
| list_functions | 49.5 | 3630 | 41% | no |
| mini_sudoku | 58.5 | 4012 | 21% | no |
| puzzle24 | 81.0 | 1796 | 2% | no |
| quantum_lock | 15.5 | 5661 | 64% | yes |
| rotten_oranges | 1.5 | 5312 | 54% | yes |
| self_reference | 13.0 | 5200 | 48% | yes |
| shortest_path | 60.5 | 4470 | 32% | no |
| survo | 16.5 | 5508 | 70% | yes |
| word_ladder | 3.5 | 4521 | 34% | yes |
| zebra_puzzles | 90.5 | 3470 | 8% | no |

Kept 9 of 20. Stage B order, highest screen pass@1 first: survo, quantum_lock,
self_reference, jugs, word_ladder, arc_1d, game_of_life, kakurasu, rotten_oranges.

## Notes

- In most kept families, the bulk of failures are answers **cut off at the 6,144-token
  cap** (34–96%), not finished wrong answers. There, a low pass@1 partly measures whether
  reasoning fits the budget. Lessons that shorten reasoning could legitimately help.
- Kept families average 4,500–6,100 tokens per answer. At the speed test's throughput, stage
  B costs roughly 7 GPU-hours per family (both models), about 60 for all nine.
