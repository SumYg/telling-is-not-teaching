# Test 1, stage B: gap and headroom

The full protocol from PREREGISTRATION.md runs one family at a time, highest screen pass@1
first. The table grows as families finish. A family qualifies if the 1.7B's pass@1 is 5–30%
and the 8B's pass@1 is at least 15 points higher. Go needs at least 3 qualifying families.

| Family | 1.7B pass@1 | 1.7B pass@8 | 8B pass@1 | Gap (points) | Gap 95% CI | 1.7B tokens/answer | Qualifies | Preferred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| survo | 14.3 | 44.7 | 64.1 | +49.8 | [+46.1, +53.3] | 5687 | yes | yes |
| quantum_lock | 19.3 | 53.3 | 38.4 | +19.1 | [+15.2, +23.0] | 5566 | yes | no |
| self_reference | 13.3 | 47.3 | 26.6 | +13.2 | [+9.5, +17.0] | 5107 | no | no |
| jugs | 11.7 | 40.3 | 24.2 | +12.5 | [+9.3, +15.8] | 5653 | no | no |
| word_ladder | 4.8 | 17.3 | 22.2 | +17.4 | [+14.1, +21.0] | 4665 | no | no |

**Outcome: 2 of 5 families qualify, below the 3 needed for go.** The stopping rule ended
stage B after 5 families. The three misses were narrow:

- self_reference: gap +13.2 and jugs: gap +12.5, against the 15 required.
- word_ladder: the 1.7B's pass@1 is 4.8%, against the 5% floor.

The pre-registered response is to adjust difficulty once and re-test. Test 2 runs on the two
qualifying families (survo, quantum_lock) in the meantime.

"Preferred": the 1.7B's pass@8 is below the 8B's pass@1, so sampling alone does not close
the gap. In quantum_lock the 1.7B's pass@8 (53.3) is above the 8B's pass@1 (38.4). There,
plain RL could likely close the gap by sharpening what the model already does, so survo is
the cleaner test of learning.

Stopping rule (pre-registered 2026-09-28): stage B stops once 3 families qualify, or after
the first 5.

## Cost

About 5–6 GPU-hours per family.

| Family | 1.7B (2,400 samples) | 8B (1,200 samples) |
| --- | --- | --- |
| survo | 3.0 h, 1,274 tokens/s | 2.2 h, 712 tokens/s |
| quantum_lock | 2.9 h, 1,289 tokens/s | 2.8 h, 649 tokens/s |
