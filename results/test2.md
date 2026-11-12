# Test 2: do the learner's own lessons help on other problems?

Protocol in PREREGISTRATION.md: 200 experience problems (seed 2000), lessons from each
failure, condensed to at most 10. Then 500 new problems (seed 3000), 2 samples each, with and
without the playbook. Pass line: at least +5 points with the 95% CI above 0, on at least
2 families. Families: survo and quantum_lock, the two that qualified in test 1.

## survo (code `8d08a0d`)

| Playbook | pass@1 | Gain (points) | Cut off at cap | Finished but wrong | Mean tokens |
| --- | --- | --- | --- | --- | --- |
| none | 17.1% | | 68.7% | 14.2% | 5,600 |
| written by the 1.7B | 10.7% | −6.4 | 76.7% | 12.6% | 5,742 |
| written by the 8B | 8.2% | −8.9 | 81.4% | 10.4% | 5,869 |

**Both playbooks hurt.** With only two families, test 2 cannot reach its pass line, whatever
quantum_lock shows.

**Why:** the drop comes almost entirely from more answers cut off at the 6,144-token cap.
Finished-but-wrong answers went down slightly. Both playbooks are generic advice to verify
and cross-check more (see `runs/test2/survo/playbook_*.txt`). The model already ran out of
budget on 69% of attempts, so advice to be more thorough costs tokens it does not have.
The 8B's more "systematic" lessons hurt most.

A likely design flaw: about four in five failures were cut off. Yet the lesson writer saw
only "your final answer: None" and the correct answer. It never learned that the attempt
ran out of budget, so it wrote lessons about correctness, not efficiency.

Minor: survo's 1.7B playbook was cut mid-way through lesson 10 by a text cap, since fixed
(`dc3539b`). It does not change the result.

## quantum_lock

| Playbook | pass@1 | Gain (points) | 95% CI | Cut off at cap | Finished but wrong |
| --- | --- | --- | --- | --- | --- |
| none | 20.9% | | | 57.8% | 21.3% |
| written by the 1.7B | 22.3% | +1.4 | [−1.6, +4.4] | 54.5% | 23.2% |
| written by the 8B | 22.6% | +1.7 | [−1.3, +4.9] | | |

Small, non-significant gains. The 1.7B's playbook here already includes efficiency advice
("minimize redundant steps"), and cut-offs fell slightly.

## Outcome

**Test 2 is a no-go:** no family gains at least 5 points. Per the pre-registration, Project 1
as designed stops. Test 2b (budget-aware lessons on survo) was pre-registered before running,
and tests whether the failure comes from the lesson-writing design.

# Test 2b: budget-aware lessons (survo, code `866c40f`)

The writer was told when an attempt ran out of budget (135 of 162 failures), and asked for
lessons that shorten the path to an answer. The no-playbook baseline is reused from test 2.

| Playbook | pass@1 | Gain (points) | 95% CI | Cut off at cap | Finished but wrong | Mean tokens |
| --- | --- | --- | --- | --- | --- | --- |
| none | 17.1% | | | 68.7% | 14.2% | 5,600 |
| budget-aware, by the 1.7B | 9.3% | −7.8 | [−10.3, −5.3] | 83.1% | 7.6% | 5,906 |
| budget-aware, by the 8B | 9.6% | −7.5 | [−9.9, −5.1] | 81.8% | 8.6% | 5,851 |

**No-go.** Lessons about reaching the answer in fewer steps made answers longer still, and
cut-offs rose from 69% to 82–83%. The harm does not depend on lesson content. Any list of
10 lessons in the system prompt lengthens this thinking model's reasoning. A likely
mechanism, not yet checked in the traces, is that the model works through every lesson
while thinking.

**What this means:** for a small thinking model under a tight token budget, in-context
lessons carry a cost that exceeded their benefit in both designs. Per the pre-registration,
this is reported as a negative result. The pattern bears directly on the text-vs-weights
question: a lesson compiled into the weights costs no prompt, so it may not trigger this
deliberation. Whether it helps is untested.
