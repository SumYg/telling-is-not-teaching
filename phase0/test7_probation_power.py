"""After test 7b (not pre-registered): how many answers would a generation-based probation check need to catch test 7's
forgetting on survo?

    python phase0/test7_probation_power.py

Resamples test 7's saved survo evaluations (500 problems x 2 samples) of the base model and of a family's round-1 model:
m problems drawn with replacement, one sample each from both models. Prints the spread of the difference and how often a
one-sided 5% test would flag the model as worse than the base.
"""
import json
import random

BASE = "runs/test2/survo/eval_none.jsonl"
MODELS = {"course_schedule": -4.6, "self_reference": -3.7, "word_ladder": +1.0}  # forgetting on survo in test 7


def load(path):
    return [[s >= 0.999 for s in json.loads(line)["scores"]] for line in open(path, encoding="utf-8") if line.strip()]


def main():
    base = load(BASE)
    for family, full in MODELS.items():
        trained = load(f"runs/test7/{family}/r1_s1/eval_on_survo.jsonl")
        rng = random.Random(0)
        out = []
        for m in (100, 250, 500, 1000):
            diffs = []
            for _ in range(2000):
                idx = [rng.randrange(len(base)) for _ in range(m)]
                diffs.append(100 * sum(rng.choice(trained[i]) - rng.choice(base[i]) for i in idx) / m)
            mean = sum(diffs) / len(diffs)
            sd = (sum((x - mean) ** 2 for x in diffs) / (len(diffs) - 1)) ** 0.5
            flagged = sum(x < -1.645 * sd for x in diffs) / len(diffs)
            out.append(f"{m} answers: SE {sd:.1f}, flagged {100 * flagged:.0f}%")
        print(f"{family} on survo (test 7: {full:+.1f}): " + "; ".join(out))


if __name__ == "__main__":
    main()
