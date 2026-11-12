"""Checks test 5's arm selection and bootstrap. Run inside the research venv: python tests/test_test5_gate.py"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "phase0"))

import test5_gate as T  # noqa: E402


def attempt(correct, finished=True, text="t"):
    return {"text": text, "length": 10, "score": 1.0 if correct else 0.0, "finished": finished}


RECORDS = [
    # problem 0: easy (4 of 8 correct) -> in R-all only
    {"index": 0, "n_correct": 4, "attempts": [attempt(True, text="e0")] * 4 + [attempt(False)] * 4},
    # problem 1: surprising (1 of 8) -> in R-all and R-gate
    {"index": 1, "n_correct": 1, "attempts": [attempt(True, text="s1")] + [attempt(False)] * 7},
    # problem 2: surprising (2 of 8), one correct attempt cut off -> only the finished one counts
    {"index": 2, "n_correct": 2, "attempts": [attempt(True, text="s2"), attempt(True, False)] + [attempt(False)] * 6},
    # problem 3: never solved -> nothing
    {"index": 3, "n_correct": 0, "attempts": [attempt(False)] * 8},
]
ENTRIES = [{"question": f"q{i}"} for i in range(4)]


def test_arms():
    all_ = T.arm_examples(RECORDS, ENTRIES, "all")
    gate = T.arm_examples(RECORDS, ENTRIES, "gate")
    rand = T.arm_examples(RECORDS, ENTRIES, "random")
    assert len(all_) == 6 and sorted(gate) == [("q1", "s1"), ("q2", "s2")]
    assert len(rand) == len(gate) and all(r in all_ for r in rand)
    assert rand == T.arm_examples(RECORDS, ENTRIES, "random")  # fixed seed


def test_boot():
    mean, lo, hi = T.boot([0, 0, 0, 0], [1, 1, 1, 1], [0, 0], [0.5, 0.5])
    assert abs(mean - 1.5) < 1e-9 and lo == hi == mean
    mean, lo, hi = T.boot([0, 1], [1, 0], [], [])
    assert mean == 0 and lo <= 0 <= hi


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
