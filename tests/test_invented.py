"""Checks for the invented task families. Run inside WSL: python tests/test_invented.py"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "phase0"))

from invented import apply_program, drifting_stream, score_answer  # noqa: E402


def test_correct_answer_scores_one():
    for entry in drifting_stream("A", n=50, seed=1):
        assert score_answer(entry["answer"], entry) == 1.0
        assert score_answer("[" + entry["answer"][1:-1] + " ]", entry) == 1.0  # spacing tolerated


def test_wrong_or_unparsable_answers_score_zero():
    entry = drifting_stream("A", n=1, seed=1)[0]
    for bad in (None, "", "not a list", "[", "[99, 99]", "(1, 2)"):
        assert score_answer(bad, entry) == 0.0


def test_drift_changes_the_program_once():
    stream = drifting_stream("B", n=100, drift_at=40, seed=2)
    before = {tuple(e["metadata"]["program"]) for e in stream[:40]}
    after = {tuple(e["metadata"]["program"]) for e in stream[40:]}
    assert len(before) == 1 and len(after) == 1 and before != after
    assert all(e["metadata"]["phase"] == (i >= 40) for i, e in enumerate(stream))


def test_streams_are_reproducible():
    assert drifting_stream("C", n=20, seed=3) == drifting_stream("C", n=20, seed=3)
    assert drifting_stream("C", n=20, seed=3) != drifting_stream("C", n=20, seed=4)


def test_apply_program_composes_left_to_right():
    assert apply_program(["reverse", "add_1"], [1, 2, 9]) == [0, 3, 2]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
