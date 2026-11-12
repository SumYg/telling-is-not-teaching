"""Checks the awake/sleep loop with a stub model. Run inside WSL: python tests/test_lifetime.py"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "phase0"))

from invented import drifting_stream  # noqa: E402
from lifetime import run_lifetime  # noqa: E402


def rule(entry):
    meta = entry["metadata"]
    return f"{meta['family']}: {'+'.join(meta['program'])}"


def stub_solver(entries, playbook, seed):
    """Correct exactly when the playbook states the entry's current rule."""
    return [(entry["answer"], 1.0) if rule(entry) in playbook else ("[]", 0.0)
            for entry in entries]


def stub_writer(failures):
    """One perfect lesson per family, from its most recent failure."""
    latest = {entry["metadata"]["family"]: entry for entry, _ in failures}
    return [(family, rule(entry)) for family, entry in latest.items()]


def test_stable_rule_is_learned_after_the_first_sleep():
    stream = drifting_stream("A", n=100, seed=0)
    result = run_lifetime(stream, stub_solver, stub_writer, tau=5, sleep_every=25)
    assert result.scores[:25] == [0.0] * 25 and result.scores[25:] == [1.0] * 75
    assert [e[0] for e in result.events] == ["promote"]


def test_drift_removes_the_stale_lesson_and_learns_the_new_rule():
    stream = drifting_stream("A", n=100, drift_at=50, seed=0)
    result = run_lifetime(stream, stub_solver, stub_writer, tau=5, sleep_every=25)
    assert sum(result.scores[25:50]) == 25 and sum(result.scores[50:75]) == 0
    assert sum(result.scores[75:]) == 25
    assert [e[0] for e in result.events] == ["promote", "remove", "promote"]


def test_a_threshold_above_any_gain_never_promotes():
    stream = drifting_stream("A", n=50, seed=0)
    result = run_lifetime(stream, stub_solver, stub_writer, tau=101, sleep_every=25)
    assert sum(result.scores) == 0 and result.events == []


def test_on_a_tie_the_incumbent_lesson_stays():
    a = drifting_stream("A", n=40, seed=0)
    b = drifting_stream("B", n=40, seed=0)
    stream = [x for pair in zip(a, b) for x in pair]  # interleaved, both lessons gain 100
    result = run_lifetime(stream, stub_solver, stub_writer, tau=5, sleep_every=20, capacity=1)
    assert [e[0] for e in result.events] == ["promote"]


def partial_solver(entries, playbook, seed):
    """Family A is half solvable without a lesson (even input sum), so its lessons gain less."""
    out = []
    for entry in entries:
        easy = entry["metadata"]["family"] == "A" and sum(entry["metadata"]["input"]) % 2 == 0
        ok = easy or rule(entry) in playbook
        out.append((entry["answer"], 1.0) if ok else ("[]", 0.0))
    return out


def test_a_stronger_candidate_evicts_a_weaker_lesson_at_capacity():
    stream = drifting_stream("A", n=20, seed=0) + drifting_stream("B", n=20, seed=0)
    result = run_lifetime(stream, partial_solver, stub_writer, tau=5, sleep_every=20,
                          capacity=1)
    kinds = [(e[0], e[2]) for e in result.events]
    assert kinds == [("promote", "A"), ("evict", "A"), ("promote", "B")]
    assert result.events[0][4] < 100 and result.events[2][4] == 100


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
