"""Checks test 6's report on synthetic records (CPU). Run inside the research venv: python tests/test_test6_iterate.py"""
import json
import pathlib
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "phase0"))

import test6_iterate as T  # noqa: E402

N = 20  # evaluation problems


def evals(pass_rate):
    """N problems x 2 samples, the first pass_rate share of problems correct in both samples, the rest cut off."""
    k = round(pass_rate * N)
    return [{"scores": [1.0, 1.0] if i < k else [0.0, 0.0], "answers": ["a", "a"] if i < k else [None, None],
             "lengths": [100, 100] if i < k else [6144, 6144]} for i in range(N)]


def write(path, records, examples=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    if examples is not None:
        (path.parent / "train_log.json").write_text(json.dumps({"examples": examples}))


def setup(root, base, rounds):
    T.ROOT, T.TEST5, T.TEST2 = root / "test6", root / "test5", root / "test2"
    write(T.TEST2 / "survo" / "eval_none.jsonl", evals(base))
    write(T.TEST2 / "quantum_lock" / "eval_none.jsonl", evals(0.2))
    for (k, d), rate in rounds.items():
        write(T.eval_file("survo", k, d), evals(rate), examples=100 + k)
    write(T.eval_file("survo", 3, "s1", "quantum_lock"), evals(0.2))


ARGS = SimpleNamespace(family="survo", max_tokens=6144)


def test_paths_reuse_test5_for_round_1():
    T.ROOT, T.TEST5 = pathlib.Path("runs/test6"), pathlib.Path("runs/test5")
    assert T.model_dir("survo", 1, "t5") == pathlib.Path("runs/test5/survo/all")
    assert T.model_dir("survo", 3, "s2") == pathlib.Path("runs/test6/survo/r3_s2")
    assert T.practice_file("survo", 1) == pathlib.Path("runs/test5/survo/experience.jsonl")
    assert T.practice_file("survo", 2) == pathlib.Path("runs/test6/survo/r2/practice.jsonl")
    assert T.eval_file("survo", 3, "s1", "quantum_lock").name == "eval_on_quantum_lock.jsonl"


def test_pass_and_compounding():
    with tempfile.TemporaryDirectory() as tmp:
        setup(pathlib.Path(tmp), 0.1, {(1, "t5"): 0.2, (1, "s2"): 0.3, (2, "s1"): 0.4, (3, "s1"): 0.5, (3, "s2"): 0.6})
        T.stage_report(ARGS)
        text = (T.ROOT / "survo" / "report.md").read_text()
        assert "base pass@1 10.0" in text
        assert "Round 1 minus base: +15.0" in text                      # mean of 20% and 30%, minus 10%
        assert "**Main, round 3 minus base: +45.0" in text and "self-training PASSES" in text
        assert "**Compounding, round 3 minus round 1: +30.0" in text and "rounds ADD" in text
        assert "on quantum_lock, minus base: +0.0" in text
        assert "| 3 | s2 | 103 | 60.0 | 40.0% | 100.0% |" in text      # the table row of one training


def test_no_pass_and_partial_results():
    with tempfile.TemporaryDirectory() as tmp:
        setup(pathlib.Path(tmp), 0.2, {(1, "t5"): 0.2, (1, "s2"): 0.2, (2, "s1"): 0.2, (3, "s1"): 0.2, (3, "s2"): 0.25})
        T.stage_report(ARGS)
        text = (T.ROOT / "survo" / "report.md").read_text()
        assert "self-training does not pass" in text and "rounds do not add" in text
    with tempfile.TemporaryDirectory() as tmp:
        setup(pathlib.Path(tmp), 0.2, {(1, "t5"): 0.2, (2, "s1"): 0.3})  # round 1 half done, no round 3 yet
        T.stage_report(ARGS)
        text = (T.ROOT / "survo" / "report.md").read_text()
        assert "Round 2 minus base: +10.0" in text and "Round 1 minus base" not in text and "Main" not in text


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
