"""Checks test 7's split, statistics and report on synthetic records (CPU). Run inside the research venv:
python tests/test_test7_families.py"""
import json
import pathlib
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "phase0"))

import test7_families as T  # noqa: E402

N = 20  # evaluation problems
ARGS = SimpleNamespace(max_tokens=6144, min_families=6)


def evals(pass_rate):
    """N problems x 2 samples, the first pass_rate share of problems correct in both samples, the rest cut off."""
    k = round(pass_rate * N)
    return [{"scores": [1.0, 1.0] if i < k else [0.0, 0.0], "answers": ["a", "a"] if i < k else [None, None],
             "lengths": [100, 100] if i < k else [6144, 6144]} for i in range(N)]


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def setup(root, round1, probes, too_few=()):
    """Base pass@1 20% everywhere; round 1 at the given rates; the probe signal per family."""
    T.ROOT, T.TEST2, T.TEST5, T.TEST6 = root / "test7", root / "test2", root / "test5", root / "test6"
    for f in T.FAMILIES:
        write(T.base_file(f), evals(0.2))
        if f in too_few:
            T.model_dir(f).mkdir(parents=True, exist_ok=True)
            (T.model_dir(f) / "too_few.json").write_text(json.dumps({"examples": 5}))
            continue
        if f in round1:
            for path in T.round1_files(f):
                write(path, evals(round1[f]))
            write(T.round1_files(f, T.anchor(f))[0], evals(0.2))
        if f in probes:
            sig = {"family": f, "n_success": 100, "finished_accuracy": 0.5, "reliability": 2.0, "base_pass1": 0.2,
                   "probe": probes[f], "success_gain": 0.01, "familiarity": -0.3}
            (T.ROOT / f).mkdir(parents=True, exist_ok=True)
            (T.ROOT / f / "signals.json").write_text(json.dumps(sig))


def test_paths_reuse_tests_2_5_and_6():
    T.ROOT, T.TEST2, T.TEST5, T.TEST6 = (pathlib.Path(p) for p in ("runs/test7", "runs/test2", "runs/test5", "runs/test6"))
    assert T.practice_file("survo") == pathlib.Path("runs/test5/survo/experience.jsonl")
    assert T.practice_file("jugs") == pathlib.Path("runs/test7/jugs/experience.jsonl")
    assert T.base_file("quantum_lock") == pathlib.Path("runs/test2/quantum_lock/eval_none.jsonl")
    assert T.round1_files("survo") == [pathlib.Path("runs/test5/survo/all/eval.jsonl"),
                                       pathlib.Path("runs/test6/survo/r1_s2/eval.jsonl")]
    assert T.round1_files("survo", "quantum_lock") == [pathlib.Path("runs/test5/survo/all/eval_on_quantum_lock.jsonl")]
    assert T.round1_files("jugs", "survo") == [pathlib.Path("runs/test7/jugs/r1_s1/eval_on_survo.jsonl")]
    assert T.anchor("survo") == "quantum_lock" and T.anchor("jugs") == "survo"


def test_halves_split_by_problem():
    entries = [{"question": f"q{i}"} for i in range(4)]
    att = lambda s, f: {"text": f"t{s}{f}", "score": s, "finished": f}  # noqa: E731
    records = [{"index": i, "attempts": [att(1.0, True), att(0.0, True), att(1.0, False)], "n_correct": 2}
               for i in range(4)]
    split = T.halves(records, entries)
    assert [q for q, _ in split["A"]["right"]] == ["q0", "q2"] and [q for q, _ in split["B"]["wrong"]] == ["q1", "q3"]
    assert all(len(split[h][k]) == 2 for h in "AB" for k in ("right", "wrong"))  # unfinished attempts are left out


def test_ranks_spearman_and_exact_permutation_p():
    assert T.ranks([1, 2, 2, 3]) == [0.0, 1.5, 1.5, 3.0]
    x = list(range(8))
    assert abs(T.spearman(x, x) - 1) < 1e-12 and abs(T.spearman(x, x[::-1]) + 1) < 1e-12
    rho, p = T.permutation_p(x, x)
    assert abs(rho - 1) < 1e-12 and abs(p - 1 / 40320) < 1e-12
    assert T.permutation_p(x, x[::-1])[1] == 1.0                       # one-sided: the wrong direction never passes
    assert abs(T.permutation_p(x, x[::-1], one_sided=False)[1] - 2 / 40320) < 1e-12


def test_report_probe_predicts():
    with tempfile.TemporaryDirectory() as tmp:
        round1 = {f: 0.2 + 0.05 * i for i, f in enumerate(T.FAMILIES)}       # gains 0, 5, ..., 35 points
        setup(pathlib.Path(tmp), round1, {f: 0.001 * i for i, f in enumerate(T.FAMILIES)})
        T.stage_report(ARGS)
        text = (T.ROOT / "report.md").read_text()
        assert "**Families evaluated:** 8 of 8" in text
        assert "- probe: rho +1.00, p 0.0000 (one-sided), 8 families" in text
        assert "**Main: the probe PREDICTS which families gain** (rho +1.00" in text
        assert "| mini_sudoku | 20.0 | 55.0 | +35.0" in text
        assert "keeps 7 families, total gain +140.0; keeping all +140.0" in text


def test_report_probe_does_not_predict_or_cannot_be_tested():
    with tempfile.TemporaryDirectory() as tmp:
        round1 = {f: 0.2 + 0.05 * i for i, f in enumerate(T.FAMILIES)}
        setup(pathlib.Path(tmp), round1, {f: -0.001 * i for i, f in enumerate(T.FAMILIES)})
        T.stage_report(ARGS)
        assert "**Main: the probe does not predict which families gain** (rho -1.00" in (T.ROOT / "report.md").read_text()
    with tempfile.TemporaryDirectory() as tmp:
        families = T.FAMILIES[:5]
        setup(pathlib.Path(tmp), {f: 0.3 for f in families}, {f: 0.001 for f in families}, too_few=("word_ladder",))
        T.stage_report(ARGS)
        text = (T.ROOT / "report.md").read_text()
        assert "**Main: not testable**" in text and "not trained (5 examples)" in text
        assert "**Families evaluated:** 5 of 8" in text


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
