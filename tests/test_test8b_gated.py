"""Checks test 8b's report on synthetic records (CPU). Run inside the research venv: python tests/test_test8b_gated.py"""
import json
import pathlib
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "phase0"))

import test7_families as T  # noqa: E402
import test8_union as U  # noqa: E402
import test8b_gated as G  # noqa: E402

N = 20
ARGS = SimpleNamespace(forgive=False)


def evals(pass_rate):
    k = round(pass_rate * N)
    return [{"scores": [1.0, 1.0] if i < k else [0.0, 0.0], "answers": ["a", "a"] if i < k else [None, None],
             "lengths": [100, 100] if i < k else [6144, 6144]} for i in range(N)]


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def setup(root, own, full, gated):
    """Base 20% everywhere; the own adapters, test 8's full union and the gated union at the given rates."""
    T.ROOT, T.TEST2, T.TEST5, T.TEST6 = root / "test7", root / "test2", root / "test5", root / "test6"
    U.ROOT, G.ROOT = root / "test8", root / "test8b"
    for f in T.FAMILIES:
        write(T.base_file(f), evals(0.2))
        for path in T.round1_files(f):
            write(path, evals(own))
        write(U.eval_file(f), evals(full))
        write(G.eval_file(f), evals(gated))


def test_kept_families_are_test7_gainers():
    assert set(G.KEPT) == {"survo", "self_reference", "jugs", "course_schedule", "mini_sudoku"}


def test_report_helps_and_reference():
    with tempfile.TemporaryDirectory() as tmp:
        setup(pathlib.Path(tmp), own=0.4, full=0.3, gated=0.4)          # gated beats full everywhere by 10 points
        G.stage_report(ARGS)
        text = (G.ROOT / "report.md").read_text()
        assert "**Main: the gate HELPS**" in text
        # reference: own (40%) for the 5 kept families, base (20%) for the 3 left out -> gated is +0 and +20 above it
        assert "gated minus reference: +7.5 [" in text and "HOLDS" in text
    with tempfile.TemporaryDirectory() as tmp:
        setup(pathlib.Path(tmp), own=0.4, full=0.3, gated=0.3)
        G.stage_report(ARGS)
        assert "the gate is not shown to help" in (G.ROOT / "report.md").read_text()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
