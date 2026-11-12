"""Checks test 8's stratified interval and report on synthetic records (CPU). Run inside the research venv:
python tests/test_test8_union.py"""
import json
import pathlib
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "phase0"))

import test7_families as T  # noqa: E402
import test8_union as U  # noqa: E402

N = 20
ARGS = SimpleNamespace(forgive=False)


def evals(pass_rate):
    k = round(pass_rate * N)
    return [{"scores": [1.0, 1.0] if i < k else [0.0, 0.0], "answers": ["a", "a"] if i < k else [None, None],
             "lengths": [100, 100] if i < k else [6144, 6144]} for i in range(N)]


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def setup(root, own, union, families=T.FAMILIES):
    """Base 20% everywhere; each family's own adapter and the union adapter at the given rates."""
    T.ROOT, T.TEST2, T.TEST5, T.TEST6 = root / "test7", root / "test2", root / "test5", root / "test6"
    U.ROOT = root / "test8"
    for f in T.FAMILIES:
        write(T.base_file(f), evals(0.2))
        for path in T.round1_files(f):
            write(path, evals(own))
    for f in families:
        write(U.eval_file(f), evals(union))


def test_stratified_boot_means_over_families():
    d, lo, hi = U.stratified_boot([([0.0] * 10, [0.1] * 10), ([0.0] * 10, [0.3] * 10)])
    assert abs(d - 0.2) < 1e-12 and abs(lo - 0.2) < 1e-12 and abs(hi - 0.2) < 1e-12  # no spread within families


def test_report_holds_and_not():
    with tempfile.TemporaryDirectory() as tmp:
        setup(pathlib.Path(tmp), own=0.3, union=0.3)
        U.stage_report(ARGS)
        text = (U.ROOT / "report.md").read_text()
        assert "union minus own adapters: **+0.0 [+0.0, +0.0]**" in text and "one adapter HOLDS the gains" in text
    with tempfile.TemporaryDirectory() as tmp:
        setup(pathlib.Path(tmp), own=0.4, union=0.3)                     # 10 points worse in every family
        U.stage_report(ARGS)
        text = (U.ROOT / "report.md").read_text()
        assert "does not hold the gains" in text and "clearly worse" in text
    with tempfile.TemporaryDirectory() as tmp:
        setup(pathlib.Path(tmp), own=0.3, union=0.3, families=T.FAMILIES[:3])
        U.stage_report(ARGS)
        assert "**Not complete:** 3 of 8 families evaluated." in (U.ROOT / "report.md").read_text()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
