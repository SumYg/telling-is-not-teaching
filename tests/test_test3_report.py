"""Checks the test 3 streams and report on synthetic results. Run: python tests/test_test3_report.py"""
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "phase0"))

from test3_controller import lifetime_stream, stage_report  # noqa: E402


def write_run(root, kind, tau, seed, accuracy, n=200):
    hits = round(accuracy * n)
    scores = [1.0] * hits + [0.0] * (n - hits)
    path = root / kind / f"tau{tau:g}_seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kind": kind, "tau": tau, "seed": seed, "scores": scores,
                                "events": [], "replays": 0}))


def test_room_when_the_best_threshold_differs_and_matters():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        for seed in (0, 1):
            write_run(root, "stable_few", 0, seed, 0.90)
            write_run(root, "stable_few", 10, seed, 0.80)
            write_run(root, "drift", 0, seed, 0.60)
            write_run(root, "drift", 10, seed, 0.75)
        assert stage_report(None, root) is True


def test_no_room_when_one_threshold_is_best_everywhere():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        for seed in (0, 1):
            write_run(root, "stable_few", 0, seed, 0.90)
            write_run(root, "stable_few", 10, seed, 0.80)
            write_run(root, "drift", 0, seed, 0.75)
            write_run(root, "drift", 10, seed, 0.60)
        assert stage_report(None, root) is False


def test_missing_seeds_are_left_out_of_the_pairing():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        write_run(root, "stable_few", 0, 0, 0.90)
        write_run(root, "stable_few", 10, 0, 0.80)
        write_run(root, "drift", 0, 0, 0.60)
        write_run(root, "drift", 10, 0, 0.75)
        write_run(root, "drift", 10, 1, 0.75)  # seed 1 only for one threshold
        assert stage_report(None, root) is True


def test_lifetime_types_have_the_expected_shape():
    assert len(lifetime_stream("stable_few", 100, 0)) == 100
    drift = lifetime_stream("drift", 100, 0)
    assert [e["metadata"]["phase"] for e in drift].count(1) == 50
    many = lifetime_stream("stable_many", 120, 0)
    assert len({e["metadata"]["family"] for e in many}) == 12
    mixed = lifetime_stream("mixed", 120, 0)
    drifting = {e["metadata"]["family"] for e in mixed if e["metadata"]["phase"] == 1}
    assert len(drifting) == 6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
