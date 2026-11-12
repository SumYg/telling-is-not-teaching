"""Checks test 7b's contrast and report on synthetic records (CPU). Run inside the research venv:
python tests/test_test7_check.py"""
import json
import pathlib
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "phase0"))

import test7_check as C  # noqa: E402
import test7_families as T  # noqa: E402

ARGS = SimpleNamespace(min_attempts=2, min_families=6)


def setup(root, check, outcome):
    """Base log-probs -1 everywhere; each family's model raises right attempts by check[f] relative to wrong ones."""
    C.ROOT = root
    for f in T.FAMILIES:
        (root / f).mkdir(parents=True, exist_ok=True)
        (root / f / "base_logprobs.json").write_text(json.dumps({"right": [-1.0] * 3, "wrong": [-1.0] * 3}))
        model = {"right": [-1.0 + check[f]] * 3, "wrong": [-1.0] * 3}
        n_gain = len(C.gain_models(f))
        (root / f / "model_logprobs.json").write_text(json.dumps({"gain": [model] * n_gain, "forgetting": [model]}))
    T.family_result = lambda f: {"gain": [outcome[f], 0, 0], "forgetting": [outcome[f], 0, 0]}


def test_contrast_is_right_gain_minus_wrong_gain():
    base = {"right": [-1.0, -2.0], "wrong": [-1.0, -1.0]}
    trained = {"right": [-0.5, -1.5], "wrong": [-1.25, -0.75]}
    assert abs(C.contrast(base, trained) - 0.5) < 1e-12              # right +0.5 on average, wrong 0


def test_models_follow_test7():
    assert len(C.gain_models("survo")) == 2 and len(C.forgetting_models("survo")) == 1
    assert C.gain_models("jugs") == C.forgetting_models("jugs") == [T.model_dir("jugs") / "merged"]


def test_report_predicts_and_does_not():
    with tempfile.TemporaryDirectory() as tmp:
        setup(pathlib.Path(tmp), {f: 0.001 * i for i, f in enumerate(T.FAMILIES)},
              {f: float(i) for i, f in enumerate(T.FAMILIES)})
        C.stage_report(ARGS)
        text = (C.ROOT / "report.md").read_text()
        assert "**Main, the forgetting check PREDICTS the measured forgetting** (rho +1.00" in text
        assert "**Secondary, the gain check PREDICTS the measured gain** (rho +1.00" in text
    with tempfile.TemporaryDirectory() as tmp:
        setup(pathlib.Path(tmp), {f: -0.001 * i for i, f in enumerate(T.FAMILIES)},
              {f: float(i) for i, f in enumerate(T.FAMILIES)})
        C.stage_report(ARGS)
        assert "the forgetting check does not predict" in (C.ROOT / "report.md").read_text()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
