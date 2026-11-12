"""Checks the offline time test's report on synthetic results (CPU): python tests/test_time_report.py"""
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "kaggle_swe"))

import time_report  # noqa: E402


def write(root, arm, task, resolved, error=""):
    d = pathlib.Path(root) / "results" / arm / task
    d.mkdir(parents=True, exist_ok=True)
    row = {"instance_id": task, "resolved": resolved, "repo": "fastapi", "tool_calls": 10, "error": error}
    (d / "task_results.jsonl").write_text(json.dumps(row) + "\n")


def test_pairs_tasks_across_sessions():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        for i in range(10):  # session a: tasks 0-9, more time solves 2 more
            write(a, "default", f"t{i}", i < 3)
            write(a, "more_time", f"t{i}", i < 5)
        for i in range(10, 20):  # session b: tasks 10-19, the default also ran t20, which more_time never reached
            write(b, "default", f"t{i}", False, error="timeout")
            write(b, "more_time", f"t{i}", False)
        write(b, "default", "t20", True)
        text = time_report.report([a, b])
        assert "Tasks with a result in both arms: 20 (default 21, more_time 20)" in text
        assert "**Main (more_time minus default): +10.0 points" in text and "not shown" not in text.split("\n")[6]
        assert "solved only with more time 2, only with the default 0" in text
        assert "| default | 3 (15.0%) | 10 |" in text


if __name__ == "__main__":
    test_pairs_tasks_across_sessions()
    print("ok test_pairs_tasks_across_sessions")
