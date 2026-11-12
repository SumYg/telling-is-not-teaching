"""Checks kaggle_swe/skill_report.py on synthetic results (CPU). Run: python tests/test_skill_report.py"""
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "kaggle_swe"))
import skill_report as R  # noqa: E402


def write(root, arm, task, resolved, calls, skill=None, error=None):
    d = pathlib.Path(root, "results", arm, task)
    (d / "traces").mkdir(parents=True)
    row = {"instance_id": task, "repo": "a/x" if task < "t5" else "b/y", "resolved": resolved, "tool_calls": len(calls),
           "error": error}
    (d / "task_results.jsonl").write_text(json.dumps(row) + "\n")
    steps = []
    for name in calls:
        obs = {"content": json.dumps(skill if name == "run_skill_script" and skill else {"status": "ok"})}
        steps.append({"tool_calls": [{"function_name": name, "arguments": {}}], "observation": obs})
    (d / "traces" / f"trace_{task}.json").write_text(json.dumps({"steps": steps}))


def build(skill_solved, base_solved, advice_solved, n=20):
    root = tempfile.mkdtemp()
    for i in range(n):
        t = f"t{i:02d}"
        write(root, "base", t, i < base_solved, ["read_file", "edit_file"])
        write(root, "advice", t, i < advice_solved, ["read_file", "read_file", "edit_file"])
        write(root, "skill", t, i < skill_solved, ["run_skill_script", "edit_file"],
              skill={"stdout": "=== X\n raise ValueError(\"error\")", "stderr": ""})
    return root


def test_go_and_tool_matters():
    text = R.report(build(skill_solved=10, base_solved=4, advice_solved=4))
    assert "Tasks with a result in all three arms: 20" in text
    assert "**Main (skill minus base): +30.0 points" in text and "**GO**" in text
    assert "the tool matters" in text
    assert "Skill use: called in 20 of 20 runs, 20 calls, 0 returned errors" in text  # output quoting "error" is not a failure


def test_no_go_and_missing_tasks():
    root = build(skill_solved=5, base_solved=5, advice_solved=5)
    write(root, "base", "t99", True, ["edit_file"])  # only in one arm: not compared
    text = R.report(root)
    assert "NO-GO" in text and "Tasks with a result in all three arms: 20" in text and "'base': 1" in text


def test_calls_before_first_edit_and_skill_errors():
    steps = [{"tool_calls": [{"function_name": "read_file"}]}, {"tool_calls": [{"function_name": "run_skill_script"}],
             "observation": {"content": json.dumps({"error": "x", "error_code": "SCRIPT_NOT_FOUND"})}},
             {"tool_calls": [{"function_name": "edit_file"}]}]
    assert R.calls_before_first_edit(steps) == 2
    assert R.skill_calls(steps) == (1, 1)
    assert R.calls_before_first_edit([{"tool_calls": [{"function_name": "read_file"}]}]) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
