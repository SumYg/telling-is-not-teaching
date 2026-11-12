"""Checks the SWE go/no-go helpers, kernel builder and report. Run: python tests/test_gonogo.py"""
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "kaggle_swe" / "gonogo"))

import build_kernels  # noqa: E402
import gonogo  # noqa: E402
import report  # noqa: E402

TASK = {"instance_id": "x_1", "repo": "a/x", "problem_statement": "It breaks.", "patch": "--- fix"}


def test_ended_by():
    assert gonogo.ended_by({"error": "litellm.ContextWindowExceededError: ..."}) == "context"
    assert gonogo.ended_by({"error": "Agent exceeded session timeout (6 min)"}) == "time"
    assert gonogo.ended_by({"error": "", "tool_calls": 100}) == "tool_calls"
    assert gonogo.ended_by({"error": None, "tool_calls": 12, "total_llm_calls": 100}) == "turns"
    assert gonogo.ended_by({"error": "Sandbox execution error: boom"}) == "other_error"
    assert gonogo.ended_by({"error": "", "tool_calls": 12, "total_llm_calls": 14}) == "finished"


def test_lesson_prompt_without_patch():
    row = {"error": "Agent exceeded session timeout (6 min)", "tool_calls": 18}
    prompt = gonogo.lesson_prompt(TASK, row, "", "")
    assert "ran out of the 6-minute time limit after 18 tool calls" in prompt
    assert "you produced no patch" in prompt
    assert "(no patch)" in prompt and "(tests not run: no patch)" in prompt
    assert "--- fix" in prompt and "It breaks." in prompt


def test_lesson_prompt_clips_long_test_output_from_the_end():
    row = {"error": "", "tool_calls": 5}
    tests = "noise\n" * 5000 + "FAILED tests/test_a.py::test_b"
    prompt = gonogo.lesson_prompt(TASK, row, "diff", tests)
    assert "your patch failed the tests" in prompt
    assert "FAILED tests/test_a.py::test_b" in prompt
    assert len(prompt) < 12000


def test_usable_playbook():
    assert gonogo.usable_playbook("1. Read the failing test first.\n2. Edit once.")
    assert gonogo.usable_playbook("  1) Read the failing test first.")
    assert not gonogo.usable_playbook("")
    assert not gonogo.usable_playbook("I could not find any general lessons.")


def test_add_lessons_and_pick_tasks():
    text = gonogo.add_lessons("You are an agent.\n\n", "1. Read the test first.")
    assert text == "You are an agent.\n\n## Lessons From Past Tasks\n1. Read the test first.\n"
    tasks = [{"instance_id": "b_2", "repo": "r/b"}, {"instance_id": "a_1", "repo": "r/a"},
             {"instance_id": "c_3", "repo": "r/c"}]
    assert gonogo.pick_tasks(tasks, ("r/a", "r/b")) == ["a_1", "b_2"]


def test_condense_merges_chunks_in_one_batch_then_once_more():
    calls = []

    def fake_generate(setup, prompts, **kwargs):
        calls.append(len(prompts))
        return [None if "lesson 29." in p else "1. merged a\n2. merged b" for p in prompts]

    original = gonogo.generate
    gonogo.generate = fake_generate
    try:
        playbook = gonogo.condense(None, [f"lesson {i}." for i in range(65)])
    finally:
        gonogo.generate = original
    assert calls == [3, 1]  # three chunks of up to 30 at once, then the final merge
    # chunk 1 was cut off, so its first 10 lessons went into the final merge as they were
    assert playbook == "1. merged a\n2. merged b"


def test_fill_sets_stage_and_lessons():
    source = (pathlib.Path(gonogo.__file__)).read_text()
    filled = build_kernels.fill(source, "evaluate", "1. Say \"hi\".\n2. Go.\n")
    namespace = {}
    exec(compile(filled.split("\nON_KAGGLE")[0], "filled", "exec"), namespace)
    assert namespace["STAGE"] == "evaluate"
    assert namespace["LESSONS"] == "1. Say \"hi\".\n2. Go."
    compile(filled, "filled", "exec")


def write_results(root, resolved):
    """resolved[(condition, sample)] = list of 0/1 per task."""
    for (condition, sample), outcomes in resolved.items():
        run = root / "results" / f"{condition}_s{sample}"
        (run / "traces").mkdir(parents=True)
        rows = [{"instance_id": f"t_{i}", "resolved": bool(o), "error": "", "tool_calls": 10,
                 "total_llm_calls": 11} for i, o in enumerate(outcomes)]
        (run / "task_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (run / "traces" / "trace_t_0.json").write_text(json.dumps(
            {"steps": [{"source": "agent", "metrics": {"prompt_tokens": 1000 if condition == "base" else 1500}}]}))


def test_report_go_and_no_go():
    n = 48
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        base = [1] * 12 + [0] * (n - 12)
        better = [1] * 24 + [0] * (n - 24)
        write_results(root, {("base", 1): base, ("base", 2): base,
                             ("lessons", 1): better, ("lessons", 2): better})
        text, go = report.report(root)
        assert go and "+25.0 points" in text and "1,500" in text
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        same = [1] * 12 + [0] * (n - 12)
        write_results(root, {("base", 1): same, ("base", 2): same,
                             ("lessons", 1): same, ("lessons", 2): same})
        text, go = report.report(root)
        assert not go and "NO-GO" in text


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
