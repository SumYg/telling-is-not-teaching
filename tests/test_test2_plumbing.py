"""Runs all four test 2 stages with a stand-in model, to check files and reports end to end.

Run inside WSL from the repo root:  python tests/test_test2_plumbing.py
"""
import argparse
import json
import pathlib
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "phase0"))

import test2_lessons  # noqa: E402


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        return "\n".join(m["content"] for m in messages)


class FakeLLM:
    """Answers wrongly, writes a fixed lesson (every other one cut off mid-thought), and
    condenses to a one-line playbook."""

    def get_tokenizer(self):
        return FakeTokenizer()

    def generate(self, prompts, params, use_tqdm=True):
        outs = []
        for i, prompt in enumerate(prompts):
            if "Merge them into at most 10" in prompt:
                text = "<think>merging</think>1. Check every constraint before answering."
            elif "Write ONE general lesson" in prompt and i % 2:
                text = "<think>still thinking when the token cap hit"
            elif "Write ONE general lesson" in prompt:
                text = "<think>why it failed</think>Check every constraint before answering."
            else:
                text = "<think>solving</think><answer>not the answer</answer>"
            sample = SimpleNamespace(text=text, token_ids=[0] * 7, finish_reason="stop")
            outs.append(SimpleNamespace(outputs=[sample] * params.n))
        return outs


def test_all_stages_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        test2_lessons.ROOT = pathlib.Path(tmp)
        test2_lessons.load_llm = lambda model, args: FakeLLM()
        args = argparse.Namespace(family="survo", writer=test2_lessons.LEARNER, n_experience=6,
                                  n_eval=5, samples=2, thinking=True, max_tokens=64,
                                  gpu_memory_utilization=0.5)
        test2_lessons.stage_experience(args)
        test2_lessons.stage_write(args)
        test2_lessons.stage_write(argparse.Namespace(**{**vars(args),
                                                        "writer": "Qwen/Qwen3-8B-AWQ"}))
        test2_lessons.stage_evaluate(args)
        test2_lessons.stage_report(args)

        out = pathlib.Path(tmp) / "survo"
        lessons = json.loads((out / "lessons_Qwen__Qwen3-1.7B.json").read_text())
        assert len(lessons) == 3 and all("<think>" not in l for l in lessons)  # cut-off ones dropped
        playbook = (out / "playbook_Qwen__Qwen3-1.7B.txt").read_text()
        assert playbook.startswith("1. ") and "<think>" not in playbook
        for name in ("none", "Qwen__Qwen3-1.7B", "Qwen__Qwen3-8B-AWQ"):
            records = [json.loads(l) for l in (out / f"eval_{name}.jsonl").read_text().split("\n") if l]
            assert len(records) == 5 and all(len(r["scores"]) == 2 for r in records)
        report = (pathlib.Path(tmp) / "report.md").read_text()
        assert report.count("| survo |") == 2 and "+0.0" in report


def test_efficient_variant_tells_the_writer_what_happened():
    cut = {"answers": [None], "lengths": [6144]}
    unformatted = {"answers": [None], "lengths": [900]}
    wrong = {"answers": ["42"], "lengths": [900]}
    assert "ran out of the token budget" in test2_lessons.outcome_text(cut, 6144)
    assert "without giving an answer" in test2_lessons.outcome_text(unformatted, 6144)
    assert "you answered 42" in test2_lessons.outcome_text(wrong, 6144)

    seen = []

    class RecordingLLM(FakeLLM):
        def generate(self, prompts, params, use_tqdm=True):
            seen.extend(prompts)
            return super().generate(prompts, params, use_tqdm)

    with tempfile.TemporaryDirectory() as tmp:
        test2_lessons.ROOT = pathlib.Path(tmp)
        test2_lessons.load_llm = lambda model, args: RecordingLLM()
        args = argparse.Namespace(family="survo", writer=test2_lessons.LEARNER, n_experience=6,
                                  n_eval=5, samples=2, thinking=True, max_tokens=64,
                                  gpu_memory_utilization=0.5, variant="efficient")
        test2_lessons.stage_experience(args)
        test2_lessons.stage_write(args)
    lesson_prompts = [p for p in seen if "Write ONE general lesson" in p]
    assert lesson_prompts and all("What happened:" in p and "FEWER reasoning steps" in p
                                  for p in lesson_prompts)
    assert any("Drop any lesson that only adds checking" in p for p in seen)


if __name__ == "__main__":
    test_all_stages_end_to_end()
    print("ok test_all_stages_end_to_end")
    test_efficient_variant_tells_the_writer_what_happened()
    print("ok test_efficient_variant_tells_the_writer_what_happened")
