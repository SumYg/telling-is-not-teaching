"""Shared prompt, sampling and scoring helpers for the Phase 0 tests."""
import json
import pathlib
import subprocess

from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer

CORRECT = 0.999  # a sample counts as correct only with a full score

# Qwen3 model-card sampling settings.
SAMPLING = {
    True: dict(temperature=0.6, top_p=0.95, top_k=20),   # thinking on
    False: dict(temperature=0.7, top_p=0.8, top_k=20),   # thinking off
}


def system_prompt(playbook=None):
    prompt = SYSTEM_PROMPTS["default"]
    if playbook:
        prompt += "\n\nLessons from past experience with this kind of problem:\n" + playbook
    return prompt


def chat(tokenizer, system, user, thinking):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def score(dataset, entry, text):
    answer = extract_answer(text)
    return answer, dataset.score_answer(answer=answer, entry=entry)


def model_slug(model):
    return model.replace("/", "__")


def write_jsonl(path, records):
    path = pathlib.Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    tmp.replace(path)  # readers only ever see complete files


def read_jsonl(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line]


def code_version():
    """Git commit of the code, and whether the run code has uncommitted changes."""
    git = ["git", "-c", "safe.directory=*"]  # the repo lives on the Windows side of WSL
    try:
        commit = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        status = subprocess.run(git + ["status", "--porcelain", "--", "phase0", "scripts"],
                                capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": commit, "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def save_config(out_dir, name, args):
    """Record the arguments and code version next to a run's results."""
    config = {**vars(args), "code": code_version()}
    (pathlib.Path(out_dir) / name).write_text(json.dumps(config, indent=2))
