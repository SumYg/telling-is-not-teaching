"""Build a submission directory from the official sample, optionally with lessons appended.

    python kaggle_swe/make_submission.py OUT_DIR [--model gemma-4-e2b-it] [--lessons FILE]

The sample's placeholder LoRA adapters are removed, along with the `adapter:` lines that
reference them. `--model` swaps the model alias, for local debugging with a small Gemma; the
competition requires gemma-4-31b-it-qat-w4a16-ct. `--lessons` appends a playbook to the root
agent's system prompt.
"""
import argparse
import pathlib
import re
import shutil

SAMPLE = pathlib.Path.home() / "data" / "gemma4" / "sample_submission"
COMPETITION_MODEL = "gemma-4-31b-it-qat-w4a16-ct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--model", default=COMPETITION_MODEL)
    parser.add_argument("--lessons", help="text file with a numbered list of lessons")
    args = parser.parse_args()

    out = pathlib.Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(SAMPLE, out, ignore=shutil.ignore_patterns("adapters"))

    for yaml_file in [out / "agent.yaml", *out.glob("sub_agents/*.yaml")]:
        text = re.sub(r"(?m)^adapter:.*\n", "", yaml_file.read_text())
        yaml_file.write_text(text.replace(COMPETITION_MODEL, args.model))

    if args.lessons:
        lessons = pathlib.Path(args.lessons).read_text().strip()
        system = out / "prompts" / "system.md"
        system.write_text(system.read_text().rstrip()
                          + "\n\n## Lessons From Past Tasks\n" + lessons + "\n")
    print(f"submission written to {out}")


if __name__ == "__main__":
    main()
