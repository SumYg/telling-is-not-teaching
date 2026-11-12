"""Download Gemma 4 Developer Agent competition files into ~/data/gemma4, selectively.

The full data is 22.4 GB, almost all repository snapshots. Experiments run on Kaggle, where
the data is already mounted, so locally we fetch only what debugging needs.

    python kaggle_swe/download.py small                 # docs, task list, samples (~3 MB)
    python kaggle_swe/download.py support               # sandbox wheels (28 MB)
    python kaggle_swe/download.py tasks ID ...          # one task's snapshot, graph, embeddings
    python kaggle_swe/download.py wheelhouse            # local harness dataset (0.87 GB)

Run inside WSL after `source scripts/env.sh`. Files already present are skipped.
"""
import json
import pathlib
import subprocess
import sys
import time
import zipfile

COMPETITION = "gemma-4-developer-agent"
WHEELHOUSE = "metric/gemma-4-developer-agent-wheelhouse"
DATA = pathlib.Path.home() / "data" / "gemma4"
SMALL = ("HARNESS_README.md", "tasks.jsonl", "sample_submission/", "docker/", "sandbox/")
SUPPORT = ("wheels/",)  # graphs and embeddings come per task, via "tasks"


def run(cmd, attempts=5, done=None):
    """Run a Kaggle CLI command, retrying transient server errors (Kaggle returns
    intermittent 500s). `done` optionally checks success directly, since some calls exit
    non-zero even though the file arrived. Returns stdout, or None after the last attempt."""
    for attempt in range(1, attempts + 1):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 or (done and done()):
            return result.stdout
        error = (result.stderr or result.stdout).strip().splitlines()
        print(f"  attempt {attempt}: {error[-1][:120] if error else 'no output'}", flush=True)
        time.sleep(15 * attempt)
    return None


def list_files():
    names, token = [], None
    while True:
        cmd = ["kaggle", "competitions", "files", COMPETITION, "--csv", "--page-size", "200"]
        cmd += ["--page-token", token] if token else []
        out = run(cmd)
        if out is None:
            raise SystemExit("could not list the competition files")
        token = None
        for line in out.splitlines():
            if line.startswith("Next Page Token"):
                token = line.split("=", 1)[1].strip()
            elif line and not line.startswith("name,"):
                names.append(line.split(",")[0])
        if not token:
            return names


def fetch(name):
    """Download one file; returns False if it never arrived."""
    target = DATA / name
    zipped = target.parent / (target.name + ".zip")
    if target.exists():
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "competitions", "download", COMPETITION, "-f", name, "-p",
         str(target.parent), "-q"], done=lambda: target.exists() or zipped.exists())
    if zipped.exists():  # the CLI zips some files on the way down
        with zipfile.ZipFile(zipped) as z:
            z.extractall(target.parent)
        zipped.unlink()
    return target.exists()


def main():
    what, args = sys.argv[1], sys.argv[2:]
    if what == "wheelhouse":
        subprocess.run(["kaggle", "datasets", "download", WHEELHOUSE, "-p",
                        str(DATA / "wheelhouse"), "--unzip", "-q"], check=True)
        return
    names = list_files()
    if what == "small":
        chosen = [n for n in names if n.startswith(SMALL)]
    elif what == "support":
        chosen = [n for n in names if n.startswith(SUPPORT)]
    elif what == "tasks":  # files are named by instance id or by base commit
        tasks = {t["instance_id"]: t for t in map(json.loads, (DATA / "tasks.jsonl").read_text().splitlines())}
        keys = [k for a in args for k in (a, tasks[a]["base_commit"])]
        chosen = [n for n in names if n.startswith(("snapshots/", "graphs/", "embeddings/"))
                  and any(k in n for k in keys)]
    else:
        raise SystemExit(f"unknown selection: {what}")
    failed = []
    for i, name in enumerate(chosen, 1):
        if not fetch(name):
            failed.append(name)
        if i % 25 == 0 or i == len(chosen):
            print(f"{what}: {i}/{len(chosen)}", flush=True)
    if failed:
        print(f"{len(failed)} files failed, rerun to retry: {' '.join(failed)}")


if __name__ == "__main__":
    main()
