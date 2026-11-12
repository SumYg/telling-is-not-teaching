"""Show where identifiers (or a piece of text) are defined and used in the repository, in one call.

    context.py NAME [NAME ...]

The skill runtime starts this script from a temporary directory, so the repository is found from PWD (the sandbox shell's
working directory) or /workspace. Output is capped at MAX_CHARS characters.
"""
import os
import re
import subprocess
import sys

MAX_CHARS = 12000
MAX_DEF_LINES = 120
MAX_DEFS = 3
MAX_USES = 10
MAX_TESTS = 5
IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def repo_root():
    """The task workspace. The skill runtime runs this script from a temporary directory, but the sandbox started the
    process in the workspace: bash sets PWD, the local sandbox sets PYTHONPATH to it, and the parent shell's working
    directory is it. The docker sandbox (the scorer's) uses /workspace."""
    candidates = [os.environ.get("PWD")]
    candidates += [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    try:
        candidates.append(os.readlink(f"/proc/{os.getppid()}/cwd"))
    except OSError:
        pass
    candidates += ["/workspace", os.getcwd()]
    for path in candidates:
        if path and os.path.isdir(os.path.join(path, ".git")):
            return path
    for path in candidates:
        if path and os.path.isdir(path) and any(f.endswith(".py") for f in os.listdir(path)):
            return path
    return candidates[1]


def git_grep(root, args):
    try:
        out = subprocess.run(["git", "grep", "-n", "-I", *args, "--", "*.py"], cwd=root, capture_output=True,
                             text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    hits = []
    for line in out.splitlines():
        path, _, rest = line.partition(":")
        num, _, code = rest.partition(":")
        if num.isdigit():
            hits.append((path, int(num), code))
    return hits


def is_test(path):
    name = os.path.basename(path)
    return name.startswith("test") or name.endswith("_test.py") or "/tests/" in "/" + path or "/test/" in "/" + path


def read_lines(root, path):
    try:
        with open(os.path.join(root, path), encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()
    except OSError:
        return []


def block(lines, start):
    """The definition starting at 0-based line `start`: decorators above it and every deeper-indented line below."""
    first = start
    while first > 0 and lines[first - 1].lstrip().startswith("@"):
        first -= 1
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    while end < len(lines) and end - first < MAX_DEF_LINES:
        line = lines[end]
        if line.strip() and len(line) - len(line.lstrip()) <= indent and not line.lstrip().startswith((")", "]", "}")):
            break
        end += 1
    return first, end


def numbered(lines, first, end):
    return "\n".join(f"{i + 1:5d}  {lines[i]}" for i in range(first, end))


def show_name(root, name):
    short = name.split(".")[-1]
    out = [f"=== {name}"]
    pattern = rf"^\s*(async\s+def|def|class)\s+{re.escape(short)}\b|^{re.escape(short)}\s*[:=]"
    defs = git_grep(root, ["-E", pattern])
    defs.sort(key=lambda h: (is_test(h[0]), h[0], h[1]))
    if not defs:
        out.append("no definition found")
    for path, num, _ in defs[:MAX_DEFS]:
        lines = read_lines(root, path)
        if not lines:
            continue
        first, end = block(lines, num - 1)
        more = " (cut at 120 lines)" if end - first >= MAX_DEF_LINES else ""
        out.append(f"--- defined in {path}:{num}{more}")
        out.append(numbered(lines, first, end))
    if len(defs) > MAX_DEFS:
        out.append(f"(+{len(defs) - MAX_DEFS} more definitions: " + ", ".join(f"{p}:{n}" for p, n, _ in defs[MAX_DEFS:MAX_DEFS + 8]) + ")")
    def_lines = {(p, n) for p, n, _ in defs}
    uses = [h for h in git_grep(root, ["-w", "-F", short]) if (h[0], h[1]) not in def_lines]
    code_uses = [h for h in uses if not is_test(h[0])]
    if code_uses:
        out.append(f"--- used in ({len(code_uses)} lines outside tests; first {min(len(code_uses), MAX_USES)})")
        out += [f"{p}:{n}: {c.strip()[:160]}" for p, n, c in code_uses[:MAX_USES]]
    tests = sorted({p for p, _, _ in uses if is_test(p)})
    if tests:
        out.append("--- tests mentioning it: " + ", ".join(tests[:MAX_TESTS]) + (" ..." if len(tests) > MAX_TESTS else ""))
    return "\n".join(out)


def show_text(root, text):
    out = [f"=== text: {text!r}"]
    hits = git_grep(root, ["-F", text])
    if not hits:
        return "\n".join(out + ["no matches"])
    hits.sort(key=lambda h: (is_test(h[0]), h[0], h[1]))
    out += [f"{p}:{n}: {c.strip()[:160]}" for p, n, c in hits[:15]]
    for path, num, _ in hits[:2]:
        lines = read_lines(root, path)
        if lines:
            out.append(f"--- around {path}:{num}")
            out.append(numbered(lines, max(0, num - 13), min(len(lines), num + 12)))
    return "\n".join(out)


def main(argv):
    names = [a.strip() for a in argv if a.strip()]
    if not names:
        print("usage: context.py NAME [NAME ...]  (identifiers or a short piece of text)")
        return
    root = repo_root()
    parts = [show_name(root, n) if IDENT.match(n) else show_text(root, n) for n in names]
    text = "\n\n".join(parts)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + f"\n[output cut at {MAX_CHARS} characters; ask for fewer names]"
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
