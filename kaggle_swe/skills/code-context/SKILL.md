---
name: code-context
description: Find where Python identifiers or an error message are defined and used in the repository, in one call. Prints whole definitions with line numbers, call sites, and the test files that mention them.
---
# code-context

Run `scripts/context.py` with the identifiers from the problem statement (function, class, method or constant names) or a
short error message in quotes:

    run_skill_script(skill_name="code-context", file_path="scripts/context.py", args=["Table", "_render"])

`file_path` is always `scripts/context.py`, this skill's own script, never a file of the repository. The repository files
and identifiers you want to look at go in `args`.

For each identifier it prints:
- where it is defined (file and line) and the whole definition with line numbers (up to 120 lines);
- up to 10 places that use it;
- the test files that mention it.

For text that is not an identifier (for example an error message), it prints the matching lines and the code around the
first matches. One call replaces several `grep` and small `read_file` calls.
