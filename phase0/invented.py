"""Invented task families with hidden rules that can drift partway through a lifetime.

Each family applies a secret list transformation: a short program of primitives such as
reverse, rotate or add-k. The base model cannot know the program. It can only learn it from
feedback, since the correct answer is revealed after each attempt. So any gain is real
learning, and a rule change makes old lessons wrong, which tests revision.

    stream = drifting_stream("A", n=500, drift_at=250, seed=0)
    entry = stream[0]           # {"question", "answer", "metadata": {"program", "phase"}}
    score_answer("[1, 2]", entry)
"""
import ast
import random

PRIMITIVES = {
    "reverse": lambda xs: xs[::-1],
    "rotate_left_1": lambda xs: xs[1:] + xs[:1],
    "rotate_left_2": lambda xs: xs[2:] + xs[:2],
    "rotate_right_1": lambda xs: xs[-1:] + xs[:-1],
    "swap_pairs": lambda xs: [xs[i ^ 1] if (i ^ 1) < len(xs) else xs[i] for i in range(len(xs))],
    "add_1": lambda xs: [(x + 1) % 10 for x in xs],
    "add_3": lambda xs: [(x + 3) % 10 for x in xs],
    "sort_ascending": sorted,
    "sort_descending": lambda xs: sorted(xs, reverse=True),
    "double": lambda xs: [(2 * x) % 10 for x in xs],
}


def random_program(rng, length=2):
    """A composition of distinct primitives, applied left to right."""
    return rng.sample(sorted(PRIMITIVES), length)


def apply_program(program, xs):
    for name in program:
        xs = list(PRIMITIVES[name](list(xs)))
    return xs


def make_entry(family, program, xs, phase):
    question = (f"Apply secret transformation {family} to the list {xs}. "
                f"Answer with the resulting list only, for example [1, 2, 3].")
    return {"question": question, "answer": str(apply_program(program, xs)),
            "metadata": {"family": family, "program": program, "phase": phase, "input": xs}}


def drifting_stream(family, n, drift_at=None, seed=0, length=2):
    """n problems for one family. If drift_at is set, the program changes at that index."""
    rng = random.Random(f"{family}-{seed}")
    programs = [random_program(rng, length)]
    if drift_at is not None:
        new = random_program(rng, length)
        while new == programs[0]:
            new = random_program(rng, length)
        programs.append(new)
    entries = []
    for i in range(n):
        phase = 1 if drift_at is not None and i >= drift_at else 0
        xs = [rng.randrange(10) for _ in range(rng.randint(5, 7))]
        entries.append(make_entry(family, programs[phase], xs, phase))
    return entries


def score_answer(answer, entry):
    """1.0 for the exact list, else 0.0. Tolerates spacing and brackets written as text."""
    if answer is None:
        return 0.0
    try:
        value = ast.literal_eval(answer.strip())
    except (ValueError, SyntaxError):
        return 0.0
    return 1.0 if isinstance(value, list) and str(value) == entry["answer"] else 0.0
