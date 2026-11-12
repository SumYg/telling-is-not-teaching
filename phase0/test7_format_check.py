"""Post-hoc check on test 7 (not pre-registered): how much of each gain is the answer's layout?

    python phase0/test7_format_check.py

Rescores every saved answer with the family's own scorer after fixing two layout slips only: a literal backslash-n
written where a line break belongs (mini_sudoku and survo grids), and true or false in lower case (course_schedule).
Everything else about the scoring is unchanged. Prints the gains and the forgetting as scored and with the slips
forgiven, the signals' rank correlations with the forgiven gains, and test 6's survo round 3 both ways.
"""
import json
import pathlib

import test7_families as T
from common import CORRECT, read_jsonl
from test4_weights import dataset
from test5_gate import boot

BS_N, NL = chr(92) + "n", chr(10)  # a literal backslash-n, and a real line break


def variants(answer):
    out = {answer, answer.replace(BS_N, NL)}
    if answer.strip().lower() in ("true", "false"):
        out.add(answer.strip().capitalize())
    return out


def rates(data, entries, path, forgive):
    """Per-problem pass rates of one evaluation file, optionally with the two slips forgiven."""
    per = []
    for r, e in zip(read_jsonl(path), entries):
        ok = [s >= CORRECT or (forgive and a is not None and
                               max(data.score_answer(answer=v, entry=e) for v in variants(str(a))) >= CORRECT)
              for a, s in zip(r["answers"], r["scores"])]
        per.append(sum(ok) / len(ok))
    return per


def mean_rates(data, entries, paths, forgive):
    return [sum(c) / len(c) for c in zip(*[rates(data, entries, p, forgive) for p in paths])]


def gain(data, entries, base, trained, forgive):
    return [100 * x for x in boot(rates(data, entries, base, forgive), mean_rates(data, entries, trained, forgive), [], [])]


def show(t):
    return f"{t[0]:+5.1f} [{t[1]:+5.1f}, {t[2]:+5.1f}]"


def main():
    forgiven = {}
    print("| Family | Gain as scored | Gain, slips forgiven | Forgetting as scored | Forgetting, slips forgiven |")
    print("| --- | --- | --- | --- | --- |")
    for f in T.FAMILIES:
        data, entries = dataset(f, 500, 3000)
        own = [gain(data, entries, T.base_file(f), T.round1_files(f), forgive) for forgive in (False, True)]
        a = T.anchor(f)
        data_a, entries_a = dataset(a, 500, 3000)
        other = [gain(data_a, entries_a, T.base_file(a), T.round1_files(f, a), forgive) for forgive in (False, True)]
        forgiven[f] = own[1][0]
        print(f"| {f} | {show(own[0])} | {show(own[1])} | {show(other[0])} ({a}) | {show(other[1])} |")
    print("\nSignals against the forgiven gains (Spearman, exact two-sided p):")
    for name in T.SIGNALS:
        x = [json.loads((T.ROOT / f / "signals.json").read_text())[name] for f in T.FAMILIES]
        rho, p = T.permutation_p(x, [forgiven[f] for f in T.FAMILIES], one_sided=False)
        print(f"- {name}: rho {rho:+.2f}, p {p:.3f}")
    data, entries = dataset("survo", 500, 3000)
    r3 = [pathlib.Path("runs/test6/survo") / d / "eval.jsonl" for d in ("r3_s1", "r3_s2")]
    for forgive in (False, True):
        print(f"\nTest 6, survo round 3 minus base, {'slips forgiven' if forgive else 'as scored'}: "
              f"{show(gain(data, entries, T.base_file('survo'), r3, forgive))}")


if __name__ == "__main__":
    main()
