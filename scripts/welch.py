"""Welch's two-sample 95% interval for a difference of means, in plain Python (no scipy).

    python scripts/welch.py --a 0.13 0.13 0.12 0.10 --b 0.15 0.13 0.10 0.12      # b minus a

Used for the leaderboard series (arc3/LESSONS_REPEAT.md, kaggle_swe/LEADERBOARD_AB.md). The t quantile comes from
integrating Student's t density with Simpson's rule and bisecting; `python scripts/welch.py --check` compares it with
table values.
"""
import argparse
import math


def t_cdf(x, df, steps=4000):
    """P(T <= x) for Student's t with df degrees of freedom (df may be fractional), x >= 0."""
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
    f = lambda u: c * (1 + u * u / df) ** (-(df + 1) / 2)  # noqa: E731
    h = x / steps
    total = f(0) + f(x) + sum((4 if i % 2 else 2) * f(i * h) for i in range(1, steps))
    return 0.5 + total * h / 3


def t_quantile(p, df):
    """The x with P(T <= x) = p, for p in (0.5, 1)."""
    lo, hi = 0.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, df) < p else (lo, mid)
    return (lo + hi) / 2


def welch(a, b, level=0.95):
    """Mean of b minus mean of a, with Welch's interval and degrees of freedom."""
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    var = lambda v: sum((x - mean(v)) ** 2 for x in v) / (len(v) - 1)  # noqa: E731
    va, vb = var(a) / len(a), var(b) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    d = mean(b) - mean(a)
    half = t_quantile(1 - (1 - level) / 2, df) * se
    return d, d - half, d + half, df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=float, nargs="+")
    parser.add_argument("--b", type=float, nargs="+")
    parser.add_argument("--check", action="store_true", help="compare t quantiles with table values")
    args = parser.parse_args()
    if args.check:
        for df, table in ((3, 3.1824), (5, 2.5706), (10, 2.2281), (30, 2.0423)):
            print(f"df {df}: t(0.975) {t_quantile(0.975, df):.4f}, table {table}")
        return
    d, lo, hi, df = welch(args.a, args.b)
    print(f"b minus a: {d:+.4f}, 95% interval [{lo:+.4f}, {hi:+.4f}], Welch df {df:.2f}")


if __name__ == "__main__":
    main()
