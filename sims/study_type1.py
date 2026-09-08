"""Type I error under a true null: does each test reject at exactly alpha?

Spec §4. Every scenario runs at least 10,000 replications and reports the
realized rejection rate with its Monte Carlo standard error, because you cannot
judge "5.1%" without knowing whether the noise is 0.1% or 1%.

Two design choices worth stating:

* **The tests are called, not re-implemented.** Each replication invokes the same
  ``expkit.inference`` function a user would call. A study that re-derives the
  arithmetic validates the study, not the library.
* **Continuous data is zero-inflated lognormal**, not Gaussian. Testing a t-test
  on normal data proves only that the derivation was copied correctly. The
  question that matters is whether it holds up on revenue, which is 88% zeros
  with a heavy tail -- so that is what it is fed.
"""

from __future__ import annotations

import sys

import numpy as np

from expkit.inference.fixed import two_proportion_test, welch_test
from expkit.metrics.cuped import cuped_test
from expkit.metrics.ratio import ratio_test
from sims._report import banner, rate, savefig, write_results

SEED = 20240601
ALPHA = 0.05
REPS_BINARY = 20_000
REPS_CONTINUOUS = 10_000

BINARY_SIZES = (200, 1_000, 5_000, 50_000)
CONTINUOUS_SIZES = (200, 1_000, 5_000)
RATIO_SIZES = (500, 2_000)
CUPED_SIZES = (500, 2_000)
BASELINE_RATE = 0.10
CUPED_RHO = 0.7


def binary_type1(rng: np.random.Generator, n: int, reps: int) -> tuple:
    """Both arms share one true rate, so every rejection is a false positive."""
    successes = rng.binomial(n, BASELINE_RATE, size=(reps, 2))
    rejections = 0
    for s_c, s_t in successes:
        if s_c in (0, n) and s_t in (0, n) and s_c == s_t:
            continue  # degenerate: no test is defined, and it cannot reject
        result = two_proportion_test(
            successes_control=int(s_c),
            n_control=n,
            successes_treatment=int(s_t),
            n_treatment=n,
            alpha=ALPHA,
        )
        rejections += result.significant
    return rate(f"binary n={n:,}", rejections, reps), n


def continuous_type1(rng: np.random.Generator, n: int, reps: int) -> tuple:
    """Zero-inflated lognormal revenue, identical distribution in both arms."""
    rejections = 0
    for _ in range(reps):
        draws = np.where(rng.random((2, n)) < 0.12, rng.lognormal(3.0, 0.9, size=(2, n)), 0.0)
        result = welch_test(control=draws[0], treatment=draws[1], alpha=ALPHA)
        rejections += result.significant
    return rate(f"continuous n={n:,}", rejections, reps), n


def ratio_type1(rng: np.random.Generator, n: int, reps: int) -> tuple:
    """Clustered ratio metric, identical in both arms, delta-method variance."""

    def arm():
        propensity = rng.beta(2.0, 8.0, n)
        sessions = 1 + rng.poisson(3.0, n)
        return rng.binomial(sessions, propensity).astype(float), sessions.astype(float)

    rejections = 0
    for _ in range(reps):
        n_c, d_c = arm()
        n_t, d_t = arm()
        rejections += ratio_test(
            numerator_control=n_c,
            denominator_control=d_c,
            numerator_treatment=n_t,
            denominator_treatment=d_t,
            alpha=ALPHA,
        ).significant
    return rate(f"ratio n={n:,}", rejections, reps), n


def cuped_type1(rng: np.random.Generator, n: int, reps: int) -> tuple:
    """CUPED-adjusted outcome under the null.

    ``guard=False``: the covariate is pre-treatment by construction, so every
    raise would be a false alarm rather than a rejection. The guard's own
    calibration is measured in ``study_cuped.py``; mixing it in here would
    conflate two different rates.
    """

    def arm():
        x = rng.normal(0.0, 1.0, n)
        y = CUPED_RHO * x + rng.normal(0.0, np.sqrt(1 - CUPED_RHO**2), n)
        return y, x

    rejections = 0
    for _ in range(reps):
        y_c, x_c = arm()
        y_t, x_t = arm()
        result, _ = cuped_test(
            y_control=y_c,
            x_control=x_c,
            y_treatment=y_t,
            x_treatment=x_t,
            alpha=ALPHA,
            guard=False,
        )
        rejections += result.significant
    return rate(f"CUPED n={n:,}", rejections, reps), n


def main() -> int:
    banner(f"TYPE I ERROR UNDER THE NULL (alpha={ALPHA})")
    rng = np.random.default_rng(SEED)
    rows, failures = [], []

    for n in BINARY_SIZES:
        r, size = binary_type1(rng, n, REPS_BINARY)
        rows.append(("binary", size, r))
    for n in CONTINUOUS_SIZES:
        r, size = continuous_type1(rng, n, REPS_CONTINUOUS)
        rows.append(("continuous", size, r))
    for n in RATIO_SIZES:
        r, size = ratio_type1(rng, n, REPS_CONTINUOUS)
        rows.append(("ratio", size, r))
    for n in CUPED_SIZES:
        r, size = cuped_type1(rng, n, REPS_CONTINUOUS)
        rows.append(("cuped", size, r))

    print(f"\n  {'test':<12}{'n/arm':>9}{'rejection rate':>17}{'MC SE':>10}{'sigma from 5%':>15}")
    for family, n, r in rows:
        flag = "" if r.within(ALPHA, 3.0) else "   <-- OUTSIDE 3 MC SE"
        if flag:
            failures.append(r.label)
        print(
            f"  {family:<12}{n:>9,}{r.rate:>17.4f}{r.mc_se:>10.4f}"
            f"{r.sigma_from(ALPHA):>15.2f}{flag}"
        )

    print(
        "\n  A rate near 5.1% is not automatically a bug: at 20,000 replications one\n"
        "  Monte Carlo SE is about 0.15%, so 5.1% sits well inside the noise. The\n"
        "  error bar is what makes that judgement possible."
    )

    fig, ax = _chart(rows)
    savefig(fig, "study_type1")
    write_results(
        "study_type1",
        {
            "alpha": ALPHA,
            "seed": SEED,
            "replications": {"binary": REPS_BINARY, "other": REPS_CONTINUOUS},
            "scenarios": [
                {"family": f, "n_per_arm": n, **{k: v for k, v in r.__dict__.items()}}
                for f, n, r in rows
            ],
            "all_within_3_mc_se": not failures,
            "outside_3_mc_se": failures,
        },
    )
    if failures:
        print(f"\n  FAIL: {len(failures)} scenario(s) outside 3 Monte Carlo SE of alpha.")
        return 1
    print("\n  PASS: every test rejects at the nominal rate within Monte Carlo error.")
    return 0


def _chart(rows):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.4))
    colors = {"binary": "#2b6cb0", "continuous": "#c05621", "ratio": "#6b46c1", "cuped": "#2c7a7b"}
    markers = {"binary": "o", "continuous": "s", "ratio": "^", "cuped": "D"}
    labels = []
    for i, (family, n, r) in enumerate(rows):
        ax.errorbar(
            i,
            r.rate,
            yerr=1.96 * r.mc_se,
            fmt=markers[family],
            color=colors[family],
            capsize=4,
            markersize=7,
        )
        labels.append(f"{family}\n{n:,}")
    ax.axhline(ALPHA, color="#2f855a", linestyle="--", linewidth=1.4, label="nominal alpha = 0.05")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("false-positive rate")
    ax.set_title("Type I error under the null, every estimator, " "with 95% Monte Carlo intervals")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    sys.exit(main())
