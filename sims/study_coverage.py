"""Do 95% confidence intervals actually cover the truth 95% of the time?

Spec §4, and the home of this repository's headline artifact.

Part A checks coverage for every estimator against a known true effect. Part B is
the comparison worth publishing: the same ratio metric analysed two ways, sweeping
how many sessions each user contributes. The delta-method interval holds nominal
coverage throughout. The naive interval -- the one you get by treating sessions as
independent observations -- falls further below nominal the more clustered the data
becomes, and it never announces that it is doing so.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from expkit.inference.fixed import two_proportion_test, welch_test
from expkit.metrics.ratio import ratio_estimate, ratio_test
from sims._report import banner, rate, savefig, write_results

SEED = 20240602
ALPHA = 0.05
NOMINAL = 1.0 - ALPHA
REPS = 10_000
SWEEP_REPS = 4_000

N_BINARY = 4_000
N_CONTINUOUS = 2_000
N_RATIO_UNITS = 2_000

P_CONTROL, BINARY_EFFECT = 0.10, 0.01
PURCHASE_RATE, LOG_MEAN, LOG_SD, REVENUE_LIFT = 0.12, 3.0, 0.9, 0.05
BETA_A, BETA_B, RATIO_LIFT = 2.0, 8.0, 0.05
SESSIONS_SWEEP = (1.0, 2.0, 4.0, 8.0, 16.0)


def _covers(result, truth: float) -> bool:
    return bool(result.ci_lower <= truth <= result.ci_upper)


def binary_coverage(rng, reps):
    truth = BINARY_EFFECT
    hits = 0
    for _ in range(reps):
        s_c = rng.binomial(N_BINARY, P_CONTROL)
        s_t = rng.binomial(N_BINARY, P_CONTROL + BINARY_EFFECT)
        hits += _covers(
            two_proportion_test(
                successes_control=int(s_c),
                n_control=N_BINARY,
                successes_treatment=int(s_t),
                n_treatment=N_BINARY,
                alpha=ALPHA,
            ),
            truth,
        )
    return rate("binary (two-proportion)", hits, reps), truth


def continuous_coverage(rng, reps):
    """True mean of the zero-inflated lognormal is exact, not estimated."""
    mean_control = PURCHASE_RATE * math.exp(LOG_MEAN + LOG_SD**2 / 2.0)
    truth = mean_control * REVENUE_LIFT
    hits = 0
    for _ in range(reps):
        buy = rng.random((2, N_CONTINUOUS)) < PURCHASE_RATE
        amt = rng.lognormal(LOG_MEAN, LOG_SD, size=(2, N_CONTINUOUS))
        draws = np.where(buy, amt, 0.0)
        hits += _covers(
            welch_test(control=draws[0], treatment=draws[1] * (1 + REVENUE_LIFT), alpha=ALPHA),
            truth,
        )
    return rate("continuous (Welch)", hits, reps), truth


def _ratio_arm(rng, n_units, sessions_mean, lift):
    propensity = np.clip(rng.beta(BETA_A, BETA_B, n_units) * (1 + lift), 0.0, 1.0)
    sessions = 1 + rng.poisson(sessions_mean - 1.0, n_units)
    clicks = rng.binomial(sessions, propensity)
    return clicks.astype(float), sessions.astype(float)


def ratio_coverage(rng, reps, sessions_mean, naive):
    base = BETA_A / (BETA_A + BETA_B)
    truth = base * RATIO_LIFT
    hits = 0
    for _ in range(reps):
        n_c, d_c = _ratio_arm(rng, N_RATIO_UNITS, sessions_mean, 0.0)
        n_t, d_t = _ratio_arm(rng, N_RATIO_UNITS, sessions_mean, RATIO_LIFT)
        hits += _covers(
            ratio_test(
                numerator_control=n_c,
                denominator_control=d_c,
                numerator_treatment=n_t,
                denominator_treatment=d_t,
                alpha=ALPHA,
                naive=naive,
            ),
            truth,
        )
    label = "ratio (naive, denominator unit)" if naive else "ratio (delta method)"
    return rate(f"{label}", hits, reps), truth


def variance_recovery(rng, reps=3_000):
    """Does the analytic delta-method variance match the empirical spread of R?"""
    ratios, analytic = [], []
    for _ in range(reps):
        num, den = _ratio_arm(rng, N_RATIO_UNITS, 4.0, 0.0)
        est = ratio_estimate(num, den)
        ratios.append(est.ratio)
        analytic.append(est.standard_error)
    empirical_sd = float(np.std(ratios, ddof=1))
    predicted_sd = float(np.mean(analytic))
    return {
        "empirical_sd_of_ratio": empirical_sd,
        "mean_delta_method_se": predicted_sd,
        "ratio_predicted_to_empirical": predicted_sd / empirical_sd,
        "replications": reps,
    }


def main() -> int:
    rng = np.random.default_rng(SEED)

    banner(f"PART A - CI COVERAGE AT NOMINAL {NOMINAL:.0%}")
    rows = []
    rows.append(binary_coverage(rng, REPS))
    rows.append(continuous_coverage(rng, REPS))
    rows.append(ratio_coverage(rng, REPS, 4.0, naive=False))
    rows.append(ratio_coverage(rng, REPS, 4.0, naive=True))

    print(f"\n  {'estimator':<34}{'coverage':>11}{'MC SE':>9}{'sigma from 95%':>16}")
    for r, _ in rows:
        print(f"  {r.label:<34}{r.rate:>11.4f}{r.mc_se:>9.4f}{r.sigma_from(NOMINAL):>16.2f}")

    banner("VARIANCE RECOVERY - analytic delta method vs empirical spread")
    recovery = variance_recovery(rng)
    print(
        f"\n  empirical sd of R over {recovery['replications']:,} resamples: "
        f"{recovery['empirical_sd_of_ratio']:.6f}"
    )
    print(f"  mean analytic delta-method SE:                {recovery['mean_delta_method_se']:.6f}")
    print(
        f"  ratio predicted / empirical:                  "
        f"{recovery['ratio_predicted_to_empirical']:.4f}"
    )

    banner("PART B - COVERAGE vs CLUSTERING (the headline)")
    sweep = []
    for sessions_mean in SESSIONS_SWEEP:
        delta, _ = ratio_coverage(rng, SWEEP_REPS, sessions_mean, naive=False)
        naive, _ = ratio_coverage(rng, SWEEP_REPS, sessions_mean, naive=True)
        sweep.append({"sessions_per_user": sessions_mean, "delta": delta, "naive": naive})
        print(
            f"  {sessions_mean:>5.0f} sessions/user   "
            f"delta method {delta.rate:.4f} +/- {delta.mc_se:.4f}   "
            f"naive {naive.rate:.4f} +/- {naive.mc_se:.4f}   "
            f"shortfall {NOMINAL - naive.rate:+.4f}"
        )

    worst = min(sweep, key=lambda s: s["naive"].rate)
    print(
        f"\n  At {worst['sessions_per_user']:.0f} sessions per user the naive interval "
        f"covers {worst['naive'].rate:.1%} of the time\n"
        f"  while claiming {NOMINAL:.0%}. One in "
        f"{1 / (1 - worst['naive'].rate):.1f} \"95% confident\" readouts is wrong,\n"
        f"  against the one in 20 the label promises. Nothing in the output says so."
    )

    savefig(_chart(sweep), "study_coverage")
    delta_ok = all(s["delta"].within(NOMINAL, 3.0) for s in sweep)
    write_results(
        "study_coverage",
        {
            "alpha": ALPHA,
            "seed": SEED,
            "nominal": NOMINAL,
            "replications": {"table": REPS, "sweep": SWEEP_REPS},
            "part_a": [{"truth": t, **r.__dict__} for r, t in rows],
            "variance_recovery": recovery,
            "part_b_sweep": [
                {
                    "sessions_per_user": s["sessions_per_user"],
                    "delta_method": s["delta"].__dict__,
                    "naive": s["naive"].__dict__,
                }
                for s in sweep
            ],
            "delta_method_holds_nominal": delta_ok,
            "worst_naive_coverage": worst["naive"].rate,
        },
    )
    if not delta_ok:
        print("\n  FAIL: delta-method coverage left the nominal band.")
        return 1
    print("\n  PASS: delta-method coverage is nominal at every clustering level.")
    return 0


def _chart(sweep):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = [s["sessions_per_user"] for s in sweep]
    for key, color, marker, label in (
        ("delta", "#2b6cb0", "o", "delta method (randomization unit)"),
        ("naive", "#c53030", "s", "naive (denominator unit)"),
    ):
        y = [s[key].rate for s in sweep]
        err = [1.96 * s[key].mc_se for s in sweep]
        ax.errorbar(
            x,
            y,
            yerr=err,
            fmt=marker + "-",
            color=color,
            capsize=4,
            label=label,
            markersize=7,
            linewidth=1.8,
        )
    ax.axhline(NOMINAL, color="#2f855a", linestyle="--", linewidth=1.4, label="nominal 95%")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.0f}" for v in x])
    ax.set_xlabel("mean sessions per user (more sessions = more within-user clustering)")
    ax.set_ylabel("realized coverage of a nominal 95% CI")
    ax.set_title("Ratio-metric CI coverage: what the unit-of-analysis error costs")
    ax.legend(frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    sys.exit(main())
