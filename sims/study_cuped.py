"""Does CUPED remove the variance it promises, without moving the estimate?

Spec §4. Two claims are on trial:

1. The realized variance reduction equals ``1 - rho^2``.
2. CUPED reduces variance *only*. If it also shifts the point estimate, it is not
   a variance-reduction technique, it is a bias.

Both are measured across a rho sweep. A third section demonstrates the failure the
guard exists to prevent: adjusting on a covariate the treatment moved, which
shrinks the estimate toward zero while making the interval tighter -- a readout
that looks more trustworthy precisely because it is wrong.

The sweep also measures something the guard's author does not get to assert: its
false-alarm rate. Every replication here uses a genuinely pre-treatment covariate,
so any time the guard raises, it is wrong. Across the full sweep that rate should
land on the D11 threshold of 0.001, and the study checks that it does.
"""

from __future__ import annotations

import sys

import numpy as np

from expkit.metrics.cuped import (
    COVARIATE_GUARD_THRESHOLD,
    CovariateContamination,
    cuped_adjust,
)
from sims._report import banner, rate, savefig, write_results

SEED = 20240604
ALPHA = 0.05
REPS = 2_000
N_PER_ARM = 2_000
TRUE_EFFECT = 0.05
RHOS = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9)


def _draw(rng, rho, n, effect, contaminate=0.0):
    """Covariate first, then outcome. ``contaminate`` leaks the treatment into X."""
    x = rng.normal(0.0, 1.0, n)
    y = rho * x + rng.normal(0.0, np.sqrt(1.0 - rho**2), n) + effect
    return y, x + contaminate


def sweep_point(rng, rho, reps, boot_rng):
    adjusted, unadjusted, covered, reductions, thetas = [], [], 0, [], []
    false_alarms = 0
    for _ in range(reps):
        y_c, x_c = _draw(rng, rho, N_PER_ARM, 0.0)
        y_t, x_t = _draw(rng, rho, N_PER_ARM, TRUE_EFFECT)
        # The covariate is pre-treatment by construction, so a raise here is a
        # false alarm. Count it, then re-run unguarded so the sweep statistics
        # cover every replication rather than only the ones that got through.
        try:
            r = cuped_adjust(y_control=y_c, x_control=x_c, y_treatment=y_t, x_treatment=x_t)
        except CovariateContamination:
            false_alarms += 1
            r = cuped_adjust(
                y_control=y_c,
                x_control=x_c,
                y_treatment=y_t,
                x_treatment=x_t,
                guard=False,
            )
        adjusted.append(r.estimate)
        unadjusted.append(r.unadjusted_estimate)
        reductions.append(r.realized_variance_reduction)
        thetas.append(r.theta)
        half = 1.959963984540054 * r.standard_error
        covered += r.estimate - half <= TRUE_EFFECT <= r.estimate + half
    adjusted = np.asarray(adjusted)
    unadjusted = np.asarray(unadjusted)
    var_adj = float(adjusted.var(ddof=1))
    var_raw = float(unadjusted.var(ddof=1))
    reduction = 1.0 - var_adj / var_raw
    bias = float(adjusted.mean() - TRUE_EFFECT)

    # The realized reduction is a ratio of two variances estimated from the same
    # replications, so its sampling error has no tidy closed form. Bootstrap it,
    # rather than compare against a tolerance picked to make the test pass.
    boot = boot_rng.integers(0, reps, size=(300, reps))
    boot_reduction = 1.0 - (
        adjusted[boot].var(axis=1, ddof=1) / unadjusted[boot].var(axis=1, ddof=1)
    )
    reduction_mc_se = float(boot_reduction.std(ddof=1))
    return {
        "rho": rho,
        "predicted_reduction": rho**2,
        "realized_reduction_across_reps": reduction,
        "reduction_mc_se": reduction_mc_se,
        "mean_within_rep_reduction": float(np.mean(reductions)),
        "mean_theta": float(np.mean(thetas)),
        "bias": bias,
        "bias_mc_se": float(adjusted.std(ddof=1) / np.sqrt(reps)),
        "mean_adjusted_estimate": float(adjusted.mean()),
        "mean_unadjusted_estimate": float(unadjusted.mean()),
        "coverage": rate(f"CUPED CI coverage rho={rho}", covered, reps),
        "guard_false_alarms": false_alarms,
    }


def contamination_demo(rng, reps=1_000, rho=0.6):
    """A post-treatment covariate: same machinery, silently wrong answer."""
    clean, dirty, dirty_se, clean_se = [], [], [], []
    for _ in range(reps):
        y_c, x_c = _draw(rng, rho, N_PER_ARM, 0.0)
        y_t, x_t = _draw(rng, rho, N_PER_ARM, TRUE_EFFECT, contaminate=TRUE_EFFECT)
        ok = cuped_adjust(
            y_control=y_c,
            x_control=x_c,
            y_treatment=y_t,
            x_treatment=x_t - TRUE_EFFECT,  # the honest pre-period version
            guard=False,  # guard calibration is measured in the sweep, not here
        )
        bad = cuped_adjust(
            y_control=y_c, x_control=x_c, y_treatment=y_t, x_treatment=x_t, guard=False
        )
        clean.append(ok.estimate)
        dirty.append(bad.estimate)
        clean_se.append(ok.standard_error)
        dirty_se.append(bad.standard_error)
    return {
        "replications": reps,
        "rho": rho,
        "true_effect": TRUE_EFFECT,
        "clean_mean_estimate": float(np.mean(clean)),
        "contaminated_mean_estimate": float(np.mean(dirty)),
        "clean_mean_se": float(np.mean(clean_se)),
        "contaminated_mean_se": float(np.mean(dirty_se)),
        "bias_fraction_of_true_effect": float(np.mean(dirty) / TRUE_EFFECT - 1.0),
    }


def main() -> int:
    rng = np.random.default_rng(SEED)

    banner(f"CUPED - variance reduction vs theory (true effect {TRUE_EFFECT:+.3f})")
    # A separate generator for the bootstrap: a diagnostic must not perturb the
    # data stream of the experiment it is diagnosing.
    boot_rng = np.random.default_rng(SEED + 1)
    rows = [sweep_point(rng, rho, REPS, boot_rng) for rho in RHOS]
    print(
        f"\n  {'rho':>5}{'theory':>9}{'realized':>10}{'MC SE':>8}{'sigma':>7}"
        f"{'theta':>8}{'bias':>10}{'bias sig':>9}{'CI cover':>10}"
    )
    for r in rows:
        sigma = abs(r["realized_reduction_across_reps"] - r["predicted_reduction"]) / max(
            r["reduction_mc_se"], 1e-12
        )
        print(
            f"  {r['rho']:>5.2f}{r['predicted_reduction']:>9.4f}"
            f"{r['realized_reduction_across_reps']:>10.4f}{r['reduction_mc_se']:>8.4f}"
            f"{sigma:>7.2f}{r['mean_theta']:>8.4f}"
            f"{r['bias']:>+10.5f}{abs(r['bias']) / r['bias_mc_se']:>9.2f}"
            f"{r['coverage'].rate:>10.4f}"
        )

    total_alarms = sum(r["guard_false_alarms"] for r in rows)
    alarm_rate = rate("covariate guard false alarms", total_alarms, REPS * len(RHOS))
    worst_reduction = max(
        abs(r["realized_reduction_across_reps"] - r["predicted_reduction"])
        / max(r["reduction_mc_se"], 1e-12)
        for r in rows
    )
    worst_bias_sigma = max(abs(r["bias"]) / r["bias_mc_se"] for r in rows)
    print(f"\n  largest gap from theory, in Monte Carlo SEs: {worst_reduction:.2f}")
    print(f"  largest bias, in Monte Carlo SEs:            {worst_bias_sigma:.2f}")
    print("\n  The estimate does not move. Only its variance does -- which is the whole")
    print("  claim: at rho=0.9 the same experiment resolves the same effect with about")
    print(
        f"  {1 - RHOS[-1] ** 2:.0%} of the variance, worth roughly a {RHOS[-1] ** 2:.0%} "
        f"cut in required sample size."
    )

    banner("COVARIATE GUARD - false-alarm rate on covariates known to be clean")
    print(
        f"\n  raised on {alarm_rate.successes} of {alarm_rate.trials:,} clean covariates: "
        f"{alarm_rate.rate:.5f} +/- {alarm_rate.mc_se:.5f}"
    )
    print(
        f"  D11 threshold:                                        "
        f"{COVARIATE_GUARD_THRESHOLD:.5f}"
    )
    print(
        f"  distance from threshold:                              "
        f"{alarm_rate.sigma_from(COVARIATE_GUARD_THRESHOLD):.2f} Monte Carlo SE"
    )
    print("\n  The guard is calibrated, not free. Run 1,000 CUPED-adjusted metrics and it")
    print("  will block one clean covariate. That is the price of catching the")
    print("  contaminated ones, and it is the same trade the SRM threshold makes.")

    banner("WHEN CUPED DOES NOT HELP, AND WHEN IT HURTS")
    print(
        f"\n  rho=0 gives {rows[0]['realized_reduction_across_reps']:.1%} reduction. A "
        f"pre-period covariate uncorrelated with the\n  outcome buys nothing -- and new "
        f"users have no pre-period at all, so on a\n  signup-flow experiment CUPED is "
        f"simply unavailable."
    )
    demo = contamination_demo(rng)
    print("\n  Contaminated covariate (treatment moved X), guard bypassed:")
    print(f"    true effect                {demo['true_effect']:+.5f}")
    print(
        f"    clean covariate estimate   {demo['clean_mean_estimate']:+.5f}  "
        f"(mean SE {demo['clean_mean_se']:.5f})"
    )
    print(
        f"    contaminated estimate      {demo['contaminated_mean_estimate']:+.5f}  "
        f"(mean SE {demo['contaminated_mean_se']:.5f})"
    )
    print(
        f"    bias                       {demo['bias_fraction_of_true_effect']:+.1%} "
        f"of the true effect"
    )
    print("\n  The interval got tighter and the answer got wrong. That combination is why")
    print("  the guard raises rather than warns.")

    savefig(_chart(rows), "study_cuped")
    reduction_ok = worst_reduction < 3.0
    unbiased_ok = worst_bias_sigma < 3.0
    guard_ok = alarm_rate.within(COVARIATE_GUARD_THRESHOLD, 3.0)
    write_results(
        "study_cuped",
        {
            "alpha": ALPHA,
            "seed": SEED,
            "replications": REPS,
            "n_per_arm": N_PER_ARM,
            "true_effect": TRUE_EFFECT,
            "sweep": [{**r, "coverage": r["coverage"].__dict__} for r in rows],
            "contamination_demo": demo,
            "guard_false_alarm_rate": alarm_rate.__dict__,
            "guard_threshold": COVARIATE_GUARD_THRESHOLD,
            "max_reduction_gap": worst_reduction,
            "max_bias_sigma": worst_bias_sigma,
            "reduction_matches_theory": reduction_ok,
            "estimator_unbiased": unbiased_ok,
            "guard_calibrated": guard_ok,
        },
    )
    if not (reduction_ok and unbiased_ok and guard_ok):
        print("\n  FAIL: reduction, unbiasedness, or guard calibration did not hold.")
        return 1
    print("\n  PASS: reduction tracks 1-rho^2, the estimator stays unbiased, and the")
    print("  covariate guard fires at its stated rate.")
    return 0


def _chart(rows):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    rhos = [r["rho"] for r in rows]

    axes[0].plot(
        rhos,
        [r["predicted_reduction"] for r in rows],
        "-",
        color="#2f855a",
        linewidth=2,
        label="theory: 1 - rho^2",
    )
    axes[0].errorbar(
        rhos,
        [r["realized_reduction_across_reps"] for r in rows],
        yerr=[1.96 * r["reduction_mc_se"] for r in rows],
        fmt="o",
        color="#2b6cb0",
        markersize=7,
        capsize=4,
        label="realized across replications",
    )
    axes[0].set_xlabel("corr(outcome, pre-period covariate)")
    axes[0].set_ylabel("variance reduction")
    axes[0].set_title("CUPED removes the variance it promises")
    axes[0].legend(frameon=False, loc="upper left")

    bias = [r["bias"] for r in rows]
    err = [1.96 * r["bias_mc_se"] for r in rows]
    axes[1].errorbar(rhos, bias, yerr=err, fmt="o", color="#2b6cb0", capsize=4, markersize=7)
    axes[1].axhline(0.0, color="#2f855a", linestyle="--", linewidth=1.4, label="unbiased")
    axes[1].set_xlabel("corr(outcome, pre-period covariate)")
    axes[1].set_ylabel("bias in the estimated effect")
    axes[1].set_title("...and does not move the point estimate")
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    sys.exit(main())
