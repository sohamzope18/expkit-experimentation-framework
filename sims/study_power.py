"""Does the empirical power curve match what the design module predicted?

Spec §4. The design module (§3.1) makes a falsifiable claim: run this many units
and you will detect this effect this often. This study runs the experiments and
checks.

The continuous arm is fed zero-inflated lognormal revenue rather than Gaussian
data, with the true variance computed in closed form from the generating
parameters. That makes the comparison a genuine test of the design formula against
data that violates its assumptions -- which is the only version worth running.

It found one. A *multiplicative* treatment effect -- revenue up 15% -- scales the
treatment arm's variance by (1 + lift)^2, so the arms are no longer homoscedastic.
The standard sample-size formula takes a single ``variance`` argument and assumes
both arms share it, so it over-predicts power: at a 15% lift it promises 20.3% and
delivers 18.2%. Supplying the average of the two arm variances restores agreement
to within Monte Carlo error. The study reports both predictions so the gap is
visible rather than dismissed as noise.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from expkit.design.power import power_for_n_binary, power_for_n_continuous
from expkit.inference.fixed import two_proportion_test, welch_test
from sims._report import banner, rate, savefig, write_results

SEED = 20240603
ALPHA = 0.05
REPS = 5_000

N_BINARY = 2_000
P_CONTROL = 0.10
BINARY_EFFECTS = (0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030)

N_CONTINUOUS = 2_000
PURCHASE_RATE, LOG_MEAN, LOG_SD = 0.12, 3.0, 0.9
REVENUE_LIFTS = (0.00, 0.03, 0.06, 0.09, 0.12, 0.15)


def revenue_moments() -> tuple[float, float]:
    """Exact mean and variance of the zero-inflated lognormal, not estimated."""
    mean = PURCHASE_RATE * math.exp(LOG_MEAN + LOG_SD**2 / 2.0)
    second = PURCHASE_RATE * math.exp(2 * LOG_MEAN + 2 * LOG_SD**2)
    return mean, second - mean**2


def binary_power(rng, effect, reps):
    rejections = 0
    for _ in range(reps):
        s_c = rng.binomial(N_BINARY, P_CONTROL)
        s_t = rng.binomial(N_BINARY, P_CONTROL + effect)
        rejections += two_proportion_test(
            successes_control=int(s_c),
            n_control=N_BINARY,
            successes_treatment=int(s_t),
            n_treatment=N_BINARY,
            alpha=ALPHA,
        ).significant
    return rate(f"binary d={effect:.3f}", rejections, reps)


def continuous_power(rng, lift, reps):
    rejections = 0
    for _ in range(reps):
        buy = rng.random((2, N_CONTINUOUS)) < PURCHASE_RATE
        amt = rng.lognormal(LOG_MEAN, LOG_SD, size=(2, N_CONTINUOUS))
        draws = np.where(buy, amt, 0.0)
        rejections += welch_test(
            control=draws[0], treatment=draws[1] * (1 + lift), alpha=ALPHA
        ).significant
    return rate(f"continuous lift={lift:.2f}", rejections, reps)


def main() -> int:
    rng = np.random.default_rng(SEED)
    mean_rev, var_rev = revenue_moments()

    banner("BINARY - empirical power vs analytic prediction")
    binary_rows = []
    print(f"\n  {'effect':>8}{'predicted':>12}{'empirical':>12}{'MC SE':>9}{'sigma':>8}")
    for effect in BINARY_EFFECTS:
        predicted = (
            power_for_n_binary(
                n_control=N_BINARY, p_control=P_CONTROL, absolute_effect=effect, alpha=ALPHA
            )
            if effect > 0
            else ALPHA
        )
        observed = binary_power(rng, effect, REPS)
        binary_rows.append({"effect": effect, "predicted": predicted, "observed": observed})
        print(
            f"  {effect:>8.3f}{predicted:>12.4f}{observed.rate:>12.4f}"
            f"{observed.mc_se:>9.4f}{observed.sigma_from(predicted):>8.2f}"
        )

    banner("CONTINUOUS - empirical power vs analytic prediction")
    print(
        f"  true revenue mean {mean_rev:.4f}, true control variance {var_rev:.2f} "
        f"(closed form, zero-inflated lognormal)"
    )
    print("  a multiplicative lift scales the treatment variance by (1+lift)^2, so the")
    print("  equal-variance formula and the heteroscedasticity-aware one diverge.")
    continuous_rows = []
    print(
        f"\n  {'lift':>7}{'abs effect':>12}{'equal-var':>11}{'het-aware':>11}"
        f"{'empirical':>11}{'MC SE':>8}{'sigma':>7}"
    )
    for lift in REVENUE_LIFTS:
        absolute = mean_rev * lift
        variance_treatment = var_rev * (1.0 + lift) ** 2
        if lift > 0:
            equal_var = power_for_n_continuous(
                n_control=N_CONTINUOUS,
                variance=var_rev,
                absolute_effect=absolute,
                alpha=ALPHA,
            )
            het_aware = power_for_n_continuous(
                n_control=N_CONTINUOUS,
                variance=(var_rev + variance_treatment) / 2.0,
                absolute_effect=absolute,
                alpha=ALPHA,
            )
        else:
            equal_var = het_aware = ALPHA
        observed = continuous_power(rng, lift, REPS)
        continuous_rows.append(
            {
                "lift": lift,
                "absolute_effect": absolute,
                "predicted_equal_variance": equal_var,
                "predicted": het_aware,
                "observed": observed,
            }
        )
        print(
            f"  {lift:>7.2f}{absolute:>12.4f}{equal_var:>11.4f}{het_aware:>11.4f}"
            f"{observed.rate:>11.4f}{observed.mc_se:>8.4f}"
            f"{observed.sigma_from(het_aware):>7.2f}"
        )

    worst = max(continuous_rows, key=lambda r: r["predicted_equal_variance"] - r["predicted"])
    print(
        f"\n  At a {worst['lift']:.0%} lift the equal-variance formula promises "
        f"{worst['predicted_equal_variance']:.1%} power and the\n"
        f"  experiment delivers {worst['observed'].rate:.1%}. The formula is not wrong; the "
        f"assumption\n  that both arms share one variance is. Pass the averaged variance "
        f"and it agrees."
    )

    binary_bias = float(
        np.mean([r["observed"].rate - r["predicted"] for r in binary_rows if r["effect"] > 0])
    )
    print(
        f"\n  Binary note: empirical power runs {binary_bias:+.4f} above the design "
        f"prediction on\n  average. That is the D9 trade showing up where it was predicted "
        f"to -- the design\n  formula sizes for a pooled-variance test while the analysis "
        f"uses the unpooled\n  Wald test, which is marginally more powerful. Every point "
        f"stays inside 3 MC SE."
    )

    all_rows = binary_rows + continuous_rows
    off = [r for r in all_rows if not r["observed"].within(r["predicted"], 3.0)]
    savefig(_chart(binary_rows, continuous_rows), "study_power")
    write_results(
        "study_power",
        {
            "alpha": ALPHA,
            "seed": SEED,
            "replications": REPS,
            "binary": {
                "n_per_arm": N_BINARY,
                "p_control": P_CONTROL,
                "points": [{**r, "observed": r["observed"].__dict__} for r in binary_rows],
            },
            "continuous": {
                "n_per_arm": N_CONTINUOUS,
                "true_mean": mean_rev,
                "true_control_variance": var_rev,
                "points": [{**r, "observed": r["observed"].__dict__} for r in continuous_rows],
            },
            "binary_mean_bias_vs_prediction": binary_bias,
            "all_within_3_mc_se": not off,
        },
    )
    if off:
        print(f"\n  FAIL: {len(off)} point(s) more than 3 Monte Carlo SE from prediction.")
        return 1
    print(
        "\n  PASS: the design module's predictions hold on data that violates its\n"
        "  normality assumption, at every effect size tested."
    )
    return 0


def _chart(binary_rows, continuous_rows):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, rows, xkey, xlabel, title in (
        (
            axes[0],
            binary_rows,
            "effect",
            "absolute effect on a 10% conversion rate",
            f"Binary, n={N_BINARY:,}/arm",
        ),
        (
            axes[1],
            continuous_rows,
            "lift",
            "relative lift on revenue per user",
            f"Continuous, n={N_CONTINUOUS:,}/arm",
        ),
    ):
        x = [r[xkey] for r in rows]
        if "predicted_equal_variance" in rows[0]:
            ax.plot(
                x,
                [r["predicted_equal_variance"] for r in rows],
                "--",
                color="#c53030",
                linewidth=1.8,
                label="prediction assuming equal variance",
            )
        ax.plot(
            x,
            [r["predicted"] for r in rows],
            "-",
            color="#2f855a",
            linewidth=2,
            label="analytic prediction",
        )
        ax.errorbar(
            x,
            [r["observed"].rate for r in rows],
            yerr=[1.96 * r["observed"].mc_se for r in rows],
            fmt="o",
            color="#2b6cb0",
            capsize=4,
            markersize=6,
            label="empirical power",
        )
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("power")
    axes[0].legend(frameon=False, loc="lower right")
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle("Design module predictions vs simulated reality", y=1.02)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    sys.exit(main())
