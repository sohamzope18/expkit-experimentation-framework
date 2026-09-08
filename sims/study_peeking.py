"""The headline study: what peeking costs, and what fixing it costs.

Spec §3.6 and §4. Three numbers, in order:

1. **The damage.** Under a true null, an analyst who checks a fixed-horizon test
   at every daily checkpoint and stops at the first significant result. The
   false-positive rate is not 5%.
2. **The fix.** The identical protocol run against an asymptotic confidence
   sequence. The rate returns to nominal.
3. **The bill.** Always-valid inference is not free. For the same power, the
   sequential method needs more units than a fixed-horizon test run to
   completion, and this study reports the multiple rather than gesturing at it.

Nothing about the analyst's behaviour changes between (1) and (2). Only the
interval does.
"""

from __future__ import annotations

import sys

import numpy as np

from expkit.design.power import required_n_binary
from expkit.inference.fixed import two_proportion_test
from expkit.inference.sequential import asympcs, rho_for_target
from sims._report import banner, rate, savefig, write_results

SEED = 20240605
ALPHA = 0.05
REPS = 5_000
LOOKS = 20  # e.g. a daily check across a three-week experiment

P_CONTROL = 0.10
TRUE_EFFECT = 0.01  # a 10% relative lift, the effect the design is sized for
HORIZON_MULTIPLE = 4.0


def simulate(rng, effect, n_per_look, looks, reps, rho):
    """Accumulate both arms across looks; evaluate both procedures at every look.

    Returns, per look: the cumulative fraction of experiments a peeking analyst
    would have stopped on, and the cumulative fraction whose confidence sequence
    has crossed. Also the per-look fixed-horizon rejection rate, which is what a
    disciplined analyst running to exactly that size would see.
    """
    inc_c = rng.binomial(n_per_look, P_CONTROL, size=(reps, looks))
    inc_t = rng.binomial(n_per_look, P_CONTROL + effect, size=(reps, looks))
    cum_c, cum_t = np.cumsum(inc_c, axis=1), np.cumsum(inc_t, axis=1)

    peek_stopped = np.zeros(reps, dtype=bool)
    seq_crossed = np.zeros(reps, dtype=bool)
    peek_curve, seq_curve, fixed_curve = [], [], []
    peek_stop_n = np.full(reps, np.nan)
    seq_stop_n = np.full(reps, np.nan)

    for look in range(looks):
        n = n_per_look * (look + 1)
        fixed_hits = 0
        for i in range(reps):
            result = two_proportion_test(
                successes_control=int(cum_c[i, look]),
                n_control=n,
                successes_treatment=int(cum_t[i, look]),
                n_treatment=n,
                alpha=ALPHA,
            )
            fixed_hits += result.significant
            if not peek_stopped[i] and result.significant:
                peek_stopped[i] = True
                peek_stop_n[i] = n
            if not seq_crossed[i]:
                cs = asympcs(
                    estimate=result.estimate,
                    standard_error=result.standard_error,
                    n_total=2 * n,
                    alpha=ALPHA,
                    rho=rho,
                )
                if cs.excludes_null:
                    seq_crossed[i] = True
                    seq_stop_n[i] = n
        peek_curve.append(peek_stopped.sum())
        seq_curve.append(seq_crossed.sum())
        fixed_curve.append(fixed_hits)

    return {
        "n_grid": [n_per_look * (i + 1) for i in range(looks)],
        "peek_cumulative": peek_curve,
        "sequential_cumulative": seq_curve,
        "fixed_per_look": fixed_curve,
        "peek_stop_n": peek_stop_n,
        "sequential_stop_n": seq_stop_n,
        "reps": reps,
    }


def _n_at_power(n_grid, counts, reps, target=0.80):
    """Linear interpolation of the sample size where a curve reaches ``target``."""
    powers = [c / reps for c in counts]
    for i in range(1, len(powers)):
        if powers[i] >= target > powers[i - 1]:
            span = powers[i] - powers[i - 1]
            frac = (target - powers[i - 1]) / span if span else 0.0
            return n_grid[i - 1] + frac * (n_grid[i] - n_grid[i - 1])
    return None


def main() -> int:
    rng = np.random.default_rng(SEED)

    design = required_n_binary(
        p_control=P_CONTROL, absolute_effect=TRUE_EFFECT, alpha=ALPHA, power=0.80
    )
    n_fixed = design.n_control
    n_per_look = int(round(n_fixed * HORIZON_MULTIPLE / LOOKS))
    horizon = n_per_look * LOOKS
    rho = rho_for_target(target_n=2 * n_fixed, alpha=ALPHA)

    banner("DESIGN")
    print(
        f"\n  fixed-horizon design for a {TRUE_EFFECT:+.3f} effect at 80% power: "
        f"{n_fixed:,} per arm"
    )
    print(f"  simulated horizon: {horizon:,} per arm ({HORIZON_MULTIPLE:.0f}x the design)")
    print(f"  looks: {LOOKS} evenly spaced, every {n_per_look:,} units per arm")
    print(f"  rho tuned at the planned total sample {2 * n_fixed:,}: {rho:.6f}")

    banner("1. THE DAMAGE - peeking at a fixed-horizon test under a TRUE NULL")
    null = simulate(rng, 0.0, n_per_look, LOOKS, REPS, rho)
    peek_fpr = rate("peeking, fixed-horizon test", null["peek_cumulative"][-1], REPS)
    seq_fpr = rate("peeking, confidence sequence", null["sequential_cumulative"][-1], REPS)
    single_fpr = rate("single look at the horizon", null["fixed_per_look"][-1], REPS)

    print(f"\n  {'procedure':<38}{'false-positive rate':>21}{'MC SE':>9}")
    for r in (single_fpr, peek_fpr, seq_fpr):
        print(f"  {r.label:<38}{r.rate:>21.4f}{r.mc_se:>9.4f}")
    print(
        f"\n  Peeking {LOOKS} times inflates the false-positive rate from "
        f"{single_fpr.rate:.1%} to {peek_fpr.rate:.1%}\n"
        f"  -- {peek_fpr.rate / ALPHA:.1f}x nominal. Roughly "
        f"{peek_fpr.rate:.0%} of 'wins' found this way are noise."
    )

    banner("2. THE FIX - the identical protocol, against a confidence sequence")
    print(
        f"\n  false-positive rate {seq_fpr.rate:.4f} +/- {seq_fpr.mc_se:.4f} "
        f"against a nominal {ALPHA}"
    )
    print(
        f"\n  The analyst still looks {LOOKS} times and still stops at the first "
        f"crossing. The\n  guarantee holds because the interval was built to be valid "
        f"at every stopping\n  time, not because anyone behaved more carefully."
    )
    print(
        f"\n  Note the direction: {seq_fpr.rate:.1%} is *below* {ALPHA:.0%}, not at it, "
        f"and that is expected\n  rather than a miscalibration. A confidence sequence "
        f"guarantees\n  P(ever cross | null) <= alpha over an unbounded horizon and "
        f"every possible\n  stopping rule. This simulation spends only {LOOKS} looks "
        f"inside {HORIZON_MULTIPLE:.0f}x the design\n  size, so most of the protection "
        f"purchased is never used, and the unused\n  protection shows up as "
        f"conservatism. The guarantee is one-sided, so being\n  under nominal is "
        f"compliance; being over it would be the failure."
    )

    banner("3. THE BILL - sample size for equal power")
    alt = simulate(rng, TRUE_EFFECT, n_per_look, LOOKS, REPS, rho)
    n_fixed_80 = _n_at_power(alt["n_grid"], alt["fixed_per_look"], REPS)
    n_seq_80 = _n_at_power(alt["n_grid"], alt["sequential_cumulative"], REPS)

    print(
        f"\n  fixed-horizon test, run to completion, 80% power at   " f"{n_fixed_80:,.0f} per arm"
    )
    print(f"  confidence sequence, stopping any time, 80% power at   " f"{n_seq_80:,.0f} per arm")
    cost = n_seq_80 / n_fixed_80
    print(f"\n  cost of always-valid inference: {cost:.2f}x the sample size")
    print(
        f"\n  A single-look comparison would predict worse. The sequence is 1.55x wider\n"
        f"  than a fixed interval at its tuned size, and power scales with the square\n"
        f"  of width, so detecting at one pre-chosen moment would cost about "
        f"{1.549 ** 2:.2f}x.\n  The realized {cost:.2f}x is better than that because the "
        f"sequence gets {LOOKS} chances\n  to cross rather than one: repeated looks are "
        f"the cost under the null and a\n  partial refund under the alternative."
    )

    stopped = alt["sequential_stop_n"][~np.isnan(alt["sequential_stop_n"])]
    print(
        f"\n  Against that, the sequence stopped early when it could: of the "
        f"{len(stopped):,} runs\n  that ever crossed, the median crossing came at "
        f"{np.median(stopped):,.0f} units per arm, "
        f"{np.median(stopped) / n_fixed:.2f}x\n  the fixed design. You pay for the "
        f"right to stop early in worst-case sample\n  size and get some of it back in "
        f"the average case."
    )

    savefig(_chart(null, alt, n_fixed, REPS), "study_peeking")
    seq_ok = seq_fpr.rate <= ALPHA + 3 * seq_fpr.se_under(ALPHA)
    peek_inflated = peek_fpr.rate > 2 * ALPHA
    write_results(
        "study_peeking",
        {
            "alpha": ALPHA,
            "seed": SEED,
            "replications": REPS,
            "looks": LOOKS,
            "p_control": P_CONTROL,
            "true_effect": TRUE_EFFECT,
            "n_fixed_design": n_fixed,
            "horizon_per_arm": horizon,
            "rho": rho,
            "null": {
                "single_look": single_fpr.__dict__,
                "peeking_fixed_horizon": peek_fpr.__dict__,
                "peeking_sequential": seq_fpr.__dict__,
                "n_grid": null["n_grid"],
                "peek_cumulative": null["peek_cumulative"],
                "sequential_cumulative": null["sequential_cumulative"],
            },
            "alternative": {
                "n_grid": alt["n_grid"],
                "fixed_per_look": alt["fixed_per_look"],
                "sequential_cumulative": alt["sequential_cumulative"],
                "n_for_80pct_fixed": n_fixed_80,
                "n_for_80pct_sequential": n_seq_80,
                "sample_size_cost_multiple": cost,
                "median_crossing_n": float(np.median(stopped)),
            },
            "sequential_controls_type1": seq_ok,
            "peeking_inflates_type1": peek_inflated,
        },
    )
    if not (seq_ok and peek_inflated):
        print(
            "\n  FAIL: the sequential method did not control Type I, or the naive\n"
            "  procedure did not inflate it. Either way the study is not showing\n"
            "  what it claims."
        )
        return 1
    print(
        "\n  PASS: peeking inflates the false-positive rate, the confidence sequence\n"
        "  restores it, and the sample-size cost is quantified above."
    )
    return 0


def _chart(null, alt, n_fixed, reps):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))

    ax = axes[0]
    x = null["n_grid"]
    ax.plot(
        x,
        [c / reps for c in null["peek_cumulative"]],
        "-o",
        color="#c53030",
        markersize=4,
        linewidth=2,
        label="peeking at a fixed-horizon test",
    )
    ax.plot(
        x,
        [c / reps for c in null["sequential_cumulative"]],
        "-s",
        color="#2b6cb0",
        markersize=4,
        linewidth=2,
        label="peeking at a confidence sequence",
    )
    ax.axhline(ALPHA, color="#2f855a", linestyle="--", linewidth=1.4, label="nominal 5%")
    ax.set_xlabel("units per arm accumulated")
    ax.set_ylabel("cumulative false-positive rate")
    ax.set_title("Under a true null: what looking costs")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    x = alt["n_grid"]
    ax.plot(
        x,
        [c / reps for c in alt["fixed_per_look"]],
        "-o",
        color="#2f855a",
        markersize=4,
        linewidth=2,
        label="fixed horizon, run to completion",
    )
    ax.plot(
        x,
        [c / reps for c in alt["sequential_cumulative"]],
        "-s",
        color="#2b6cb0",
        markersize=4,
        linewidth=2,
        label="confidence sequence, stop any time",
    )
    ax.axhline(0.80, color="#718096", linestyle=":", linewidth=1.4, label="80% power")
    ax.axvline(n_fixed, color="#a0aec0", linestyle=":", linewidth=1.2)
    ax.annotate(
        "design n", xy=(n_fixed, 0.05), xytext=(n_fixed * 1.05, 0.05), fontsize=9, color="#4a5568"
    )
    ax.set_xlabel("units per arm accumulated")
    ax.set_ylabel("probability of detecting the true effect")
    ax.set_title("Under a real effect: what validity costs")
    ax.legend(frameon=False, loc="lower right")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    sys.exit(main())
