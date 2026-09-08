"""End-to-end readout: the full pipeline on one simulated experiment.

    make setup && .venv/bin/python examples/readout_walkthrough.py

Four scenarios, same code path. The last one is refused.
"""

import numpy as np

from expkit.metrics.base import MetricRole, MetricSpec, MetricType
from expkit.readout import (
    BinaryObservation,
    ContinuousObservation,
    RatioObservation,
    readout,
)
from expkit.validity.srm import SRMFailure

N = 60_000
ALPHA = 0.05


def build(rng, *, primary_lift, latency_shift, srm_leak=0):
    n_t = N + srm_leak
    x_c, x_t = rng.normal(50.0, 20.0, N), rng.normal(50.0, 20.0, n_t)

    def ratio_arm(n, lift):
        p = np.clip(rng.beta(2.0, 8.0, n) * (1 + lift), 0, 1)
        d = 1 + rng.poisson(3.0, n)
        return rng.binomial(d, p).astype(float), d.astype(float)

    num_c, den_c = ratio_arm(N, 0.0)
    num_t, den_t = ratio_arm(n_t, 0.03)

    return [
        BinaryObservation(
            MetricSpec("checkout_conversion", MetricType.BINARY, MetricRole.PRIMARY),
            successes_control=int(rng.binomial(N, 0.10)),
            n_control=N,
            successes_treatment=int(rng.binomial(n_t, 0.10 + primary_lift)),
            n_treatment=n_t,
        ),
        ContinuousObservation(
            MetricSpec(
                "page_load_ms",
                MetricType.CONTINUOUS,
                MetricRole.GUARDRAIL,
                higher_is_better=False,
            ),
            control=rng.normal(800.0, 120.0, N),
            treatment=rng.normal(800.0 + latency_shift, 120.0, n_t),
        ),
        ContinuousObservation(
            MetricSpec(
                "revenue_per_user",
                MetricType.CONTINUOUS,
                MetricRole.SECONDARY,
                cuped_covariate="pre_period_revenue",
            ),
            control=0.7 * x_c + rng.normal(0, 14.0, N),
            treatment=0.7 * x_t + rng.normal(0, 14.0, n_t) + 0.6,
            covariate_control=x_c,
            covariate_treatment=x_t,
        ),
        RatioObservation(
            MetricSpec("clicks_per_session", MetricType.RATIO, MetricRole.SECONDARY),
            numerator_control=num_c,
            denominator_control=den_c,
            numerator_treatment=num_t,
            denominator_treatment=den_t,
        ),
    ]


SCENARIOS = [
    ("Clean win", dict(primary_lift=0.006, latency_shift=0.0)),
    ("Primary up, guardrail down", dict(primary_lift=0.006, latency_shift=6.0)),
    ("Inconclusive", dict(primary_lift=0.0004, latency_shift=0.0)),
    ("Sample ratio mismatch", dict(primary_lift=0.006, latency_shift=0.0, srm_leak=1200)),
]

for title, params in SCENARIOS:
    print(f"\n{'=' * 78}\n {title}\n{'=' * 78}")
    rng = np.random.default_rng(2024)
    observations = build(rng, **params)
    counts = {"control": N, "treatment": N + params.get("srm_leak", 0)}
    try:
        result = readout(
            observations=observations,
            intended_allocation={"control": 0.5, "treatment": 0.5},
            observed_counts=counts,
            alpha=ALPHA,
            duration_days=14,
        )
    except SRMFailure as failure:
        print(f"\n  REFUSED — {failure.result.detail}")
        print("\n  No treatment effect is returned. Not a formatting choice: broken")
        print("  randomization means the arms are not exchangeable, so the comparison")
        print("  is not causal and the estimate would be meaningless at any p-value.")
        continue

    print(f"\n  DECISION: {result.recommendation.decision.value.upper()}")
    print(f"  {result.recommendation.summary}\n")
    header = (
        f"  {'role':<10}{'metric':<21}{'estimate':>11}{'95% CI':>26}"
        f"{'p':>9}{'adj p':>9}{'sig':>6}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in result.metrics:
        ci = f"[{m.test.ci_lower:+.4f}, {m.test.ci_upper:+.4f}]"
        print(
            f"  {m.spec.role.value:<10}{m.spec.name:<21}{m.test.estimate:>+11.4f}"
            f"{ci:>26}{m.test.p_value:>9.4f}{m.adjusted_p_value:>9.4f}"
            f"{str(m.significant):>6}"
        )
    cuped = [m for m in result.metrics if m.cuped]
    for m in cuped:
        print(
            f"\n  CUPED on '{m.spec.name}': rho={m.cuped.rho:+.3f}, "
            f"variance reduction {m.variance_reduction:.1%} "
            f"(SE {m.cuped.unadjusted_standard_error:.4f} -> {m.cuped.standard_error:.4f})"
        )
    print(
        f"\n  SRM p={result.srm.p_value:.4f} (threshold {result.srm.threshold:g}) · "
        f"n={result.n_control:,}/{result.n_treatment:,} · "
        f"{result.duration_days:.0f} days · alpha={result.alpha}"
    )
    for reason in result.recommendation.reasons:
        print(f"    - {reason}")
    if result.recommendation.requires_human_override:
        print("\n  >> Requires an explicit human override to proceed.")
print()
