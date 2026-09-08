"""Orchestration: validity checks, per-metric inference, and a decision.

One entry point, :func:`readout`, one structured result. The order is not
negotiable and the first step can end the whole thing:

1. **SRM.** If the realized allocation contradicts the intended one, raise
   :class:`~expkit.validity.srm.SRMFailure` and return nothing. Not a warning
   beside an effect estimate -- no effect estimate at all (D1).
2. **Balance.** Pre-period covariates are compared and reported. Never a stop.
3. **Estimation.** Each metric by its type; CUPED applied where a pre-period
   covariate was declared.
4. **Multiplicity.** Benjamini-Hochberg across the secondary family only (D3).
5. **Recommendation.** Explicit rules, stated below.

The decision logic (D4), in full, with no cases beyond these:

* A significant guardrail regression **blocks the ship** regardless of what the
  primary did, and marks the readout as requiring a human override. Guardrails are
  constraints, not terms in an objective function; trading one against the primary
  needs an exchange rate this library has no standing to invent.
* Otherwise a significant primary improvement recommends **ship**.
* A significant primary regression recommends **do not ship**.
* Anything else is **inconclusive**: the experiment did not resolve the question,
  which is a different statement from "there is no effect".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from expkit.inference.fixed import TestResult, two_proportion_test, welch_test
from expkit.inference.multiple import benjamini_hochberg
from expkit.metrics.base import MetricRole, MetricSpec
from expkit.metrics.cuped import CupedResult, cuped_test
from expkit.metrics.ratio import ratio_test
from expkit.validity.checks import BalanceResult, check_balance
from expkit.validity.srm import SRMResult, check_srm

__all__ = [
    "Decision",
    "BinaryObservation",
    "ContinuousObservation",
    "RatioObservation",
    "MetricReadout",
    "Recommendation",
    "Readout",
    "readout",
]


class Decision(StrEnum):
    SHIP = "ship"
    DO_NOT_SHIP = "do not ship"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class BinaryObservation:
    spec: MetricSpec
    successes_control: int
    n_control: int
    successes_treatment: int
    n_treatment: int


@dataclass(frozen=True)
class ContinuousObservation:
    spec: MetricSpec
    control: np.ndarray
    treatment: np.ndarray
    covariate_control: np.ndarray | None = None
    covariate_treatment: np.ndarray | None = None


@dataclass(frozen=True)
class RatioObservation:
    spec: MetricSpec
    numerator_control: np.ndarray
    denominator_control: np.ndarray
    numerator_treatment: np.ndarray
    denominator_treatment: np.ndarray


Observation = BinaryObservation | ContinuousObservation | RatioObservation


@dataclass(frozen=True)
class MetricReadout:
    """One metric's full result, including how multiplicity was handled."""

    spec: MetricSpec
    test: TestResult
    adjusted_p_value: float
    significant: bool
    correction: str
    cuped: CupedResult | None = None

    @property
    def variance_reduction(self) -> float:
        return self.cuped.realized_variance_reduction if self.cuped else 0.0

    @property
    def is_regression(self) -> bool:
        """Significant *and* moving in the harmful direction."""
        return self.significant and self.spec.is_regression(self.test.estimate)

    @property
    def is_improvement(self) -> bool:
        return self.significant and not self.spec.is_regression(self.test.estimate)


@dataclass(frozen=True)
class Recommendation:
    decision: Decision
    summary: str
    reasons: tuple[str, ...]
    requires_human_override: bool


@dataclass(frozen=True)
class Readout:
    """Everything the experiment produced, and what to do about it."""

    srm: SRMResult
    balance: tuple[BalanceResult, ...]
    metrics: tuple[MetricReadout, ...]
    n_control: int
    n_treatment: int
    realized_allocation: dict[str, float]
    intended_allocation: dict[str, float]
    alpha: float
    duration_days: float | None
    recommendation: Recommendation
    warnings: tuple[str, ...] = field(default=())

    def by_role(self, role: MetricRole) -> tuple[MetricReadout, ...]:
        return tuple(m for m in self.metrics if m.spec.role is role)

    @property
    def primary(self) -> MetricReadout:
        return self.by_role(MetricRole.PRIMARY)[0]


def _estimate(obs: Observation, alpha: float) -> tuple[TestResult, CupedResult | None]:
    if isinstance(obs, BinaryObservation):
        return (
            two_proportion_test(
                successes_control=obs.successes_control,
                n_control=obs.n_control,
                successes_treatment=obs.successes_treatment,
                n_treatment=obs.n_treatment,
                alpha=alpha,
            ),
            None,
        )

    if isinstance(obs, RatioObservation):
        return (
            ratio_test(
                numerator_control=obs.numerator_control,
                denominator_control=obs.denominator_control,
                numerator_treatment=obs.numerator_treatment,
                denominator_treatment=obs.denominator_treatment,
                alpha=alpha,
            ),
            None,
        )

    if isinstance(obs, ContinuousObservation):
        has_covariate = obs.covariate_control is not None and obs.covariate_treatment is not None
        if obs.spec.cuped_covariate and not has_covariate:
            raise ValueError(
                f"metric '{obs.spec.name}' declares the CUPED covariate "
                f"'{obs.spec.cuped_covariate}' but no covariate arrays were supplied"
            )
        if has_covariate:
            # A CovariateContamination raised here is deliberately not caught:
            # adjusting on a contaminated covariate biases the estimate toward zero
            # while tightening the interval, so silently falling back to the
            # unadjusted test would hide that the declared covariate is invalid.
            return cuped_test(
                y_control=obs.control,
                x_control=obs.covariate_control,
                y_treatment=obs.treatment,
                x_treatment=obs.covariate_treatment,
                alpha=alpha,
            )
        return welch_test(control=obs.control, treatment=obs.treatment, alpha=alpha), None

    raise TypeError(f"unsupported observation type: {type(obs).__name__}")


def _recommend(metrics: tuple[MetricReadout, ...]) -> Recommendation:
    """Apply D4. No cases beyond the four documented in the module docstring."""
    primary = next(m for m in metrics if m.spec.role is MetricRole.PRIMARY)
    guardrail_regressions = [
        m for m in metrics if m.spec.role is MetricRole.GUARDRAIL and m.is_regression
    ]

    if guardrail_regressions:
        reasons = [
            f"guardrail '{m.spec.name}' regressed by {m.test.estimate:+.5f} "
            f"[{m.test.ci_lower:+.5f}, {m.test.ci_upper:+.5f}], p={m.test.p_value:.4f}"
            for m in guardrail_regressions
        ]
        if primary.is_improvement:
            reasons.insert(
                0,
                f"primary '{primary.spec.name}' improved by {primary.test.estimate:+.5f} "
                f"[{primary.test.ci_lower:+.5f}, {primary.test.ci_upper:+.5f}], "
                f"p={primary.test.p_value:.4f}",
            )
            summary = (
                f"Do not ship without an explicit override. The primary metric improved "
                f"significantly, but {len(guardrail_regressions)} guardrail(s) regressed "
                f"significantly. Guardrails are constraints, not quantities to trade "
                f"against the primary; this readout will not price that exchange for you."
            )
        else:
            summary = (
                f"Do not ship. {len(guardrail_regressions)} guardrail(s) regressed "
                f"significantly."
            )
        return Recommendation(Decision.DO_NOT_SHIP, summary, tuple(reasons), True)

    if primary.is_improvement:
        return Recommendation(
            Decision.SHIP,
            f"Ship. The primary metric '{primary.spec.name}' improved significantly and "
            f"no guardrail regressed.",
            (
                f"primary '{primary.spec.name}': {primary.test.estimate:+.5f} "
                f"[{primary.test.ci_lower:+.5f}, {primary.test.ci_upper:+.5f}], "
                f"p={primary.test.p_value:.4f}",
            ),
            False,
        )

    if primary.is_regression:
        return Recommendation(
            Decision.DO_NOT_SHIP,
            f"Do not ship. The primary metric '{primary.spec.name}' regressed " f"significantly.",
            (
                f"primary '{primary.spec.name}': {primary.test.estimate:+.5f} "
                f"[{primary.test.ci_lower:+.5f}, {primary.test.ci_upper:+.5f}], "
                f"p={primary.test.p_value:.4f}",
            ),
            False,
        )

    return Recommendation(
        Decision.INCONCLUSIVE,
        f"Inconclusive. The primary metric '{primary.spec.name}' did not move "
        f"detectably. This is not evidence of no effect -- the interval "
        f"[{primary.test.ci_lower:+.5f}, {primary.test.ci_upper:+.5f}] shows what the "
        f"experiment could and could not rule out.",
        (
            f"primary '{primary.spec.name}': {primary.test.estimate:+.5f} "
            f"[{primary.test.ci_lower:+.5f}, {primary.test.ci_upper:+.5f}], "
            f"p={primary.test.p_value:.4f}",
        ),
        False,
    )


def readout(
    *,
    observations: list[Observation],
    intended_allocation: dict[str, float],
    observed_counts: dict[str, int],
    alpha: float,
    balance_covariates: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    duration_days: float | None = None,
) -> Readout:
    """Run an experiment readout end to end.

    Raises
    ------
    SRMFailure
        If the realized allocation contradicts the intended one. No effect
        estimates are computed and none are returned.
    ValueError
        If the metric set does not contain exactly one primary metric.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not observations:
        raise ValueError("at least one metric observation is required")

    primaries = [o for o in observations if o.spec.role is MetricRole.PRIMARY]
    if len(primaries) != 1:
        raise ValueError(
            f"exactly one primary metric is required, got {len(primaries)}. "
            "A readout answers one pre-registered question; more than one primary "
            "reintroduces the multiplicity that declaring a primary was meant to remove."
        )

    # Step 1: SRM. Raises before anything is estimated.
    srm = check_srm(observed_counts, intended_allocation)

    # Step 2: balance. Reported, never a stop.
    balance = tuple(
        check_balance(name, control, treatment)
        for name, (control, treatment) in (balance_covariates or {}).items()
    )

    # Step 3: estimation.
    estimated = [(obs.spec, *_estimate(obs, alpha)) for obs in observations]

    # Step 4: multiplicity, over the secondary family only (D3).
    secondary_idx = [
        i for i, (spec, _, _) in enumerate(estimated) if spec.role is MetricRole.SECONDARY
    ]
    bh = benjamini_hochberg([estimated[i][1].p_value for i in secondary_idx], alpha)
    bh_by_index = {
        idx: (bh.adjusted_p_values[k], bh.rejected[k]) for k, idx in enumerate(secondary_idx)
    }

    metrics = []
    for i, (spec, test, cuped) in enumerate(estimated):
        if i in bh_by_index:
            adjusted_p, significant = bh_by_index[i]
            correction = f"Benjamini-Hochberg over {bh.n_tests} secondary metrics"
        else:
            adjusted_p, significant = test.p_value, test.significant
            correction = (
                "none (single pre-registered hypothesis)"
                if spec.role is MetricRole.PRIMARY
                else "none (guardrail: correction would cost power to detect harm)"
            )
        metrics.append(
            MetricReadout(
                spec=spec,
                test=test,
                adjusted_p_value=adjusted_p,
                significant=significant,
                correction=correction,
                cuped=cuped,
            )
        )
    metrics = tuple(metrics)

    total = sum(observed_counts.values())
    warnings = tuple(b.detail for b in balance if not b.balanced)

    arms = sorted(observed_counts)
    control_arm = arms[0] if "control" not in observed_counts else "control"
    treatment_arms = [a for a in arms if a != control_arm]
    return Readout(
        srm=srm,
        balance=balance,
        metrics=metrics,
        n_control=observed_counts[control_arm],
        n_treatment=sum(observed_counts[a] for a in treatment_arms),
        realized_allocation={a: observed_counts[a] / total for a in arms},
        intended_allocation=dict(intended_allocation),
        alpha=alpha,
        duration_days=duration_days,
        recommendation=_recommend(metrics),
        warnings=warnings,
    )
