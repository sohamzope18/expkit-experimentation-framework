"""Ratio metrics and the unit-of-analysis problem.

A ratio metric is ``R = sum(N) / sum(D)`` -- clicks per session, items per order --
where numerator and denominator are *both* random. The trap: the experiment
randomizes users, but the metric is naturally computed over sessions. Treating
sessions as independent observations ignores that one user's sessions are
correlated with each other, which understates the variance, which makes confidence
intervals too narrow, which makes every result look more certain than it is.

The fix is to linearize at the randomization unit. Writing ``R = Nbar / Dbar`` and
taking a first-order Taylor expansion about ``(E[N], E[D])``:

    Var(R) ~= (1 / (n * Dbar^2)) * ( Var(N) - 2*R*Cov(N, D) + R^2 * Var(D) )

which is exactly the variance of the per-unit linearized value

    L_i = (N_i - R * D_i) / Dbar

divided by ``n``. This module computes the second form: it is algebraically
identical, numerically better behaved, and it hands back ``Var(L)`` -- the per-unit
variance the design module needs for sizing a ratio metric.

``sims/study_coverage.py`` measures both estimators against known ground truth and
publishes how far below nominal the naive interval's coverage actually falls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from expkit.inference.fixed import TestResult, z_test

__all__ = ["RatioEstimate", "ratio_estimate", "naive_ratio_estimate", "ratio_test"]


@dataclass(frozen=True)
class RatioEstimate:
    """A ratio metric with its variance and the method that produced it."""

    ratio: float
    variance: float
    unit_variance: float
    n_units: int
    total_numerator: float
    total_denominator: float
    method: str

    @property
    def standard_error(self) -> float:
        return math.sqrt(self.variance)


def _as_pair(numerator, denominator) -> tuple[np.ndarray, np.ndarray]:
    num = np.asarray(numerator, dtype=float).ravel()
    den = np.asarray(denominator, dtype=float).ravel()
    if num.size != den.size:
        raise ValueError(
            f"numerator and denominator must align per unit, got {num.size} and {den.size}"
        )
    if num.size < 2:
        raise ValueError("at least two randomization units are required")
    if np.any(den < 0):
        raise ValueError("denominator values must be non-negative")
    if den.sum() <= 0:
        raise ValueError("total denominator must be positive")
    return num, den


def ratio_estimate(numerator, denominator) -> RatioEstimate:
    """Delta-method estimate at the randomization unit. **Use this one.**

    ``numerator`` and ``denominator`` hold one value per randomization unit --
    per user, not per session. Aggregating to the randomization unit before this
    call is what makes the variance correct.
    """
    num, den = _as_pair(numerator, denominator)
    n = num.size
    d_bar = den.mean()
    ratio = num.sum() / den.sum()

    # Linearized per-unit value; Var(L)/n is the delta-method variance.
    linearized = (num - ratio * den) / d_bar
    unit_variance = float(linearized.var(ddof=1))
    return RatioEstimate(
        ratio=float(ratio),
        variance=unit_variance / n,
        unit_variance=unit_variance,
        n_units=n,
        total_numerator=float(num.sum()),
        total_denominator=float(den.sum()),
        method="delta method (randomization unit)",
    )


def naive_ratio_estimate(numerator, denominator) -> RatioEstimate:
    """The wrong estimator, implemented so its failure can be measured.

    Treats each *denominator* unit as an independent observation -- the mistake
    made whenever someone computes a rate from session-level rows and uses the
    session count as ``n``. The variance becomes ``R*(1-R) / sum(D)``.

    This is exact rather than approximate for rate metrics, where each denominator
    unit contributes 0 or 1 to the numerator: for binary data the sample variance
    is fully determined by the mean, so no session-level array is needed to
    reproduce what the naive analyst would have computed.

    Provided for comparison only. Never use it for a readout.
    """
    num, den = _as_pair(numerator, denominator)
    if np.any(num > den):
        raise ValueError(
            "naive_ratio_estimate models a rate: each denominator unit must "
            "contribute at most 1 to the numerator"
        )
    total_d = den.sum()
    ratio = num.sum() / total_d
    return RatioEstimate(
        ratio=float(ratio),
        variance=float(ratio * (1.0 - ratio) / total_d),
        unit_variance=float(ratio * (1.0 - ratio)),
        n_units=int(total_d),
        total_numerator=float(num.sum()),
        total_denominator=float(total_d),
        method="naive (denominator unit) -- INVALID, for comparison only",
    )


def ratio_test(
    *,
    numerator_control,
    denominator_control,
    numerator_treatment,
    denominator_treatment,
    alpha: float,
    naive: bool = False,
) -> TestResult:
    """Two-arm comparison of a ratio metric.

    Set ``naive=True`` only to reproduce the broken estimator for the coverage
    study. The variances of two independent arms add either way; what differs is
    whether each arm's variance was computed at the right unit.
    """
    estimator = naive_ratio_estimate if naive else ratio_estimate
    control = estimator(numerator_control, denominator_control)
    treatment = estimator(numerator_treatment, denominator_treatment)
    se = math.sqrt(control.variance + treatment.variance)
    return z_test(
        estimate=treatment.ratio - control.ratio,
        standard_error=se,
        alpha=alpha,
        method=f"ratio, {control.method}",
        n_control=control.n_units,
        n_treatment=treatment.n_units,
        baseline=control.ratio,
        mean_control=control.ratio,
        mean_treatment=treatment.ratio,
    )
