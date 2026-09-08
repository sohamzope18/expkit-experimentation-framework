"""CUPED: variance reduction using a strictly pre-experiment covariate.

Given a covariate ``X`` measured *before* the experiment started, adjust the
outcome:

    theta = Cov(Y, X) / Var(X)          (estimated pooled across arms)
    Y_adj = Y - theta * (X - Xbar)

Because ``X`` is independent of assignment, subtracting a multiple of it removes
variance from ``Y`` without moving the expected difference between arms. The
theoretical reduction is ``1 - rho^2`` where ``rho = corr(Y, X)``: a covariate
correlated 0.7 with the outcome removes about half the variance, which is worth
roughly a halving of required sample size.

**theta is pooled across arms on purpose.** Estimating it per arm lets the
treatment effect leak into the adjustment. Pooling is safe because assignment is
independent of ``X``, so the between-arm component of ``Y`` contributes no
covariance with ``X`` in expectation.

**Why a post-treatment covariate is fatal.** If the treatment changed ``X``, then
``X`` carries part of the effect, and ``theta * (X - Xbar)`` subtracts part of the
effect from ``Y``. The point estimate is biased toward zero and the variance still
goes down, so the readout looks *better* -- tighter interval, smaller effect --
while being wrong. :func:`cuped_adjust` guards against the version of this that
shows up as an arm imbalance in ``X``; see the caveat on that guard below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from expkit.inference.fixed import TestResult, z_test

__all__ = [
    "CupedResult",
    "CovariateContamination",
    "cuped_adjust",
    "cuped_test",
    "COVARIATE_GUARD_THRESHOLD",
]

#: Arm-balance p-value below which a covariate is rejected as contaminated (D11).
#: Matches the SRM threshold (D1) for the same reason: the check runs on every
#: CUPED-adjusted metric, so a loose threshold produces constant false alarms.
COVARIATE_GUARD_THRESHOLD = 0.001


class CovariateContamination(ValueError):
    """Raised when the covariate differs across arms, implying it is not pre-treatment."""


@dataclass(frozen=True)
class CupedResult:
    """A CUPED-adjusted comparison and the variance it actually removed."""

    theta: float
    rho: float
    estimate: float
    standard_error: float
    unadjusted_estimate: float
    unadjusted_standard_error: float
    realized_variance_reduction: float
    predicted_variance_reduction: float
    n_control: int
    n_treatment: int

    def __str__(self) -> str:
        return (
            f"CUPED theta={self.theta:.4f} rho={self.rho:+.3f}  "
            f"variance reduction {self.realized_variance_reduction:.1%} "
            f"(theory {self.predicted_variance_reduction:.1%})"
        )


def _guard_covariate(x_control: np.ndarray, x_treatment: np.ndarray, threshold: float) -> None:
    """Reject a covariate that differs across arms.

    A genuinely pre-treatment covariate cannot be affected by assignment, so a
    significant imbalance means either the randomization is broken or the
    covariate was measured after treatment began.

    This guard is necessary but not sufficient, and saying so matters: a
    post-treatment covariate that the treatment happens not to move will pass it.
    Nothing in the data can prove a covariate is pre-treatment -- only the
    definition of the measurement window can. The guard catches the contamination
    that shows.
    """
    from scipy import stats

    p_value = float(stats.ttest_ind(x_treatment, x_control, equal_var=False).pvalue)
    if p_value < threshold:
        raise CovariateContamination(
            f"covariate differs across arms (p={p_value:.3e} < {threshold:g}): "
            f"control mean {x_control.mean():.5f}, treatment mean {x_treatment.mean():.5f}. "
            "A pre-treatment covariate cannot be affected by assignment, so this is "
            "either a post-treatment covariate or broken randomization. Adjusting on it "
            "would bias the effect toward zero while tightening the interval."
        )


def cuped_adjust(
    *,
    y_control,
    x_control,
    y_treatment,
    x_treatment,
    guard: bool = True,
    guard_threshold: float = COVARIATE_GUARD_THRESHOLD,
) -> CupedResult:
    """Adjust both arms by a pre-period covariate and report the variance removed.

    ``guard=False`` skips the contamination check. It exists for the simulation
    that *demonstrates* what a contaminated covariate does; production paths leave
    it on.
    """
    y_c = np.asarray(y_control, dtype=float).ravel()
    x_c = np.asarray(x_control, dtype=float).ravel()
    y_t = np.asarray(y_treatment, dtype=float).ravel()
    x_t = np.asarray(x_treatment, dtype=float).ravel()
    if y_c.size != x_c.size or y_t.size != x_t.size:
        raise ValueError("outcome and covariate must have one value per unit in each arm")
    if y_c.size < 2 or y_t.size < 2:
        raise ValueError("each arm needs at least two units")

    y_all = np.concatenate([y_c, y_t])
    x_all = np.concatenate([x_c, x_t])
    var_x = float(x_all.var(ddof=1))
    if var_x <= 0.0:
        raise ValueError("covariate has zero variance; no adjustment is possible")

    # Degeneracy is checked before the balance test: comparing two constant arms
    # is not a meaningful t-test, and scipy warns about the cancellation.
    if guard:
        _guard_covariate(x_c, x_t, guard_threshold)

    theta = float(np.cov(y_all, x_all, ddof=1)[0, 1] / var_x)
    rho = float(np.corrcoef(y_all, x_all)[0, 1])
    x_bar = float(x_all.mean())

    adj_c = y_c - theta * (x_c - x_bar)
    adj_t = y_t - theta * (x_t - x_bar)

    n_c, n_t = y_c.size, y_t.size
    se_adj = math.sqrt(adj_c.var(ddof=1) / n_c + adj_t.var(ddof=1) / n_t)
    se_raw = math.sqrt(y_c.var(ddof=1) / n_c + y_t.var(ddof=1) / n_t)

    return CupedResult(
        theta=theta,
        rho=rho,
        estimate=float(adj_t.mean() - adj_c.mean()),
        standard_error=se_adj,
        unadjusted_estimate=float(y_t.mean() - y_c.mean()),
        unadjusted_standard_error=se_raw,
        realized_variance_reduction=1.0 - (se_adj / se_raw) ** 2 if se_raw > 0 else 0.0,
        predicted_variance_reduction=rho**2,
        n_control=n_c,
        n_treatment=n_t,
    )


def cuped_test(
    *,
    y_control,
    x_control,
    y_treatment,
    x_treatment,
    alpha: float,
    guard: bool = True,
) -> tuple[TestResult, CupedResult]:
    """CUPED-adjusted two-arm comparison, returned with the adjustment diagnostics.

    The standard error treats ``theta`` as known rather than estimated. The extra
    uncertainty from estimating it is O(1/n) relative to the leading term and is
    negligible at experiment scale; ``sims/study_cuped.py`` confirms the intervals
    keep nominal coverage regardless.
    """
    adjustment = cuped_adjust(
        y_control=y_control,
        x_control=x_control,
        y_treatment=y_treatment,
        x_treatment=x_treatment,
        guard=guard,
    )
    result = z_test(
        estimate=adjustment.estimate,
        standard_error=adjustment.standard_error,
        alpha=alpha,
        method=f"CUPED (theta={adjustment.theta:.4f}, rho={adjustment.rho:+.3f})",
        n_control=adjustment.n_control,
        n_treatment=adjustment.n_treatment,
        baseline=float(np.mean(np.asarray(y_control, dtype=float))),
    )
    return result, adjustment
