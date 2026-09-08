"""Fixed-horizon tests. Every result carries an estimate, an interval, and a p-value.

**Variance convention (D9).** Both the p-value and the confidence interval use the
unpooled (Wald) variance, so that ``p < alpha`` holds if and only if the interval
excludes zero. The alternative -- a pooled-variance score test for the p-value and
an unpooled interval for reporting -- is marginally better calibrated under the
null but lets the two disagree at the boundary, producing readouts that call a
result significant while showing an interval containing zero. The readout's
decision logic depends on that coherence, so it is bought here.

The cost is a slight anti-conservatism at small samples. That is not asserted to be
negligible; ``sims/study_type1.py`` measures it across sample sizes and reports it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = ["TestResult", "z_test", "two_proportion_test", "welch_test"]


@dataclass(frozen=True)
class TestResult:
    """One metric's fixed-horizon readout."""

    estimate: float
    ci_lower: float
    ci_upper: float
    p_value: float
    standard_error: float
    alpha: float
    method: str
    n_control: int
    n_treatment: int
    mean_control: float | None = None
    mean_treatment: float | None = None
    relative_estimate: float | None = None

    @property
    def significant(self) -> bool:
        """``p < alpha``. Equivalent to the interval excluding zero, by D9."""
        return self.p_value < self.alpha

    def __str__(self) -> str:
        rel = f" ({self.relative_estimate:+.2%})" if self.relative_estimate is not None else ""
        return (
            f"{self.estimate:+.5f}{rel}  "
            f"[{self.ci_lower:+.5f}, {self.ci_upper:+.5f}]  p={self.p_value:.4f}"
        )


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def z_test(
    *,
    estimate: float,
    standard_error: float,
    alpha: float,
    method: str,
    n_control: int,
    n_treatment: int,
    baseline: float | None = None,
    mean_control: float | None = None,
    mean_treatment: float | None = None,
) -> TestResult:
    """Two-sided normal test for any asymptotically normal estimator.

    Shared by the ratio and CUPED metrics so that every estimator in the library
    reports through one code path, with one definition of "significant".
    """
    _validate_alpha(alpha)
    if standard_error <= 0.0 or not math.isfinite(standard_error):
        raise ValueError(f"standard_error must be positive and finite, got {standard_error}")

    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    z_stat = estimate / standard_error
    half_width = z_crit * standard_error
    return TestResult(
        estimate=float(estimate),
        ci_lower=float(estimate - half_width),
        ci_upper=float(estimate + half_width),
        p_value=float(2.0 * stats.norm.sf(abs(z_stat))),
        standard_error=float(standard_error),
        alpha=alpha,
        method=method,
        n_control=int(n_control),
        n_treatment=int(n_treatment),
        mean_control=mean_control,
        mean_treatment=mean_treatment,
        relative_estimate=(float(estimate / baseline) if baseline not in (None, 0.0) else None),
    )


def two_proportion_test(
    *,
    successes_control: int,
    n_control: int,
    successes_treatment: int,
    n_treatment: int,
    alpha: float,
) -> TestResult:
    """Difference in conversion rates, unpooled variance for both p and CI (D9)."""
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("both arms need at least one unit")
    if not 0 <= successes_control <= n_control or not 0 <= successes_treatment <= n_treatment:
        raise ValueError("successes must lie between 0 and n for each arm")

    p_c = successes_control / n_control
    p_t = successes_treatment / n_treatment
    se = math.sqrt(p_c * (1.0 - p_c) / n_control + p_t * (1.0 - p_t) / n_treatment)
    if se == 0.0:
        raise ValueError(
            "zero standard error: both arms are degenerate (all or no successes); "
            "no test is defined"
        )
    return z_test(
        estimate=p_t - p_c,
        standard_error=se,
        alpha=alpha,
        method="two-proportion z (unpooled)",
        n_control=n_control,
        n_treatment=n_treatment,
        baseline=p_c,
        mean_control=p_c,
        mean_treatment=p_t,
    )


def welch_test(
    *,
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float,
) -> TestResult:
    """Difference in means without assuming equal variances across arms.

    Welch rather than Student: a treatment that shifts the mean usually shifts the
    variance too (a pricing change moves both), and the equal-variance t-test is
    not robust to that when the arms are also unequally sized.
    """
    _validate_alpha(alpha)
    control = np.asarray(control, dtype=float).ravel()
    treatment = np.asarray(treatment, dtype=float).ravel()
    n_c, n_t = control.size, treatment.size
    if n_c < 2 or n_t < 2:
        raise ValueError("each arm needs at least two observations")

    mean_c, mean_t = float(control.mean()), float(treatment.mean())
    var_c, var_t = float(control.var(ddof=1)), float(treatment.var(ddof=1))
    se = math.sqrt(var_c / n_c + var_t / n_t)
    if se == 0.0:
        raise ValueError("zero standard error: both arms are constant")

    # Welch-Satterthwaite degrees of freedom.
    df = (var_c / n_c + var_t / n_t) ** 2 / (
        (var_c / n_c) ** 2 / (n_c - 1) + (var_t / n_t) ** 2 / (n_t - 1)
    )
    estimate = mean_t - mean_c
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    half_width = t_crit * se
    return TestResult(
        estimate=estimate,
        ci_lower=estimate - half_width,
        ci_upper=estimate + half_width,
        p_value=float(2.0 * stats.t.sf(abs(estimate / se), df)),
        standard_error=se,
        alpha=alpha,
        method="Welch t",
        n_control=n_c,
        n_treatment=n_t,
        mean_control=mean_c,
        mean_treatment=mean_t,
        relative_estimate=estimate / mean_c if mean_c != 0.0 else None,
    )
