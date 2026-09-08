"""Always-valid inference via asymptotic confidence sequences (D2).

A fixed-horizon confidence interval is valid at *one* pre-declared sample size.
Look at it earlier and the guarantee is gone -- which is why an analyst who checks
the dashboard daily and stops at the first significant result has a false-positive
rate far above alpha, without doing anything that feels wrong.

A confidence sequence is valid at *every* sample size simultaneously:

    P( there exists any t such that CS_t excludes the true value ) <= alpha

so you may look whenever you like, stop whenever you like, and the coverage
guarantee still holds. The interval for an asymptotically normal estimator is

    theta_hat_t  +/-  sigma_hat * sqrt( 2*(t*rho^2 + 1) / (t^2 * rho^2)
                                        * log( sqrt(t*rho^2 + 1) / alpha ) )

where ``sigma_hat`` is the per-unit standard deviation, i.e.
``standard_error * sqrt(t)``. Because it takes any asymptotically normal estimator
and its standard error, ratio metrics (delta method) and CUPED-adjusted outcomes
run through this same function without a separate derivation each -- the reason
D2 chose it over mSPRT.

**About rho.** AsympCS is not tuning-free, and claiming otherwise would be a worse
answer than explaining it. ``rho`` sets the sample size at which the sequence is
tightest; it is wider everywhere else. The parameter mSPRT asks for is different in
kind: a prior on the *effect size*, which you do not know and are running the
experiment to learn. ``rho`` asks for your planned sample size, which you chose
before launch and can state. :func:`rho_for_target` picks it by numerically
minimizing the width at that planned size.

The width shrinks like ``sqrt(log(t)/t)`` rather than the fixed-horizon
``sqrt(1/t)``. That extra ``log t`` is the entire price of being allowed to look,
and ``sims/study_peeking.py`` measures what it costs in sample size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import minimize_scalar

__all__ = ["SequentialResult", "rho_for_target", "asympcs_half_width", "asympcs"]


@dataclass(frozen=True)
class SequentialResult:
    """An always-valid interval, evaluated at one point in an ongoing experiment."""

    estimate: float
    ci_lower: float
    ci_upper: float
    half_width: float
    standard_error: float
    n_total: int
    alpha: float
    rho: float
    method: str = "asymptotic confidence sequence"

    @property
    def excludes_null(self) -> bool:
        """Has the sequence separated from zero? Safe to check at any time."""
        return self.ci_lower > 0.0 or self.ci_upper < 0.0

    def __str__(self) -> str:
        return (
            f"{self.estimate:+.5f}  [{self.ci_lower:+.5f}, {self.ci_upper:+.5f}]  "
            f"n={self.n_total:,}  {'CROSSED' if self.excludes_null else 'continue'}"
        )


def asympcs_half_width(*, t: float, sigma: float, alpha: float, rho: float) -> float:
    """Half-width of the confidence sequence at sample size ``t``.

    ``sigma`` is the per-unit standard deviation of the estimator, so that the
    fixed-horizon standard error would be ``sigma / sqrt(t)``.
    """
    if t <= 0:
        raise ValueError(f"t must be positive, got {t}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")

    inner = t * rho**2 + 1.0
    return sigma * math.sqrt((2.0 * inner) / (t**2 * rho**2) * math.log(math.sqrt(inner) / alpha))


def rho_for_target(*, target_n: float, alpha: float) -> float:
    """Choose ``rho`` to make the sequence tightest at the planned sample size.

    Minimized numerically rather than by a closed form: the objective is smooth and
    one-dimensional, and solving it directly removes a transcribed constant that
    nothing in the test suite would catch if it were wrong.
    """
    if target_n <= 0:
        raise ValueError(f"target_n must be positive, got {target_n}")

    def width(log_rho: float) -> float:
        return asympcs_half_width(t=target_n, sigma=1.0, alpha=alpha, rho=math.exp(log_rho))

    # Bracket generously in log space: the optimum sits near 1/sqrt(target_n).
    guess = -0.5 * math.log(target_n)
    result = minimize_scalar(width, bounds=(guess - 6.0, guess + 6.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"rho optimization failed for target_n={target_n}")
    return float(math.exp(result.x))


def asympcs(
    *,
    estimate: float,
    standard_error: float,
    n_total: int,
    alpha: float,
    rho: float,
) -> SequentialResult:
    """Always-valid interval for any asymptotically normal estimator.

    Parameters
    ----------
    estimate, standard_error
        The point estimate and its fixed-horizon standard error at this moment --
        whatever estimator produced them: difference in means, delta-method ratio,
        or a CUPED-adjusted difference.
    n_total
        Units accumulated so far, across both arms.
    rho
        From :func:`rho_for_target`, computed once at design time from the planned
        sample size. Recomputing it from the data as the experiment runs would make
        the boundary depend on the data and void the guarantee.
    """
    if n_total <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}")
    if standard_error <= 0 or not math.isfinite(standard_error):
        raise ValueError(f"standard_error must be positive and finite, got {standard_error}")

    sigma = standard_error * math.sqrt(n_total)
    half = asympcs_half_width(t=n_total, sigma=sigma, alpha=alpha, rho=rho)
    return SequentialResult(
        estimate=float(estimate),
        ci_lower=float(estimate - half),
        ci_upper=float(estimate + half),
        half_width=float(half),
        standard_error=float(standard_error),
        n_total=int(n_total),
        alpha=alpha,
        rho=float(rho),
    )
