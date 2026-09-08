"""Power, minimum detectable effect, and sample size.

Supports three metric types: binary (conversion), continuous (revenue per user),
and ratio (clicks per session, where numerator and denominator are both random).

Conventions, all recorded in ``DECISIONS.md``:

* **Two-sided only** (D5). There is no ``sides`` parameter.
* **No continuity correction** (D6), so the design matches the uncorrected
  normal-approximation test in :mod:`expkit.inference.fixed`.
* **Unequal allocation via** ``allocation_ratio = n_treatment / n_control`` (D7).
* **Pooled-null variance in the alpha term, alternative variance in the power
  term** (D8) for binary metrics. See the annotations in
  :func:`required_n_binary`.

``alpha`` and ``power`` are required keyword arguments everywhere. They have no
defaults, deliberately: a significance level is an owner decision per experiment,
not a library constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq
from scipy.stats import norm

__all__ = [
    "SampleSize",
    "required_n_binary",
    "required_n_continuous",
    "required_n_ratio",
    "power_for_n_binary",
    "power_for_n_continuous",
    "power_for_n_ratio",
    "mde_binary",
    "mde_continuous",
    "mde_ratio",
]


@dataclass(frozen=True)
class SampleSize:
    """Required sample size for one experiment design.

    ``n_control`` and ``n_treatment`` are rounded up independently, so
    ``n_treatment / n_control`` may differ from ``allocation_ratio`` by less than
    one unit per arm.
    """

    n_control: int
    n_treatment: int
    n_total: int
    absolute_effect: float
    relative_effect: float | None
    alpha: float
    power: float
    allocation_ratio: float


def _validate(alpha: float, power: float, allocation_ratio: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")
    if power <= alpha:
        raise ValueError(f"power ({power}) must exceed alpha ({alpha}) to be meaningful")
    if allocation_ratio <= 0.0:
        raise ValueError(f"allocation_ratio must be positive, got {allocation_ratio}")


def _critical_values(alpha: float, power: float) -> tuple[float, float]:
    """Return ``(z_alpha, z_beta)``.

    ``z_alpha`` is the two-sided critical value ``z_(1 - alpha/2)`` per D5.
    """
    return norm.ppf(1.0 - alpha / 2.0), norm.ppf(power)


def _resolve_effect(
    baseline: float,
    absolute_effect: float | None,
    relative_effect: float | None,
) -> tuple[float, float | None]:
    """Resolve the absolute/relative lift parameterizations against each other."""
    if (absolute_effect is None) == (relative_effect is None):
        raise ValueError("specify exactly one of absolute_effect or relative_effect")
    if relative_effect is not None:
        if baseline == 0.0:
            raise ValueError("relative_effect requires a non-zero baseline")
        return relative_effect * baseline, relative_effect
    rel = absolute_effect / baseline if baseline != 0.0 else None
    return absolute_effect, rel


# --------------------------------------------------------------------------- #
# Continuous metrics
# --------------------------------------------------------------------------- #


def required_n_continuous(
    *,
    variance: float,
    alpha: float,
    power: float,
    absolute_effect: float | None = None,
    relative_effect: float | None = None,
    baseline_mean: float | None = None,
    allocation_ratio: float = 1.0,
) -> SampleSize:
    """Sample size for a difference in means, assuming equal variance across arms.

    ``Var(delta_hat) = variance * (1 + 1/kappa) / n_control``, so

        n_control = (z_alpha + z_beta)^2 * variance * (1 + 1/kappa) / delta^2
    """
    _validate(alpha, power, allocation_ratio)
    if variance <= 0.0:
        raise ValueError(f"variance must be positive, got {variance}")
    if relative_effect is not None and baseline_mean is None:
        raise ValueError("relative_effect requires baseline_mean")
    delta, rel = _resolve_effect(
        baseline_mean if baseline_mean is not None else 0.0, absolute_effect, relative_effect
    )
    if delta == 0.0:
        raise ValueError("effect size must be non-zero")

    z_a, z_b = _critical_values(alpha, power)
    n_control = (z_a + z_b) ** 2 * variance * (1.0 + 1.0 / allocation_ratio) / delta**2
    return _assemble(n_control, delta, rel, alpha, power, allocation_ratio)


def power_for_n_continuous(
    *,
    n_control: float,
    variance: float,
    absolute_effect: float,
    alpha: float,
    allocation_ratio: float = 1.0,
) -> float:
    """Power of the two-sided test at ``n_control`` units in the control arm."""
    if n_control <= 0.0:
        raise ValueError(f"n_control must be positive, got {n_control}")
    if variance <= 0.0:
        raise ValueError(f"variance must be positive, got {variance}")
    z_a = norm.ppf(1.0 - alpha / 2.0)
    se = math.sqrt(variance * (1.0 + 1.0 / allocation_ratio) / n_control)
    # Two-sided: the far tail contributes negligibly but is included for exactness.
    return norm.sf(z_a - abs(absolute_effect) / se) + norm.cdf(-z_a - abs(absolute_effect) / se)


def mde_continuous(
    *,
    n_control: float,
    variance: float,
    alpha: float,
    power: float,
    allocation_ratio: float = 1.0,
    baseline_mean: float | None = None,
) -> tuple[float, float | None]:
    """Minimum detectable effect: ``(absolute, relative_or_None)``.

    Closed form, since the variance of a continuous metric does not depend on the
    effect size.
    """
    _validate(alpha, power, allocation_ratio)
    if n_control <= 0.0:
        raise ValueError(f"n_control must be positive, got {n_control}")
    z_a, z_b = _critical_values(alpha, power)
    delta = (z_a + z_b) * math.sqrt(variance * (1.0 + 1.0 / allocation_ratio) / n_control)
    rel = delta / baseline_mean if baseline_mean else None
    return delta, rel


# --------------------------------------------------------------------------- #
# Ratio metrics
# --------------------------------------------------------------------------- #
#
# A ratio metric is sized exactly like a continuous one once you have the
# per-randomization-unit variance of the linearized metric. That variance is the
# delta-method quantity, and it is computed in expkit.metrics.ratio (M3) — it is
# deliberately NOT reimplemented here. One variance formula, one home.


def required_n_ratio(
    *,
    baseline_ratio: float,
    unit_variance: float,
    alpha: float,
    power: float,
    absolute_effect: float | None = None,
    relative_effect: float | None = None,
    allocation_ratio: float = 1.0,
) -> SampleSize:
    """Sample size for a ratio metric, in randomization units per arm.

    ``unit_variance`` is the per-unit variance of the *linearized* ratio, i.e. the
    delta-method variance multiplied by n. Obtain it from
    :mod:`expkit.metrics.ratio` rather than deriving it here.
    """
    return required_n_continuous(
        variance=unit_variance,
        alpha=alpha,
        power=power,
        absolute_effect=absolute_effect,
        relative_effect=relative_effect,
        baseline_mean=baseline_ratio,
        allocation_ratio=allocation_ratio,
    )


def power_for_n_ratio(
    *,
    n_control: float,
    unit_variance: float,
    absolute_effect: float,
    alpha: float,
    allocation_ratio: float = 1.0,
) -> float:
    """Power for a ratio metric. See :func:`required_n_ratio` on ``unit_variance``."""
    return power_for_n_continuous(
        n_control=n_control,
        variance=unit_variance,
        absolute_effect=absolute_effect,
        alpha=alpha,
        allocation_ratio=allocation_ratio,
    )


def mde_ratio(
    *,
    n_control: float,
    baseline_ratio: float,
    unit_variance: float,
    alpha: float,
    power: float,
    allocation_ratio: float = 1.0,
) -> tuple[float, float | None]:
    """MDE for a ratio metric. See :func:`required_n_ratio` on ``unit_variance``."""
    return mde_continuous(
        n_control=n_control,
        variance=unit_variance,
        alpha=alpha,
        power=power,
        allocation_ratio=allocation_ratio,
        baseline_mean=baseline_ratio,
    )


# --------------------------------------------------------------------------- #
# Binary metrics
# --------------------------------------------------------------------------- #


def _pooled_rate(p_control: float, p_treatment: float, allocation_ratio: float) -> float:
    """Allocation-weighted rate under H0, where both arms share one rate (D8)."""
    return (p_control + allocation_ratio * p_treatment) / (1.0 + allocation_ratio)


def required_n_binary(
    *,
    p_control: float,
    alpha: float,
    power: float,
    absolute_effect: float | None = None,
    relative_effect: float | None = None,
    allocation_ratio: float = 1.0,
) -> SampleSize:
    """Sample size for a difference in proportions, uncorrected (D6).

        n_control = [ z_alpha * sqrt((1 + 1/kappa) * pbar*(1-pbar))
                    + z_beta  * sqrt(p_c*(1-p_c) + p_t*(1-p_t)/kappa) ]^2 / delta^2

    The two variance expressions are different on purpose (D8) — see the inline
    annotations below.
    """
    _validate(alpha, power, allocation_ratio)
    if not 0.0 < p_control < 1.0:
        raise ValueError(f"p_control must be in (0, 1), got {p_control}")
    delta, rel = _resolve_effect(p_control, absolute_effect, relative_effect)
    if delta == 0.0:
        raise ValueError("effect size must be non-zero")
    p_treatment = p_control + delta
    if not 0.0 < p_treatment < 1.0:
        raise ValueError(
            f"implied treatment rate {p_treatment} is outside (0, 1); effect is too large"
        )

    z_a, z_b = _critical_values(alpha, power)
    kappa = allocation_ratio

    # ALPHA TERM — pooled variance under the null (D8). The critical value is set
    # under H0, where both arms share the rate pbar.
    p_bar = _pooled_rate(p_control, p_treatment, kappa)
    alpha_term = z_a * math.sqrt((1.0 + 1.0 / kappa) * p_bar * (1.0 - p_bar))

    # POWER TERM — variance under the alternative (D8). Power is evaluated under
    # H1, where the arms have different rates and therefore different variances.
    power_term = z_b * math.sqrt(
        p_control * (1.0 - p_control) + p_treatment * (1.0 - p_treatment) / kappa
    )

    n_control = (alpha_term + power_term) ** 2 / delta**2
    return _assemble(n_control, delta, rel, alpha, power, kappa)


def power_for_n_binary(
    *,
    n_control: float,
    p_control: float,
    absolute_effect: float,
    alpha: float,
    allocation_ratio: float = 1.0,
) -> float:
    """Power of the two-sided two-proportion test, inverting :func:`required_n_binary`."""
    if n_control <= 0.0:
        raise ValueError(f"n_control must be positive, got {n_control}")
    if not 0.0 < p_control < 1.0:
        raise ValueError(f"p_control must be in (0, 1), got {p_control}")
    p_treatment = p_control + absolute_effect
    if not 0.0 < p_treatment < 1.0:
        raise ValueError(f"implied treatment rate {p_treatment} is outside (0, 1)")

    kappa = allocation_ratio
    z_a = norm.ppf(1.0 - alpha / 2.0)
    p_bar = _pooled_rate(p_control, p_treatment, kappa)  # null variance (D8)
    se_null = math.sqrt((1.0 + 1.0 / kappa) * p_bar * (1.0 - p_bar) / n_control)
    se_alt = math.sqrt(  # alternative variance (D8)
        (p_control * (1.0 - p_control) + p_treatment * (1.0 - p_treatment) / kappa) / n_control
    )
    shift = abs(absolute_effect) / se_alt
    crit = z_a * se_null / se_alt
    return norm.sf(crit - shift) + norm.cdf(-crit - shift)


def mde_binary(
    *,
    n_control: float,
    p_control: float,
    alpha: float,
    power: float,
    allocation_ratio: float = 1.0,
) -> tuple[float, float]:
    """Minimum detectable effect for a binary metric: ``(absolute, relative)``.

    Solved numerically rather than in closed form: the binary variance depends on
    the effect size, so ``delta`` appears on both sides of the equation. Reported
    for the positive direction; because the alternative variance is not symmetric
    in ``delta``, the detectable decrease differs slightly in magnitude.
    """
    _validate(alpha, power, allocation_ratio)
    if n_control <= 0.0:
        raise ValueError(f"n_control must be positive, got {n_control}")
    if not 0.0 < p_control < 1.0:
        raise ValueError(f"p_control must be in (0, 1), got {p_control}")

    def gap(delta: float) -> float:
        return (
            power_for_n_binary(
                n_control=n_control,
                p_control=p_control,
                absolute_effect=delta,
                alpha=alpha,
                allocation_ratio=allocation_ratio,
            )
            - power
        )

    lo, hi = 1e-12, (1.0 - p_control) * (1.0 - 1e-9)
    if gap(hi) < 0.0:
        raise ValueError(
            f"power={power} is unreachable at n_control={n_control} for any effect "
            f"keeping the treatment rate below 1"
        )
    delta = brentq(gap, lo, hi, xtol=1e-12, rtol=1e-12)
    return delta, delta / p_control


def _assemble(
    n_control: float,
    delta: float,
    rel: float | None,
    alpha: float,
    power: float,
    kappa: float,
) -> SampleSize:
    n_c = math.ceil(n_control)
    n_t = math.ceil(n_control * kappa)
    return SampleSize(
        n_control=n_c,
        n_treatment=n_t,
        n_total=n_c + n_t,
        absolute_effect=delta,
        relative_effect=rel,
        alpha=alpha,
        power=power,
        allocation_ratio=kappa,
    )
