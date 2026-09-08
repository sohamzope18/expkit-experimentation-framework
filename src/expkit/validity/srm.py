"""Sample Ratio Mismatch: chi-square goodness-of-fit against intended allocation.

This is a hard stop, not a warning (D1). If the p-value falls below the threshold,
:func:`check_srm` raises and the readout returns no treatment effect at all.

The reasoning is short and worth being able to state cleanly: SRM implies the
randomization or the logging is broken. Broken assignment means the arms are not
exchangeable. Non-exchangeable arms mean the comparison is not causal. So the
effect estimate is meaningless *regardless of its p-value* -- and printing it
beside a warning only invites someone to read it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chisquare

__all__ = ["SRMFailure", "SRMResult", "check_srm", "SRM_THRESHOLD"]

#: Hard-stop threshold (DECISIONS.md D1). Deliberately far stricter than 0.05:
#: this check runs on every experiment, so its false-alarm rate compounds with
#: experiment volume. At ~1,000 experiments a year this is ~1 false stop annually.
SRM_THRESHOLD = 0.001


@dataclass(frozen=True)
class SRMResult:
    """Outcome of the sample-ratio check."""

    passed: bool
    p_value: float
    chi_square: float
    threshold: float
    observed: dict[str, int]
    expected: dict[str, float]
    worst_arm: str
    worst_relative_deviation: float
    detail: str = field(default="")


class SRMFailure(RuntimeError):
    """Raised when observed arm counts contradict the intended allocation.

    Carries the full :class:`SRMResult` so a caller can report the counts without
    re-running the check.
    """

    def __init__(self, result: SRMResult):
        self.result = result
        super().__init__(result.detail)


def check_srm(
    observed: Mapping[str, int],
    allocation: Mapping[str, float],
    *,
    threshold: float = SRM_THRESHOLD,
    raise_on_failure: bool = True,
) -> SRMResult:
    """Compare observed arm counts against the intended allocation.

    Parameters
    ----------
    observed
        Realized unit count per arm.
    allocation
        Intended share per arm; must sum to 1 and cover exactly the observed arms.
    raise_on_failure
        Set ``False`` only to inspect a failure without unwinding the stack -- for
        instance when rendering a diagnostic. The readout path leaves it ``True``.

    Raises
    ------
    SRMFailure
        When ``p_value < threshold`` and ``raise_on_failure`` is set.
    """
    if set(observed) != set(allocation):
        raise ValueError(
            f"observed arms {sorted(observed)} do not match allocation arms "
            f"{sorted(allocation)}"
        )
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")

    arms = sorted(observed)
    counts = np.array([observed[a] for a in arms], dtype=float)
    shares = np.array([allocation[a] for a in arms], dtype=float)
    if not np.isclose(shares.sum(), 1.0, atol=1e-9):
        raise ValueError(f"allocation must sum to 1.0, got {shares.sum()}")
    total = counts.sum()
    if total <= 0:
        raise ValueError("no units observed")

    expected = shares * total
    statistic, p_value = chisquare(f_obs=counts, f_exp=expected)

    deviations = np.abs(counts - expected) / expected
    worst = int(np.argmax(deviations))
    passed = bool(p_value >= threshold)
    detail = (
        f"SRM check {'passed' if passed else 'FAILED'}: chi2={statistic:.2f}, "
        f"p={p_value:.3e} (threshold {threshold:g}). "
        f"Worst arm '{arms[worst]}': observed {int(counts[worst]):,}, "
        f"expected {expected[worst]:,.1f} "
        f"({deviations[worst]:+.2%} relative deviation)."
    )

    result = SRMResult(
        passed=passed,
        p_value=float(p_value),
        chi_square=float(statistic),
        threshold=threshold,
        observed={a: int(observed[a]) for a in arms},
        expected={a: float(e) for a, e in zip(arms, expected, strict=True)},
        worst_arm=arms[worst],
        worst_relative_deviation=float(counts[worst] / expected[worst] - 1.0),
        detail=detail,
    )
    if not passed and raise_on_failure:
        raise SRMFailure(result)
    return result
