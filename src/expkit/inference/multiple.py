"""Benjamini-Hochberg false discovery rate control.

Applied to the **secondary metric family only** (D3). The primary metric is a
single pre-registered hypothesis, so there is no multiplicity to correct.
Guardrails are excluded on purpose: correction costs power to detect harm, and for
a guardrail a missed regression is far more expensive than a false alarm.

BH controls the expected *proportion of rejections that are false*, which is the
right target for exploratory metrics -- unlike Bonferroni, which controls the
probability of *any* false rejection and is punishing on a family of twenty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BHResult", "benjamini_hochberg"]


@dataclass(frozen=True)
class BHResult:
    """Outcome of a Benjamini-Hochberg correction over one family."""

    p_values: tuple[float, ...]
    adjusted_p_values: tuple[float, ...]
    rejected: tuple[bool, ...]
    alpha: float
    n_tests: int
    critical_p: float | None

    def __str__(self) -> str:
        return (
            f"BH at alpha={self.alpha}: {sum(self.rejected)} of {self.n_tests} "
            f"rejected (largest p rejected: "
            f"{self.critical_p if self.critical_p is not None else 'none'})"
        )


def benjamini_hochberg(p_values, alpha: float) -> BHResult:
    """Step-up procedure: reject all p up to the largest ``p_(i) <= (i/m) * alpha``.

    Returns adjusted p-values alongside the decisions. The adjusted value is the
    smallest family-wide alpha at which that hypothesis would still be rejected,
    computed by the usual running minimum from the largest p downward, which
    enforces monotonicity -- without it an adjusted p-value can come out smaller
    than one below it in the ordering, which is incoherent.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    p = np.asarray(list(p_values), dtype=float)
    if p.size == 0:
        return BHResult((), (), (), alpha, 0, None)
    if np.any((p < 0.0) | (p > 1.0)) or np.any(np.isnan(p)):
        raise ValueError("p-values must all lie in [0, 1]")

    m = p.size
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    thresholds = (np.arange(1, m + 1) / m) * alpha

    below = np.nonzero(ranked <= thresholds)[0]
    rejected_sorted = np.zeros(m, dtype=bool)
    critical_p = None
    if below.size:
        cutoff = below[-1]
        rejected_sorted[: cutoff + 1] = True
        critical_p = float(ranked[cutoff])

    # Adjusted p-values: running minimum from the largest p downward.
    adjusted_sorted = np.minimum.accumulate((m / np.arange(m, 0, -1)) * ranked[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    rejected = np.empty(m, dtype=bool)
    adjusted = np.empty(m, dtype=float)
    rejected[order] = rejected_sorted
    adjusted[order] = adjusted_sorted

    return BHResult(
        p_values=tuple(float(x) for x in p),
        adjusted_p_values=tuple(float(x) for x in adjusted),
        rejected=tuple(bool(x) for x in rejected),
        alpha=alpha,
        n_tests=m,
        critical_p=critical_p,
    )
