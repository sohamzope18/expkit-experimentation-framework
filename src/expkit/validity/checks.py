"""Pre-period balance and outlier policy.

Unlike SRM (:mod:`expkit.validity.srm`), nothing here stops a readout. Covariate
imbalance on a correctly randomized experiment happens by chance at exactly the
rate the test says it should, so treating it as a hard stop would throw away good
experiments. These are reported as signals for a human to weigh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "BalanceResult",
    "check_balance",
    "winsorize",
    "OUTLIER_PERCENTILE",
]

#: Default winsorization percentile when trimming is explicitly enabled (D10).
#: Not applied unless a caller asks for it.
OUTLIER_PERCENTILE = 99.9


@dataclass(frozen=True)
class BalanceResult:
    """Pre-period comparability of one covariate across two arms."""

    covariate: str
    mean_control: float
    mean_treatment: float
    standardized_difference: float
    p_value: float
    balanced: bool
    detail: str

    def __str__(self) -> str:
        return self.detail


def check_balance(
    covariate: str,
    control: np.ndarray,
    treatment: np.ndarray,
    *,
    threshold: float = 0.001,
) -> BalanceResult:
    """Compare a pre-period covariate across arms. Reports; never stops.

    Reports the standardized mean difference alongside the p-value because they
    answer different questions. At experiment scale a p-value detects differences
    far too small to bias anything; the standardized difference says whether the
    imbalance is large enough to care about. Conventionally |SMD| < 0.1 is
    negligible.
    """
    control = np.asarray(control, dtype=float).ravel()
    treatment = np.asarray(treatment, dtype=float).ravel()
    if control.size < 2 or treatment.size < 2:
        raise ValueError("each arm needs at least two observations")

    mean_c, mean_t = float(control.mean()), float(treatment.mean())
    pooled_sd = float(np.sqrt((control.var(ddof=1) + treatment.var(ddof=1)) / 2.0))
    smd = (mean_t - mean_c) / pooled_sd if pooled_sd > 0 else 0.0
    p_value = float(stats.ttest_ind(treatment, control, equal_var=False).pvalue)
    balanced = bool(p_value >= threshold)

    return BalanceResult(
        covariate=covariate,
        mean_control=mean_c,
        mean_treatment=mean_t,
        standardized_difference=smd,
        p_value=p_value,
        balanced=balanced,
        detail=(
            f"{covariate}: control {mean_c:.4f} vs treatment {mean_t:.4f}, "
            f"SMD {smd:+.4f}, p={p_value:.3e} -> "
            f"{'balanced' if balanced else 'IMBALANCED (report only, not a stop)'}"
        ),
    )


def winsorize(
    control: np.ndarray,
    treatment: np.ndarray,
    *,
    percentile: float = OUTLIER_PERCENTILE,
) -> tuple[np.ndarray, np.ndarray]:
    """Cap both arms at a percentile of the **pooled** distribution (D10).

    The cutoff is computed on the arms combined, then applied identically to both.
    Computing a per-arm cutoff is the trap: if the treatment genuinely produced
    more big spenders, a per-arm cap trims exactly the effect you are trying to
    measure, and biases the estimate toward zero by an amount nobody can see.

    Off by default everywhere in this library. Trimming changes the estimand --
    you stop estimating the mean and start estimating a trimmed mean -- so it must
    be a deliberate, pre-registered choice rather than a silent default.
    """
    if not 50.0 < percentile < 100.0:
        raise ValueError(f"percentile must be in (50, 100), got {percentile}")
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    cutoff = float(np.percentile(np.concatenate([control.ravel(), treatment.ravel()]), percentile))
    return np.minimum(control, cutoff), np.minimum(treatment, cutoff)
