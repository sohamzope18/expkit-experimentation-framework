"""Metric definitions shared across estimation, correction, and the readout."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = ["MetricType", "MetricRole", "MetricSpec"]


class MetricType(StrEnum):
    """How a metric is computed, which determines its variance."""

    BINARY = "binary"
    CONTINUOUS = "continuous"
    RATIO = "ratio"


class MetricRole(StrEnum):
    """What a metric is for, which determines how multiplicity is handled (D3).

    PRIMARY
        The single pre-registered hypothesis the experiment exists to test. One
        hypothesis, so nothing to correct for.
    SECONDARY
        Exploratory or supporting metrics. Benjamini-Hochberg applies across this
        family, because this is where multiplicity actually bites.
    GUARDRAIL
        A constraint that must not degrade. Deliberately excluded from FDR
        correction: correction costs power to detect harm, and a missed regression
        is far more expensive than a false alarm.
    """

    PRIMARY = "primary"
    SECONDARY = "secondary"
    GUARDRAIL = "guardrail"


@dataclass(frozen=True)
class MetricSpec:
    """Declaration of a metric, fixed before the experiment reads out.

    ``higher_is_better`` exists so the readout can tell an improvement from a
    regression without guessing from the metric's name. A guardrail like page
    latency degrades when it goes *up*; conversion degrades when it goes *down*.
    Getting this wrong inverts every guardrail decision, so it is required rather
    than inferred.
    """

    name: str
    type: MetricType
    role: MetricRole = MetricRole.SECONDARY
    higher_is_better: bool = True
    cuped_covariate: str | None = None

    def is_regression(self, estimate: float) -> bool:
        """Does this signed effect move the metric in the harmful direction?"""
        return estimate < 0 if self.higher_is_better else estimate > 0
