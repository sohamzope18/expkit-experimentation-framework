"""Shared reporting plumbing for the simulation studies.

Two invariants live here so no individual study can violate them:

1. **Every simulated rate carries a Monte Carlo standard error.** ``rate()`` is
   the only way a study reports a proportion, and it always returns the SE
   alongside it. A claim of "5.02%" without an error bar is not a validation.
2. **Every artifact is byte-reproducible.** Floats are rounded before
   serialization and keys are sorted, so a fresh clone regenerates identical
   JSON rather than JSON that differs in the sixteenth decimal place.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display in CI; must precede the pyplot import
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
ROUND = 6


@dataclass(frozen=True)
class Rate:
    """A simulated proportion with its Monte Carlo uncertainty."""

    label: str
    successes: int
    trials: int
    rate: float
    mc_se: float
    ci_lower: float
    ci_upper: float

    def se_under(self, target: float) -> float:
        """Monte Carlo SE computed **under the target rate**, not the observed one.

        ``mc_se`` uses the observed rate, which is what an error bar on the
        estimate should show. Testing whether the estimate is consistent with a
        target is a different question, and its SE is ``sqrt(t(1-t)/R)``. The two
        barely differ near 50%, but for a rare event they differ by a lot: an
        observed 4/14,000 against a target of 0.001 looks like a 5-sigma miss with
        the observed SE and a 2.7-sigma one with the correct SE.
        """
        return math.sqrt(target * (1.0 - target) / self.trials)

    def within(self, target: float, n_se: float = 3.0) -> bool:
        """Is the estimate within ``n_se`` Monte Carlo SEs of ``target``?"""
        return abs(self.rate - target) <= n_se * self.se_under(target)

    def sigma_from(self, target: float) -> float:
        """How many Monte Carlo SEs separate the estimate from ``target``."""
        se = self.se_under(target)
        return abs(self.rate - target) / se if se > 0 else math.inf

    def __str__(self) -> str:
        return f"{self.label}: {self.rate:.4f} +/- {self.mc_se:.4f} (n={self.trials:,})"


def rate(label: str, successes: int, trials: int, z: float = 1.96) -> Rate:
    """Binomial rate with Monte Carlo SE ``sqrt(p(1-p)/R)``."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    p = successes / trials
    se = math.sqrt(p * (1.0 - p) / trials)
    return Rate(label, int(successes), int(trials), p, se, p - z * se, p + z * se)


def _clean(obj: Any) -> Any:
    """Round floats recursively so serialized artifacts are byte-stable."""
    if isinstance(obj, Rate):
        return _clean(asdict(obj))
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, ROUND)
    if hasattr(obj, "item"):  # numpy scalar
        return _clean(obj.item())
    return obj


def write_results(name: str, payload: dict) -> Path:
    """Serialize a study's numbers to ``results/<name>.json``."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(_clean(payload), indent=2, sort_keys=True) + "\n")
    return path


def savefig(fig: plt.Figure, name: str) -> Path:
    """Save a chart to ``results/<name>.png`` with metadata stripped.

    matplotlib stamps a creation date into PNG metadata by default, which would
    make every regenerated chart differ from the committed one.
    """
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return path


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n {title}\n{'=' * 72}")
