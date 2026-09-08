"""Deterministic hash-based bucketing.

    sha256(salt + ":" + unit_id) -> leading 8 bytes -> integer -> [0, 1) -> bucket

Three properties this must have, all verified in ``tests/test_assignment.py``:

1. The same unit in the same experiment lands in the same arm, always, across
   processes and across runs.
2. Different experiments (different salt) assign independently.
3. Arbitrary allocation splits are supported, not just 50/50.

**Never use the builtin** ``hash()``. It is salted per interpreter process by
default (PYTHONHASHSEED), so assignments silently change between runs — the
experiment looks fine and the data is garbage. ``hashlib`` is stable by
construction.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

import numpy as np

__all__ = ["assignment_score", "assign", "assign_many", "Allocation"]

_DIGEST_BYTES = 8
_SCALE = float(1 << (8 * _DIGEST_BYTES))

# A mapping of arm name -> share of traffic. Iteration order is part of the
# assignment: see the note in `assign`.
Allocation = Mapping[str, float]


def _validate_allocation(allocation: Allocation) -> tuple[tuple[str, ...], np.ndarray]:
    if not allocation:
        raise ValueError("allocation must contain at least one arm")
    arms = tuple(allocation.keys())
    weights = np.asarray([allocation[a] for a in arms], dtype=float)
    if np.any(weights < 0.0):
        raise ValueError(f"allocation weights must be non-negative, got {dict(allocation)}")
    total = weights.sum()
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(f"allocation weights must sum to 1.0, got {total}")
    return arms, weights


def assignment_score(unit_id: str, salt: str) -> float:
    """Map a unit into ``[0, 1)`` deterministically.

    The salt is joined with ``":"`` so that ("ab", "c") and ("a", "bc") cannot
    collide into the same digest.
    """
    digest = hashlib.sha256(f"{salt}:{unit_id}".encode()).digest()
    return int.from_bytes(digest[:_DIGEST_BYTES], "big") / _SCALE


def assign(unit_id: str, salt: str, allocation: Allocation) -> str:
    """Return the arm this unit belongs to.

    Buckets are laid out along ``[0, 1)`` in the allocation mapping's iteration
    order. That order is therefore part of the experiment definition: reordering
    the arms, or changing any weight, re-buckets units that were already
    enrolled. Fix the allocation before launch and do not edit it mid-flight.
    """
    arms, weights = _validate_allocation(allocation)
    score = assignment_score(unit_id, salt)
    edges = np.cumsum(weights)
    idx = int(np.searchsorted(edges, score, side="right"))
    # Guards the case where floating-point cumsum lands just below `score` on the
    # final edge; the last arm absorbs it.
    return arms[min(idx, len(arms) - 1)]


def assign_many(unit_ids: Iterable[str], salt: str, allocation: Allocation) -> np.ndarray:
    """Vectorized :func:`assign` over many units, for simulation studies.

    Returns an array of arm labels aligned with ``unit_ids``. Identical results to
    calling :func:`assign` per unit — the hashing is still one call per unit,
    since sha256 does not vectorize; only the bucketing is batched.
    """
    arms, weights = _validate_allocation(allocation)
    ids = list(unit_ids)
    scores = np.fromiter((assignment_score(u, salt) for u in ids), dtype=float, count=len(ids))
    edges = np.cumsum(weights)
    idx = np.searchsorted(edges, scores, side="right")
    np.clip(idx, 0, len(arms) - 1, out=idx)
    return np.asarray(arms, dtype=object)[idx]
