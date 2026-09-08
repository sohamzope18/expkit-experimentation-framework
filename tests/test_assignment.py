"""Assignment module: determinism, marginal allocation, cross-experiment independence.

Every test here is deterministic — hashing has no seed to vary — so a failure is
a real regression, never flake.
"""

import itertools
import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import chi2_contingency

from expkit.design.assignment import assign, assign_many, assignment_score

AB = {"control": 0.5, "treatment": 0.5}
UNITS = [f"unit-{i}" for i in range(20_000)]


class TestDeterminism:
    def test_same_unit_same_arm(self):
        first = [assign(u, "exp-checkout", AB) for u in UNITS[:1000]]
        second = [assign(u, "exp-checkout", AB) for u in UNITS[:1000]]
        assert first == second

    def test_stable_across_processes(self):
        """The property builtin hash() cannot provide.

        PYTHONHASHSEED randomizes str hashing per process by default. If
        assignment_score ever reaches for hash(), two workers scoring the same
        user disagree, and the experiment silently mis-assigns without erroring.
        Two subprocesses are launched with deliberately different hash seeds.
        """
        code = (
            "from expkit.design.assignment import assignment_score;"
            "print([round(assignment_score(f'unit-{i}', 'exp-checkout'), 12) "
            "for i in range(50)])"
        )
        outs = []
        for seed in ("0", "12345"):
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                check=True,
                env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            )
            outs.append(proc.stdout.strip())
        assert outs[0] == outs[1]

        in_process = str(
            [round(assignment_score(f"unit-{i}", "exp-checkout"), 12) for i in range(50)]
        )
        assert outs[0] == in_process

    def test_vectorized_matches_scalar(self):
        alloc = {"a": 0.2, "b": 0.3, "c": 0.5}
        batched = assign_many(UNITS[:2000], "exp-3arm", alloc)
        scalar = [assign(u, "exp-3arm", alloc) for u in UNITS[:2000]]
        assert list(batched) == scalar


class TestScoreDistribution:
    def test_scores_lie_in_unit_interval(self):
        scores = np.array([assignment_score(u, "exp") for u in UNITS])
        assert scores.min() >= 0.0
        assert scores.max() < 1.0

    def test_scores_are_uniform(self):
        """Kolmogorov-Smirnov against U(0,1); a biased digest would show here."""
        from scipy.stats import kstest

        scores = np.array([assignment_score(u, "exp") for u in UNITS])
        assert kstest(scores, "uniform").pvalue > 0.01


class TestMarginalAllocation:
    """Realized allocation must match target within Monte Carlo error."""

    @pytest.mark.parametrize(
        "allocation",
        [
            {"control": 0.5, "treatment": 0.5},
            {"control": 0.6, "treatment": 0.4},
            {"control": 0.9, "treatment": 0.1},
            {"a": 0.2, "b": 0.3, "c": 0.5},
            {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
        ],
    )
    def test_realized_allocation_within_monte_carlo_error(self, allocation):
        arms = assign_many(UNITS, "exp-alloc", allocation)
        n = len(UNITS)
        for arm, target in allocation.items():
            realized = float(np.mean(arms == arm))
            se = np.sqrt(target * (1 - target) / n)
            assert abs(realized - target) < 4 * se, (
                f"{arm}: realized {realized:.4f} vs target {target} "
                f"({abs(realized - target) / se:.2f} MC SE away)"
            )

    def test_every_unit_lands_in_exactly_one_arm(self):
        alloc = {"a": 0.2, "b": 0.3, "c": 0.5}
        arms = assign_many(UNITS, "exp-3arm", alloc)
        assert set(np.unique(arms)) <= set(alloc)
        assert len(arms) == len(UNITS)


class TestCrossExperimentIndependence:
    """Different salts must assign independently (spec §3.2).

    Built from 46 salts, giving C(46,2) = 1035 experiment pairs while hashing only
    46 x 2000 units. Pairs drawn from a shared salt pool are not mutually
    independent, so the acceptance band below is deliberately wider than the
    nominal binomial SE of 0.0068.
    """

    N_UNITS = 2000
    SALTS = [f"exp-{i}" for i in range(46)]

    @pytest.fixture(scope="class")
    @classmethod
    def pair_statistics(cls):
        ids = [f"unit-{i}" for i in range(cls.N_UNITS)]
        assigned = {s: (assign_many(ids, s, AB) == "treatment").astype(int) for s in cls.SALTS}
        p_values, phis = [], []
        for x, y in itertools.combinations(cls.SALTS, 2):
            a, b = assigned[x], assigned[y]
            table = np.array(
                [
                    [np.sum((a == 0) & (b == 0)), np.sum((a == 0) & (b == 1))],
                    [np.sum((a == 1) & (b == 0)), np.sum((a == 1) & (b == 1))],
                ]
            )
            result = chi2_contingency(table, correction=False)
            p_values.append(result.pvalue)
            phis.append(np.sqrt(result.statistic / cls.N_UNITS))
        return np.array(p_values), np.array(phis)

    def test_at_least_1000_pairs_examined(self, pair_statistics):
        p_values, _ = pair_statistics
        assert len(p_values) >= 1000

    def test_rejection_rate_is_nominal(self, pair_statistics):
        """Under independence, 5% of pairs should reject at alpha=0.05."""
        p_values, _ = pair_statistics
        rejected = float((p_values < 0.05).mean())
        assert 0.03 < rejected < 0.075, f"rejection rate {rejected:.4f}, expected ~0.05"

    def test_correlation_magnitude_matches_independence(self, pair_statistics):
        """E|phi| for independent binary pairs is sqrt(2/(pi*n))."""
        _, phis = pair_statistics
        expected = np.sqrt(2.0 / (np.pi * self.N_UNITS))
        assert abs(float(phis.mean()) - expected) < 0.15 * expected
        assert float(phis.max()) < 0.15


class TestValidation:
    @pytest.mark.parametrize(
        "allocation, match",
        [
            ({}, "at least one arm"),
            ({"a": 0.5, "b": 0.4}, "sum to 1.0"),
            ({"a": 1.5, "b": -0.5}, "non-negative"),
        ],
    )
    def test_rejects_invalid_allocation(self, allocation, match):
        with pytest.raises(ValueError, match=match):
            assign("unit-1", "exp", allocation)

    def test_salt_separator_prevents_collision(self):
        """("ab", "c") and ("a", "bc") must not produce the same digest."""
        assert assignment_score("c", "ab") != assignment_score("bc", "a")
