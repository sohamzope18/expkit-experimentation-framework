"""Benjamini-Hochberg step-up procedure."""

import numpy as np
import pytest

from expkit.inference.multiple import benjamini_hochberg


class TestProcedure:
    def test_matches_hand_computed_example(self):
        p = [0.001, 0.008, 0.012, 0.030, 0.040, 0.20, 0.44, 0.61, 0.79, 0.95]
        result = benjamini_hochberg(p, 0.05)
        # 0.012 <= (3/10)*0.05 = 0.015 rejects; 0.030 > (4/10)*0.05 = 0.020 does not.
        assert result.rejected == (True, True, True) + (False,) * 7
        assert result.critical_p == pytest.approx(0.012)

    def test_step_up_rejects_below_the_largest_crossing(self):
        """A p-value above its own threshold is still rejected if a later one crosses.

        This is what makes BH a step-*up* procedure, and it is the detail a naive
        per-hypothesis implementation gets wrong.
        """
        # m=3 thresholds are 0.0167, 0.0333, 0.05. The middle p-value fails its
        # own threshold, but the largest one crosses, so all three are rejected.
        result = benjamini_hochberg([0.001, 0.040, 0.045], 0.05)
        assert result.rejected == (True, True, True)
        assert 0.040 > (2 / 3) * 0.05  # would fail its own threshold in isolation
        assert 0.045 <= (3 / 3) * 0.05  # but this one crosses, lifting the rest

    def test_is_less_conservative_than_bonferroni(self):
        p = [0.001, 0.008, 0.012, 0.03, 0.04]
        bh = sum(benjamini_hochberg(p, 0.05).rejected)
        bonferroni = sum(x <= 0.05 / len(p) for x in p)
        assert bh > bonferroni

    def test_order_of_input_does_not_change_decisions(self):
        p = [0.30, 0.001, 0.04, 0.012, 0.008]
        result = benjamini_hochberg(p, 0.05)
        shuffled_idx = np.argsort(p)
        reordered = benjamini_hochberg([p[i] for i in shuffled_idx], 0.05)
        assert sum(result.rejected) == sum(reordered.rejected)
        for i, idx in enumerate(shuffled_idx):
            assert result.rejected[idx] == reordered.rejected[i]

    def test_rejects_nothing_when_all_p_are_large(self):
        assert not any(benjamini_hochberg([0.4, 0.5, 0.9], 0.05).rejected)

    def test_rejects_everything_when_all_p_are_tiny(self):
        assert all(benjamini_hochberg([1e-9, 1e-8, 1e-7], 0.05).rejected)

    def test_uniform_p_values_produce_few_rejections(self):
        """Under a global null, BH should reject rarely."""
        rng = np.random.default_rng(5)
        total = sum(
            sum(benjamini_hochberg(list(rng.random(20)), 0.05).rejected) for _ in range(200)
        )
        assert total < 200  # far below 200*20*0.05 = 200 uncorrected expectation


class TestAdjustedPValues:
    def test_adjusted_are_monotone_in_the_raw_ordering(self):
        p = [0.001, 0.008, 0.012, 0.03, 0.04, 0.2, 0.44]
        adjusted = benjamini_hochberg(p, 0.05).adjusted_p_values
        assert list(adjusted) == sorted(adjusted)

    def test_adjusted_never_below_raw_and_never_above_one(self):
        p = [0.001, 0.3, 0.02, 0.9, 0.05]
        result = benjamini_hochberg(p, 0.05)
        for raw, adj in zip(p, result.adjusted_p_values, strict=True):
            assert raw <= adj <= 1.0

    def test_rejection_agrees_with_adjusted_threshold(self):
        p = [0.001, 0.008, 0.012, 0.03, 0.04, 0.2]
        result = benjamini_hochberg(p, 0.05)
        for adj, rejected in zip(result.adjusted_p_values, result.rejected, strict=True):
            assert rejected == (adj <= 0.05)


class TestEdgeCases:
    def test_empty_family(self):
        result = benjamini_hochberg([], 0.05)
        assert result.n_tests == 0
        assert result.rejected == ()
        assert result.critical_p is None

    def test_single_hypothesis_is_uncorrected(self):
        assert benjamini_hochberg([0.049], 0.05).rejected == (True,)
        assert benjamini_hochberg([0.051], 0.05).rejected == (False,)

    @pytest.mark.parametrize("bad", [[-0.1], [1.1], [float("nan")]])
    def test_rejects_invalid_p_values(self, bad):
        with pytest.raises(ValueError, match="p-values must"):
            benjamini_hochberg(bad, 0.05)

    def test_rejects_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha must be"):
            benjamini_hochberg([0.01], 1.5)
