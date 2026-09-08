"""Balance checks report; they never stop. Winsorization is opt-in and pooled."""

import numpy as np
import pytest

from expkit.validity.checks import OUTLIER_PERCENTILE, check_balance, winsorize


class TestBalance:
    def test_balanced_covariate_passes(self):
        rng = np.random.default_rng(31)
        result = check_balance("pre_spend", rng.normal(0, 1, 5_000), rng.normal(0, 1, 5_000))
        assert result.balanced
        assert abs(result.standardized_difference) < 0.1

    def test_imbalance_is_reported_not_raised(self):
        """The difference from SRM: this returns, it does not raise."""
        rng = np.random.default_rng(32)
        result = check_balance("pre_spend", rng.normal(0, 1, 5_000), rng.normal(0.3, 1, 5_000))
        assert not result.balanced
        assert "IMBALANCED" in result.detail
        assert "not a stop" in result.detail

    def test_standardized_difference_has_the_right_scale(self):
        rng = np.random.default_rng(33)
        result = check_balance("x", rng.normal(0, 2, 50_000), rng.normal(1.0, 2, 50_000))
        assert result.standardized_difference == pytest.approx(0.5, abs=0.03)

    def test_reports_both_smd_and_p_value(self):
        """At experiment scale a p-value flags differences too small to matter."""
        rng = np.random.default_rng(34)
        result = check_balance("x", rng.normal(0, 1, 400_000), rng.normal(0.008, 1, 400_000))
        assert abs(result.standardized_difference) < 0.02  # negligible
        assert result.p_value < 0.05  # yet detectable

    def test_requires_two_observations(self):
        with pytest.raises(ValueError, match="at least two"):
            check_balance("x", np.array([1.0]), np.array([1.0, 2.0]))


class TestWinsorize:
    def test_cutoff_is_pooled_not_per_arm(self):
        """Per-arm cutoffs would trim exactly the effect being measured.

        Treatment values run twice as high as control, so the pooled 99th
        percentile sits above every control value. The correct behaviour leaves
        control untouched and caps only treatment. A per-arm cutoff would instead
        trim control at its own 99th percentile, shrinking the gap between the arms
        -- which is the effect.
        """
        control = np.arange(1000.0)
        treatment = np.arange(1000.0) * 2
        pooled_cutoff = float(np.percentile(np.concatenate([control, treatment]), 99.0))
        per_arm_control_cutoff = float(np.percentile(control, 99.0))

        w_c, w_t = winsorize(control, treatment, percentile=99.0)
        assert w_c.max() == control.max()  # untouched
        assert w_c.max() > per_arm_control_cutoff  # a per-arm rule would have cut it
        assert w_t.max() == pytest.approx(pooled_cutoff)

    def test_caps_the_upper_tail_only(self):
        rng = np.random.default_rng(35)
        data = rng.lognormal(3, 1.5, 10_000)
        low, high = winsorize(data, data, percentile=99.0)
        assert low.max() < data.max()
        assert low.min() == data.min()

    def test_preserves_shape(self):
        a, b = np.arange(10.0), np.arange(20.0)
        w_a, w_b = winsorize(a, b)
        assert w_a.shape == a.shape and w_b.shape == b.shape

    def test_default_percentile_is_conservative(self):
        assert OUTLIER_PERCENTILE == 99.9

    @pytest.mark.parametrize("percentile", [50.0, 100.0, 10.0])
    def test_rejects_invalid_percentile(self, percentile):
        with pytest.raises(ValueError, match="percentile must be"):
            winsorize(np.arange(10.0), np.arange(10.0), percentile=percentile)
