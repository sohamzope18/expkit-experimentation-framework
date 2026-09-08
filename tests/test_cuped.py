"""CUPED: the variance it removes, the estimate it must not move, and the guard."""

import numpy as np
import pytest

from expkit.metrics.cuped import (
    COVARIATE_GUARD_THRESHOLD,
    CovariateContamination,
    cuped_adjust,
    cuped_test,
)


def arms(rng, rho, n=20_000, effect=0.0, covariate_shift=0.0):
    def draw(shift, eff):
        x = rng.normal(0.0, 1.0, n)
        y = rho * x + rng.normal(0.0, np.sqrt(1 - rho**2), n) + eff
        return y, x + shift

    return draw(0.0, 0.0), draw(covariate_shift, effect)


class TestVarianceReduction:
    @pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.9])
    def test_reduction_matches_one_minus_rho_squared(self, rho):
        rng = np.random.default_rng(int(rho * 100) + 5)
        (y_c, x_c), (y_t, x_t) = arms(rng, rho)
        r = cuped_adjust(y_control=y_c, x_control=x_c, y_treatment=y_t, x_treatment=x_t)
        assert r.realized_variance_reduction == pytest.approx(rho**2, abs=0.02)
        assert r.predicted_variance_reduction == pytest.approx(rho**2, abs=0.02)

    def test_theta_recovers_the_generating_slope(self):
        rng = np.random.default_rng(21)
        (y_c, x_c), (y_t, x_t) = arms(rng, 0.7)
        r = cuped_adjust(y_control=y_c, x_control=x_c, y_treatment=y_t, x_treatment=x_t)
        assert r.theta == pytest.approx(0.7, abs=0.02)

    def test_zero_correlation_buys_nothing(self):
        rng = np.random.default_rng(22)
        (y_c, x_c), (y_t, x_t) = arms(rng, 0.0)
        r = cuped_adjust(y_control=y_c, x_control=x_c, y_treatment=y_t, x_treatment=x_t)
        assert abs(r.realized_variance_reduction) < 0.01

    def test_adjustment_shrinks_the_standard_error(self):
        rng = np.random.default_rng(23)
        (y_c, x_c), (y_t, x_t) = arms(rng, 0.8)
        r = cuped_adjust(y_control=y_c, x_control=x_c, y_treatment=y_t, x_treatment=x_t)
        assert r.standard_error < r.unadjusted_standard_error


class TestUnbiasedness:
    def test_point_estimate_is_not_moved(self):
        """CUPED trades variance, not location. If it moves the estimate, it is a bug."""
        rng = np.random.default_rng(24)
        (y_c, x_c), (y_t, x_t) = arms(rng, 0.8, n=60_000, effect=0.05)
        r = cuped_adjust(y_control=y_c, x_control=x_c, y_treatment=y_t, x_treatment=x_t)
        assert r.estimate == pytest.approx(0.05, abs=3 * r.standard_error)
        assert r.unadjusted_estimate == pytest.approx(0.05, abs=3 * r.unadjusted_standard_error)


class TestGuard:
    def test_contaminated_covariate_raises(self):
        rng = np.random.default_rng(25)
        (y_c, x_c), (y_t, x_t) = arms(rng, 0.6, effect=0.05, covariate_shift=0.05)
        with pytest.raises(CovariateContamination, match="differs across arms"):
            cuped_adjust(y_control=y_c, x_control=x_c, y_treatment=y_t, x_treatment=x_t)

    def test_guard_can_be_bypassed_for_demonstration(self):
        rng = np.random.default_rng(26)
        (y_c, x_c), (y_t, x_t) = arms(rng, 0.6, effect=0.05, covariate_shift=0.05)
        r = cuped_adjust(
            y_control=y_c, x_control=x_c, y_treatment=y_t, x_treatment=x_t, guard=False
        )
        assert r.estimate < 0.05  # biased toward zero, which is the point

    def test_contamination_biases_toward_zero_while_tightening(self):
        """The dangerous combination: a wronger answer that looks more certain."""
        rng = np.random.default_rng(27)
        (y_c, x_c), (y_t, x_t) = arms(rng, 0.6, n=40_000, effect=0.05, covariate_shift=0.05)
        dirty = cuped_adjust(
            y_control=y_c, x_control=x_c, y_treatment=y_t, x_treatment=x_t, guard=False
        )
        clean = cuped_adjust(
            y_control=y_c, x_control=x_c, y_treatment=y_t, x_treatment=x_t - 0.05, guard=False
        )
        assert abs(dirty.estimate - 0.05) > abs(clean.estimate - 0.05)
        assert dirty.standard_error < dirty.unadjusted_standard_error

    def test_threshold_matches_the_srm_reasoning(self):
        from expkit.validity.srm import SRM_THRESHOLD

        assert COVARIATE_GUARD_THRESHOLD == SRM_THRESHOLD


class TestCupedTest:
    def test_returns_result_and_diagnostics(self):
        rng = np.random.default_rng(28)
        (y_c, x_c), (y_t, x_t) = arms(rng, 0.8, n=40_000, effect=0.03)
        result, adjustment = cuped_test(
            y_control=y_c, x_control=x_c, y_treatment=y_t, x_treatment=x_t, alpha=0.05
        )
        assert result.significant
        assert "CUPED" in result.method
        assert adjustment.realized_variance_reduction > 0.5

    def test_cuped_detects_what_the_raw_test_cannot(self):
        """The whole point: same data, same effect, more power.

        Asserted over 20 independent experiments rather than one, because a
        single-experiment version of this test is a coin flip near the boundary
        and would flake on any change to the random stream.
        """
        from expkit.inference.fixed import welch_test

        raw_hits = adjusted_hits = 0
        for seed in range(20):
            rng = np.random.default_rng(100 + seed)
            (y_c, x_c), (y_t, x_t) = arms(rng, 0.9, n=3_000, effect=0.04)
            raw_hits += welch_test(control=y_c, treatment=y_t, alpha=0.05).significant
            adjusted_hits += cuped_test(
                y_control=y_c,
                x_control=x_c,
                y_treatment=y_t,
                x_treatment=x_t,
                alpha=0.05,
            )[0].significant
        assert raw_hits <= 8
        assert adjusted_hits >= 18


class TestValidation:
    def test_mismatched_arrays(self):
        with pytest.raises(ValueError, match="one value per unit"):
            cuped_adjust(
                y_control=np.ones(5),
                x_control=np.ones(4),
                y_treatment=np.ones(5),
                x_treatment=np.ones(5),
            )

    def test_constant_covariate(self):
        with pytest.raises(ValueError, match="zero variance"):
            cuped_adjust(
                y_control=np.arange(10.0),
                x_control=np.ones(10),
                y_treatment=np.arange(10.0),
                x_treatment=np.ones(10),
            )
