"""Fixed-horizon tests, and the CI/p-value coherence D9 buys."""

import numpy as np
import pytest

from expkit.inference.fixed import two_proportion_test, welch_test, z_test

ALPHA = 0.05


class TestCoherence:
    """D9: p < alpha must hold exactly when the interval excludes zero."""

    @pytest.mark.parametrize("seed", range(25))
    def test_binary_p_and_ci_agree(self, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(200, 20_000))
        p_c = float(rng.uniform(0.02, 0.5))
        result = two_proportion_test(
            successes_control=int(rng.binomial(n, p_c)),
            n_control=n,
            successes_treatment=int(rng.binomial(n, p_c * rng.uniform(0.9, 1.1))),
            n_treatment=n,
            alpha=ALPHA,
        )
        excludes_zero = result.ci_lower > 0 or result.ci_upper < 0
        assert result.significant == excludes_zero

    @pytest.mark.parametrize("seed", range(25))
    def test_continuous_p_and_ci_agree(self, seed):
        rng = np.random.default_rng(1000 + seed)
        n = int(rng.integers(50, 3_000))
        result = welch_test(
            control=rng.normal(10, 3, n),
            treatment=rng.normal(10 + rng.uniform(-0.5, 0.5), rng.uniform(2, 4), n),
            alpha=ALPHA,
        )
        excludes_zero = result.ci_lower > 0 or result.ci_upper < 0
        assert result.significant == excludes_zero

    def test_alpha_changes_both_together(self):
        kwargs = dict(
            successes_control=1000, n_control=10_000, successes_treatment=1080, n_treatment=10_000
        )
        loose = two_proportion_test(**kwargs, alpha=0.10)
        tight = two_proportion_test(**kwargs, alpha=0.01)
        assert tight.ci_upper - tight.ci_lower > loose.ci_upper - loose.ci_lower
        assert loose.p_value == pytest.approx(tight.p_value)


class TestEstimates:
    def test_binary_point_estimate_and_direction(self):
        r = two_proportion_test(
            successes_control=1000,
            n_control=10_000,
            successes_treatment=1120,
            n_treatment=10_000,
            alpha=ALPHA,
        )
        assert r.estimate == pytest.approx(0.012)
        assert r.mean_control == pytest.approx(0.10)
        assert r.mean_treatment == pytest.approx(0.112)
        assert r.relative_estimate == pytest.approx(0.12)
        assert r.ci_lower < r.estimate < r.ci_upper

    def test_welch_recovers_a_known_shift(self):
        rng = np.random.default_rng(3)
        r = welch_test(
            control=rng.normal(10, 2, 40_000), treatment=rng.normal(10.3, 2, 40_000), alpha=ALPHA
        )
        assert r.estimate == pytest.approx(0.3, abs=0.05)
        assert r.ci_lower <= 0.3 <= r.ci_upper

    def test_welch_handles_unequal_variances(self):
        """The reason Welch rather than Student: a treatment often moves both."""
        rng = np.random.default_rng(4)
        r = welch_test(
            control=rng.normal(10, 1, 5_000), treatment=rng.normal(10, 5, 20_000), alpha=ALPHA
        )
        assert not r.significant
        assert r.method == "Welch t"

    def test_z_test_shared_by_ratio_and_cuped(self):
        r = z_test(
            estimate=0.05,
            standard_error=0.01,
            alpha=ALPHA,
            method="custom",
            n_control=100,
            n_treatment=100,
            baseline=1.0,
        )
        assert r.p_value < 1e-6
        assert r.ci_lower == pytest.approx(0.05 - 1.959963984540054 * 0.01)
        assert r.relative_estimate == pytest.approx(0.05)


class TestValidation:
    def test_degenerate_binary_arms_error_rather_than_return_nan(self):
        with pytest.raises(ValueError, match="zero standard error"):
            two_proportion_test(
                successes_control=0,
                n_control=100,
                successes_treatment=0,
                n_treatment=100,
                alpha=ALPHA,
            )

    def test_successes_cannot_exceed_n(self):
        with pytest.raises(ValueError, match="between 0 and n"):
            two_proportion_test(
                successes_control=101,
                n_control=100,
                successes_treatment=50,
                n_treatment=100,
                alpha=ALPHA,
            )

    def test_welch_needs_two_observations(self):
        with pytest.raises(ValueError, match="at least two"):
            welch_test(control=np.array([1.0]), treatment=np.array([1.0, 2.0]), alpha=ALPHA)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5])
    def test_rejects_invalid_alpha(self, alpha):
        with pytest.raises(ValueError, match="alpha must be"):
            welch_test(control=np.zeros(10), treatment=np.ones(10), alpha=alpha)
