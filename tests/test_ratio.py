"""Delta-method ratio variance, and the naive estimator it replaces."""

import numpy as np
import pytest

from expkit.metrics.ratio import naive_ratio_estimate, ratio_estimate, ratio_test


def clustered_arm(rng, n_units=4_000, sessions_mean=6.0, lift=0.0):
    """Users with personal click propensities: sessions within a user correlate."""
    propensity = np.clip(rng.beta(2.0, 8.0, n_units) * (1 + lift), 0, 1)
    sessions = 1 + rng.poisson(sessions_mean - 1.0, n_units)
    return rng.binomial(sessions, propensity).astype(float), sessions.astype(float)


class TestDeltaMethod:
    def test_ratio_is_the_aggregate_not_the_mean_of_ratios(self):
        num = np.array([1.0, 0.0, 3.0])
        den = np.array([2.0, 5.0, 3.0])
        assert ratio_estimate(num, den).ratio == pytest.approx(4.0 / 10.0)

    def test_variance_matches_the_taylor_expansion_form(self):
        """Linearized Var(L)/n must equal the textbook moment expression."""
        rng = np.random.default_rng(11)
        num, den = clustered_arm(rng)
        est = ratio_estimate(num, den)

        n = num.size
        r, d_bar = est.ratio, den.mean()
        var_n, var_d = num.var(ddof=1), den.var(ddof=1)
        cov = np.cov(num, den, ddof=1)[0, 1]
        textbook = (var_n - 2 * r * cov + r**2 * var_d) / (n * d_bar**2)
        assert est.variance == pytest.approx(textbook, rel=1e-10)

    def test_analytic_variance_matches_empirical_spread(self):
        """The claim that matters: does the formula predict the real variance?"""
        rng = np.random.default_rng(12)
        ratios, ses = [], []
        for _ in range(1_500):
            num, den = clustered_arm(rng)
            est = ratio_estimate(num, den)
            ratios.append(est.ratio)
            ses.append(est.standard_error)
        empirical = float(np.std(ratios, ddof=1))
        predicted = float(np.mean(ses))
        assert predicted == pytest.approx(empirical, rel=0.06)

    def test_unit_variance_feeds_the_design_module(self):
        rng = np.random.default_rng(13)
        num, den = clustered_arm(rng)
        est = ratio_estimate(num, den)
        assert est.variance == pytest.approx(est.unit_variance / est.n_units)


class TestNaiveEstimator:
    def test_naive_understates_the_standard_error_under_clustering(self):
        rng = np.random.default_rng(14)
        num, den = clustered_arm(rng, sessions_mean=8.0)
        assert naive_ratio_estimate(num, den).standard_error < (
            ratio_estimate(num, den).standard_error
        )

    def test_both_estimators_agree_on_the_point_estimate(self):
        """Only the variance differs. The bug is invisible in the headline number."""
        rng = np.random.default_rng(15)
        num, den = clustered_arm(rng)
        assert naive_ratio_estimate(num, den).ratio == pytest.approx(ratio_estimate(num, den).ratio)

    def test_naive_is_correct_when_there_is_no_clustering(self):
        """One session per user: the denominator unit IS the randomization unit."""
        rng = np.random.default_rng(16)
        den = np.ones(20_000)
        num = rng.binomial(1, 0.2, 20_000).astype(float)
        assert naive_ratio_estimate(num, den).standard_error == pytest.approx(
            ratio_estimate(num, den).standard_error, rel=0.02
        )

    def test_naive_rejects_non_rate_data(self):
        with pytest.raises(ValueError, match="at most 1"):
            naive_ratio_estimate(np.array([5.0, 1.0]), np.array([2.0, 3.0]))


class TestRatioTest:
    def test_detects_a_real_lift(self):
        rng = np.random.default_rng(17)
        n_c, d_c = clustered_arm(rng, n_units=30_000)
        n_t, d_t = clustered_arm(rng, n_units=30_000, lift=0.10)
        result = ratio_test(
            numerator_control=n_c,
            denominator_control=d_c,
            numerator_treatment=n_t,
            denominator_treatment=d_t,
            alpha=0.05,
        )
        assert result.significant
        assert result.estimate > 0

    def test_naive_flag_produces_a_narrower_interval(self):
        rng = np.random.default_rng(18)
        n_c, d_c = clustered_arm(rng, sessions_mean=8.0)
        n_t, d_t = clustered_arm(rng, sessions_mean=8.0)
        kwargs = dict(
            numerator_control=n_c,
            denominator_control=d_c,
            numerator_treatment=n_t,
            denominator_treatment=d_t,
            alpha=0.05,
        )
        correct = ratio_test(**kwargs)
        naive = ratio_test(**kwargs, naive=True)
        assert (naive.ci_upper - naive.ci_lower) < (correct.ci_upper - correct.ci_lower)


class TestValidation:
    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="align per unit"):
            ratio_estimate(np.ones(5), np.ones(4))

    def test_negative_denominator(self):
        with pytest.raises(ValueError, match="non-negative"):
            ratio_estimate(np.ones(3), np.array([1.0, -1.0, 2.0]))

    def test_zero_total_denominator(self):
        with pytest.raises(ValueError, match="total denominator"):
            ratio_estimate(np.zeros(3), np.zeros(3))
