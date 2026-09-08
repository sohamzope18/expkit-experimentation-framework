"""Asymptotic confidence sequences: shape, tuning, and anytime validity."""

import math

import numpy as np
import pytest

from expkit.inference.fixed import two_proportion_test
from expkit.inference.sequential import asympcs, asympcs_half_width, rho_for_target

ALPHA = 0.05
Z = 1.959963984540054


class TestWidth:
    def test_wider_than_the_fixed_horizon_interval(self):
        """Anytime validity is not free; if it were, nobody would use fixed tests."""
        rho = rho_for_target(target_n=10_000, alpha=ALPHA)
        seq = asympcs_half_width(t=10_000, sigma=1.0, alpha=ALPHA, rho=rho)
        assert seq > Z / math.sqrt(10_000)

    def test_tightest_at_the_tuned_sample_size(self):
        """Tightest *relative to* the fixed-horizon interval, not in absolute width.

        Absolute width always shrinks with more data. What rho controls is where
        the anytime-validity penalty is smallest, so the comparison has to be
        against the fixed-horizon interval at the same t.
        """
        rho = rho_for_target(target_n=10_000, alpha=ALPHA)

        def penalty(t):
            return asympcs_half_width(t=t, sigma=1.0, alpha=ALPHA, rho=rho) / (Z / math.sqrt(t))

        at_target = penalty(10_000)
        for t in (2_000, 5_000, 20_000, 100_000):
            assert penalty(t) > at_target

    def test_width_ratio_is_scale_invariant(self):
        """The penalty over a fixed interval does not depend on the planned size."""
        ratios = []
        for target in (1_000, 10_000, 100_000, 1_000_000):
            rho = rho_for_target(target_n=target, alpha=ALPHA)
            seq = asympcs_half_width(t=target, sigma=1.0, alpha=ALPHA, rho=rho)
            ratios.append(seq / (Z / math.sqrt(target)))
        assert max(ratios) - min(ratios) < 1e-6
        assert ratios[0] == pytest.approx(1.549, abs=0.01)

    def test_shrinks_with_sample_size(self):
        rho = rho_for_target(target_n=10_000, alpha=ALPHA)
        widths = [
            asympcs_half_width(t=t, sigma=1.0, alpha=ALPHA, rho=rho)
            for t in (10_000, 50_000, 250_000)
        ]
        assert widths[0] > widths[1] > widths[2]

    def test_tighter_alpha_widens_the_interval(self):
        rho = rho_for_target(target_n=10_000, alpha=0.05)
        loose = asympcs_half_width(t=10_000, sigma=1.0, alpha=0.10, rho=rho)
        tight = asympcs_half_width(t=10_000, sigma=1.0, alpha=0.01, rho=rho)
        assert tight > loose


class TestRhoSelection:
    def test_rho_scales_inversely_with_root_n(self):
        r1 = rho_for_target(target_n=1_000, alpha=ALPHA)
        r2 = rho_for_target(target_n=100_000, alpha=ALPHA)
        assert r1 / r2 == pytest.approx(10.0, rel=0.02)

    def test_rho_actually_minimizes_the_width(self):
        rho = rho_for_target(target_n=10_000, alpha=ALPHA)
        best = asympcs_half_width(t=10_000, sigma=1.0, alpha=ALPHA, rho=rho)
        for factor in (0.5, 0.8, 1.25, 2.0):
            assert asympcs_half_width(t=10_000, sigma=1.0, alpha=ALPHA, rho=rho * factor) >= best


class TestAnytimeValidity:
    def test_false_positive_rate_stays_below_alpha_under_repeated_looks(self):
        """The property the whole method exists for.

        200 null experiments, each inspected at 10 checkpoints, stopping at the
        first crossing. A fixed-horizon test used this way rejects far above 5%;
        this must not. Deterministic seed, so it cannot flake.
        """
        rng = np.random.default_rng(99)
        n_per_look, looks, reps = 3_000, 10, 200
        rho = rho_for_target(target_n=2 * n_per_look * looks, alpha=ALPHA)

        seq_hits = peek_hits = 0
        for _ in range(reps):
            cum_c = np.cumsum(rng.binomial(n_per_look, 0.1, looks))
            cum_t = np.cumsum(rng.binomial(n_per_look, 0.1, looks))
            crossed = peeked = False
            for look in range(looks):
                n = n_per_look * (look + 1)
                fixed = two_proportion_test(
                    successes_control=int(cum_c[look]),
                    n_control=n,
                    successes_treatment=int(cum_t[look]),
                    n_treatment=n,
                    alpha=ALPHA,
                )
                peeked = peeked or fixed.significant
                crossed = (
                    crossed
                    or asympcs(
                        estimate=fixed.estimate,
                        standard_error=fixed.standard_error,
                        n_total=2 * n,
                        alpha=ALPHA,
                        rho=rho,
                    ).excludes_null
                )
            seq_hits += crossed
            peek_hits += peeked
        assert seq_hits / reps <= ALPHA
        assert peek_hits / reps > 2 * ALPHA  # the problem being solved

    def test_excludes_null_detects_a_real_effect(self):
        rho = rho_for_target(target_n=40_000, alpha=ALPHA)
        result = asympcs(estimate=0.02, standard_error=0.004, n_total=40_000, alpha=ALPHA, rho=rho)
        assert result.excludes_null
        assert result.ci_lower > 0

    def test_does_not_cross_on_a_null_estimate(self):
        rho = rho_for_target(target_n=40_000, alpha=ALPHA)
        assert not asympcs(
            estimate=0.0001, standard_error=0.004, n_total=40_000, alpha=ALPHA, rho=rho
        ).excludes_null


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(t=0, sigma=1.0, alpha=0.05, rho=0.01), "t must be positive"),
            (dict(t=100, sigma=0.0, alpha=0.05, rho=0.01), "sigma must be positive"),
            (dict(t=100, sigma=1.0, alpha=0.0, rho=0.01), "alpha must be"),
            (dict(t=100, sigma=1.0, alpha=0.05, rho=0.0), "rho must be positive"),
        ],
    )
    def test_rejects_bad_inputs(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            asympcs_half_width(**kwargs)

    def test_rejects_non_positive_target(self):
        with pytest.raises(ValueError, match="target_n must be positive"):
            rho_for_target(target_n=0, alpha=ALPHA)
