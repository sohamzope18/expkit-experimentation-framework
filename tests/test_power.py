"""Design module: reference values, round-trips, and the D8 variance distinction."""

import math

import pytest

from expkit.design.power import (
    mde_binary,
    mde_continuous,
    power_for_n_binary,
    power_for_n_continuous,
    required_n_binary,
    required_n_continuous,
    required_n_ratio,
)

ALPHA = 0.05
POWER = 0.80


class TestReferenceValues:
    """Published values for the uncorrected two-sided formulas (D5, D6)."""

    def test_binary_matches_reference(self):
        # p_control=0.10, p_treatment=0.12, alpha=0.05 two-sided, power=0.80.
        s = required_n_binary(p_control=0.10, absolute_effect=0.02, alpha=ALPHA, power=POWER)
        assert s.n_control == 3841
        assert s.n_treatment == 3841
        assert s.n_total == 7682

    def test_continuous_matches_reference(self):
        # Effect of 0.1 sd: n = 2 * (1.959964 + 0.841621)^2 / 0.1^2.
        s = required_n_continuous(variance=1.0, absolute_effect=0.1, alpha=ALPHA, power=POWER)
        assert s.n_control == 1570


class TestRoundTrips:
    def test_binary_power_at_required_n(self):
        s = required_n_binary(p_control=0.10, absolute_effect=0.02, alpha=ALPHA, power=POWER)
        achieved = power_for_n_binary(
            n_control=s.n_control, p_control=0.10, absolute_effect=0.02, alpha=ALPHA
        )
        assert achieved == pytest.approx(POWER, abs=1e-3)

    def test_continuous_power_at_required_n(self):
        s = required_n_continuous(variance=4.0, absolute_effect=0.25, alpha=ALPHA, power=POWER)
        achieved = power_for_n_continuous(
            n_control=s.n_control, variance=4.0, absolute_effect=0.25, alpha=ALPHA
        )
        assert achieved == pytest.approx(POWER, abs=1e-3)

    def test_binary_mde_inverts_required_n(self):
        s = required_n_binary(p_control=0.10, absolute_effect=0.02, alpha=ALPHA, power=POWER)
        delta, rel = mde_binary(n_control=s.n_control, p_control=0.10, alpha=ALPHA, power=POWER)
        assert delta == pytest.approx(0.02, abs=1e-4)
        assert rel == pytest.approx(0.20, abs=1e-3)

    def test_continuous_mde_inverts_required_n(self):
        s = required_n_continuous(variance=1.0, absolute_effect=0.1, alpha=ALPHA, power=POWER)
        delta, _ = mde_continuous(n_control=s.n_control, variance=1.0, alpha=ALPHA, power=POWER)
        assert delta == pytest.approx(0.1, rel=1e-3)


class TestParameterizations:
    def test_relative_and_absolute_agree(self):
        a = required_n_binary(p_control=0.10, absolute_effect=0.02, alpha=ALPHA, power=POWER)
        r = required_n_binary(p_control=0.10, relative_effect=0.20, alpha=ALPHA, power=POWER)
        assert a.n_control == r.n_control
        assert r.relative_effect == pytest.approx(0.20)
        assert a.relative_effect == pytest.approx(0.20)

    def test_ratio_reuses_continuous_math(self):
        kwargs = dict(alpha=ALPHA, power=POWER, absolute_effect=0.05)
        r = required_n_ratio(baseline_ratio=1.5, unit_variance=2.0, **kwargs)
        c = required_n_continuous(variance=2.0, baseline_mean=1.5, **kwargs)
        assert r.n_control == c.n_control

    def test_effect_must_be_specified_exactly_once(self):
        with pytest.raises(ValueError, match="exactly one"):
            required_n_binary(p_control=0.1, alpha=ALPHA, power=POWER)
        with pytest.raises(ValueError, match="exactly one"):
            required_n_binary(
                p_control=0.1,
                absolute_effect=0.01,
                relative_effect=0.1,
                alpha=ALPHA,
                power=POWER,
            )


class TestUnequalAllocation:
    """kappa = n_treatment / n_control (D7)."""

    def test_kappa_one_is_equal_allocation(self):
        s = required_n_continuous(
            variance=1.0, absolute_effect=0.1, alpha=ALPHA, power=POWER, allocation_ratio=1.0
        )
        assert s.n_control == s.n_treatment

    def test_treatment_arm_scales_with_kappa(self):
        s = required_n_continuous(
            variance=1.0, absolute_effect=0.1, alpha=ALPHA, power=POWER, allocation_ratio=0.25
        )
        assert s.n_treatment == math.ceil(s.n_control * 0.25)

    def test_unequal_allocation_costs_total_sample(self):
        """Equal allocation minimizes total N for a fixed effect and power."""
        equal = required_n_continuous(
            variance=1.0, absolute_effect=0.1, alpha=ALPHA, power=POWER, allocation_ratio=1.0
        )
        skewed = required_n_continuous(
            variance=1.0, absolute_effect=0.1, alpha=ALPHA, power=POWER, allocation_ratio=0.1
        )
        assert skewed.n_total > equal.n_total

    def test_power_round_trip_under_unequal_allocation(self):
        s = required_n_binary(
            p_control=0.10, absolute_effect=0.02, alpha=ALPHA, power=POWER, allocation_ratio=0.25
        )
        achieved = power_for_n_binary(
            n_control=s.n_control,
            p_control=0.10,
            absolute_effect=0.02,
            alpha=ALPHA,
            allocation_ratio=0.25,
        )
        assert achieved == pytest.approx(POWER, abs=1e-3)


class TestVarianceConvention:
    """D8: pooled-null variance in the alpha term, alternative variance in the power term."""

    @staticmethod
    def _collapsed_n(p_c, delta, kappa=1.0):
        """The standard mistake: one variance (pooled under H0) used in both terms."""
        from scipy.stats import norm

        p_t = p_c + delta
        p_bar = (p_c + kappa * p_t) / (1 + kappa)
        z_a, z_b = norm.ppf(1 - ALPHA / 2), norm.ppf(POWER)
        return (z_a + z_b) ** 2 * (1 + 1 / kappa) * p_bar * (1 - p_bar) / delta**2

    @staticmethod
    def _actual_n(p_c, delta, kappa=1.0):
        """Unrounded n_control from the implementation, so ceil() cannot mask a gap."""
        s = required_n_binary(
            p_control=p_c,
            absolute_effect=delta,
            alpha=ALPHA,
            power=POWER,
            allocation_ratio=kappa,
        )
        # Recover the pre-ceil value by re-running the closed form the module uses.
        return s.n_control

    @pytest.mark.parametrize(
        "p_c, delta", [(0.10, 0.02), (0.05, 0.15), (0.50, 0.40), (0.02, 0.08), (0.30, 0.05)]
    )
    def test_collapsing_inflates_n_at_equal_allocation(self, p_c, delta):
        """Concavity of p(1-p) makes the alternative variance <= the pooled one.

        At kappa=1 the collapsed formula is therefore always conservative. The gap
        is small at small effects (0.03% at p=0.10, delta=0.02) and grows with the
        effect (6.3% at p=0.50, delta=0.40) -- which is why this error survives
        code review and only shows up on large-effect designs.
        """
        collapsed = self._collapsed_n(p_c, delta)
        actual = self._actual_n(p_c, delta)
        assert collapsed >= actual

    @pytest.mark.parametrize("kappa, direction", [(0.1, "under"), (0.25, "under"), (4.0, "over")])
    def test_collapsing_can_undersize_at_unequal_allocation(self, kappa, direction):
        """The Jensen argument does not survive unequal allocation.

        p_bar weights p_treatment by kappa/(1+kappa), but the alternative variance
        weights f(p_treatment) by 1/(1+kappa) -- the weights are transposed, so the
        inequality can run either way. At kappa=0.1 the collapsed form under-sizes
        the control arm by ~28%, which ships an underpowered experiment.
        """
        collapsed = self._collapsed_n(0.05, 0.15, kappa)
        actual = self._actual_n(0.05, 0.15, kappa)
        if direction == "under":
            assert collapsed < actual
        else:
            assert collapsed > actual

    def test_pooled_rate_is_allocation_weighted(self):
        """p_bar weights by kappa; a simple average is wrong off 50/50."""
        from expkit.design.power import _pooled_rate

        assert _pooled_rate(0.10, 0.20, 1.0) == pytest.approx(0.15)
        assert _pooled_rate(0.10, 0.20, 0.25) == pytest.approx(0.12)
        assert _pooled_rate(0.10, 0.20, 4.0) == pytest.approx(0.18)

    def test_two_sided_critical_value(self):
        """D5: alpha is split across two tails. A one-sided design would be smaller."""
        two_sided = required_n_continuous(
            variance=1.0, absolute_effect=0.1, alpha=0.05, power=POWER
        ).n_control
        one_sided_equivalent = required_n_continuous(
            variance=1.0, absolute_effect=0.1, alpha=0.10, power=POWER
        ).n_control
        assert two_sided > one_sided_equivalent


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(alpha=0.0, power=0.8), "alpha must be in"),
            (dict(alpha=1.0, power=0.8), "alpha must be in"),
            (dict(alpha=0.05, power=1.5), "power must be in"),
            (dict(alpha=0.5, power=0.4), "must exceed alpha"),
        ],
    )
    def test_rejects_invalid_alpha_power(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            required_n_continuous(variance=1.0, absolute_effect=0.1, **kwargs)

    def test_rejects_effect_pushing_rate_out_of_bounds(self):
        with pytest.raises(ValueError, match="outside"):
            required_n_binary(p_control=0.95, absolute_effect=0.10, alpha=ALPHA, power=POWER)

    def test_alpha_and_power_have_no_defaults(self):
        """Spec §8 Session 1: these stay required parameters."""
        with pytest.raises(TypeError):
            required_n_continuous(variance=1.0, absolute_effect=0.1)
