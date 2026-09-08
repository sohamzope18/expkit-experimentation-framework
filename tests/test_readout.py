"""End-to-end readout: the hard stop, the correction policy, and the D4 decision."""

import numpy as np
import pytest

from expkit.metrics.base import MetricRole, MetricSpec, MetricType
from expkit.readout import (
    BinaryObservation,
    ContinuousObservation,
    Decision,
    RatioObservation,
    readout,
)
from expkit.validity.srm import SRMFailure

N = 40_000
ALPHA = 0.05


def primary(successes_control=4_000, successes_treatment=4_180, n=N):
    return BinaryObservation(
        MetricSpec("checkout_conversion", MetricType.BINARY, MetricRole.PRIMARY),
        successes_control=successes_control,
        n_control=n,
        successes_treatment=successes_treatment,
        n_treatment=n,
    )


def guardrail(shift, n=N, seed=1):
    """A latency guardrail. Note that at n=40,000 per arm the standard error is
    about 0.85ms, so even a 1ms shift is detectable -- guardrail(0.0) is the only
    reliably null case, which is itself worth knowing when reading these tests."""
    rng = np.random.default_rng(seed)
    return ContinuousObservation(
        MetricSpec(
            "page_load_ms", MetricType.CONTINUOUS, MetricRole.GUARDRAIL, higher_is_better=False
        ),
        control=rng.normal(800.0, 120.0, n),
        treatment=rng.normal(800.0 + shift, 120.0, n),
    )


def run(observations, counts=None, **kwargs):
    return readout(
        observations=observations,
        intended_allocation={"control": 0.5, "treatment": 0.5},
        observed_counts=counts or {"control": N, "treatment": N},
        alpha=ALPHA,
        **kwargs,
    )


class TestHardStop:
    def test_srm_raises_before_any_estimate_is_produced(self):
        with pytest.raises(SRMFailure):
            run([primary()], counts={"control": N + 1_100, "treatment": N})

    def test_srm_failure_returns_no_readout_object(self):
        """There is no partial result to accidentally read an effect out of."""
        result = None
        try:
            result = run([primary()], counts={"control": N + 1_100, "treatment": N})
        except SRMFailure:
            pass
        assert result is None

    def test_passing_srm_is_recorded_on_the_readout(self):
        assert run([primary()]).srm.passed


class TestDecisionLogic:
    """D4, exactly as specified. No cases beyond these four."""

    def test_primary_up_and_no_guardrail_harm_ships(self):
        result = run([primary(), guardrail(0.0)])
        assert result.recommendation.decision is Decision.SHIP
        assert not result.recommendation.requires_human_override

    def test_primary_up_but_guardrail_down_blocks_and_escalates(self):
        """The case that separates real analysts from tutorial followers."""
        result = run([primary(), guardrail(12.0)])
        assert result.recommendation.decision is Decision.DO_NOT_SHIP
        assert result.recommendation.requires_human_override
        joined = " ".join(result.recommendation.reasons)
        assert "page_load_ms" in joined
        assert "checkout_conversion" in joined  # both sides are surfaced

    def test_primary_regression_blocks_without_escalation(self):
        result = run([primary(successes_treatment=3_820), guardrail(0.0)])
        assert result.recommendation.decision is Decision.DO_NOT_SHIP
        assert not result.recommendation.requires_human_override

    def test_flat_primary_is_inconclusive_not_no_effect(self):
        result = run([primary(successes_treatment=4_010), guardrail(0.0)])
        assert result.recommendation.decision is Decision.INCONCLUSIVE
        assert "not evidence of no effect" in result.recommendation.summary

    def test_guardrail_direction_respects_higher_is_better(self):
        """Latency going up is harm; conversion going up is not."""
        rng = np.random.default_rng(2)
        faster = ContinuousObservation(
            MetricSpec(
                "page_load_ms", MetricType.CONTINUOUS, MetricRole.GUARDRAIL, higher_is_better=False
            ),
            control=rng.normal(800.0, 120.0, N),
            treatment=rng.normal(788.0, 120.0, N),
        )
        result = run([primary(), faster])
        assert result.recommendation.decision is Decision.SHIP

    def test_guardrail_blocks_even_when_it_is_not_the_only_signal(self):
        result = run([primary(), guardrail(12.0), guardrail(15.0, seed=9)])
        assert result.recommendation.decision is Decision.DO_NOT_SHIP
        assert len(result.recommendation.reasons) == 3  # primary + two guardrails


class TestCorrectionPolicy:
    """D3: three families. Primary and guardrails uncorrected, secondaries BH."""

    def _mixed(self):
        return [
            primary(),
            guardrail(0.0),
            BinaryObservation(
                MetricSpec("addtocart", MetricType.BINARY, MetricRole.SECONDARY),
                successes_control=12_000,
                n_control=N,
                successes_treatment=12_180,
                n_treatment=N,
            ),
            BinaryObservation(
                MetricSpec("newsletter", MetricType.BINARY, MetricRole.SECONDARY),
                successes_control=800,
                n_control=N,
                successes_treatment=840,
                n_treatment=N,
            ),
        ]

    def test_primary_is_uncorrected(self):
        result = run(self._mixed())
        assert result.primary.adjusted_p_value == result.primary.test.p_value
        assert "single pre-registered" in result.primary.correction

    def test_guardrails_are_uncorrected(self):
        result = run(self._mixed())
        rail = result.by_role(MetricRole.GUARDRAIL)[0]
        assert rail.adjusted_p_value == rail.test.p_value
        assert "power to detect harm" in rail.correction

    def test_secondaries_are_bh_corrected(self):
        result = run(self._mixed())
        for metric in result.by_role(MetricRole.SECONDARY):
            assert metric.adjusted_p_value >= metric.test.p_value
            assert "Benjamini-Hochberg" in metric.correction

    def test_correction_only_counts_the_secondary_family(self):
        """Adding a guardrail must not weaken the secondary correction."""
        base = run(self._mixed())
        with_extra_rail = run(self._mixed() + [guardrail(0.0, seed=2)])
        for a, b in zip(
            base.by_role(MetricRole.SECONDARY),
            with_extra_rail.by_role(MetricRole.SECONDARY),
            strict=True,
        ):
            assert a.adjusted_p_value == pytest.approx(b.adjusted_p_value)


class TestMetricTypes:
    def test_ratio_metric_flows_through(self):
        rng = np.random.default_rng(3)

        def arm(lift):
            p = np.clip(rng.beta(2.0, 8.0, 10_000) * (1 + lift), 0, 1)
            d = 1 + rng.poisson(3.0, 10_000)
            return rng.binomial(d, p).astype(float), d.astype(float)

        n_c, d_c = arm(0.0)
        n_t, d_t = arm(0.10)
        result = run(
            [
                primary(),
                RatioObservation(
                    MetricSpec("clicks_per_session", MetricType.RATIO, MetricRole.SECONDARY),
                    numerator_control=n_c,
                    denominator_control=d_c,
                    numerator_treatment=n_t,
                    denominator_treatment=d_t,
                ),
            ]
        )
        assert result.by_role(MetricRole.SECONDARY)[0].significant

    def test_cuped_covariate_is_applied_and_reported(self):
        rng = np.random.default_rng(4)
        x_c, x_t = rng.normal(0, 1, N), rng.normal(0, 1, N)
        y_c = 0.8 * x_c + rng.normal(0, 0.6, N)
        y_t = 0.8 * x_t + rng.normal(0, 0.6, N) + 0.02
        result = run(
            [
                primary(),
                ContinuousObservation(
                    MetricSpec(
                        "revenue",
                        MetricType.CONTINUOUS,
                        MetricRole.SECONDARY,
                        cuped_covariate="pre_period_revenue",
                    ),
                    control=y_c,
                    treatment=y_t,
                    covariate_control=x_c,
                    covariate_treatment=x_t,
                ),
            ]
        )
        revenue = result.by_role(MetricRole.SECONDARY)[0]
        assert revenue.cuped is not None
        assert revenue.variance_reduction > 0.5

    def test_declared_covariate_without_data_is_an_error(self):
        with pytest.raises(ValueError, match="no covariate arrays"):
            run(
                [
                    primary(),
                    ContinuousObservation(
                        MetricSpec(
                            "revenue",
                            MetricType.CONTINUOUS,
                            MetricRole.SECONDARY,
                            cuped_covariate="pre_period_revenue",
                        ),
                        control=np.zeros(100),
                        treatment=np.ones(100),
                    ),
                ]
            )


class TestReadoutContents:
    def test_reports_run_statistics(self):
        result = run([primary()], duration_days=14.0)
        assert result.n_control == N
        assert result.n_treatment == N
        assert result.realized_allocation == {"control": 0.5, "treatment": 0.5}
        assert result.duration_days == 14.0
        assert result.alpha == ALPHA

    def test_balance_results_are_attached_as_warnings_when_imbalanced(self):
        rng = np.random.default_rng(6)
        result = run(
            [primary()],
            balance_covariates={"pre_spend": (rng.normal(0, 1, 5_000), rng.normal(0.4, 1, 5_000))},
        )
        assert len(result.balance) == 1
        assert not result.balance[0].balanced
        assert result.warnings  # reported, but the readout still returned


class TestValidation:
    def test_requires_exactly_one_primary(self):
        with pytest.raises(ValueError, match="exactly one primary"):
            run([primary(), primary()])

    def test_requires_at_least_one_primary(self):
        with pytest.raises(ValueError, match="exactly one primary"):
            run([guardrail(0.0)])

    def test_requires_at_least_one_metric(self):
        with pytest.raises(ValueError, match="at least one metric"):
            run([])

    def test_rejects_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha must be"):
            readout(
                observations=[primary()],
                intended_allocation={"control": 0.5, "treatment": 0.5},
                observed_counts={"control": N, "treatment": N},
                alpha=1.5,
            )
