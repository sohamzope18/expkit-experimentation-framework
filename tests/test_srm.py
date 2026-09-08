"""SRM must be a hard stop, not a warning (D1)."""

import pytest

from expkit.validity.srm import SRM_THRESHOLD, SRMFailure, check_srm


class TestHardStop:
    def test_balanced_counts_pass(self):
        result = check_srm(
            {"control": 50_120, "treatment": 49_880}, {"control": 0.5, "treatment": 0.5}
        )
        assert result.passed
        assert result.p_value >= SRM_THRESHOLD

    def test_mismatch_raises_rather_than_warns(self):
        with pytest.raises(SRMFailure):
            check_srm({"control": 50_600, "treatment": 49_400}, {"control": 0.5, "treatment": 0.5})

    def test_failure_carries_the_evidence(self):
        """A caller must be able to report counts without re-running the check."""
        with pytest.raises(SRMFailure) as exc:
            check_srm({"control": 50_600, "treatment": 49_400}, {"control": 0.5, "treatment": 0.5})
        result = exc.value.result
        assert not result.passed
        assert result.observed == {"control": 50_600, "treatment": 49_400}
        assert result.expected == {"control": 50_000.0, "treatment": 50_000.0}
        assert result.worst_arm == "control"
        assert result.worst_relative_deviation == pytest.approx(0.012)
        assert "FAILED" in result.detail

    def test_inspection_mode_does_not_raise(self):
        result = check_srm(
            {"control": 50_600, "treatment": 49_400},
            {"control": 0.5, "treatment": 0.5},
            raise_on_failure=False,
        )
        assert not result.passed

    def test_threshold_is_far_stricter_than_conventional(self):
        """D1: 0.001, not 0.05. This check runs on every experiment."""
        assert SRM_THRESHOLD == 0.001

    def test_deviation_that_trips_0_05_but_not_0_001_passes(self):
        """The gap between the two thresholds is the false-alarm budget."""
        counts = {"control": 50_400, "treatment": 49_600}  # p ~ 0.011
        loose = check_srm(
            counts, {"control": 0.5, "treatment": 0.5}, threshold=0.05, raise_on_failure=False
        )
        assert not loose.passed
        assert check_srm(counts, {"control": 0.5, "treatment": 0.5}).passed


class TestUnequalAllocation:
    def test_ninety_ten_split_passes_when_realized(self):
        assert check_srm(
            {"control": 90_100, "treatment": 9_900}, {"control": 0.9, "treatment": 0.1}
        ).passed

    def test_ninety_ten_split_fails_when_treatment_leaks(self):
        with pytest.raises(SRMFailure):
            check_srm({"control": 90_000, "treatment": 10_600}, {"control": 0.9, "treatment": 0.1})

    def test_three_arms(self):
        assert check_srm(
            {"a": 33_400, "b": 33_300, "c": 33_300}, {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
        ).passed


class TestValidation:
    def test_arm_mismatch_is_an_error_not_a_failure(self):
        with pytest.raises(ValueError, match="do not match"):
            check_srm({"control": 100}, {"control": 0.5, "treatment": 0.5})

    def test_allocation_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            check_srm({"a": 100, "b": 100}, {"a": 0.5, "b": 0.4})

    def test_rejects_empty_experiment(self):
        with pytest.raises(ValueError, match="no units"):
            check_srm({"a": 0, "b": 0}, {"a": 0.5, "b": 0.5})

    @pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1])
    def test_rejects_invalid_threshold(self, threshold):
        with pytest.raises(ValueError, match="threshold must be"):
            check_srm({"a": 100, "b": 100}, {"a": 0.5, "b": 0.5}, threshold=threshold)
