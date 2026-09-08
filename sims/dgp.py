"""Data-generating processes with known ground truth.

Every process here exposes the true parameter it was built around, so a study can
check that an estimator recovers it. All randomness flows through an explicitly
passed ``numpy.random.Generator`` -- the legacy global ``numpy.random`` API is
banned repo-wide by ``tests/test_scaffold.py``, because its shared process state
makes results depend on execution order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "binary_arms",
    "revenue_arms",
    "RatioSample",
    "clustered_ratio_arm",
    "cuped_arms",
]


def binary_arms(
    rng: np.random.Generator,
    *,
    n_control: int,
    n_treatment: int,
    p_control: float,
    absolute_effect: float = 0.0,
    reps: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Success counts for both arms, vectorized over ``reps`` replications.

    Draws counts directly from the binomial rather than simulating individual
    Bernoulli trials: the count is a sufficient statistic for the rate, so this
    is exact, not an approximation, and it makes 10,000 replications instant.
    """
    p_treatment = p_control + absolute_effect
    control = rng.binomial(n_control, p_control, size=reps)
    treatment = rng.binomial(n_treatment, p_treatment, size=reps)
    return control, treatment


def revenue_arms(
    rng: np.random.Generator,
    *,
    n_control: int,
    n_treatment: int,
    reps: int,
    purchase_rate: float = 0.12,
    log_mean: float = 3.0,
    log_sd: float = 0.9,
    relative_effect: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Revenue per user: zero-inflated lognormal, the realistic hard case.

    Most users spend nothing; spenders follow a heavy right tail. This breaks the
    normality assumption a t-test is derived under, which is exactly why the Type I
    study uses it rather than drawing from a Gaussian -- testing a t-test on normal
    data proves nothing about the data it will actually meet.

    Returned arrays have shape ``(reps, n)``.
    """

    def draw(n: int, lift: float) -> np.ndarray:
        purchased = rng.random((reps, n)) < purchase_rate
        amounts = rng.lognormal(log_mean, log_sd, size=(reps, n))
        return np.where(purchased, amounts, 0.0) * (1.0 + lift)

    return draw(n_control, 0.0), draw(n_treatment, relative_effect)


@dataclass(frozen=True)
class RatioSample:
    """Per-randomization-unit numerator and denominator, plus the truth."""

    numerator: np.ndarray  # shape (reps, n_units)
    denominator: np.ndarray
    true_ratio: float


def clustered_ratio_arm(
    rng: np.random.Generator,
    *,
    n_units: int,
    reps: int,
    sessions_mean: float = 4.0,
    beta_a: float = 2.0,
    beta_b: float = 8.0,
    relative_effect: float = 0.0,
) -> RatioSample:
    """Clicks per session, randomized by user -- the unit-of-analysis trap.

    Each user draws a personal click propensity from Beta(a, b) and a random
    number of sessions. Sessions belonging to one user are therefore correlated,
    which is precisely what a session-level variance calculation ignores.

    Ground truth: with sessions independent of propensity,
    ``E[sum N] / E[sum D] = E[p] = a / (a + b)``.
    """
    if not 0.0 <= relative_effect:
        raise ValueError("relative_effect must be non-negative for a Beta propensity")
    propensity = rng.beta(beta_a, beta_b, size=(reps, n_units))
    propensity = np.clip(propensity * (1.0 + relative_effect), 0.0, 1.0)
    sessions = 1 + rng.poisson(sessions_mean - 1.0, size=(reps, n_units))
    clicks = rng.binomial(sessions, propensity)
    true_ratio = beta_a / (beta_a + beta_b) * (1.0 + relative_effect)
    return RatioSample(clicks.astype(float), sessions.astype(float), true_ratio)


def cuped_arms(
    rng: np.random.Generator,
    *,
    n_control: int,
    n_treatment: int,
    reps: int,
    rho: float,
    absolute_effect: float = 0.0,
    outcome_sd: float = 1.0,
    covariate_sd: float = 1.0,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Outcome and strictly pre-period covariate with correlation ``rho``.

    The covariate is drawn before any treatment is applied and the effect is added
    only to the outcome, so ``X`` is uncontaminated by construction. Returns
    ``((y_control, x_control), (y_treatment, x_treatment))``, each ``(reps, n)``.
    """
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must be in (-1, 1), got {rho}")

    def draw(n: int, effect: float) -> tuple[np.ndarray, np.ndarray]:
        x = rng.normal(0.0, covariate_sd, size=(reps, n))
        noise = rng.normal(0.0, outcome_sd * np.sqrt(1.0 - rho**2), size=(reps, n))
        y = rho * (outcome_sd / covariate_sd) * x + noise + effect
        return y, x

    return draw(n_control, 0.0), draw(n_treatment, absolute_effect)
