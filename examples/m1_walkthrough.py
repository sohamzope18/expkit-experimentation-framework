"""Runnable walkthrough of what M1 delivers: design (§3.1) and assignment (§3.2).

    make setup && .venv/bin/python examples/m1_walkthrough.py

Everything printed here is computed live. Nothing is hard-coded.
"""

import itertools
import subprocess
import sys

import numpy as np
from scipy.stats import chi2_contingency, kstest

from expkit.design.assignment import assign, assign_many, assignment_score
from expkit.design.power import (
    mde_binary,
    mde_continuous,
    power_for_n_binary,
    required_n_binary,
    required_n_continuous,
    required_n_ratio,
)

ALPHA, POWER = 0.05, 0.80


def rule(title):
    print(f"\n{'=' * 74}\n {title}\n{'=' * 74}")


# --------------------------------------------------------------------------- #
rule("1. SAMPLE SIZING — three metric types, two-sided, no continuity correction")

binary = required_n_binary(p_control=0.10, relative_effect=0.20, alpha=ALPHA, power=POWER)
print("  BINARY   conversion 10.0% -> 12.0% (+20% relative)")
print(
    f"           {binary.n_control:>7,} control  {binary.n_treatment:>7,} treatment"
    f"  {binary.n_total:>8,} total"
)

continuous = required_n_continuous(
    baseline_mean=12.40, variance=45.0, relative_effect=0.03, alpha=ALPHA, power=POWER
)
print("\n  CONTINUOUS   revenue/user  mean 12.40, var 45.0, +3% relative")
print(
    f"           {continuous.n_control:>7,} control  {continuous.n_treatment:>7,} treatment"
    f"  {continuous.n_total:>8,} total"
)

ratio = required_n_ratio(
    baseline_ratio=2.85, unit_variance=6.1, relative_effect=0.05, alpha=ALPHA, power=POWER
)
print("\n  RATIO    clicks/session  R=2.85, delta-method unit variance 6.1, +5% relative")
print(
    f"           {ratio.n_control:>7,} control  {ratio.n_treatment:>7,} treatment"
    f"  {ratio.n_total:>8,} total"
)
print("           (unit_variance comes from expkit.metrics.ratio in M3 — the delta-method")
print("            formula has one home and is not duplicated in the design module)")

# --------------------------------------------------------------------------- #
rule("2. ROUND-TRIP — power(required_n) must return the power you asked for")

achieved = power_for_n_binary(
    n_control=binary.n_control, p_control=0.10, absolute_effect=0.02, alpha=ALPHA
)
print(f"  required_n_binary(power={POWER})           -> n_control = {binary.n_control:,}")
print(f"  power_for_n_binary(n={binary.n_control:,})            -> power     = {achieved:.4f}")

delta, rel = mde_binary(n_control=binary.n_control, p_control=0.10, alpha=ALPHA, power=POWER)
print(
    f"  mde_binary(n={binary.n_control:,})                    -> {delta:.5f} absolute"
    f" ({rel:+.2%} relative)"
)
mde_c, _ = mde_continuous(n_control=continuous.n_control, variance=45.0, alpha=ALPHA, power=POWER)
print(
    f"  mde_continuous(n={continuous.n_control:,})                -> {mde_c:.5f} absolute"
    f" (asked for {0.03 * 12.40:.5f})"
)

# --------------------------------------------------------------------------- #
rule("3. THE D8 TRAP — pooled-null variance vs alternative variance")

from scipy.stats import norm  # noqa: E402

z_a, z_b = norm.ppf(1 - ALPHA / 2), norm.ppf(POWER)


def collapsed(p_c, d, k=1.0):
    """The common mistake: one pooled variance in both terms."""
    p_t = p_c + d
    p_bar = (p_c + k * p_t) / (1 + k)
    return (z_a + z_b) ** 2 * (1 + 1 / k) * p_bar * (1 - p_bar) / d**2


def correct(p_c, d, k=1.0):
    return required_n_binary(
        p_control=p_c, absolute_effect=d, alpha=ALPHA, power=POWER, allocation_ratio=k
    ).n_control


print("  Equal allocation — p(1-p) is concave, so the collapse is always conservative:")
print(f"  {'baseline':>10} {'effect':>8} {'correct':>10} {'collapsed':>10} {'error':>9}")
for p_c, d in [(0.10, 0.02), (0.05, 0.15), (0.50, 0.40)]:
    c, x = correct(p_c, d), collapsed(p_c, d)
    print(f"  {p_c:>10.2f} {d:>8.2f} {c:>10,} {x:>10,.0f} {x / c - 1:>+8.2%}")

print("\n  Unequal allocation — the weights on the two terms transpose, Jensen dies,")
print("  and the same mistake now UNDER-sizes the experiment:")
print(f"  {'kappa':>10} {'split':>10} {'correct':>10} {'collapsed':>10} {'error':>9}")
for k in [0.10, 0.25, 1.00, 4.00]:
    c, x = correct(0.05, 0.15, k), collapsed(0.05, 0.15, k)
    share = k / (1 + k)
    print(
        f"  {k:>10.2f} {f'{1 - share:.0%}/{share:.0%}':>10} {c:>10,} {x:>10,.0f}"
        f" {x / c - 1:>+8.2%}"
    )
print("\n  At kappa=0.10 the collapsed formula asks for 28% fewer control units than")
print("  needed. The experiment runs, reports 'not significant', and the team")
print("  concludes the feature does nothing.")

# --------------------------------------------------------------------------- #
rule("4. ASSIGNMENT — determinism across processes")

code = (
    "from expkit.design.assignment import assignment_score;"
    "print([round(assignment_score(f'unit-{i}', 'exp-checkout'), 12) for i in range(4)])"
)
runs = {}
for seed in ("0", "12345"):
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
    ).stdout.strip()
    runs[seed] = out
    print(f"  PYTHONHASHSEED={seed:<6} {out}")
print(f"  identical: {runs['0'] == runs['12345']}")
print("  builtin hash() would differ here — silently, with no error, in production.")

# --------------------------------------------------------------------------- #
rule("5. ASSIGNMENT — arbitrary splits hit their target within Monte Carlo error")

units = [f"unit-{i}" for i in range(20_000)]
for alloc in (
    {"control": 0.50, "treatment": 0.50},
    {"control": 0.90, "treatment": 0.10},
    {"control": 0.40, "variant_a": 0.30, "variant_b": 0.30},
):
    arms = assign_many(units, "exp-ramp", alloc)
    parts = []
    for arm, target in alloc.items():
        realized = float(np.mean(arms == arm))
        se = np.sqrt(target * (1 - target) / len(units))
        drift = abs(realized - target) / se
        parts.append(f"{arm}={realized:.4f} (target {target:.2f}, {drift:.1f} SE)")
    print("  " + "  ".join(parts))

scores = np.array([assignment_score(u, "exp-ramp") for u in units])
print(
    f"\n  scores uniform on [0,1): KS p = {kstest(scores, 'uniform').pvalue:.3f}"
    f"   range [{scores.min():.5f}, {scores.max():.5f}]"
)
sample, ab = units[:500], {"control": 0.5, "treatment": 0.5}
agrees = all(
    assign(u, "exp-ramp", ab) == a
    for u, a in zip(sample, assign_many(sample, "exp-ramp", ab), strict=True)
)
print(f"  scalar assign() agrees with vectorized assign_many(): {agrees}")

# --------------------------------------------------------------------------- #
rule("6. ASSIGNMENT — different experiments assign independently")

n_units, salts = 2000, [f"exp-{i}" for i in range(46)]
ids = [f"unit-{i}" for i in range(n_units)]
assigned = {
    s: (assign_many(ids, s, {"control": 0.5, "treatment": 0.5}) == "treatment").astype(int)
    for s in salts
}
p_values, phis = [], []
for x, y in itertools.combinations(salts, 2):
    a, b = assigned[x], assigned[y]
    table = np.array(
        [
            [np.sum((a == 0) & (b == 0)), np.sum((a == 0) & (b == 1))],
            [np.sum((a == 1) & (b == 0)), np.sum((a == 1) & (b == 1))],
        ]
    )
    r = chi2_contingency(table, correction=False)
    p_values.append(r.pvalue)
    phis.append(np.sqrt(r.statistic / n_units))
p_values, phis = np.array(p_values), np.array(phis)
expected_phi = np.sqrt(2 / (np.pi * n_units))
print(f"  experiment pairs examined:      {len(p_values):,}")
print(f"  pairs rejecting at alpha=0.05:  {(p_values < 0.05).mean():.4f}   (expect 0.05)")
print(f"  mean |phi| across pairs:        {phis.mean():.4f}   (expect {expected_phi:.4f})")
print(f"  max  |phi| across pairs:        {phis.max():.4f}")
print("\n  A unit's arm in one experiment tells you nothing about its arm in another.")
print(f"\n{'=' * 74}\n")
