# DECISIONS

Every non-obvious choice in this repository, with its rationale and the alternative that was
rejected. Nothing in `src/` implements a threshold, estimator, or stopping rule that is not
recorded here first.

Status legend: **LOCKED** = owner-confirmed. **PROPOSED** = recommended, awaiting owner sign-off
before the milestone that consumes it.

---

## D1 — SRM hard-stop threshold — LOCKED

**Decision:** `readout()` raises when the SRM chi-square p-value falls below **0.001**.

**Rationale:** the SRM check runs on *every* experiment, so its false-alarm rate compounds with
experiment volume. At roughly 1,000 experiments per year, α = 0.001 yields about one spurious hard
stop annually — rare enough that a trip is treated as real and investigated. At the conventional
0.05 the same volume produces ~50 false stops a year, analysts learn to override the check, and a
hard stop stops meaning anything.

**Why it is a stop and not a warning:** SRM implies the randomization or the logging is broken.
A broken assignment means the arms are not exchangeable, which means the comparison is not causal,
which means the effect estimate is meaningless *regardless of its p-value*. Reporting an effect
alongside a warning invites someone to read the effect.

**Rejected:** 0.01 (more sensitive to genuine leaks, but ~10 false stops per 1,000 experiments);
0.0001 (near-zero false alarms, at the cost of missing mild real mismatch such as a small
bot-filtering leak).

**Consumed by:** M2, `src/expkit/validity/srm.py`.

---

## D2 — Sequential method — PROPOSED

**Decision:** implement **asymptotic confidence sequences (AsympCS)** as the single always-valid
method. Do not also implement mSPRT.

**Rationale:** AsympCS applies to any asymptotically normal estimator. The ratio metric (delta-method
variance, D-M3) and the CUPED-adjusted outcome (M4) therefore flow through one code path rather than
three per-metric derivations, and any estimator added later inherits sequential inference for free.
It also carries no tuning constant that has to be justified.

**Rejected:** mSPRT. More widely recognized from the commercial platforms (Optimizely, VWO), but it
requires choosing and defending a mixing variance, and each metric type needs its own derivation.
The recognition is not worth the two extra things to defend.

**Cost being bought:** always-valid inference buys the right to stop at any time without inflating
the false-positive rate, and pays for it in power — for equal power it needs more samples than a
fixed-horizon test run to completion. `sims/study_peeking.py` quantifies the exchange rate.

**Consumed by:** M5, `src/expkit/inference/sequential.py`.

---

## D3 — Multiple-testing family definition — PROPOSED

**Decision:** three families.
- **Primary** — uncorrected. One pre-registered hypothesis; there is no multiplicity to correct.
- **Secondary** — Benjamini-Hochberg FDR at the configured level.
- **Guardrails** — excluded from correction entirely.

**Rationale for excluding guardrails:** the error that matters is asymmetric. For a guardrail you
care about power to *detect harm*; a false alarm costs one investigation, a miss ships a regression
to users. BH correction reduces exactly the sensitivity a guardrail exists to provide. Correcting
guardrails optimizes the wrong error.

**Rejected:** one family across all metrics. Cleaner in principle — a readout is a single ship
decision, so all its tests could share one error budget — but it trades away harm sensitivity, which
is the wrong direction for a constraint metric.

**Consumed by:** M6, `src/expkit/inference/multiple.py`.

---

## D4 — Guardrail conflict in the recommendation — PROPOSED

**Decision:** when the primary is up and significant and a guardrail is down and significant,
`readout()` recommends **do not ship, escalate to owner**. The output names the offending guardrail,
its point estimate and CI, and the primary estimate, and flags the decision as requiring an explicit
human override. No significant guardrail degradation is traded against primary lift automatically.

**Rationale:** guardrails are constraints, not terms in an objective function. Automating the
trade-off requires an exchange rate between, say, conversion and latency that the readout has no
standing to invent. Escalation puts the trade where the information to make it exists.

**Rejected (recorded as the upgrade path):** pre-registered non-inferiority margins per guardrail —
ship when the CI on harm lies entirely inside a margin declared before launch. This is the stronger
design where the margins have a real source, because it makes the trade-off explicit and testable
in advance. It is rejected here only because inventing margins with no organization behind them
produces a rule that cannot be falsified.

**Consumed by:** M6, `src/expkit/readout.py`.

---

## D5 — Two-sided tests — LOCKED

**Decision:** all tests are two-sided. Sample sizing uses `z_(1-alpha/2)`. There is no `sides`
parameter; one-sided inference is not implemented.

**Rationale:** a one-sided test structurally cannot detect that the treatment hurt, and that is the
result the organization most needs to see. The one-sided design is only honest if you can pre-commit
to taking no action on a large negative effect — and nobody can. Paying ~20% more sample for the
ability to see harm is the correct trade.

**Rejected:** one-sided at the same alpha. ~20% smaller sample for equal power, defensible only when
the decision is genuinely one-directional and harm detection is fully delegated to guardrails.

**Consumed by:** M1, `src/expkit/design/power.py`; M2, `src/expkit/inference/fixed.py`.

---

## D6 — No continuity correction — LOCKED

**Decision:** the two-proportion sample-size formula applies no continuity correction.

**Rationale:** consistency between the design module and the test it sizes for. `inference/fixed.py`
implements an uncorrected normal-approximation two-proportion test; correcting the design while
leaving the analysis uncorrected makes `required_n` systematically conservative relative to the test
actually run, and `sims/study_power.py` would surface that as an empirical power curve sitting above
the analytic prediction. The design and the analysis must make the same approximation.

**Rejected:** the Fleiss continuity correction (inflates n by about `2/(|delta| * kappa)` per arm).
More accurate against Fisher's exact test and conservative at small n — but only coherent if the
analysis is corrected too.

**Consumed by:** M1, `src/expkit/design/power.py`.

---

## D7 — Unequal allocation via the ratio kappa — LOCKED

**Decision:** unequal allocation enters as `kappa = n_treatment / n_control`. The formulas solve for
`n_control`, then `n_treatment = kappa * n_control`; both and the total are reported. `kappa = 1`
collapses to the familiar equal-allocation expression.

**Rationale:** this is the form the sample-size literature states (Fleiss), so each formula is
checkable line by line against a published reference rather than against a rearrangement of one.

**Rejected:** parameterizing by treatment share `p` of a total `N`, with variance entering as
`1/(N*p) + 1/(N*(1-p))`. Closer to how a ramp is actually configured, but not directly checkable
against the reference formulas.

**Consumed by:** M1, `src/expkit/design/power.py`.

---

## D8 — Pooled-null vs alternative variance — LOCKED (spec-mandated, recorded for defensibility)

**Decision:** in the two-proportion sample-size formula the two variances appear in different terms
and are not interchangeable:

- the **alpha term** uses the **pooled variance under the null**, `pbar*(1-pbar)` with
  `pbar = (p_control + kappa*p_treatment)/(1+kappa)` — because the critical value is set under H0,
  where both arms share one rate;
- the **power term** uses the **alternative variance**, `p_c*(1-p_c) + p_t*(1-p_t)/kappa` — because
  power is evaluated under H1, where the arms have different rates and therefore different variances.

Both call sites are annotated in the source. Collapsing them to a single pooled variance is the
standard error here, and the direction of the resulting bias is worth knowing precisely:

- **At equal allocation it always inflates `n`.** `p*(1-p)` is concave, so by Jensen the alternative
  variance never exceeds the pooled one. The collapsed formula is conservative. The gap is 0.03% at
  `p=0.10, delta=0.02` and 6.3% at `p=0.50, delta=0.40` — small enough at typical effect sizes to
  survive code review unnoticed.
- **At unequal allocation it can under-size, badly.** `pbar` weights `p_treatment` by
  `kappa/(1+kappa)`, while the alternative variance weights `f(p_treatment)` by `1/(1+kappa)`. The
  weights are transposed, so Jensen does not apply and the inequality can run either way. At
  `kappa=0.1` the collapsed form asks for 28% fewer control units than are needed, and the
  experiment ships underpowered.

Both directions are pinned by `tests/test_power.py::TestVarianceConvention`.

**Consumed by:** M1, `src/expkit/design/power.py`.

---

## D9 — Unpooled (Wald) variance for both the p-value and the interval — LOCKED

**Decision:** `inference/fixed.py` computes the two-proportion p-value and the confidence interval from
the same unpooled variance, so `p < alpha` holds **if and only if** the interval excludes zero.

**Rationale:** the readout's decision logic reads significance off one metric and reports an interval
next to it. If the p-value came from a pooled-variance score test while the interval used the unpooled
variance — textbook practice, and marginally better calibrated under the null — the two can disagree at
the boundary, and the readout would call a result significant while displaying an interval containing
zero. There is no defensible way to explain that to a stakeholder. Coherence is worth the trade.

**The cost, measured rather than asserted:** the unpooled test is very slightly anti-conservative, and
because the *design* module sizes for a pooled-variance test (D6, D8), empirical power runs a little
above the design prediction. `sims/study_power.py` reports the gap: **+0.0055 on average**, with every
point inside 3 Monte Carlo SE. `sims/study_type1.py` confirms Type I stays at nominal from n=200 to
n=50,000.

**Rejected:** a Miettinen–Nurminen score interval obtained by inverting the pooled test. That would make
design, test, and interval all mutually consistent and is the theoretically cleanest option, but it needs
a constrained-MLE root solve whose failure modes are subtle. Recorded as the upgrade path.

**Consumed by:** M2, `src/expkit/inference/fixed.py`.

---

## D10 — Outlier policy: off by default, pooled cutoff when enabled — LOCKED

**Decision:** no automatic trimming anywhere in the library. `validity/checks.winsorize` exists, must be
called deliberately, and caps both arms at a percentile of the **pooled** distribution (default 99.9).

**Rationale:** trimming changes the estimand. You stop estimating the mean and start estimating a trimmed
mean, and a readout that does that silently is reporting an answer to a question nobody asked. It has to
be a pre-registered choice.

**Why the cutoff is pooled:** a per-arm cutoff is the trap. If the treatment genuinely produced more big
spenders, capping each arm at its own percentile trims exactly the effect being measured and biases the
estimate toward zero — invisibly. Pinned by `tests/test_checks.py::TestWinsorize`.

**Rejected:** winsorizing by default at 99%. Common in practice and it does stabilize revenue metrics,
but a default that changes the estimand is the wrong kind of convenience.

**Consumed by:** M2, `src/expkit/validity/checks.py`.

---

## D11 — CUPED covariate guard: hard raise at p < 0.001 — LOCKED

**Decision:** `cuped_adjust` raises `CovariateContamination` when the covariate differs across arms at
p < 0.001 — the same threshold as SRM (D1), for the same reason: the check runs on every CUPED-adjusted
metric, so a loose threshold produces constant false alarms.

**Rationale:** a post-treatment covariate carries part of the treatment effect, so subtracting
`theta * (X - Xbar)` removes part of the effect from the outcome. The estimate is biased toward zero
*and* the variance still falls, so the readout looks **more** trustworthy — tighter interval, smaller
effect — while being wrong. `sims/study_cuped.py` measures it: with the treatment leaking into the
covariate, the estimated effect comes back at **60% below truth** with an unchanged standard error.
A warning is not adequate for a failure mode that makes output look better.

**The honest limit of this guard:** it is necessary, not sufficient. A post-treatment covariate that the
treatment happens not to move will pass. Nothing in the data can prove a covariate is pre-treatment —
only the definition of the measurement window can. Documented in the function itself.

**False-alarm rate, measured:** across 14,000 replications with covariates that are pre-treatment by
construction, the guard raised **15 times — 0.00107, within 0.27 Monte Carlo SE of the 0.001 threshold**.
Calibrated, but not free: roughly one clean covariate blocked per thousand adjusted metrics.

**Rejected:** an attestation flag the caller sets to assert the covariate is pre-period. Zero false
alarms, and zero protection — it moves the failure from the data to a checkbox.

**Consumed by:** M4, `src/expkit/metrics/cuped.py`.

---

## Open

None. Every decision this specification flagged as the owner's has been made and recorded above:
O1–O4 became D5–D8 (M1), O5 became D10 (M2), and O6 became D11 (M4).

`alpha` and `power` remain **required parameters with no defaults** throughout the design module, and
there is no one-sided option anywhere in the library (D5).

## Decisions still marked PROPOSED

D2, D3, and D4 were delegated rather than chosen by the owner. They are implemented and their
simulations pass, but the rejected alternative is recorded in each so both sides can be argued. Sign
them off — or overrule one — before presenting this work; the interview value is in defending them as
your positions.
