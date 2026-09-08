"""Thin Streamlit interface over :func:`expkit.readout.readout`.

Deliberately thin: this module builds inputs, calls the library, and formats what
comes back. No statistics live here. If a number appears on screen, ``expkit``
computed it -- otherwise the UI could disagree with the library and nobody would
know which one was wrong.

Run with ``make app``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from expkit.metrics.base import MetricRole, MetricSpec, MetricType
from expkit.readout import (
    BinaryObservation,
    ContinuousObservation,
    Decision,
    readout,
)
from expkit.validity.srm import SRMFailure

st.set_page_config(page_title="expkit readout", page_icon="📊", layout="wide")

SCENARIOS = {
    "Clean win": dict(primary_lift=0.006, guardrail_shift=1.0, srm_leak=0),
    "Primary up, guardrail down": dict(primary_lift=0.006, guardrail_shift=12.0, srm_leak=0),
    "Inconclusive": dict(primary_lift=0.0005, guardrail_shift=1.0, srm_leak=0),
    "Primary regression": dict(primary_lift=-0.006, guardrail_shift=1.0, srm_leak=0),
    "Sample ratio mismatch": dict(primary_lift=0.006, guardrail_shift=1.0, srm_leak=900),
}


def build_observations(rng, n, params):
    p_control = 0.10
    n_t = n + params["srm_leak"]
    return [
        BinaryObservation(
            MetricSpec("checkout_conversion", MetricType.BINARY, MetricRole.PRIMARY),
            successes_control=int(rng.binomial(n, p_control)),
            n_control=n,
            successes_treatment=int(rng.binomial(n_t, p_control + params["primary_lift"])),
            n_treatment=n_t,
        ),
        ContinuousObservation(
            MetricSpec(
                "page_load_ms",
                MetricType.CONTINUOUS,
                MetricRole.GUARDRAIL,
                higher_is_better=False,
            ),
            control=rng.normal(800.0, 120.0, n),
            treatment=rng.normal(800.0 + params["guardrail_shift"], 120.0, n_t),
        ),
        BinaryObservation(
            MetricSpec("addtocart_rate", MetricType.BINARY, MetricRole.SECONDARY),
            successes_control=int(rng.binomial(n, 0.30)),
            n_control=n,
            successes_treatment=int(rng.binomial(n_t, 0.304)),
            n_treatment=n_t,
        ),
        BinaryObservation(
            MetricSpec("newsletter_signup", MetricType.BINARY, MetricRole.SECONDARY),
            successes_control=int(rng.binomial(n, 0.02)),
            n_control=n,
            successes_treatment=int(rng.binomial(n_t, 0.021)),
            n_treatment=n_t,
        ),
    ]


st.title("expkit — experiment readout")
st.caption(
    "Every number on this page is computed by the library. The interface formats; "
    "it does not calculate."
)

with st.sidebar:
    st.header("Experiment")
    scenario = st.selectbox("Scenario", list(SCENARIOS))
    n_per_arm = st.select_slider(
        "Units per arm", options=[10_000, 25_000, 50_000, 100_000], value=50_000
    )
    alpha = st.select_slider("Alpha", options=[0.01, 0.05, 0.10], value=0.05)
    duration = st.number_input("Duration (days)", 1, 90, 14)
    seed = st.number_input("Seed", 0, 10_000, 7)
    st.divider()
    st.caption(
        "Alpha has no default in the library — it is a required argument on every "
        "call, so it is a control here rather than a constant."
    )

rng = np.random.default_rng(int(seed))
params = SCENARIOS[scenario]
observations = build_observations(rng, int(n_per_arm), params)
counts = {
    "control": int(n_per_arm),
    "treatment": int(n_per_arm) + params["srm_leak"],
}

try:
    result = readout(
        observations=observations,
        intended_allocation={"control": 0.5, "treatment": 0.5},
        observed_counts=counts,
        alpha=float(alpha),
        duration_days=float(duration),
    )
except SRMFailure as failure:
    st.error("### Readout refused — sample ratio mismatch", icon="🛑")
    st.write(failure.result.detail)
    st.info(
        "No treatment effect is shown, and this is not a formatting choice. SRM means "
        "the randomization or the logging is broken, so the arms are not exchangeable "
        "and the comparison is not causal. The effect estimate would be meaningless "
        "regardless of its p-value, so the library declines to compute one.",
        icon="ℹ️",
    )
    st.stop()

banner = {
    Decision.SHIP: st.success,
    Decision.DO_NOT_SHIP: st.error,
    Decision.INCONCLUSIVE: st.warning,
}[result.recommendation.decision]
banner(f"### {result.recommendation.decision.value.upper()}")
st.write(result.recommendation.summary)
for reason in result.recommendation.reasons:
    st.markdown(f"- {reason}")
if result.recommendation.requires_human_override:
    st.warning("This decision requires an explicit human override to proceed.", icon="⚠️")

st.divider()
left, right = st.columns([3, 2])

with left:
    st.subheader("Metrics")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "metric": m.spec.name,
                    "role": m.spec.role.value,
                    "estimate": round(m.test.estimate, 6),
                    "95% CI": f"[{m.test.ci_lower:+.5f}, {m.test.ci_upper:+.5f}]",
                    "relative": (
                        f"{m.test.relative_estimate:+.2%}"
                        if m.test.relative_estimate is not None
                        else "—"
                    ),
                    "p": round(m.test.p_value, 5),
                    "adjusted p": round(m.adjusted_p_value, 5),
                    "significant": m.significant,
                    "correction": m.correction,
                }
                for m in result.metrics
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

with right:
    st.subheader("Validity")
    st.metric(
        "SRM p-value",
        f"{result.srm.p_value:.4f}",
        delta=f"threshold {result.srm.threshold:g}",
        delta_color="off",
    )
    st.caption(result.srm.detail)
    st.subheader("Run")
    st.write(
        pd.DataFrame(
            {
                "arm": list(result.realized_allocation),
                "units": [
                    result.n_control if a == "control" else result.n_treatment
                    for a in result.realized_allocation
                ],
                "realized": [f"{v:.4%}" for v in result.realized_allocation.values()],
                "intended": [
                    f"{result.intended_allocation[a]:.4%}" for a in result.realized_allocation
                ],
            }
        ).set_index("arm")
    )
    st.caption(f"alpha {result.alpha} · duration {result.duration_days:.0f} days")

st.divider()
st.caption(
    "Multiplicity: the primary metric is uncorrected (one pre-registered hypothesis), "
    "secondary metrics carry Benjamini-Hochberg FDR control, and guardrails are "
    "deliberately uncorrected — correcting them would cost power to detect harm. "
    "See DECISIONS.md D3."
)
