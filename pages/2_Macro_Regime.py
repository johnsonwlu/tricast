"""Macro regime page: composite gauge, indicator cards, current tilt."""

import plotly.graph_objects as go
import streamlit as st

from stock_scenarios import macro_regime, ui

ui.page_setup("🌍 Macro Regime")

try:
    macro_state = ui.cached_macro_state()
except Exception as e:
    st.error(
        f"Macro data unavailable: {e}\n\n"
        "Make sure FRED_API_KEY is set in .env "
        "(free key: https://fred.stlouisfed.org/docs/api/api_key.html)"
    )
    st.stop()

score = macro_state["score"]
gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=score,
    number={"valueformat": "+.2f"},
    title={"text": f"Regime: {macro_state['regime']}"},
    gauge={
        "axis": {"range": [-1, 1]},
        "bar": {"color": "#333"},
        "steps": [
            {"range": [-1, -0.3], "color": "rgba(214,39,40,0.5)"},
            {"range": [-0.3, 0.3], "color": "rgba(150,150,150,0.4)"},
            {"range": [0.3, 1], "color": "rgba(44,160,44,0.5)"},
        ],
    },
))
gauge.update_layout(height=300, margin=dict(t=60, b=10))
st.plotly_chart(gauge, width="stretch")

# Tilt currently applied
priors = {"bear": 25, "base": 50, "bull": 25}
tilted = macro_regime.tilt_probabilities(priors, score)
shift = tilted["bull"] - priors["bull"]
st.markdown(
    f"**Probability tilt applied:** bull {shift:+d}pp / bear {-shift:+d}pp "
    f"→ bear {tilted['bear']}% / base {tilted['base']}% / bull {tilted['bull']}%"
)

st.divider()
cols = st.columns(len(macro_state["signals"]))
SIGNAL_LABEL = {1: ":green[+1 expansionary]", 0: ":gray[0 neutral]", -1: ":red[−1 contractionary]"}
for col, sig in zip(cols, macro_state["signals"]):
    with col, st.container(border=True):
        st.markdown(f"**{sig['name']}**")
        st.metric("Value", sig["value"])
        st.markdown(SIGNAL_LABEL[sig["signal"]])
        st.caption(sig["detail"])

if st.button("🔄 Refresh macro data"):
    ui.cached_macro_state.clear()
    st.rerun()

ui.disclaimer()
