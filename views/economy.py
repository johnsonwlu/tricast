"""Macro regime page: composite gauge, indicator cards, current tilt."""

import plotly.graph_objects as go
import streamlit as st

from tricast import macro_regime, ui

ui.page_setup(
    "The economy right now",
    "Five widely-watched signals, boiled down to one question: is the wider "
    "economy helping or hurting stocks at the moment?",
)

# Everyday name + what it means, keyed by the internal indicator name.
PLAIN = {
    "Yield curve (10y-2y)": (
        "Long vs short-term interest rates",
        "Normally, lending money for 10 years pays more than lending it for 2. "
        "When that flips negative, a recession has often followed."),
    "CPI YoY": (
        "Inflation",
        "How fast prices are rising compared with a year ago. Both very high "
        "and negative inflation tend to unsettle markets."),
    "Unemployment 3m trend": (
        "Unemployment trend",
        "Whether unemployment has risen or fallen over the past three months. "
        "A rising trend is one of the earliest recession warnings."),
    "Fed funds 6m trend": (
        "Interest rate direction",
        "Whether the central bank has been raising or cutting rates lately. "
        "Rising rates make borrowing costlier and usually weigh on stocks."),
    "VIX": (
        "Market nerves",
        "Often called the fear gauge: how big a swing traders expect soon. "
        "Above roughly 25 means markets are jumpy."),
}

SIGNAL_PLAIN = {
    1:  ("Helping stocks", "good"),
    0:  ("Neutral", "muted"),
    -1: ("Weighing on stocks", "poor"),
}

try:
    macro_state = ui.cached_macro_state()
except Exception as e:
    st.error(
        f"Couldn't load the economic data: {e}\n\n"
        "This page needs a free FRED API key in your .env file "
        "(get one at https://fred.stlouisfed.org/docs/api/api_key.html). "
        "Everything else in the app works without it."
    )
    st.stop()

score = macro_state["score"]
sentence, tone = ui.regime_plain(macro_state)

with st.container(border=True):
    c = st.columns([1.15, 3])
    c[0].markdown(
        ui.pill(f"Economy: {macro_state['regime'].split(' (')[0]}", tone),
        unsafe_allow_html=True)
    c[1].markdown(f"<span class='tc-note'>{sentence}</span>",
                  unsafe_allow_html=True)

left, right = st.columns([1.15, 1])

with left:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"valueformat": "+.2f", "font": {"size": 34}},
        title={"text": "Overall reading", "font": {"size": 15}},
        gauge={
            "axis": {"range": [-1, 1], "tickcolor": "#B9B2A8"},
            "bar": {"color": "#4A78D8", "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [-1, -0.3], "color": "rgba(166,68,62,0.22)"},
                {"range": [-0.3, 0.3], "color": "rgba(107,101,94,0.16)"},
                {"range": [0.3, 1], "color": "rgba(31,122,92,0.22)"},
            ],
        },
    ))
    gauge.update_layout(height=270, margin=dict(t=50, b=10, l=30, r=30),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#2B2A28"))
    st.plotly_chart(gauge, width="stretch")
    st.caption("−1 means every signal is flashing caution; +1 means all five "
               "look supportive. Most of the time it sits somewhere in between.")

with right:
    st.markdown("#### What this does to the forecasts")
    priors = {"bear": 25, "base": 50, "bull": 25}
    tilted = macro_regime.tilt_probabilities(priors, score)
    shift = tilted["bull"] - priors["bull"]
    if shift == 0:
        st.markdown("<span class='tc-note'>Right now the economy is neutral "
                    "enough that it leaves the chances untouched.</span>",
                    unsafe_allow_html=True)
    else:
        direction = "brighter" if shift > 0 else "gloomier"
        st.markdown(
            f"<span class='tc-note'>Because the economy currently looks "
            f"{direction} than usual, every stock's optimistic case is nudged "
            f"<b>{shift:+d} points</b> and its pessimistic case "
            f"<b>{-shift:+d}</b>.</span>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='tc-row'><span class='lbl'>If things go well</span>"
        f"<span class='val'>{priors['bull']}% → {tilted['bull']}%</span></div>"
        f"<div class='tc-row'><span class='lbl'>Most likely</span>"
        f"<span class='val'>{priors['base']}% → {tilted['base']}%</span></div>"
        f"<div class='tc-row'><span class='lbl'>If things go poorly</span>"
        f"<span class='val'>{priors['bear']}% → {tilted['bear']}%</span></div>",
        unsafe_allow_html=True)
    st.caption("The nudge is capped at 10 points, so the economy can lean the "
               "forecast — it can never take it over.")

st.divider()
st.markdown("#### The five signals")

cols = st.columns(len(macro_state["signals"]))
for col, sig in zip(cols, macro_state["signals"]):
    label, meaning = PLAIN.get(sig["name"], (sig["name"], ""))
    text, tone = SIGNAL_PLAIN[sig["signal"]]
    with col, st.container(border=True):
        st.markdown(
            f"<span class='tc-cardtitle'>{label}</span>"
            f"<span class='tc-cardsub'>{sig['name']}</span>",
            unsafe_allow_html=True)
        st.metric("Latest reading", sig["value"], label_visibility="collapsed")
        st.markdown(ui.pill(text, tone), unsafe_allow_html=True)
        st.caption(sig["detail"])
        if meaning:
            st.caption(meaning)

if st.button("Refresh economic data"):
    ui.cached_macro_state.clear()
    st.rerun()

ui.disclaimer()
