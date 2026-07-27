"""Per-stock detail: fan chart, scenario cards, advice panel."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tricast import calibration, config, errors, store, ui

ui.page_setup(
    "Stock detail",
    "The full picture for one stock: where it's been, where the simulation "
    "thinks it could go, and what that means.",
)

tickers = store.watchlist_all()
if not tickers:
    st.info("Your watchlist is empty — add a stock on the home page first.")
    st.stop()

default = st.session_state.get("selected_ticker", tickers[0])
ticker = st.selectbox("Choose a stock", tickers,
                      index=tickers.index(default) if default in tickers else 0)

try:
    report = ui.cached_report(ticker)
except errors.PermanentTickerError as e:
    st.error(e.friendly())
    if st.button(f"Remove {ticker} from my watchlist"):
        store.watchlist_remove(ticker)
        st.session_state.pop("selected_ticker", None)
        st.rerun()
    st.stop()
except Exception as e:
    st.error(f"Couldn't load {ticker} right now: {e}")
    st.stop()

ui.explainer()

# --- The company today ---
f = report["fundamentals"]
st.subheader(f.get("shortName") or ticker)
with st.container(border=True):
    st.markdown("<span class='tc-term'>The company today</span>",
                unsafe_allow_html=True)
    # Three per row: the plain-English labels are longer than the old jargon
    # ones and get truncated at six across.
    r1 = st.columns(3)
    r1[0].metric("Price today", f"${report['spot']:.2f}", help=ui.HELP["spot"])
    r1[1].metric("Sales growth",
                 f"{f['revenueGrowth'] * 100:.1f}%" if f.get("revenueGrowth") else "—",
                 help=ui.HELP["rev_growth"])
    r1[2].metric("Profit margin",
                 f"{f['profitMargins'] * 100:.1f}%" if f.get("profitMargins") else "—",
                 help=ui.HELP["margin"])
    r2 = st.columns(3)
    r2[0].metric("Price vs profit", f"{f['forwardPE']:.1f}" if f.get("forwardPE") else "—",
                 help=ui.HELP["fwd_pe"])
    r2[1].metric("Swings vs market", f"{f['beta']:.2f}" if f.get("beta") else "—",
                 help=ui.HELP["beta"])
    r2[2].metric("Analyst target",
                 f"${f['targetMeanPrice']:.0f}" if f.get("targetMeanPrice") else "—",
                 help=ui.HELP["analyst_target"])

# --- Risk-adjusted metrics (present once risk scoring is merged) ---
rk = report.get("risk")
if rk:
    sentence, tone = ui.risk_sentence(rk)
    with st.container(border=True):
        st.markdown(
            f"<span class='tc-term'>Is the reward worth the risk?</span><br>"
            f"{ui.pill(sentence, tone)}", unsafe_allow_html=True)
        rm = st.columns(4)
        rm[0].metric("Reward for the risk", f"{rk['sharpe']:.2f}",
                     help=ui.HELP["sharpe"])
        rm[1].metric("Ignoring upward swings", f"{rk['sortino']:.2f}",
                     help=ui.HELP["sortino"])
        rm[2].metric("Chance of losing money", f"{rk['prob_loss_pct']:.0f}%",
                     help=ui.HELP["prob_loss"])
        rm[3].metric("Typical bad-case loss", f"−{rk['cvar5_pct']:.0f}%",
                     help=ui.HELP["cvar"])

# --- Fan chart: 2y history + 12-month simulated cone ---
st.subheader("Where it could go from here")
st.markdown(
    "<span class='tc-note'>The solid line is what actually happened. The shaded "
    "fan is the range of futures the simulation produced — the darker middle "
    "band is where outcomes landed about half the time.</span>",
    unsafe_allow_html=True)

hist_dates = pd.to_datetime(report["history"]["dates"])
hist_close = report["history"]["close"]
future_dates = pd.bdate_range(hist_dates[-1], periods=report["horizon_days"] + 1)[1:]
cone = report["cone"]

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist_dates, y=hist_close, name="What happened",
                         line=dict(color="#4A78D8", width=2.2)))
fig.add_trace(go.Scatter(x=future_dates, y=cone["p90"], name="p90",
                         line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=future_dates, y=cone["p10"], name="Wider range",
                         fill="tonexty", fillcolor="rgba(74,120,216,0.10)",
                         line=dict(width=0), hoverinfo="skip"))
fig.add_trace(go.Scatter(x=future_dates, y=cone["p75"], line=dict(width=0),
                         showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=future_dates, y=cone["p25"], name="Most likely range",
                         fill="tonexty", fillcolor="rgba(74,120,216,0.24)",
                         line=dict(width=0), hoverinfo="skip"))
fig.add_trace(go.Scatter(x=future_dates, y=cone["p50"], name="Middle path",
                         line=dict(color="#4A78D8", dash="dash", width=1.6)))

for name, color in (("bear", "#A6443E"), ("base", "#6B655E"), ("bull", "#1F7A5C")):
    target = report["scenarios"][name]["target"]
    # anchored inside the plot: "right" pushes the text past the edge and clips it
    fig.add_hline(y=target, line_dash="dot", line_color=color, opacity=.55,
                  annotation_text=f"{ui.scenario_label(name)} · ${target:,.0f}",
                  annotation_position="top left",
                  annotation_font=dict(size=12, color=color))

fig.update_layout(
    height=470, margin=dict(t=20, b=10, l=10, r=30),
    yaxis_title="Price per share ($)", hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#2B2A28"),
    # traceorder defaults to reversed once fills are involved, which puts the
    # real price history last in the legend
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                traceorder="normal"),
)
fig.update_xaxes(gridcolor="#EFEBE4", zeroline=False)
fig.update_yaxes(gridcolor="#EFEBE4", zeroline=False)
st.plotly_chart(fig, width="stretch")

# Calibration provenance: is the cone width learned or neutral?
_vs = report.get("vol_scale", 1.0)
if _vs != 1.0:
    _when = (calibration.metadata().get("fitted_at") or "")[:10]
    st.caption(
        f"The width of this fan has been **tuned against history** (×{_vs:.2f}"
        f"{f', last checked {_when}' if _when else ''}) — earlier versions of the "
        "model spread it too wide. Re-tune with `scripts/calibrate.py --write`."
    )
else:
    st.caption("The width of this fan has **not been tuned** against history yet — "
               "run `scripts/calibrate.py --write` to fit it.")

# --- Scenario cards ---
st.subheader("The three stories")
analysis = report.get("analysis")
probs = (
    {k: v["probability_pct"] for k, v in analysis["scenarios"].items()}
    if analysis else report["tilted_probabilities"]
)
TONE = {"bull": "good", "base": "calm", "bear": "poor"}
cards = st.columns(3)
for col, name in zip(cards, ("bull", "base", "bear")):
    s = report["scenarios"][name]
    with col, st.container(border=True):
        st.markdown(
            f"#### {ui.scenario_label(name)}\n"
            f"<span class='tc-term'>the {ui.scenario_term(name)}</span>",
            unsafe_allow_html=True)
        st.markdown(
            ui.pill(f"{probs[name]}% chance · {ui.chance_words(probs[name])}",
                    TONE[name]),
            unsafe_allow_html=True)
        st.metric("Price in 12 months", f"${s['target']:,.2f}",
                  f"{s['return_pct']:+.1f}% vs today", help=ui.HELP["target"])
        st.caption(ui.scenario_help(name))
        if analysis:
            st.write(analysis["scenarios"][name]["narrative"])
        else:
            st.caption("Run the commentary below to get a written explanation.")

# --- Advice panel ---
if analysis:
    st.subheader("What it adds up to")
    with st.container(border=True):
        st.markdown(ui.advice_pill(analysis["advice"]), unsafe_allow_html=True)
        st.markdown(f"<span class='tc-note'>{ui.plain_advice(analysis['advice'])}"
                    f"</span>", unsafe_allow_html=True)
        st.write(analysis["advice_reasoning"])
        st.markdown("**What could go wrong:**")
        for risk in analysis["key_risks"]:
            st.markdown(f"- {risk}")
        st.caption(
            f"Written by {analysis.get('model', '?')} · "
            f"~${analysis.get('cost_usd', 0):.3f} · "
            f"{ui.age_str(analysis.get('_created_at'))}. The model explains the "
            "numbers; it can't invent the price targets."
        )
    with st.expander("For the curious: how the economy shifted these chances"):
        st.caption(
            "Starting chances come from the simulation alone. The economic "
            "reading nudges them, and the AI may nudge them a little further — "
            "but never by more than 10 points."
        )
        st.json({"before adjustment": report["tilted_probabilities"],
                 "final": probs})

# --- Actions ---
st.subheader("Update this report")
c1, c2 = st.columns(2)
if c1.button("Refresh prices (free, instant)", width="stretch"):
    ui.cached_report.clear()
    ui.cached_macro_state.clear()
    st.rerun()

llm_label = (
    "Write the commentary (free, runs on your own computer)"
    if config.LLM_PROVIDER == "ollama"
    else "Write the commentary (~$0.03, uses the Claude API)"
)
if c2.button(llm_label, width="stretch"):
    from tricast import pipeline
    with st.spinner("Reading the numbers and writing it up…"):
        try:
            pipeline.build_report(ticker, run_llm=True)
            ui.cached_report.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Couldn't finish the commentary: {e}")

ui.disclaimer()
