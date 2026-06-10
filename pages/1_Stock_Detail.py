"""Per-stock detail: fan chart, scenario cards, advice panel."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stock_scenarios import store, ui

ui.page_setup("🔍 Stock Detail")

tickers = store.watchlist_all()
if not tickers:
    st.info("Watchlist is empty — add tickers on the home page first.")
    st.stop()

default = st.session_state.get("selected_ticker", tickers[0])
ticker = st.selectbox("Ticker", tickers,
                      index=tickers.index(default) if default in tickers else 0)

try:
    report = ui.cached_report(ticker)
except Exception as e:
    st.error(f"Failed to load {ticker}: {e}")
    st.stop()

f = report["fundamentals"]
st.subheader(f.get("shortName") or ticker)
m = st.columns(6)
m[0].metric("Price", f"${report['spot']:.2f}")
m[1].metric("Fwd P/E", f"{f['forwardPE']:.1f}" if f.get("forwardPE") else "—")
m[2].metric("Rev growth", f"{f['revenueGrowth'] * 100:.1f}%" if f.get("revenueGrowth") else "—")
m[3].metric("Margin", f"{f['profitMargins'] * 100:.1f}%" if f.get("profitMargins") else "—")
m[4].metric("Beta", f"{f['beta']:.2f}" if f.get("beta") else "—")
m[5].metric("Analyst target", f"${f['targetMeanPrice']:.0f}" if f.get("targetMeanPrice") else "—")

# --- Fan chart: 2y history + 12-month simulated cone ---
hist_dates = pd.to_datetime(report["history"]["dates"])
hist_close = report["history"]["close"]
future_dates = pd.bdate_range(hist_dates[-1], periods=report["horizon_days"] + 1)[1:]
cone = report["cone"]

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist_dates, y=hist_close, name="History",
                         line=dict(color="#1f77b4")))
fig.add_trace(go.Scatter(x=future_dates, y=cone["p90"], name="P90",
                         line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=future_dates, y=cone["p10"], name="P10–P90",
                         fill="tonexty", fillcolor="rgba(31,119,180,0.12)",
                         line=dict(width=0)))
fig.add_trace(go.Scatter(x=future_dates, y=cone["p75"], line=dict(width=0),
                         showlegend=False))
fig.add_trace(go.Scatter(x=future_dates, y=cone["p25"], name="P25–P75",
                         fill="tonexty", fillcolor="rgba(31,119,180,0.25)",
                         line=dict(width=0)))
fig.add_trace(go.Scatter(x=future_dates, y=cone["p50"], name="Median path",
                         line=dict(color="#1f77b4", dash="dash")))

for name, color in (("bear", "red"), ("base", "gray"), ("bull", "green")):
    target = report["scenarios"][name]["target"]
    fig.add_hline(y=target, line_dash="dot", line_color=color,
                  annotation_text=f"{name} ${target:.0f}",
                  annotation_position="right")

fig.update_layout(height=480, margin=dict(t=30, b=10),
                  yaxis_title="Price ($)", hovermode="x unified")
st.plotly_chart(fig, width="stretch")

# --- Scenario cards ---
analysis = report.get("analysis")
probs = (
    {k: v["probability_pct"] for k, v in analysis["scenarios"].items()}
    if analysis else report["tilted_probabilities"]
)
cards = st.columns(3)
for col, name, emoji in zip(cards, ("bear", "base", "bull"), ("🐻", "⚖️", "🐂")):
    s = report["scenarios"][name]
    with col, st.container(border=True):
        st.markdown(f"#### {emoji} {name.title()} — {probs[name]}%")
        st.metric("12-mo target", f"${s['target']:.2f}", f"{s['return_pct']:+.1f}% vs spot")
        if analysis:
            st.write(analysis["scenarios"][name]["narrative"])
        else:
            st.caption("Run analysis for the narrative.")

# --- Advice panel ---
if analysis:
    with st.container(border=True):
        st.markdown(f"### Advice: {ui.advice_badge(analysis['advice'])}")
        st.write(analysis["advice_reasoning"])
        st.markdown("**Key risks:**")
        for risk in analysis["key_risks"]:
            st.markdown(f"- {risk}")
        st.caption(
            f"Model {analysis.get('model', '?')} · ~${analysis.get('cost_usd', 0):.3f} "
            f"· {ui.age_str(analysis.get('_created_at'))}"
        )
    with st.expander("Debug: probabilities vs prior"):
        st.json({"tilted_prior": report["tilted_probabilities"], "final": probs})

# --- Actions ---
c1, c2 = st.columns(2)
if c1.button("🔄 Refresh data (free)"):
    ui.cached_report.clear()
    ui.cached_macro_state.clear()
    st.rerun()
if c2.button("🤖 Re-run analysis (~$0.03, calls Claude)"):
    from stock_scenarios import pipeline
    with st.spinner("Running Claude analysis…"):
        try:
            pipeline.build_report(ticker, run_llm=True)
            ui.cached_report.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Analysis failed: {e}")

ui.disclaimer()
