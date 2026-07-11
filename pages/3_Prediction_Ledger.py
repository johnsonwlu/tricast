"""Prediction ledger: every forecast the app has made, tracked to maturity."""

import pandas as pd
import streamlit as st

from stock_scenarios import ledger, ui

ui.page_setup("📒 Prediction Ledger")

st.caption(
    "Every report logs its forecast here (one per ticker per day). After the "
    "12-month horizon passes, predictions are scored: which band did the price "
    "actually land in, and how good were the stated probabilities (Brier score, "
    "lower = better, 0 = perfect)."
)

preds = ledger.all_predictions()
if not preds:
    st.info("No predictions recorded yet — open a stock report and one will be logged.")
    st.stop()

matured = [p for p in preds if p["outcome"]]
open_preds = [p for p in preds if not p["outcome"]]

# --- Calibration summary (matured only) ---
stats = ledger.summary()
if stats["n_scored"]:
    c = st.columns(4)
    c[0].metric("Scored predictions", stats["n_scored"])
    c[1].metric("Mean Brier", f"{stats['mean_brier']:.3f}")
    c[2].metric("Naive baseline", f"{stats['baseline_brier']:.3f}")
    c[3].metric("Beats baseline?", "✅ yes" if stats["beats_baseline"] else "❌ no")
else:
    st.info(
        f"{len(open_preds)} open prediction(s), none matured yet — scoring starts "
        "12 months after the first prediction. Run `scripts/score_predictions.py` "
        "(or reload this page) once horizons pass."
    )

# --- Open predictions with interim tracking ---
if open_preds:
    st.subheader(f"Open ({len(open_preds)})")
    rows = []
    for p in open_preds:
        try:
            current = ui.cached_report(p["ticker"])["spot"]
            tracking = ledger.interim_position(p, current)
        except Exception:
            current, tracking = None, "—"
        rows.append({
            "Ticker": p["ticker"],
            "Predicted": p["pred_date"],
            "Matures": p["horizon_end"],
            "Spot then": f"${p['spot']:.2f}",
            "Now": f"${current:.2f}" if current else "—",
            "Tracking band": tracking,
            "P(bear/base/bull)": f"{p['p_bear']}/{p['p_base']}/{p['p_bull']}",
            "Targets": f"{p['bear_target']:.0f} / {p['base_target']:.0f} / {p['bull_target']:.0f}",
            "Model": p["model"],
            "Regime": p["regime"],
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# --- Matured predictions ---
if matured:
    st.subheader(f"Matured ({len(matured)})")
    rows = [{
        "Ticker": p["ticker"],
        "Predicted": p["pred_date"],
        "Spot then": f"${p['spot']:.2f}",
        "Price at maturity": f"${p['outcome_price']:.2f}",
        "Outcome": p["outcome"],
        "P(outcome) stated": f"{p['p_' + p['outcome']]}%",
        "Brier": f"{p['brier']:.3f}",
        "Model": p["model"],
    } for p in matured]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.bar_chart(pd.Series(stats["outcome_counts"], name="outcomes"))

if st.button("🔄 Score matured predictions now"):
    with st.spinner("Scoring…"):
        newly = ledger.score_matured()
    st.success(f"Scored {len(newly)} prediction(s)." if newly
               else "Nothing matured yet.")
    st.rerun()

ui.disclaimer()
