"""Prediction ledger: every forecast the app has made, tracked to maturity."""

import pandas as pd
import streamlit as st

from tricast import ledger, ui

ui.page_setup(
    "Track record",
    "Every forecast this app has made, kept honest by checking it against what "
    "actually happened.",
)

st.markdown(
    "<span class='tc-note'>Anyone can sound confident about the future. The "
    "only way to know whether these forecasts are worth anything is to write "
    "them down and score them later. Each report logs its prediction here, and "
    "12 months on it gets marked right or wrong — no quiet editing.</span>",
    unsafe_allow_html=True)

preds = ledger.all_predictions()
if not preds:
    st.info("Nothing recorded yet. Open any stock report and its forecast gets "
            "logged here automatically.")
    st.stop()

matured = [p for p in preds if p["outcome"]]
open_preds = [p for p in preds if not p["outcome"]]

# --- Calibration summary (matured only) ---
stats = ledger.summary()
if stats["n_scored"]:
    st.markdown("#### How the finished forecasts did")
    with st.container(border=True):
        c = st.columns(4)
        c[0].metric("Forecasts scored", stats["n_scored"],
                    help="How many have reached their 12-month finish line.")
        c[1].metric("Accuracy score", f"{stats['mean_brier']:.3f}",
                    help=ui.HELP["brier"])
        c[2].metric("Score to beat", f"{stats['baseline_brier']:.3f}",
                    help="What you'd get by lazily guessing the same 25/50/25 "
                         "chances for every stock, every time. Beating this is "
                         "the bar for the model being worth anything.")
        beats = stats["beats_baseline"]
        c[3].markdown(
            "<span class='tc-term'>Better than lazy guessing?</span><br>"
            + ui.pill("Yes — it's adding value" if beats
                      else "Not yet — no better than guessing",
                      "good" if beats else "caution"),
            unsafe_allow_html=True)
else:
    st.info(
        f"{len(open_preds)} forecast(s) are still running, and none have reached "
        "their 12-month finish line yet — so there's nothing to score. That's "
        "expected: the first one matures a year after it was made. (The "
        "**backtest** takes the shortcut of replaying the model on past years "
        "if you want an answer sooner.)"
    )

# --- Open predictions with interim tracking ---
if open_preds:
    st.markdown(f"#### Still running ({len(open_preds)})")
    st.caption("Where each forecast is tracking so far. This is a progress "
               "check, not the final result — plenty can change before the "
               "12 months are up.")
    rows = []
    for p in open_preds:
        try:
            current = ui.cached_report(p["ticker"])["spot"]
            tracking = ledger.interim_position(p, current)
        except Exception:
            current, tracking = None, "—"
        rows.append({
            "Stock": p["ticker"],
            "Forecast made": p["pred_date"],
            "Finishes": p["horizon_end"],
            "Price then": f"${p['spot']:.2f}",
            "Price now": f"${current:.2f}" if current else "—",
            "Tracking toward": {"bull": "If things go well",
                                "base": "Most likely",
                                "bear": "If things go poorly"}.get(tracking, tracking),
            "Chances (well/likely/poorly)":
                f"{p['p_bull']}% / {p['p_base']}% / {p['p_bear']}%",
            "Prices (well/likely/poorly)":
                f"${p['bull_target']:,.0f} / ${p['base_target']:,.0f} / ${p['bear_target']:,.0f}",
            "Written by": p["model"],
            "Economy then": p["regime"],
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# --- Matured predictions ---
if matured:
    st.markdown(f"#### Finished ({len(matured)})")
    OUTCOME_PLAIN = {"bull": "Things went well",
                     "base": "Landed in the middle",
                     "bear": "Things went poorly"}
    rows = [{
        "Stock": p["ticker"],
        "Forecast made": p["pred_date"],
        "Price then": f"${p['spot']:.2f}",
        "Price 12 months on": f"${p['outcome_price']:.2f}",
        "What happened": OUTCOME_PLAIN.get(p["outcome"], p["outcome"]),
        "Chance it gave that": f"{p['p_' + p['outcome']]}%",
        "Accuracy score": f"{p['brier']:.3f}",
        "Written by": p["model"],
    } for p in matured]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("A good forecast isn't one where the most likely case always "
               "happens — it's one where things it called 70% likely happen "
               "about 70% of the time.")
    st.bar_chart(pd.Series(stats["outcome_counts"], name="outcomes"))

if st.button("Check for finished forecasts"):
    with st.spinner("Scoring…"):
        newly = ledger.score_matured()
    st.success(f"Scored {len(newly)} forecast(s)." if newly
               else "Nothing has reached its finish line yet.")
    st.rerun()

ui.disclaimer()
