"""Watchlist home page."""

import streamlit as st

from tricast import store, ui

ui.page_setup("📈 Tricast — Watchlist")

# Macro banner (degrades gracefully if FRED key is missing)
try:
    macro_state = ui.cached_macro_state()
    ui.regime_banner(macro_state)
except Exception as e:
    st.warning(f"Macro data unavailable: {e}")

# Add ticker
with st.form("add_ticker", clear_on_submit=True):
    col1, col2 = st.columns([4, 1])
    new_ticker = col1.text_input("Add ticker", placeholder="e.g. NVDA",
                                 label_visibility="collapsed")
    if st.form_submit_button("Add") and new_ticker:
        from tricast.data import market
        t = new_ticker.strip().upper()
        if market.is_valid_ticker(t):
            store.watchlist_add(t)
            ui.cached_report.clear()
            st.rerun()
        else:
            st.error(f"'{t}' doesn't look like a valid ticker.")

tickers = store.watchlist_all()
if not tickers:
    st.info("Watchlist is empty — add a ticker above to get started.")
else:
    for ticker in tickers:
        try:
            report = ui.cached_report(ticker)
        except Exception as e:
            st.error(f"{ticker}: failed to load ({e})")
            continue

        analysis = report.get("analysis")
        probs = (
            {k: v["probability_pct"] for k, v in analysis["scenarios"].items()}
            if analysis else report["tilted_probabilities"]
        )
        closes = report["history"]["close"]
        chg_1d = (closes[-1] / closes[-2] - 1) * 100 if len(closes) > 1 else 0.0
        chg_1m = (closes[-1] / closes[-22] - 1) * 100 if len(closes) > 22 else 0.0

        with st.container(border=True):
            cols = st.columns([1.2, 1, 1, 2.2, 1, 1, 1, 0.6])
            cols[0].markdown(f"### {ticker}")
            cols[1].metric("Price", f"${report['spot']:.2f}", f"{chg_1d:+.1f}% 1d")
            cols[2].metric("1 month", f"{chg_1m:+.1f}%")
            cols[3].markdown(
                f"🐻 bear **{probs['bear']}%** → ${report['scenarios']['bear']['target']:.0f}  \n"
                f"⚖️ base **{probs['base']}%** → ${report['scenarios']['base']['target']:.0f}  \n"
                f"🐂 bull **{probs['bull']}%** → ${report['scenarios']['bull']['target']:.0f}"
            )
            if analysis:
                cols[4].markdown(ui.advice_badge(analysis["advice"]))
                cols[5].caption(f"analyzed {ui.age_str(analysis.get('_created_at'))}")
            else:
                cols[4].caption("not analyzed")
            if cols[6].button("Details", key=f"detail_{ticker}"):
                st.session_state["selected_ticker"] = ticker
                st.switch_page("pages/1_Stock_Detail.py")
            if cols[7].button("✕", key=f"rm_{ticker}", help=f"Remove {ticker}"):
                store.watchlist_remove(ticker)
                st.rerun()

ui.disclaimer()
