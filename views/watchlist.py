"""Watchlist home page."""

import streamlit as st

from tricast import errors, store, ui

ui.page_setup(
    "Your watchlist",
    "Three possible futures for each stock you're following, with how likely "
    "each one looks.",
)

ui.explainer()

# Macro banner (degrades gracefully if FRED key is missing)
try:
    macro_state = ui.cached_macro_state()
    ui.regime_banner(macro_state)
except Exception as e:
    st.warning(f"Couldn't load economic data right now: {e}")

# Add ticker
with st.form("add_ticker", clear_on_submit=True):
    col1, col2 = st.columns([4, 1])
    new_ticker = col1.text_input(
        "Add ticker", placeholder="Add a stock symbol, e.g. NVDA",
        label_visibility="collapsed")
    if col2.form_submit_button("Add stock", width="stretch") and new_ticker:
        from tricast.data import market
        t = new_ticker.strip().upper()
        if not market.is_valid_ticker(t):
            st.error(f"Couldn't find a stock with the symbol '{t}'. "
                     "Double-check the spelling — it's the short code, like AAPL.")
        else:
            # existence isn't enough: reject anything too newly listed to model,
            # so it can't become a permanently broken row further down the page
            try:
                market.check_addable(t)
            except errors.PermanentTickerError as e:
                st.error(e.friendly())
            else:
                store.watchlist_add(t)
                ui.cached_report.clear()
                st.rerun()

tickers = store.watchlist_all()

# Bulk analyze the whole watchlist
if tickers:
    from tricast import config
    llm_label = (f"Write commentary for all {len(tickers)} (free, runs on your own computer)"
                 if config.LLM_PROVIDER == "ollama"
                 else f"Write commentary for all {len(tickers)} (~${0.03 * len(tickers):.2f})")
    if st.button(llm_label):
        from tricast import bulk
        prog = st.progress(0.0, "Starting…")

        def _cb(done, total, ticker, row):
            status = row["advice"] or "?" if row["ok"] else "failed"
            prog.progress(done / total, f"[{done}/{total}] {ticker}: {status}")

        results = bulk.analyze_many(tickers, run_llm=True, progress_cb=_cb)
        s = bulk.summarize(results)
        prog.empty()
        msg = f"Finished {s['ok']} of {s['n']}. Recommendations: {s['advice_counts']}"
        if s["total_cost_usd"]:
            msg += f" · ${s['total_cost_usd']:.2f}"
        (st.warning if s["failed"] else st.success)(msg)
        if s["failures"]:
            st.caption("Didn't finish: " + ", ".join(t for t, _ in s["failures"]))
        ui.cached_report.clear()
        st.rerun()

if not tickers:
    st.info("Your watchlist is empty. Add a stock symbol above to get started — "
            "try NVDA, AAPL, or VOO if you're not sure where to begin.")
else:
    dropped = []
    for ticker in tickers:
        try:
            report = ui.cached_report(ticker)
        except errors.PermanentTickerError as e:
            # can never succeed on a retry, so don't leave it sitting there
            # broken with no way to get rid of it
            store.watchlist_remove(ticker)
            dropped.append(e.friendly())
            continue
        except Exception as e:
            # might just be a network blip — keep it, but make it removable
            with st.container(border=True):
                st.markdown(f"### {ticker}")
                st.markdown(ui.pill("Couldn't load right now", "caution"),
                            unsafe_allow_html=True)
                st.markdown(f"<span class='tc-note'>{e}</span>",
                            unsafe_allow_html=True)
                st.caption("This usually clears up on its own — try refreshing. "
                           "If it keeps happening, you can remove it.")
                if st.button("Remove", key=f"rm_err_{ticker}",
                             help=f"Remove {ticker} from your watchlist"):
                    store.watchlist_remove(ticker)
                    st.rerun()
            continue

        analysis = report.get("analysis")
        probs = (
            {k: v["probability_pct"] for k, v in analysis["scenarios"].items()}
            if analysis else report["tilted_probabilities"]
        )
        closes = report["history"]["close"]
        chg_1d = (closes[-1] / closes[-2] - 1) * 100 if len(closes) > 1 else 0.0
        chg_1m = (closes[-1] / closes[-22] - 1) * 100 if len(closes) > 22 else 0.0
        name = report["fundamentals"].get("shortName") or ""

        with st.container(border=True):
            left, mid, right = st.columns([1.5, 2.1, 1.25])

            with left:
                st.markdown(f"### {ticker}")
                if name:
                    st.markdown(f"<span class='tc-term'>{name}</span>",
                                unsafe_allow_html=True)
                st.markdown(
                    f"<div style='margin-top:.5rem;font-size:1.5rem;font-weight:650'>"
                    f"${report['spot']:.2f}</div>"
                    f"<span class='tc-note'>"
                    f"{ui.change_words(chg_1d, 'day')} today ({chg_1d:+.1f}%)<br>"
                    f"{ui.change_words(chg_1m, 'month')} over the past month "
                    f"({chg_1m:+.1f}%)</span>",
                    unsafe_allow_html=True)

            with mid:
                st.markdown("<span class='tc-term'>Where this could be in "
                            "12 months</span>", unsafe_allow_html=True)
                rows = []
                for key in ("bull", "base", "bear"):
                    target = report["scenarios"][key]["target"]
                    rows.append(
                        f"<div class='tc-row'>"
                        f"<span class='lbl'>{ui.scenario_label(key)}</span>"
                        f"<span class='val'>${target:,.0f}</span>"
                        f"<span class='pct'>· {probs[key]}% chance</span></div>")
                st.markdown("".join(rows), unsafe_allow_html=True)

            with right:
                if analysis:
                    st.markdown(ui.advice_pill(analysis["advice"]),
                                unsafe_allow_html=True)
                    st.markdown(
                        f"<span class='tc-note'>{ui.plain_advice(analysis['advice'])}"
                        f"</span>", unsafe_allow_html=True)
                else:
                    st.markdown(ui.pill("No commentary yet", "muted"),
                                unsafe_allow_html=True)
                rk = report.get("risk")
                if rk:
                    label, tone = ui.risk_label(rk)
                    st.markdown(
                        f"<div style='margin-top:.45rem'>"
                        f"{ui.pill(label, tone, wrap=True)}</div>"
                        f"<span class='tc-note'>{ui.loss_note(rk)}</span>",
                        unsafe_allow_html=True)

                if st.button("See details", key=f"detail_{ticker}",
                             width="stretch"):
                    st.session_state["selected_ticker"] = ticker
                    st.switch_page("views/stock_detail.py")
                if st.button("Remove", key=f"rm_{ticker}",
                             help=f"Remove {ticker} from your watchlist",
                             width="stretch"):
                    store.watchlist_remove(ticker)
                    st.rerun()

    for message in dropped:
        st.warning(f"Removed from your watchlist — {message}")

ui.disclaimer()
