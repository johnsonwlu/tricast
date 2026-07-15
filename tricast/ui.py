"""Shared Streamlit helpers used by all pages."""

from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from tricast import config

load_dotenv(config.PROJECT_ROOT / ".env")

REGIME_COLORS = {"Expansionary": "green", "Neutral": "gray", "Contractionary": "red"}
ADVICE_COLORS = {"buy": "green", "hold": "orange", "avoid": "red"}


def page_setup(title: str):
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)


def disclaimer():
    st.divider()
    st.caption(config.DISCLAIMER)


def regime_banner(macro_state: dict):
    color = REGIME_COLORS.get(macro_state["regime"], "gray")
    st.markdown(
        f"**Macro regime:** :{color}[{macro_state['regime']}] "
        f"(score {macro_state['score']:+.2f})"
    )


def advice_badge(advice: str) -> str:
    color = ADVICE_COLORS.get(advice, "gray")
    return f":{color}[**{advice.upper()}**]"


def age_str(created_at: float | None) -> str:
    if not created_at:
        return "—"
    hours = (datetime.now(timezone.utc).timestamp() - created_at) / 3600
    if hours < 1:
        return f"{int(hours * 60)}m ago"
    if hours < 48:
        return f"{hours:.0f}h ago"
    return f"{hours / 24:.0f}d ago"


@st.cache_data(ttl=900, show_spinner="Loading macro data…")
def cached_macro_state():
    from tricast import pipeline
    return pipeline.get_macro_state()


@st.cache_data(ttl=900, show_spinner="Building report…")
def cached_report(ticker: str):
    from tricast import pipeline
    return pipeline.build_report(ticker, run_llm=False)
