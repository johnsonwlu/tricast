"""Shared Streamlit helpers: the visual system and the plain-language layer.

Two jobs live here, deliberately together:

1. **Look** — one soft, warm card style and a small set of semantic tones
   (good / caution / poor / calm) so every page reads as the same app. The
   palette lives in `.streamlit/config.toml`; this module only adds the pieces
   Streamlit's theme can't express (cards, pills, spacing).

2. **Language** — the app is for someone new to stock picking, so the UI never
   says "P25-P75 band" or "prob. of loss" on its own. Every finance term is
   translated to an everyday phrase, with the real term kept alongside in small
   text (so you learn it) and an explanation one hover away. All of that copy is
   centralized here rather than scattered across pages, so the vocabulary stays
   consistent and is testable.
"""

from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv

from tricast import config

load_dotenv(config.PROJECT_ROOT / ".env")

# --- Semantic tones -------------------------------------------------------
# Colors are intentionally muted: tints, not alarm lights.
TONES = {
    "good":    {"fg": "#1F7A5C", "bg": "#E9F4EF", "br": "#C4E1D3"},
    "caution": {"fg": "#8F6516", "bg": "#FBF2DE", "br": "#EDDCB2"},
    "poor":    {"fg": "#A6443E", "bg": "#FAEBE9", "br": "#EECCC8"},
    "calm":    {"fg": "#3A66BF", "bg": "#ECF1FB", "br": "#CCDBF3"},
    "muted":   {"fg": "#6B655E", "bg": "#F1EEE9", "br": "#E2DCD3"},
}

REGIME_COLORS = {"Expansionary": "green", "Neutral": "gray", "Contractionary": "red"}
ADVICE_COLORS = {"buy": "green", "hold": "orange", "avoid": "red"}
ADVICE_TONES = {"buy": "good", "hold": "caution", "avoid": "poor"}

# --- Plain-language vocabulary -------------------------------------------
# (everyday phrase, the real finance term, what it actually means)
SCENARIOS = {
    "bull": ("If things go well", "bull case",
             "The optimistic outcome: roughly the top quarter of what the "
             "simulation thinks could happen over the next 12 months."),
    "base": ("Most likely", "base case",
             "The middle-of-the-road outcome: the simulation's central range, "
             "where about half of all simulated outcomes landed."),
    "bear": ("If things go poorly", "bear case",
             "The pessimistic outcome: roughly the bottom quarter of what the "
             "simulation thinks could happen over the next 12 months."),
}

# Tooltips for every jargon word that survives onto the screen.
HELP = {
    "chance": "How often this outcome happened across 10,000 simulated versions "
              "of the next 12 months. It is an estimate, not a promise.",
    "target": "The typical price within this outcome, 12 months from now.",
    "spot": "What one share costs right now.",
    "sharpe": "Reward for the risk taken. It compares the expected gain against "
              "how bumpy the ride is. Higher is better: above ~0.75 is strong, "
              "below 0 means you'd expect to do worse than a savings account.",
    "sortino": "Like the reward-for-risk score, but it only counts the downward "
               "swings — it doesn't punish a stock for jumping upward.",
    "prob_loss": "How often this stock ended up worth less than today's price "
                 "across the simulations.",
    "cvar": "When things go badly, how bad it typically gets — the average loss "
            "across the worst 5% of simulated outcomes.",
    "volatility": "How much the price swings around. Higher means a bumpier, "
                  "less predictable ride.",
    "beta": "How much this moves compared to the overall market. 1.0 means it "
            "moves with the market; 2.0 means it swings about twice as hard.",
    "fwd_pe": "Price compared to the profit the company is expected to earn. "
              "Roughly: how many years of expected profit you're paying for.",
    "rev_growth": "How much the company's sales grew compared to a year ago.",
    "margin": "How much of each sales dollar the company keeps as profit.",
    "analyst_target": "The average price professional analysts expect in 12 "
                      "months. Included as one input, not treated as truth.",
    "regime": "Whether the wider economy currently looks supportive of stocks, "
              "based on interest rates, inflation, jobs and market nerves.",
    "brier": "A score for how good past forecasts turned out to be. Lower is "
             "better; 0 would be perfect.",
    "cone": "The shaded fan shows the range of prices the simulation produced. "
            "The darker middle band is where outcomes landed about half the time.",
}


def page_setup(title: str, subtitle: str | None = None):
    """Per-page chrome. `st.set_page_config` deliberately lives in app.py: with
    st.navigation the entrypoint and the page run in the same script pass, so
    calling it here too would raise."""
    _inject_css()
    st.title(title)
    if subtitle:
        st.markdown(f"<p class='tc-sub'>{subtitle}</p>", unsafe_allow_html=True)


def _inject_css():
    """Soft cards, gentle pills, roomier text. Kept small and defensive: if a
    Streamlit internal changes, the app still works — it just looks plainer."""
    st.markdown("""
    <style>
      .block-container { padding-top: 2.4rem; max-width: 1280px; }
      h1 { font-weight: 700; letter-spacing: -.02em; }
      h2, h3 { letter-spacing: -.01em; }
      .tc-sub { color:#6B655E; font-size:1.02rem; margin:-.5rem 0 1.1rem; }

      /* Bordered containers -> soft raised cards */
      [data-testid="stVerticalBlockBorderWrapper"] {
        background:#FFFFFF; border:1px solid #E7E1D8; border-radius:18px;
        box-shadow:0 1px 2px rgba(43,42,40,.04), 0 6px 18px rgba(43,42,40,.045);
        transition: box-shadow .18s ease, border-color .18s ease;
      }
      [data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow:0 2px 4px rgba(43,42,40,.05), 0 10px 26px rgba(43,42,40,.07);
        border-color:#DDD5C9;
      }

      /* Metrics: quieter labels, friendlier numbers */
      [data-testid="stMetricLabel"] p { color:#6B655E; font-size:.86rem; font-weight:500; }
      [data-testid="stMetricValue"] { font-weight:650; letter-spacing:-.02em; }

      /* Buttons: rounded and calm */
      .stButton > button {
        border-radius:12px; font-weight:600; padding:.42rem 1rem;
        border:1px solid #E0D9CE; transition: transform .12s ease, box-shadow .12s ease;
      }
      .stButton > button:hover { transform: translateY(-1px); box-shadow:0 4px 12px rgba(43,42,40,.08); }

      /* Pills + scenario rows */
      /* nowrap keeps short chips ("Worth buying") on one line; anything longer
         must opt into wrapping or it overflows its card */
      .tc-pill { display:inline-block; padding:.2rem .68rem; border-radius:999px;
                 font-size:.86rem; font-weight:650; border:1px solid;
                 white-space:nowrap; max-width:100%; vertical-align:top; }
      .tc-pill-wrap { white-space:normal; overflow-wrap:anywhere; }
      .tc-row { display:flex; align-items:baseline; gap:.55rem; margin:.3rem 0; }
      .tc-row .lbl { color:#4A443E; min-width:9.5rem; }
      .tc-row .val { font-weight:650; font-variant-numeric:tabular-nums; }
      .tc-row .pct { color:#6B655E; font-size:.9rem; }
      .tc-term { color:#8C857C; font-size:.82rem; }
      .tc-note { color:#6B655E; font-size:.92rem; }
      /* keeps a row of signal cards aligned when their titles wrap differently */
      .tc-cardtitle { font-weight:650; min-height:3.1rem; display:block; }
      .tc-cardsub { color:#8C857C; font-size:.82rem; min-height:2.6rem; display:block; }
    </style>
    """, unsafe_allow_html=True)


def pill(text: str, tone: str = "muted", wrap: bool = False) -> str:
    """A short status chip. Pass wrap=True for anything longer than ~3 words —
    otherwise it stays on one line and runs outside its container."""
    t = TONES.get(tone, TONES["muted"])
    cls = "tc-pill tc-pill-wrap" if wrap else "tc-pill"
    return (f"<span class='{cls}' style='color:{t['fg']};background:{t['bg']};"
            f"border-color:{t['br']}'>{text}</span>")


def explainer():
    """The 'I've never picked a stock before' primer. Collapsed by default so it
    never gets in the way of someone who already knows this."""
    with st.expander("New here? How to read this in 30 seconds"):
        st.markdown("""
This app does **not** know the future. It runs the last five years of a
stock's daily ups and downs through 10,000 simulated versions of the next
12 months, then sorts what came out into three stories:

- **If things go well** — the optimistic quarter of outcomes.
- **Most likely** — the middle half, where outcomes landed most often.
- **If things go poorly** — the pessimistic quarter of outcomes.

Each one shows a **chance** (how often it happened in the simulations) and a
**price** (what a share was typically worth in that story).

**Three things worth knowing before you lean on any of it:**

1. A high "if things go well" number is not free — it usually comes with an
   equally large downside. Check the *chance of losing money* before you get
   excited about the upside.
2. The chances come from history repeating itself in new orders. Genuinely new
   events — a crash, a scandal, a breakthrough — are not in there.
3. The written commentary comes from an AI model. It explains the numbers; it
   doesn't get to invent them. It can still be wrong, so treat it as a starting
   point for your own thinking, not advice.
        """)


# --- Plain-language formatting helpers ------------------------------------

def scenario_label(name: str) -> str:
    return SCENARIOS[name][0]


def scenario_term(name: str) -> str:
    return SCENARIOS[name][1]


def scenario_help(name: str) -> str:
    return SCENARIOS[name][2]


# What counts as "a lot" depends entirely on the window: a 4% day is dramatic,
# a 4% month is unremarkable. Thresholds are (flat-below, little-below, lot-above).
CHANGE_SCALES = {
    "day":   (0.25, 1.2, 3.0),
    "month": (1.0, 4.0, 10.0),
    "year":  (3.0, 12.0, 30.0),
}


def change_words(pct: float, scale: str = "day") -> str:
    """Turn a percent move into words, so a number isn't the only signal.
    `scale` is the time window the percentage covers — without it a modest
    monthly drift would be described as dramatically as a one-day swing."""
    flat, little, lot = CHANGE_SCALES.get(scale, CHANGE_SCALES["day"])
    a = abs(pct)
    if a < flat:
        return "roughly flat"
    size = "a lot" if a >= lot else ("a little" if a < little else "moderately")
    return f"{'up' if pct > 0 else 'down'} {size}"


def chance_words(pct: float) -> str:
    """Describe a probability in everyday terms."""
    if pct >= 65:
        return "very likely"
    if pct >= 45:
        return "about as likely as not"
    if pct >= 25:
        return "possible"
    return "unlikely"


def plain_advice(advice: str) -> str:
    """One friendly sentence explaining what the recommendation means."""
    return {
        "buy": "The model thinks the likely reward justifies the risk here.",
        "hold": "Nothing wrong with it, but the reward doesn't clearly beat the "
                "risk right now — no rush either way.",
        "avoid": "The likely reward doesn't look worth the risk at today's price.",
    }.get(advice, "No recommendation yet.")


RISK_TONES = {"strong": "good", "fair": "calm", "weak": "caution", "poor": "poor"}
RISK_WORDS = {
    "strong": "Good reward for risk",
    "fair": "Fair reward for risk",
    "weak": "Weak reward for risk",
    "poor": "Poor reward for risk",
}


def risk_label(rk: dict) -> tuple[str, str]:
    """Short chip-sized summary -> (label, tone). Kept to three words so it
    fits a narrow watchlist card; the loss figure goes alongside as text."""
    label = rk.get("label", "fair")
    return RISK_WORDS.get(label, "Reward for risk"), RISK_TONES.get(label, "muted")


def loss_note(rk: dict) -> str:
    return f"Loses money in {rk.get('prob_loss_pct', 0):.0f}% of simulations"


def risk_sentence(rk: dict) -> tuple[str, str]:
    """Full one-line summary -> (sentence, tone), for wide layouts only."""
    words, tone = risk_label(rk)
    return f"{words} · {loss_note(rk).lower()}", tone


def advice_badge(advice: str) -> str:
    """Inline colored badge (kept as markdown so it composes inside f-strings)."""
    color = ADVICE_COLORS.get(advice, "gray")
    return f":{color}[**{advice.upper()}**]"


def advice_pill(advice: str) -> str:
    label = {"buy": "Worth buying", "hold": "Hold / no rush",
             "avoid": "Better to avoid"}.get(advice, advice.title())
    return pill(label, ADVICE_TONES.get(advice, "muted"))


def regime_plain(macro_state: dict) -> tuple[str, str]:
    """(sentence, tone) describing the economy in everyday language."""
    regime = macro_state.get("regime", "Neutral")
    if regime.startswith("Expansionary"):
        return ("The wider economy currently looks supportive of stocks.", "good")
    if regime.startswith("Contractionary"):
        return ("The wider economy currently looks like a headwind for stocks.", "poor")
    return ("The wider economy looks neither especially helpful nor harmful "
            "for stocks right now.", "muted")


def disclaimer():
    st.divider()
    st.caption(config.DISCLAIMER)


def regime_banner(macro_state: dict):
    sentence, tone = regime_plain(macro_state)
    with st.container(border=True):
        c = st.columns([1.1, 3])
        c[0].markdown(
            pill(f"Economy: {macro_state['regime'].split(' (')[0]}", tone),
            unsafe_allow_html=True)
        c[1].markdown(
            f"<span class='tc-note'>{sentence} "
            f"<span class='tc-term'>(score {macro_state['score']:+.2f})</span></span>",
            unsafe_allow_html=True)


def age_str(created_at: float | None) -> str:
    if not created_at:
        return "—"
    hours = (datetime.now(timezone.utc).timestamp() - created_at) / 3600
    if hours < 1:
        return f"{int(hours * 60)}m ago"
    if hours < 48:
        return f"{hours:.0f}h ago"
    return f"{hours / 24:.0f}d ago"


@st.cache_data(ttl=900, show_spinner="Loading economic data…")
def cached_macro_state():
    from tricast import pipeline
    return pipeline.get_macro_state()


@st.cache_data(ttl=900, show_spinner="Crunching the numbers…")
def cached_report(ticker: str):
    from tricast import pipeline
    return pipeline.build_report(ticker, run_llm=False)
