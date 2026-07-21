"""Central knobs for the scenario engine. Everything tunable lives here."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "cache.db"
load_dotenv(PROJECT_ROOT / ".env")  # must run before any os.environ reads below

# --- Quant model ---
LOOKBACK_YEARS = 5
MIN_HISTORY_DAYS = 504          # 2 trading years; refuse to analyze below this
HORIZON_DAYS = 252              # 12-month forecast
N_PATHS = 10_000
RNG_SEED = 42
BLOCK_SIZE = 21                 # block-bootstrap length (days); ~1 trading month
DRIFT_CAP_ANNUAL = 0.25         # cap each drift component to +/-25%/yr
RISK_FREE_ANNUAL = 0.043        # risk-free rate for Sharpe/Sortino (~1y T-bill)
# Terminal distribution partition: below P25 = bear, P25-P75 = base, above = bull
BAND_LOWER_PCT = 25
BAND_UPPER_PCT = 75
# Representative target inside each band (median of the band)
BEAR_TARGET_PCT = 12.5
BASE_TARGET_PCT = 50.0
BULL_TARGET_PCT = 87.5
CONE_PERCENTILES = (10, 25, 50, 75, 90)

# --- Macro regime ---
FRED_SERIES = {
    "T10Y2Y": "10y-2y Treasury spread",
    "CPIAUCSL": "CPI (all urban)",
    "UNRATE": "Unemployment rate",
    "FEDFUNDS": "Fed funds rate",
}
REGIME_EXPANSIONARY = 0.3       # composite score >= this
REGIME_CONTRACTIONARY = -0.3    # composite score <= this
TILT_MAX_PP = 10                # max bull/bear probability shift, percentage points

# --- LLM ---
# Provider: "ollama" (local, free, no key) or "anthropic" (paid API key).
# Override via LLM_PROVIDER in .env.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
MODEL_ID = "claude-sonnet-4-6"  # anthropic model; flip to "claude-fable-5" for deeper analysis
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3")
MAX_TOKENS = 4000
LLM_ADJUST_MAX_PP = 10          # LLM may move each probability at most this far

# --- Caching TTLs (seconds) ---
FUNDAMENTALS_TTL = 24 * 3600
MACRO_TTL = 24 * 3600

DISCLAIMER = (
    "This tool is for personal research only — it is **not financial advice**. "
    "Probabilities and price targets are model estimates derived from historical "
    "data and an LLM's interpretation; they can be badly wrong. Past performance "
    "does not guarantee future results."
)
