"""Market data via yfinance, cached in SQLite so repeat runs stay offline."""

import logging
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from tricast import config, store

log = logging.getLogger(__name__)

# Subset of yfinance .info worth keeping: enough for the LLM, small enough to cache.
_FUNDAMENTAL_KEYS = [
    "shortName", "sector", "industry", "marketCap",
    "trailingPE", "forwardPE", "priceToBook",
    "profitMargins", "operatingMargins", "revenueGrowth", "earningsGrowth",
    "debtToEquity", "freeCashflow", "dividendYield", "beta",
    "targetMeanPrice", "targetHighPrice", "targetLowPrice",
    "recommendationKey", "numberOfAnalystOpinions",
    "currentPrice", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
]


def is_valid_ticker(ticker: str) -> bool:
    """Cheap existence check before adding to the watchlist."""
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        return not hist.empty
    except Exception:
        return False


def get_prices(ticker: str, db_path=config.DB_PATH) -> pd.DataFrame:
    """Daily OHLCV, lookback per config. Incremental: only fetches dates newer
    than the cache. Returns a DataFrame indexed by date with a Close column."""
    ticker = ticker.upper()
    last = store.prices_last_date(ticker, db_path=db_path)
    today = date.today()

    if last is None:
        start = today - timedelta(days=int(config.LOOKBACK_YEARS * 365.25) + 14)
        _fetch_and_cache(ticker, start, db_path)
    elif date.fromisoformat(last) < today - timedelta(days=1):
        _fetch_and_cache(ticker, date.fromisoformat(last) + timedelta(days=1), db_path)
    else:
        log.info("prices cache hit: %s (through %s)", ticker, last)

    rows = store.prices_load(ticker, db_path=db_path)
    if not rows:
        raise ValueError(f"No price data available for {ticker!r}")
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def _fetch_and_cache(ticker: str, start: date, db_path) -> None:
    log.info("fetching prices: %s from %s", ticker, start)
    hist = yf.Ticker(ticker).history(start=start.isoformat(), auto_adjust=True)
    if hist.empty:
        return
    rows = [
        (idx.date().isoformat(), float(r["Open"]), float(r["High"]),
         float(r["Low"]), float(r["Close"]), int(r["Volume"]))
        for idx, r in hist.iterrows()
    ]
    store.prices_upsert(ticker, rows, db_path=db_path)


def get_fundamentals(ticker: str, db_path=config.DB_PATH) -> dict:
    """Fundamentals + analyst targets subset, cached with a 24h TTL."""
    ticker = ticker.upper()
    cached = store.fundamentals_get(ticker, db_path=db_path)
    if cached is not None:
        log.info("fundamentals cache hit: %s", ticker)
        return cached
    log.info("fetching fundamentals: %s", ticker)
    info = yf.Ticker(ticker).info or {}
    payload = {k: info.get(k) for k in _FUNDAMENTAL_KEYS}
    store.fundamentals_put(ticker, payload, db_path=db_path)
    return payload


def get_vix(db_path=config.DB_PATH) -> float:
    """Latest VIX close (cached like any other ticker, 1y history)."""
    df = get_prices("^VIX", db_path=db_path)
    return float(df["close"].iloc[-1])


def get_spot(ticker: str, db_path=config.DB_PATH) -> float:
    return float(get_prices(ticker, db_path=db_path)["close"].iloc[-1])
