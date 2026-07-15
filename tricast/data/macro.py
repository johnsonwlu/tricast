"""FRED macro series via fredapi, cached in SQLite with a 24h freshness TTL."""

import logging
import os

import pandas as pd

from tricast import config, store

log = logging.getLogger(__name__)


def get_series(series_id: str, db_path=config.DB_PATH) -> pd.Series:
    """Return the series as a pandas Series indexed by date (last 5 years)."""
    if not store.macro_is_fresh(series_id, db_path=db_path):
        _fetch_and_cache(series_id, db_path)
    else:
        log.info("macro cache hit: %s", series_id)
    points = store.macro_load(series_id, db_path=db_path)
    if not points:
        raise ValueError(f"No data for FRED series {series_id!r}")
    s = pd.Series(
        [v for _, v in points],
        index=pd.to_datetime([d for d, _ in points]),
        name=series_id,
    )
    return s.dropna()


def _fetch_and_cache(series_id: str, db_path) -> None:
    from fredapi import Fred  # deferred: import requires a key to be useful

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and add it to .env"
        )
    log.info("fetching FRED series: %s", series_id)
    fred = Fred(api_key=api_key)
    data = fred.get_series(series_id)
    data = data[data.index >= data.index.max() - pd.DateOffset(years=5)]
    points = [(idx.date().isoformat(), float(v)) for idx, v in data.items() if pd.notna(v)]
    store.macro_put(series_id, points, db_path=db_path)


def get_all_series(db_path=config.DB_PATH) -> dict[str, pd.Series]:
    return {sid: get_series(sid, db_path=db_path) for sid in config.FRED_SERIES}
