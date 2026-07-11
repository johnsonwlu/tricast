"""SQLite persistence: watchlist, data caches, and saved analyses."""

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from stock_scenarios import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlist (
    ticker TEXT PRIMARY KEY,
    added_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume INTEGER,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS fundamentals (
    ticker TEXT PRIMARY KEY,
    fetched_at REAL NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS macro_series (
    series_id TEXT NOT NULL,
    date TEXT NOT NULL,
    value REAL,
    PRIMARY KEY (series_id, date)
);
CREATE TABLE IF NOT EXISTS macro_meta (
    series_id TEXT PRIMARY KEY,
    fetched_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS analyses (
    ticker TEXT NOT NULL,
    inputs_hash TEXT NOT NULL,
    created_at REAL NOT NULL,
    report_json TEXT NOT NULL,
    PRIMARY KEY (ticker, inputs_hash)
);
CREATE TABLE IF NOT EXISTS predictions (
    ticker TEXT NOT NULL,
    pred_date TEXT NOT NULL,        -- one prediction per ticker per day
    created_at REAL NOT NULL,
    horizon_end TEXT NOT NULL,
    spot REAL NOT NULL,
    band_lower REAL NOT NULL,       -- P25 of terminal dist: below = bear happened
    band_upper REAL NOT NULL,       -- P75 of terminal dist: above = bull happened
    bear_target REAL, base_target REAL, bull_target REAL,
    p_bear INTEGER NOT NULL, p_base INTEGER NOT NULL, p_bull INTEGER NOT NULL,
    model TEXT, regime TEXT, macro_score REAL,
    cone_json TEXT,                 -- percentile cone for interim tracking
    outcome TEXT,                   -- NULL until matured: bear|base|bull
    outcome_price REAL,
    brier REAL,
    PRIMARY KEY (ticker, pred_date)
);
"""


@contextmanager
def connect(db_path: Path | str = config.DB_PATH):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


# --- Watchlist ---

def watchlist_add(ticker: str, db_path=config.DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO watchlist (ticker, added_at) VALUES (?, ?)",
            (ticker.upper(), time.time()),
        )


def watchlist_remove(ticker: str, db_path=config.DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker.upper(),))


def watchlist_all(db_path=config.DB_PATH) -> list[str]:
    with connect(db_path) as conn:
        rows = conn.execute("SELECT ticker FROM watchlist ORDER BY added_at").fetchall()
    return [r["ticker"] for r in rows]


# --- Price cache ---

def prices_last_date(ticker: str, db_path=config.DB_PATH) -> str | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT MAX(date) AS d FROM prices WHERE ticker = ?", (ticker,)
        ).fetchone()
    return row["d"]


def prices_upsert(ticker: str, rows: list[tuple], db_path=config.DB_PATH) -> None:
    """rows: (date_iso, open, high, low, close, volume)"""
    with connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO prices (ticker, date, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(ticker, *r) for r in rows],
        )


def prices_load(ticker: str, db_path=config.DB_PATH) -> list[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute(
            "SELECT date, open, high, low, close, volume FROM prices "
            "WHERE ticker = ? ORDER BY date",
            (ticker,),
        ).fetchall()


# --- Fundamentals cache (24h TTL) ---

def fundamentals_get(ticker: str, ttl=config.FUNDAMENTALS_TTL, db_path=config.DB_PATH) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT fetched_at, payload_json FROM fundamentals WHERE ticker = ?", (ticker,)
        ).fetchone()
    if row and time.time() - row["fetched_at"] < ttl:
        return json.loads(row["payload_json"])
    return None


def fundamentals_put(ticker: str, payload: dict, db_path=config.DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO fundamentals (ticker, fetched_at, payload_json) VALUES (?, ?, ?)",
            (ticker, time.time(), json.dumps(payload)),
        )


# --- Macro series cache (24h TTL on freshness) ---

def macro_is_fresh(series_id: str, ttl=config.MACRO_TTL, db_path=config.DB_PATH) -> bool:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT fetched_at FROM macro_meta WHERE series_id = ?", (series_id,)
        ).fetchone()
    return bool(row) and time.time() - row["fetched_at"] < ttl


def macro_put(series_id: str, points: list[tuple[str, float]], db_path=config.DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO macro_series (series_id, date, value) VALUES (?, ?, ?)",
            [(series_id, d, v) for d, v in points],
        )
        conn.execute(
            "INSERT OR REPLACE INTO macro_meta (series_id, fetched_at) VALUES (?, ?)",
            (series_id, time.time()),
        )


def macro_load(series_id: str, db_path=config.DB_PATH) -> list[tuple[str, float]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT date, value FROM macro_series WHERE series_id = ? ORDER BY date",
            (series_id,),
        ).fetchall()
    return [(r["date"], r["value"]) for r in rows]


# --- Saved analyses (LLM cost control) ---

def analysis_get(ticker: str, inputs_hash: str, db_path=config.DB_PATH) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT created_at, report_json FROM analyses WHERE ticker = ? AND inputs_hash = ?",
            (ticker, inputs_hash),
        ).fetchone()
    if row:
        report = json.loads(row["report_json"])
        report["_created_at"] = row["created_at"]
        return report
    return None


def analysis_latest(ticker: str, db_path=config.DB_PATH) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT created_at, report_json FROM analyses WHERE ticker = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (ticker,),
        ).fetchone()
    if row:
        report = json.loads(row["report_json"])
        report["_created_at"] = row["created_at"]
        return report
    return None


def analysis_put(ticker: str, inputs_hash: str, report: dict, db_path=config.DB_PATH) -> None:
    report = {k: v for k, v in report.items() if not k.startswith("_")}
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO analyses (ticker, inputs_hash, created_at, report_json) "
            "VALUES (?, ?, ?, ?)",
            (ticker, inputs_hash, time.time(), json.dumps(report)),
        )
