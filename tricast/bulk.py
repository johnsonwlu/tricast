"""Bulk analysis: run reports across many tickers in one pass.

Sequential by design — the local model serializes on one host anyway, and
sequential avoids SQLite write contention. Each ticker is isolated: one failure
(bad ticker, model timeout) is recorded and the batch continues. Unchanged
inputs are served from the analysis cache, so re-running a whole watchlist is
cheap and only pays for stocks whose data actually moved.
"""

import logging
from collections import Counter

from tricast import config, pipeline, store

log = logging.getLogger(__name__)


def analyze_many(tickers: list[str], run_llm: bool = True,
                 progress_cb=None, db_path=config.DB_PATH) -> list[dict]:
    """Build a report for each ticker. `progress_cb(done, total, ticker, row)`
    is called after each one (for CLI/dashboard progress). Returns one row per
    ticker with ok/advice/cost/error."""
    results = []
    n = len(tickers)
    for i, ticker in enumerate(tickers):
        ticker = ticker.upper()
        try:
            report = pipeline.build_report(ticker, run_llm=run_llm, db_path=db_path)
            analysis = report.get("analysis")
            row = {
                "ticker": ticker,
                "ok": True,
                "advice": analysis["advice"] if analysis else None,
                "cost_usd": float(analysis.get("cost_usd", 0.0)) if analysis else 0.0,
                "error": None,
            }
        except Exception as e:
            log.warning("bulk analyze failed for %s: %s", ticker, e)
            row = {"ticker": ticker, "ok": False, "advice": None,
                   "cost_usd": 0.0, "error": str(e)}
        results.append(row)
        if progress_cb:
            progress_cb(i + 1, n, ticker, row)
    return results


def analyze_watchlist(run_llm: bool = True, progress_cb=None,
                      db_path=config.DB_PATH) -> list[dict]:
    """Convenience: bulk-analyze every ticker on the watchlist."""
    return analyze_many(store.watchlist_all(db_path=db_path), run_llm=run_llm,
                        progress_cb=progress_cb, db_path=db_path)


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if r["ok"]]
    return {
        "n": len(results),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "advice_counts": dict(Counter(r["advice"] for r in ok if r["advice"])),
        "total_cost_usd": round(sum(r["cost_usd"] for r in results), 4),
        "failures": [(r["ticker"], r["error"]) for r in results if not r["ok"]],
    }
