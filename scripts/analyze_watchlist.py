"""Bulk-analyze the whole watchlist (or a given list) in one run.

Examples:
  python scripts/analyze_watchlist.py                 # LLM analysis, whole watchlist
  python scripts/analyze_watchlist.py AAPL MSFT NVDA  # just these
  python scripts/analyze_watchlist.py --no-llm        # quant only (free, fast)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from tricast import bulk, store  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tickers", nargs="*", help="tickers (default: whole watchlist)")
    p.add_argument("--no-llm", action="store_true", help="quant only, skip the LLM")
    args = p.parse_args()

    tickers = [t.upper() for t in args.tickers] or store.watchlist_all()
    if not tickers:
        print("Watchlist is empty and no tickers given. Nothing to do.")
        return

    def cb(done, total, ticker, row):
        mark = row["advice"] or "-" if row["ok"] else f"FAILED ({row['error']})"
        print(f"  [{done}/{total}] {ticker}: {mark}")

    print(f"Analyzing {len(tickers)} ticker(s){' (quant only)' if args.no_llm else ''}…")
    results = bulk.analyze_many(tickers, run_llm=not args.no_llm, progress_cb=cb)

    s = bulk.summarize(results)
    print(f"\nDone: {s['ok']} ok, {s['failed']} failed. Advice: {s['advice_counts']}")
    if s["total_cost_usd"]:
        print(f"Total LLM cost: ${s['total_cost_usd']:.3f}")
    if s["failures"]:
        print("Failures:", json.dumps(s["failures"], indent=2))


if __name__ == "__main__":
    main()
