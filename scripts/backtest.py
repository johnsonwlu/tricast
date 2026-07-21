"""Backtest the quant scenario engine against history.

Examples:
  python scripts/backtest.py AAPL MSFT NVDA
  python scripts/backtest.py AAPL --start 2015-01-01 --freq W --n-paths 3000
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from tricast import backtest  # noqa: E402  (path setup first)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tickers", nargs="+")
    p.add_argument("--start", default="2016-01-01", help="earliest as-of date")
    p.add_argument("--end", default=None, help="latest as-of date")
    p.add_argument("--freq", default="MS", help="as-of spacing (MS=monthly, W=weekly)")
    p.add_argument("--n-paths", type=int, default=5000)
    args = p.parse_args()

    results = backtest.run_backtest(
        args.tickers, start=args.start, end=args.end,
        freq=args.freq, n_paths=args.n_paths,
    )
    print("\n" + backtest.format_summary(backtest.summarize_backtest(results)))


if __name__ == "__main__":
    main()
