"""CLI smoke test: python scripts/run_analysis.py NVDA [--data-only|--llm]"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--data-only", action="store_true",
                        help="fetch and print data, skip simulation")
    parser.add_argument("--llm", action="store_true",
                        help="include the (paid) Claude analysis")
    args = parser.parse_args()

    from tricast import pipeline
    from tricast.data import macro, market

    if args.data_only:
        prices = market.get_prices(args.ticker)
        print(f"\n{args.ticker}: {len(prices)} days, "
              f"{prices.index[0].date()} -> {prices.index[-1].date()}, "
              f"last close {prices['close'].iloc[-1]:.2f}")
        fund = market.get_fundamentals(args.ticker)
        print(json.dumps(fund, indent=2))
        for sid in ("T10Y2Y", "CPIAUCSL", "UNRATE", "FEDFUNDS"):
            s = macro.get_series(sid)
            print(f"{sid}: latest {s.iloc[-1]:.2f} @ {s.index[-1].date()}")
        print(f"VIX: {market.get_vix():.1f}")
        return

    report = pipeline.build_report(args.ticker, run_llm=args.llm)
    # cone/history are huge; print the decision-relevant parts
    slim = {k: v for k, v in report.items() if k not in ("cone", "history")}
    print(json.dumps(slim, indent=2, default=str))


if __name__ == "__main__":
    main()
