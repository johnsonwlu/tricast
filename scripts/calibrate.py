"""Refit the model's learned dispersion correction (vol_scale) from history.

This is the self-correcting loop in one command: it replays the quant engine
across past dates, measures how wide the simulated cone actually is, and fits
the single scalar that shrinks it to its nominal calibration — then persists it
so every future prediction uses the improved value. Re-run it periodically (it's
an offline monthly-ish job) and the model keeps tracking reality.

To guard the one fitted parameter against overfitting, tickers are split into a
train set (fits the scalar) and a held-out set (reports coverage the fit never
saw). Histories are fetched once and reused across the grid, so the sweep is
cheap after the initial download.

Examples:
  python scripts/calibrate.py                       # default diversified basket
  python scripts/calibrate.py AAPL MSFT NVDA XOM JPM KO --write
  python scripts/calibrate.py --start 2015-01-01 --n-paths 4000 --write
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from tricast import backtest, calibration, config  # noqa: E402

# Deliberately diversified across sectors/betas so the fit isn't a tech-only
# artifact (the user's live watchlist is mostly high-beta semis).
DEFAULT_BASKET = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "KO",
                  "JNJ", "WMT", "CAT", "SPY"]
DEFAULT_GRID = [round(0.60 + 0.05 * i, 2) for i in range(10)]  # 0.60 .. 1.05


def _prefetch(tickers):
    """Load each ticker's full history once; return a dict-backed price_loader
    so the grid sweep never re-downloads."""
    cache = {}
    for t in tickers:
        try:
            cache[t] = backtest._load_long_history(t)
        except Exception as e:  # noqa: BLE001
            logging.warning("skip %s: %s", t, e)
    if not cache:
        sys.exit("No histories could be loaded — check tickers / network.")
    return cache, (lambda t: cache[t])


def _coverage_line(tag, summ):
    return (f"  {tag:<9} n={summ['n']:<5} "
            f"P25-P75 {summ['coverage_p25_p75']:.1%} (want 50%)  "
            f"P10-P90 {summ['coverage_p10_p90']:.1%} (want 80%)  "
            f"meanPIT {summ['mean_pit']:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tickers", nargs="*", default=DEFAULT_BASKET,
                   help="basket to calibrate on (default: diversified 10)")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--freq", default="MS")
    p.add_argument("--n-paths", type=int, default=4000)
    p.add_argument("--holdout", type=int, default=3,
                   help="how many tickers to hold out for validation")
    p.add_argument("--write", action="store_true",
                   help="persist the fitted value to calibration.json")
    args = p.parse_args()

    tickers = args.tickers or DEFAULT_BASKET
    holdout = tickers[-args.holdout:] if 0 < args.holdout < len(tickers) else []
    train = [t for t in tickers if t not in holdout]
    print(f"Train:   {train}\nHoldout: {holdout or '(none)'}\n")

    cache, loader = _prefetch(tickers)
    train = [t for t in train if t in cache]
    holdout = [t for t in holdout if t in cache]

    def evaluate(scale, subset):
        results = backtest.run_backtest(
            subset, start=args.start, freq=args.freq,
            n_paths=args.n_paths, vol_scale=scale, price_loader=loader,
        )
        return backtest.summarize_backtest(results)

    print(f"Sweeping vol_scale over {DEFAULT_GRID} on train set…\n")
    best, table = calibration.fit_vol_scale(
        lambda s: evaluate(s, train), DEFAULT_GRID)

    print("  vol_scale   P25-P75   P10-P90   meanPIT   n")
    for r in table:
        star = " <-- best" if r["vol_scale"] == best else ""
        print(f"  {r['vol_scale']:>6.2f}     {r['coverage_p25_p75']:>6.1%}   "
              f"{r['coverage_p10_p90']:>6.1%}   {r['mean_pit']:>6.3f}   "
              f"{r['n']}{star}")

    print(f"\nBest vol_scale on train: {best}\n")
    print("Validation (fit never saw these):")
    base_train = evaluate(1.0, train)
    fit_train = evaluate(best, train)
    print(_coverage_line("train@1.0", base_train))
    print(_coverage_line(f"train@{best}", fit_train))
    if holdout:
        base_hold = evaluate(1.0, holdout)
        fit_hold = evaluate(best, holdout)
        print(_coverage_line("hold@1.0", base_hold))
        print(_coverage_line(f"hold@{best}", fit_hold))

    record = {
        "vol_scale": best,
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_tickers": train,
        "holdout_tickers": holdout,
        "grid": DEFAULT_GRID,
        "start": args.start,
        "freq": args.freq,
        "n_paths": args.n_paths,
        "train_coverage_p25_p75": {"before": base_train["coverage_p25_p75"],
                                   "after": fit_train["coverage_p25_p75"]},
    }
    if holdout:
        record["holdout_coverage_p25_p75"] = {
            "before": base_hold["coverage_p25_p75"],
            "after": fit_hold["coverage_p25_p75"]}

    if args.write:
        calibration.save(record)
        print(f"\nWrote {config.CALIBRATION_PATH} — future predictions now use "
              f"vol_scale={best}.")
    else:
        print("\n(dry run — re-run with --write to persist. Record would be:)")
        import json
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
