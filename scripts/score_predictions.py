"""Score matured predictions and print calibration stats.

Run any time (e.g. monthly): python scripts/score_predictions.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from stock_scenarios import ledger  # noqa: E402  (path setup first)


def main():
    scored = ledger.score_matured()
    if scored:
        print(f"Newly scored {len(scored)} prediction(s):")
        for s in scored:
            print(f"  {s['ticker']} {s['pred_date']}: {s['outcome']} "
                  f"@ {s['price']:.2f} (Brier {s['brier']})")
    else:
        print("No predictions matured since last scoring.")

    stats = ledger.summary()
    print("\nCalibration summary:")
    print(json.dumps(stats, indent=2))
    if stats["n_scored"]:
        verdict = "BEATS" if stats["beats_baseline"] else "does NOT beat"
        print(f"\nModel {verdict} the naive 25/50/25 baseline "
              f"({stats['mean_brier']} vs {stats['baseline_brier']}, lower is better).")


if __name__ == "__main__":
    main()
