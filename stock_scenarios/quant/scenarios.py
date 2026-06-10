"""Map a terminal price distribution to bull/base/bear bands.

The partition at P25/P75 makes prior probabilities an exact 25/50/25 split
(sum always 100); targets are the median of each band so they represent the
band rather than its extreme edge. The information content is in band width
(volatility), target-vs-spot (drift), and the downstream macro/LLM tilts.
"""

import numpy as np

from stock_scenarios import config


def build_scenarios(terminal: np.ndarray, spot: float) -> dict:
    bear_target = float(np.percentile(terminal, config.BEAR_TARGET_PCT))
    base_target = float(np.percentile(terminal, config.BASE_TARGET_PCT))
    bull_target = float(np.percentile(terminal, config.BULL_TARGET_PCT))

    prior_bear = config.BAND_LOWER_PCT                      # 25
    prior_bull = 100 - config.BAND_UPPER_PCT                # 25
    prior_base = 100 - prior_bear - prior_bull              # 50

    def pct_vs_spot(target: float) -> float:
        return round((target / spot - 1) * 100, 1)

    return {
        "bear": {
            "target": round(bear_target, 2),
            "return_pct": pct_vs_spot(bear_target),
            "prior_prob": prior_bear,
        },
        "base": {
            "target": round(base_target, 2),
            "return_pct": pct_vs_spot(base_target),
            "prior_prob": prior_base,
        },
        "bull": {
            "target": round(bull_target, 2),
            "return_pct": pct_vs_spot(bull_target),
            "prior_prob": prior_bull,
        },
    }
