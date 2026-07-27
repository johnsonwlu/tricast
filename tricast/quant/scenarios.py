"""Map a terminal price distribution to bull/base/bear scenarios.

Scenarios are defined by **return thresholds**, not percentiles. The earlier
version partitioned the distribution at P25/P75, which made the prior
probabilities an exact 25/50/25 split for every stock ever analyzed — the odds
were a property of the method, not of the company. A stable index fund and a
2.5-beta semiconductor came out identical, so the probabilities told you
nothing and the Brier score could never beat the naive baseline it was
measured against.

Now a scenario is a plain claim about the world — "loses 10% or more", "gains
20% or more" — and its probability is simply how often the simulation produced
that outcome. Those numbers genuinely differ per stock, which is the whole
point: a volatile name really is likelier to fall 10% than a diversified fund.
Targets are the median outcome *within* each region, so they represent the
region rather than its edge.
"""

import numpy as np

from tricast import config


def band_bounds(spot: float) -> dict:
    """The price boundaries between scenarios. Fixed fractions of spot, so a
    prediction can be scored against the same thresholds it was made under."""
    return {
        "lower_price": round(spot * (1 + config.BEAR_RETURN_PCT / 100), 2),
        "upper_price": round(spot * (1 + config.BULL_RETURN_PCT / 100), 2),
        "lower_return_pct": config.BEAR_RETURN_PCT,
        "upper_return_pct": config.BULL_RETURN_PCT,
    }


def _integer_pcts(fracs: dict[str, float]) -> dict[str, int]:
    """Round shares to integers that still sum to exactly 100 (largest
    remainder). Downstream code — the macro tilt, the LLM bounds check, the
    Brier score — all assume the three add up."""
    raw = {k: v * 100 for k, v in fracs.items()}
    out = {k: int(np.floor(v)) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[:100 - sum(out.values())]:
        out[k] += 1
    return out


def build_scenarios(terminal: np.ndarray, spot: float) -> dict:
    terminal = np.asarray(terminal, dtype=float)
    bounds = band_bounds(spot)
    lo, hi = bounds["lower_price"], bounds["upper_price"]

    masks = {
        "bear": terminal <= lo,
        "base": (terminal > lo) & (terminal < hi),
        "bull": terminal >= hi,
    }
    probs = _integer_pcts({k: float(m.mean()) for k, m in masks.items()})

    # A region can legitimately be empty (a low-volatility fund may produce no
    # path that loses 10%). Fall back to the threshold price itself so the
    # target is still meaningful and never NaN.
    fallback = {"bear": lo, "base": float(np.median(terminal)), "bull": hi}

    def target_for(name: str) -> float:
        vals = terminal[masks[name]]
        return float(np.median(vals)) if vals.size else fallback[name]

    out = {}
    for name in ("bear", "base", "bull"):
        target = target_for(name)
        out[name] = {
            "target": round(target, 2),
            "return_pct": round((target / spot - 1) * 100, 1),
            "prior_prob": probs[name],
        }
    return out
