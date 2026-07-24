"""Learned calibration parameters, refit from the backtester.

The self-correcting loop: the backtest (backtest.py) *measures* that the
simulated cone is systematically too wide — across 1,150 historical predictions
the P25-P75 band covered ~65% of outcomes instead of 50%, because the 5-year
lookback spans high-volatility years (2020, 2022). Rather than hand-tune the
model, `scripts/calibrate.py` *fits* a single scalar `vol_scale` that shrinks
the sampled return dispersion until the model's own bands land at their nominal
frequencies, and persists it here. The live engine (`montecarlo.simulate`)
loads it automatically on every prediction, so re-running the calibrator is all
it takes for the model to keep improving as new data arrives.

One scalar is a deliberate choice: with a personal-sized dataset, fitting many
parameters would overfit and lie to you. A single dispersion multiplier, fit
against thousands of pseudo-observations and validated on held-out tickers, is
about as much learning as the data honestly supports.
"""

import json
from functools import lru_cache

from tricast import config


@lru_cache(maxsize=1)
def _load() -> dict:
    """Read the persisted calibration file once (cached). Missing/corrupt file
    -> empty dict, so the engine falls back to the neutral default."""
    try:
        with open(config.CALIBRATION_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def load_vol_scale() -> float:
    """The learned dispersion multiplier applied to sampled returns. Defaults to
    1.0 (no correction) until a calibration exists; clamped to a sane range so a
    bad fit can never silently distort every prediction."""
    v = _load().get("vol_scale", config.DEFAULT_VOL_SCALE)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return config.DEFAULT_VOL_SCALE
    return min(max(v, config.VOL_SCALE_MIN), config.VOL_SCALE_MAX)


def metadata() -> dict:
    """Full persisted record (vol_scale, when it was fit, on what, before/after
    coverage) for display in the UI / CLI. Empty dict if never calibrated."""
    return dict(_load())


def save(record: dict) -> None:
    """Persist a calibration record and invalidate the read cache so the next
    prediction picks it up immediately."""
    config.CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CALIBRATION_PATH, "w") as f:
        json.dump(record, f, indent=2)
    _load.cache_clear()


def fit_vol_scale(evaluate, grid) -> tuple[float, list[dict]]:
    """Grid-search the dispersion multiplier against measured calibration.

    `evaluate(vol_scale) -> summary` must return a backtest summary dict (see
    backtest.summarize_backtest) with 'coverage_p25_p75' and 'mean_pit'. Returns
    (best_scale, table). "Best" minimizes the distance of the central band's
    coverage from its nominal 0.50, tie-broken by centering (mean PIT -> 0.5) —
    so we only ever shrink dispersion, never rig the median.
    """
    table = []
    for s in grid:
        summ = evaluate(s)
        table.append({
            "vol_scale": round(float(s), 4),
            "coverage_p25_p75": summ.get("coverage_p25_p75"),
            "coverage_p10_p90": summ.get("coverage_p10_p90"),
            "mean_pit": summ.get("mean_pit"),
            "n": summ.get("n"),
        })
    scored = [r for r in table if r["coverage_p25_p75"] is not None]
    if not scored:
        raise ValueError("no evaluable grid points (empty backtest?)")
    best = min(scored, key=lambda r: (abs(r["coverage_p25_p75"] - 0.50),
                                      abs((r["mean_pit"] or 0.5) - 0.5)))
    return best["vol_scale"], table
