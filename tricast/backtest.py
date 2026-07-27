"""Historical backtest of the quant scenario engine.

The live prediction ledger takes a year to produce its first scored point. A
backtest answers the same question *today*: if we had run this model on past
dates, were its probability bands calibrated against what actually happened?

Method (strictly no look-ahead): for each (ticker, as-of date) we simulate using
only price history up to that date, then compare the simulated terminal
distribution to the *known* realized price `horizon` trading days later.

The key statistic is the **Probability Integral Transform (PIT)**: the fraction
of simulated terminal prices below the realized price. If the simulation is
calibrated, PIT values are Uniform(0,1) across many points — so 25% of outcomes
land below the P25 band (bear), 50% in the middle (base), 25% above P75 (bull),
and the P10–P90 cone contains 80% of outcomes. Systematic deviation is
diagnostic: mean PIT < 0.5 means the model is too optimistic (outcomes keep
landing below its median); cone coverage < 80% means it understates volatility.

Scope: this validates the quant distribution — the falsifiable numeric core.
It runs drift historical-only (`analyst_target=None`), because point-in-time
analyst targets aren't reconstructable from yfinance. Macro-tilt and LLM
backtesting are separate follow-ups; see README.
"""

import logging
from collections import Counter

import numpy as np
import pandas as pd

from tricast import config, ledger
from tricast.quant import montecarlo, scenarios

log = logging.getLogger(__name__)

NOMINAL = {"bear": 0.25, "base": 0.50, "bull": 0.25}


def _load_long_history(ticker: str) -> pd.Series:
    """Max available adjusted daily closes (split/dividend adjusted, so this is
    a total-return proxy). Backtest-only: bypasses the app's 5y cache because we
    need history far enough back to have a 5y lookback *before* old as-of dates."""
    import yfinance as yf

    df = yf.Ticker(ticker).history(period="max", auto_adjust=True)
    if df.empty:
        raise ValueError(f"No history for {ticker!r}")
    s = df["Close"].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.sort_index()


def run_backtest(
    tickers: list[str],
    start: str = "2016-01-01",
    end: str | None = None,
    freq: str = "MS",                 # as-of dates: month-start by default
    horizon: int = config.HORIZON_DAYS,
    n_paths: int = 5_000,
    seed: int = config.RNG_SEED,
    block: int | None = None,
    vol_scale: float | None = None,
    price_loader=_load_long_history,
) -> list[dict]:
    """Replay the quant engine across (ticker, as-of date) pairs. Returns one
    result row per evaluable pair. `price_loader` is injectable for testing.
    `block` overrides the bootstrap block length (block=1 = old IID) for A/B.
    `vol_scale` overrides the learned dispersion multiplier (1.0 = uncorrected),
    which is what the calibrator sweeps to fit it."""
    results = []
    for ticker in tickers:
        try:
            closes = price_loader(ticker)
        except Exception as e:
            log.warning("skip %s: %s", ticker, e)
            continue

        as_ofs = pd.date_range(start, end or closes.index[-1], freq=freq)
        for d in as_ofs:
            hist = closes[closes.index <= d]
            pos = len(hist) - 1
            if pos < config.MIN_HISTORY_DAYS:
                continue                          # not enough lookback yet
            if pos + horizon >= len(closes):
                continue                          # outcome not yet realized
            outcome_price = float(closes.iloc[pos + horizon])
            outcome_date = closes.index[pos + horizon]

            sim = montecarlo.simulate(
                hist, analyst_target=None, horizon=horizon,
                n_paths=n_paths, seed=seed, block=block, vol_scale=vol_scale,
            )
            terminal = sim["terminal"]
            spot = sim["spot"]
            # classify against the same return thresholds the app forecasts on,
            # and record the probability the model gave that outcome so the
            # scenario odds can be scored for calibration, not just coverage
            bands = scenarios.build_scenarios(terminal, spot)
            bounds = scenarios.band_bounds(spot)
            band = ledger.classify_outcome(
                outcome_price, bounds["lower_price"], bounds["upper_price"])
            probs = {k: bands[k]["prior_prob"] for k in NOMINAL}
            pit = float((terminal < outcome_price).mean())
            p50 = float(np.percentile(terminal, 50))

            results.append({
                "ticker": ticker,
                "as_of": pd.Timestamp(closes.index[pos]).date().isoformat(),
                "outcome_date": pd.Timestamp(outcome_date).date().isoformat(),
                "spot": round(spot, 2),
                "outcome_price": round(outcome_price, 2),
                "realized_return_pct": round((outcome_price / spot - 1) * 100, 1),
                "band": band,
                "p_bear": probs["bear"], "p_base": probs["base"],
                "p_bull": probs["bull"],
                "brier": ledger.brier_score(probs, band),
                "pit": round(pit, 4),
                "median_forecast": round(p50, 2),
                "median_abs_pct_err": round(abs(p50 / outcome_price - 1) * 100, 1),
            })
    log.info("backtest produced %d points across %d tickers", len(results), len(tickers))
    return results


def summarize_backtest(results: list[dict]) -> dict:
    """Calibration statistics. A well-specified model matches every nominal."""
    n = len(results)
    if n == 0:
        return {"n": 0}

    pit = np.array([r["pit"] for r in results])
    band_counts = Counter(r["band"] for r in results)

    realized_freq = {b: round(band_counts.get(b, 0) / n, 3) for b in NOMINAL}
    # Reliability: when the model says "18% chance of the bear case", does the
    # bear case happen ~18% of the time? Comparing against the old fixed
    # 25/50/25 would now be meaningless, since the odds vary per stock.
    mean_predicted = {
        b: round(float(np.mean([r[f"p_{b}"] for r in results])) / 100, 3)
        for b in NOMINAL if all(f"p_{b}" in r for r in results)
    }
    # PIT-based coverage of the model's own intervals
    cov_50 = float(np.mean((pit >= 0.25) & (pit <= 0.75)))   # P25–P75, nominal 0.50
    cov_80 = float(np.mean((pit >= 0.10) & (pit <= 0.90)))   # P10–P90, nominal 0.80

    # 10-bin PIT histogram — flat means calibrated; skew reveals bias direction
    hist, _ = np.histogram(pit, bins=10, range=(0, 1))

    # Do the stock-specific odds actually beat the fixed 25/50/25 they replaced?
    # This is the falsifiable test of the whole scenario-probability change:
    # varying the numbers is only an improvement if it lowers the Brier score.
    briers = [r["brier"] for r in results if "brier" in r]
    mean_brier = round(float(np.mean(briers)), 4) if briers else None
    naive = {"bear": 25, "base": 50, "bull": 25}
    baseline_brier = round(
        float(np.mean([ledger.brier_score(naive, r["band"]) for r in results])), 4)

    mean_pit = float(pit.mean())
    if mean_pit < 0.45:
        bias = "optimistic (outcomes land below the model's median)"
    elif mean_pit > 0.55:
        bias = "pessimistic (outcomes land above the model's median)"
    else:
        bias = "roughly centered"

    return {
        "n": n,
        "realized_band_freq": realized_freq,
        "mean_predicted_prob": mean_predicted,
        "nominal_band_freq": NOMINAL,
        "coverage_p25_p75": round(cov_50, 3),      # want ~0.50
        "coverage_p10_p90": round(cov_80, 3),      # want ~0.80
        "mean_pit": round(mean_pit, 3),            # want ~0.50
        "pit_bias": bias,
        "mean_brier": mean_brier,                  # lower is better
        "baseline_brier": baseline_brier,          # fixed 25/50/25
        "beats_baseline": (mean_brier is not None and mean_brier < baseline_brier),
        "pit_histogram": hist.tolist(),
        "median_abs_pct_err": round(
            float(np.median([r["median_abs_pct_err"] for r in results])), 1),
    }


def format_summary(summary: dict) -> str:
    if summary.get("n", 0) == 0:
        return "No backtest points produced (check tickers / date range)."
    s = summary
    lines = [
        f"Backtest calibration  (n = {s['n']} predictions)",
        "-" * 52,
        "Scenario reliability   predicted   realized   (want these to match)",
        f"  bear (<= {config.BEAR_RETURN_PCT:+.0f}%)      "
        f"{s['mean_predicted_prob'].get('bear', 0):>6.1%}     {s['realized_band_freq']['bear']:>6.1%}",
        f"  base                 "
        f"{s['mean_predicted_prob'].get('base', 0):>6.1%}     {s['realized_band_freq']['base']:>6.1%}",
        f"  bull (>= {config.BULL_RETURN_PCT:+.0f}%)      "
        f"{s['mean_predicted_prob'].get('bull', 0):>6.1%}     {s['realized_band_freq']['bull']:>6.1%}",
        "",
        f"  P25–P75 interval covers {s['coverage_p25_p75']:.1%} of outcomes (want 50%)",
        f"  P10–P90 interval covers {s['coverage_p10_p90']:.1%} of outcomes (want 80%)",
        f"  mean PIT = {s['mean_pit']:.3f} (want 0.50) -> {s['pit_bias']}",
        f"  median |forecast error| = {s['median_abs_pct_err']:.1f}%",
        "",
        f"  scenario Brier = {s['mean_brier']} vs naive 25/50/25 {s['baseline_brier']}"
        f"  -> {'BEATS baseline' if s['beats_baseline'] else 'no better than naive'}",
        "",
        f"  PIT histogram (10 bins, flat = calibrated): {s['pit_histogram']}",
    ]
    return "\n".join(lines)
