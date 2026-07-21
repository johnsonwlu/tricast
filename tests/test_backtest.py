import numpy as np
import pandas as pd

from tricast import backtest, config


def _random_walk(n=3500, daily_vol=0.015, drift=0.0, seed=0):
    """Driftless (or specified-drift) geometric random walk — the case the
    bootstrap simulator should be well-calibrated on."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, daily_vol, n)
    prices = 100 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range("2008-01-01", periods=n)
    return pd.Series(prices, index=idx)


def test_no_lookahead_outcome_indexing(monkeypatch):
    """Outcome must be exactly `horizon` trading days after the as-of close,
    and the simulation must never see data past the as-of date."""
    series = _random_walk()
    seen = {}

    from tricast.quant import montecarlo
    real_simulate = montecarlo.simulate

    def spy_simulate(closes, **kw):
        seen["last_date"] = closes.index[-1]
        return real_simulate(closes, **kw)

    monkeypatch.setattr(montecarlo, "simulate", spy_simulate)

    d = pd.Timestamp("2016-01-01")
    results = backtest.run_backtest(
        ["X"], start="2016-01-01", end="2016-01-01", freq="MS",
        n_paths=500, price_loader=lambda _: series,
    )
    assert len(results) == 1
    r = results[0]
    # the sim only saw history up to the as-of date — never the future
    assert seen["last_date"] <= d
    # outcome is exactly `horizon` trading days after the as-of close
    hist = series[series.index <= d]
    as_of_pos = len(hist) - 1
    expected_outcome = series.iloc[as_of_pos + config.HORIZON_DAYS]
    assert r["outcome_price"] == round(float(expected_outcome), 2)


def test_pit_and_band_consistency():
    series = _random_walk()
    results = backtest.run_backtest(
        ["X"], start="2015-01-01", freq="MS", n_paths=2000,
        price_loader=lambda _: series,
    )
    assert results
    for r in results:
        # band and PIT must agree
        if r["band"] == "bear":
            assert r["pit"] < 0.25
        elif r["band"] == "bull":
            assert r["pit"] > 0.75
        else:
            assert 0.25 <= r["pit"] <= 0.75


def test_calibrated_on_random_walk():
    """On driftless random walks the bootstrap should be ~calibrated. This needs
    *independent* series — overlapping windows of a single path are correlated
    and one lucky realization biases the whole result. So each 'ticker' gets its
    own fresh walk, and outcomes aggregate across independent paths."""
    walks = {f"X{i}": _random_walk(n=2600, seed=100 + i) for i in range(80)}
    results = backtest.run_backtest(
        list(walks), start="2012-01-01", freq="6MS", n_paths=1500,
        price_loader=lambda t: walks[t],
    )
    summary = backtest.summarize_backtest(results)
    assert summary["n"] > 150
    # statistical check across independent paths — should straddle the nominals
    assert 0.42 <= summary["mean_pit"] <= 0.58
    assert 0.40 <= summary["coverage_p25_p75"] <= 0.60
    assert 0.72 <= summary["coverage_p10_p90"] <= 0.88


def test_summary_empty():
    assert backtest.summarize_backtest([])["n"] == 0
    assert "No backtest" in backtest.format_summary({"n": 0})
