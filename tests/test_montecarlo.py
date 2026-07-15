import numpy as np
import pandas as pd
import pytest

from tricast import config
from tricast.quant import montecarlo


def make_closes(n=1300, daily_vol=0.02, seed=1):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, daily_vol, n)
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices)


def test_refuses_short_history():
    with pytest.raises(ValueError, match="history"):
        montecarlo.simulate(make_closes(n=100))


def test_seeded_determinism():
    closes = make_closes()
    a = montecarlo.simulate(closes, seed=7)
    b = montecarlo.simulate(closes, seed=7)
    assert np.array_equal(a["terminal"], b["terminal"])


def test_terminal_shape_and_positivity():
    sim = montecarlo.simulate(make_closes(), n_paths=2000)
    assert sim["terminal"].shape == (2000,)
    assert (sim["terminal"] > 0).all()
    assert len(sim["cone"]["p50"]) == config.HORIZON_DAYS


def test_higher_vol_widens_distribution():
    lo = montecarlo.simulate(make_closes(daily_vol=0.01), n_paths=5000)
    hi = montecarlo.simulate(make_closes(daily_vol=0.04), n_paths=5000)
    # normalize by spot — the two synthetic series end at different price levels
    spread_lo = (np.percentile(lo["terminal"], 90) - np.percentile(lo["terminal"], 10)) / lo["spot"]
    spread_hi = (np.percentile(hi["terminal"], 90) - np.percentile(hi["terminal"], 10)) / hi["spot"]
    assert spread_hi > spread_lo


def test_drift_caps():
    closes = make_closes()
    returns = np.log(closes / closes.shift(1)).dropna()
    spot = float(closes.iloc[-1])
    # absurd analyst target: implied drift must be capped at +25%/yr per component
    mu = montecarlo.compute_drift(returns, spot, analyst_target=spot * 100)
    assert mu <= config.DRIFT_CAP_ANNUAL


def test_no_analyst_target_uses_historical_only():
    closes = make_closes()
    returns = np.log(closes / closes.shift(1)).dropna()
    spot = float(closes.iloc[-1])
    mu_none = montecarlo.compute_drift(returns, spot, analyst_target=None)
    mu_hist = float(np.clip(returns.mean() * 252, -0.25, 0.25))
    assert mu_none == pytest.approx(mu_hist)
