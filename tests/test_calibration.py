import json

import numpy as np
import pandas as pd
import pytest

from tricast import calibration, config
from tricast.quant import montecarlo


@pytest.fixture(autouse=True)
def _clean_calibration_cache():
    """The loader is lru_cached at module scope; clear it around every test so a
    saved/monkeypatched value can never leak into another test (or into the rest
    of the suite, which assumes the neutral default)."""
    calibration._load.cache_clear()
    yield
    calibration._load.cache_clear()


def _random_walk(n=600, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.02, n)
    return pd.Series(100 * np.exp(np.cumsum(rets)))


def test_vol_scale_shrinks_dispersion_linearly():
    """Halving vol_scale should roughly halve the terminal log-price spread —
    the whole mechanism by which the learned factor narrows the cone."""
    closes = _random_walk()
    full = montecarlo.simulate(closes, vol_scale=1.0, n_paths=8000, seed=1)
    half = montecarlo.simulate(closes, vol_scale=0.5, n_paths=8000, seed=1)
    spread_full = np.std(np.log(full["terminal"]))
    spread_half = np.std(np.log(half["terminal"]))
    assert spread_half == pytest.approx(0.5 * spread_full, rel=0.05)


def test_vol_scale_one_is_identity():
    """vol_scale=1.0 must reproduce the un-scaled simulation exactly, so the
    correction is a no-op until something is actually learned (backward compat)."""
    closes = _random_walk(seed=3)
    a = montecarlo.simulate(closes, vol_scale=1.0, seed=7)
    b = montecarlo.simulate(closes, vol_scale=1.0, seed=7)
    assert np.array_equal(a["terminal"], b["terminal"])
    assert a["vol_scale"] == 1.0


def test_simulate_reports_applied_scale():
    closes = _random_walk(seed=5)
    sim = montecarlo.simulate(closes, vol_scale=0.8, seed=2)
    assert sim["vol_scale"] == 0.8


def test_load_default_and_clamp(tmp_path, monkeypatch):
    # missing file -> neutral default
    monkeypatch.setattr(config, "CALIBRATION_PATH", tmp_path / "none.json")
    calibration._load.cache_clear()
    assert calibration.load_vol_scale() == config.DEFAULT_VOL_SCALE

    # absurd learned value gets clamped, never distorts predictions
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"vol_scale": 99.0}))
    monkeypatch.setattr(config, "CALIBRATION_PATH", path)
    calibration._load.cache_clear()
    assert calibration.load_vol_scale() == config.VOL_SCALE_MAX


def test_save_roundtrip_invalidates_cache(tmp_path, monkeypatch):
    path = tmp_path / "cal.json"
    monkeypatch.setattr(config, "CALIBRATION_PATH", path)
    calibration._load.cache_clear()
    assert calibration.load_vol_scale() == config.DEFAULT_VOL_SCALE  # primes cache

    calibration.save({"vol_scale": 0.75, "fitted_at": "now"})
    # cache must have been invalidated so the new value is seen immediately
    assert calibration.load_vol_scale() == 0.75
    assert calibration.metadata()["fitted_at"] == "now"


def test_fit_picks_grid_point_closest_to_nominal():
    """Given synthetic backtest summaries where the too-wide cone tightens as
    vol_scale drops, the fit must select the scale whose central-band coverage
    is nearest 0.50 — not the smallest scale."""
    # coverage falls through 0.50 between 0.8 and 0.75; 0.80 is the closest.
    cov = {1.0: 0.66, 0.9: 0.60, 0.85: 0.55, 0.8: 0.51, 0.75: 0.44, 0.7: 0.38}

    def evaluate(s):
        return {"coverage_p25_p75": cov[s], "coverage_p10_p90": 0.8,
                "mean_pit": 0.5, "n": 100}

    best, table = calibration.fit_vol_scale(evaluate, list(cov))
    assert best == 0.8
    assert len(table) == len(cov)


def test_fit_raises_on_empty_backtest():
    def evaluate(s):
        return {"n": 0}  # summarize_backtest returns this for no points

    with pytest.raises(ValueError):
        calibration.fit_vol_scale(evaluate, [1.0, 0.9])
