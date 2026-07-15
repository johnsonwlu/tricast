import pandas as pd
import pytest

from tricast import macro_regime

PRIORS = {"bear": 25, "base": 50, "bull": 25}


def make_series(t10y2y, cpi_yoy_pct, unrate_chg_3m, ff_chg_6m):
    """Build minimal FRED-shaped series producing the given latest signals."""
    idx = pd.date_range("2024-01-01", periods=14, freq="MS")
    cpi_base = 300.0
    cpi = pd.Series([cpi_base] * 13 + [cpi_base * (1 + cpi_yoy_pct / 100)], index=idx)
    unrate = pd.Series([4.0] * 10 + [4.0, 4.0, 4.0, 4.0 + unrate_chg_3m], index=idx)
    ff = pd.Series([5.0] * 7 + [5.0] * 6 + [5.0 + ff_chg_6m], index=idx)
    t = pd.Series([t10y2y] * 14, index=idx)
    return {"T10Y2Y": t, "CPIAUCSL": cpi, "UNRATE": unrate, "FEDFUNDS": ff}


def test_expansionary_regime():
    series = make_series(t10y2y=1.0, cpi_yoy_pct=2.0, unrate_chg_3m=-0.2, ff_chg_6m=-0.5)
    signals = macro_regime.compute_signals(series, vix=12.0)
    score = macro_regime.composite_score(signals)
    assert score == 1.0
    assert macro_regime.regime_label(score) == "Expansionary"


def test_contractionary_regime():
    series = make_series(t10y2y=-0.5, cpi_yoy_pct=6.0, unrate_chg_3m=0.5, ff_chg_6m=1.0)
    signals = macro_regime.compute_signals(series, vix=35.0)
    score = macro_regime.composite_score(signals)
    assert score == -1.0
    assert macro_regime.regime_label(score) == "Contractionary"


def test_neutral_regime():
    series = make_series(t10y2y=0.2, cpi_yoy_pct=3.0, unrate_chg_3m=0.0, ff_chg_6m=0.0)
    signals = macro_regime.compute_signals(series, vix=20.0)
    score = macro_regime.composite_score(signals)
    assert macro_regime.regime_label(score) == "Neutral"


@pytest.mark.parametrize("score", [-1.0, -0.4, 0.0, 0.4, 1.0])
def test_tilt_sum_preserved_and_bounded(score):
    tilted = macro_regime.tilt_probabilities(PRIORS, score)
    assert sum(tilted.values()) == 100
    assert abs(tilted["bull"] - PRIORS["bull"]) <= 10
    assert abs(tilted["bear"] - PRIORS["bear"]) <= 10
    assert tilted["base"] == PRIORS["base"]


def test_tilt_direction():
    up = macro_regime.tilt_probabilities(PRIORS, 1.0)
    assert up == {"bear": 15, "base": 50, "bull": 35}
    down = macro_regime.tilt_probabilities(PRIORS, -1.0)
    assert down == {"bear": 35, "base": 50, "bull": 15}
