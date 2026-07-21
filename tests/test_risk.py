import numpy as np

from tricast import risk


def test_zero_volatility_constant_outcome():
    # every path lands at +10% -> no risk; sharpe defined as 0 (vol=0 guard)
    terminal = np.full(1000, 110.0)
    m = risk.risk_metrics(terminal, spot=100.0)
    assert m["expected_return_pct"] == 10.0
    assert m["volatility_pct"] == 0.0
    assert m["prob_loss_pct"] == 0
    assert m["sharpe"] == 0.0


def test_low_vol_beats_high_vol_at_equal_mean():
    """The whole point: with the SAME expected return, the lower-volatility
    distribution must score a higher Sharpe (better risk-adjusted)."""
    rng = np.random.default_rng(0)
    mean_ret = 0.12
    calm = 100 * (1 + rng.normal(mean_ret, 0.10, 20000))
    wild = 100 * (1 + rng.normal(mean_ret, 0.45, 20000))
    m_calm = risk.risk_metrics(calm, 100.0)
    m_wild = risk.risk_metrics(wild, 100.0)
    assert abs(m_calm["expected_return_pct"] - m_wild["expected_return_pct"]) < 2
    assert m_calm["sharpe"] > m_wild["sharpe"]
    assert m_calm["prob_loss_pct"] < m_wild["prob_loss_pct"]


def test_high_expected_return_can_still_be_worse_risk_adjusted():
    """A volatile name with a HIGHER expected return can have a LOWER Sharpe —
    exactly the AMD-vs-VOO situation the score is meant to expose."""
    rng = np.random.default_rng(1)
    voo = 100 * (1 + rng.normal(0.12, 0.14, 20000))     # modest return, low vol
    amd = 100 * (1 + rng.normal(0.30, 0.55, 20000))     # high return, high vol
    m_voo = risk.risk_metrics(voo, 100.0)
    m_amd = risk.risk_metrics(amd, 100.0)
    assert m_amd["expected_return_pct"] > m_voo["expected_return_pct"]  # AMD wins on raw
    assert m_voo["sharpe"] > m_amd["sharpe"]                            # VOO wins risk-adjusted


def test_cvar_is_positive_loss_with_downside():
    rng = np.random.default_rng(2)
    terminal = 100 * (1 + rng.normal(0.05, 0.30, 10000))
    m = risk.risk_metrics(terminal, 100.0)
    assert m["cvar5_pct"] > 0            # expected shortfall reported as a positive loss
    assert 0 <= m["prob_loss_pct"] <= 100


def test_score_label_bands():
    assert risk.score_label(1.0) == "strong"
    assert risk.score_label(0.5) == "fair"
    assert risk.score_label(0.1) == "weak"
    assert risk.score_label(-0.3) == "poor"
