"""Scenario probabilities must carry information about the *stock*.

The previous percentile partition made every prior exactly 25/50/25, so the
probabilities described the method rather than the company. These tests pin the
property that replaced it: a riskier distribution must produce a higher chance
of the bear case."""

import numpy as np
import pytest

from tricast import config, macro_regime
from tricast.quant import scenarios


def _terminal(mean_ret, vol, n=40000, seed=0):
    """Lognormal-ish terminal prices from a $100 spot."""
    rng = np.random.default_rng(seed)
    return 100 * np.exp(rng.normal(np.log1p(mean_ret) - vol**2 / 2, vol, n))


def test_probabilities_differ_between_a_calm_and_a_volatile_stock():
    calm = scenarios.build_scenarios(_terminal(0.10, 0.15), 100.0)
    wild = scenarios.build_scenarios(_terminal(0.10, 0.55), 100.0)
    assert wild["bear"]["prior_prob"] > calm["bear"]["prior_prob"] + 10
    assert wild["bull"]["prior_prob"] > calm["bull"]["prior_prob"]
    assert calm["base"]["prior_prob"] > wild["base"]["prior_prob"] + 10


def test_probabilities_are_no_longer_the_fixed_partition():
    """The exact defect being fixed: identical 25/50/25 for every stock."""
    calm = scenarios.build_scenarios(_terminal(0.10, 0.15), 100.0)
    assert [calm[b]["prior_prob"] for b in ("bear", "base", "bull")] != [25, 50, 25]


def test_probabilities_are_integers_summing_to_100():
    for vol in (0.05, 0.2, 0.45, 0.9):
        s = scenarios.build_scenarios(_terminal(0.08, vol), 100.0)
        probs = [s[b]["prior_prob"] for b in ("bear", "base", "bull")]
        assert all(isinstance(p, int) for p in probs)
        assert sum(probs) == 100


def test_probability_matches_the_share_of_simulated_outcomes():
    terminal = _terminal(0.10, 0.30, seed=3)
    s = scenarios.build_scenarios(terminal, 100.0)
    expected = (terminal <= 100 * (1 + config.BEAR_RETURN_PCT / 100)).mean() * 100
    assert abs(s["bear"]["prior_prob"] - expected) <= 1     # rounding only


def test_targets_sit_inside_their_own_region():
    s = scenarios.build_scenarios(_terminal(0.10, 0.35, seed=5), 100.0)
    assert s["bear"]["return_pct"] <= config.BEAR_RETURN_PCT
    assert s["bull"]["return_pct"] >= config.BULL_RETURN_PCT
    assert config.BEAR_RETURN_PCT < s["base"]["return_pct"] < config.BULL_RETURN_PCT


def test_empty_region_falls_back_to_the_threshold_not_nan():
    """A very low-volatility asset may produce no outcome that loses 10%."""
    terminal = np.full(5000, 112.0)          # every path lands at +12%
    s = scenarios.build_scenarios(terminal, 100.0)
    assert s["bear"]["prior_prob"] == 0
    assert s["bull"]["prior_prob"] == 0
    assert s["base"]["prior_prob"] == 100
    for b in ("bear", "base", "bull"):
        assert np.isfinite(s[b]["target"])


def test_band_bounds_are_fixed_fractions_of_spot():
    b = scenarios.band_bounds(200.0)
    assert b["lower_price"] == 180.0         # -10%
    assert b["upper_price"] == 240.0         # +20%


# --- the tilt bug this change would otherwise have exposed ----------------

@pytest.mark.parametrize("priors,score", [
    ({"bear": 4, "base": 60, "bull": 36}, 1.0),    # strong bullish tilt
    ({"bear": 40, "base": 57, "bull": 3}, -1.0),   # strong bearish tilt
    ({"bear": 0, "base": 100, "bull": 0}, 1.0),
])
def test_tilt_never_produces_a_negative_probability(priors, score):
    """Unreachable while priors were always 25/50/25; reachable now."""
    out = macro_regime.tilt_probabilities(priors, score)
    assert all(v >= 0 for v in out.values()), out
    assert sum(out.values()) == 100


def test_tilt_still_shifts_normally_when_there_is_room():
    out = macro_regime.tilt_probabilities({"bear": 30, "base": 40, "bull": 30}, 1.0)
    assert out == {"bear": 20, "base": 40, "bull": 40}
