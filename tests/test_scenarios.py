import numpy as np

from stock_scenarios.quant.scenarios import build_scenarios


def test_probabilities_sum_to_100_and_targets_ordered():
    rng = np.random.default_rng(3)
    terminal = rng.lognormal(np.log(100), 0.3, 10_000)
    bands = build_scenarios(terminal, spot=100.0)
    total = sum(b["prior_prob"] for b in bands.values())
    assert total == 100
    assert bands["bear"]["target"] < bands["base"]["target"] < bands["bull"]["target"]


def test_return_pct_consistent_with_targets():
    rng = np.random.default_rng(4)
    terminal = rng.lognormal(np.log(50), 0.2, 10_000)
    bands = build_scenarios(terminal, spot=50.0)
    for b in bands.values():
        expected = round((b["target"] / 50.0 - 1) * 100, 1)
        assert b["return_pct"] == expected
