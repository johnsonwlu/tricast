import json

import pytest

from tricast import ledger, store


@pytest.fixture
def db(tmp_path):
    return tmp_path / "test.db"


def make_report(ticker="TEST", probs=None, analysis=False):
    report = {
        "ticker": ticker,
        "spot": 100.0,
        "scenarios": {
            "bear": {"target": 80.0}, "base": {"target": 105.0}, "bull": {"target": 140.0},
        },
        "tilted_probabilities": probs or {"bear": 23, "base": 50, "bull": 27},
        "macro": {"regime": "Neutral", "score": 0.2},
        "cone": {p: [100.0 + i for i in range(252)] for p in ("p10", "p25", "p50", "p75", "p90")},
        "analysis": None,
    }
    report["cone"]["p25"] = [90.0] * 252
    report["cone"]["p75"] = [120.0] * 252
    if analysis:
        report["analysis"] = {
            "scenarios": {k: {"probability_pct": v} for k, v in
                          {"bear": 20, "base": 55, "bull": 25}.items()},
            "model": "ollama/test",
        }
    return report


def test_record_is_idempotent_per_day(db):
    ledger.record_prediction(make_report(), db_path=db)
    ledger.record_prediction(make_report(), db_path=db)
    assert len(ledger.all_predictions(db_path=db)) == 1


def test_record_prefers_llm_probabilities(db):
    ledger.record_prediction(make_report(analysis=True), db_path=db)
    p = ledger.all_predictions(db_path=db)[0]
    assert (p["p_bear"], p["p_base"], p["p_bull"]) == (20, 55, 25)
    assert p["model"] == "ollama/test"


def test_classify_outcome():
    assert ledger.classify_outcome(85, 90, 120) == "bear"
    assert ledger.classify_outcome(100, 90, 120) == "base"
    assert ledger.classify_outcome(125, 90, 120) == "bull"


def test_brier_score_perfect_and_worst():
    assert ledger.brier_score({"bear": 100, "base": 0, "bull": 0}, "bear") == 0.0
    assert ledger.brier_score({"bear": 100, "base": 0, "bull": 0}, "bull") == 2.0
    # naive prior when base occurs: .25^2 + .5^2 + .25^2 with base miss .25 -> 0.375
    assert ledger.brier_score({"bear": 25, "base": 50, "bull": 25}, "base") == 0.375


def test_score_matured_and_summary(db, monkeypatch):
    import pandas as pd

    ledger.record_prediction(make_report(), db_path=db)
    # backdate so the prediction is due
    with store.connect(db) as conn:
        conn.execute("UPDATE predictions SET pred_date='2024-01-02', horizon_end='2025-01-01'")

    fake_prices = pd.DataFrame(
        {"close": [130.0]}, index=pd.to_datetime(["2024-12-31"])
    )
    monkeypatch.setattr(
        "tricast.data.market.get_prices", lambda t, db_path=None: fake_prices
    )
    scored = ledger.score_matured(db_path=db)
    assert len(scored) == 1
    assert scored[0]["outcome"] == "bull"  # 130 > band_upper 120

    stats = ledger.summary(db_path=db)
    assert stats["n_scored"] == 1
    assert stats["outcome_counts"]["bull"] == 1
    assert 0 <= stats["mean_brier"] <= 2


def test_interim_position(db):
    ledger.record_prediction(make_report(), db_path=db)
    p = ledger.all_predictions(db_path=db)[0]
    assert ledger.interim_position(p, 85.0) == "bear"
    assert ledger.interim_position(p, 100.0) == "base"
    assert ledger.interim_position(p, 125.0) == "bull"
