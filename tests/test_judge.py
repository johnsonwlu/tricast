from unittest.mock import patch

from tricast import judge
from tricast.judge import JudgeScore, expected_return_pct


def test_expected_return_math():
    quant = {
        "bear": {"return_pct": -20.0},
        "base": {"return_pct": 8.0},
        "bull": {"return_pct": 44.0},
    }
    probs = {"bear": 25, "base": 50, "bull": 25}
    # -20*.25 + 8*.5 + 44*.25 = -5 + 4 + 11 = 10
    assert expected_return_pct(quant, probs) == 10.0


def _fake_score(**kw):
    base = dict(factual_accuracy=4, grounding=3, internal_consistency=4,
                specificity=3, overall=3, issues=[])
    base.update(kw)
    return JudgeScore(**base)


def test_judge_analysis_builds_input_and_returns_score():
    payload = {"quant_scenarios": {}, "spot": 100}
    analysis = {"scenarios": {}, "advice": "hold", "advice_reasoning": "x", "key_risks": []}

    def fake_structured(provider, system, judge_input, schema, model=None):
        # judge must see both the inputs and the analyst output
        assert "inputs_given_to_analyst" in judge_input
        assert "analyst_output" in judge_input
        assert "probability_weighted_expected_return_pct" in judge_input
        return _fake_score(), "judge/test", 0.01

    with patch("tricast.judge.structured_call", side_effect=fake_structured):
        score, model, cost = judge.judge_analysis(payload, analysis, 5.0, provider="ollama")
    assert score.factual_accuracy == 4
    assert model == "judge/test"


def test_panel_and_summary_aggregate(monkeypatch):
    def fake_build(ticker, run_llm=False, db_path=None):
        return {"ticker": ticker}

    def fake_payload(report):
        return {"quant_scenarios": {"bear": {"return_pct": -20}, "base": {"return_pct": 8},
                                    "bull": {"return_pct": 44}}}

    def fake_analyze(payload, provider=None):
        return {
            "scenarios": {b: {"probability_pct": p, "narrative": "n"}
                          for b, p in (("bear", 25), ("base", 50), ("bull", 25))},
            "advice": "hold", "advice_reasoning": "r", "key_risks": [],
            "model": "ollama/qwen",
        }

    scores = iter([
        (_fake_score(factual_accuracy=2, issues=["AAPL: spot vs target reversed"]), "j", 0.01),
        (_fake_score(factual_accuracy=5, issues=[]), "j", 0.01),
    ])
    monkeypatch.setattr(judge.pipeline, "build_report", fake_build)
    monkeypatch.setattr(judge.pipeline, "_llm_payload", fake_payload)
    monkeypatch.setattr(judge.analyst, "analyze", fake_analyze)
    monkeypatch.setattr(judge, "judge_analysis", lambda *a, **k: next(scores))

    rows = judge.run_judge_panel(["AAPL", "MSFT"], judge_provider="ollama")
    assert len(rows) == 2
    assert rows[0]["expected_return_pct"] == 10.0

    summary = judge.summarize_judge(rows)
    assert summary["n"] == 2
    assert summary["mean_scores"]["factual_accuracy"] == 3.5  # (2+5)/2
    assert summary["total_issues"] == 1
    assert "spot vs target" in summary["issues"][0]


def test_summary_empty():
    assert judge.summarize_judge([])["n"] == 0
    assert "No judged" in judge.format_summary({"n": 0})
