import json
from unittest.mock import patch

from tricast.llm.analyst import analyze, enforce_bounds, inputs_hash


def test_within_bounds_passes_through():
    tilted = {"bear": 29, "base": 50, "bull": 21}
    proposed = {"bear": 25, "base": 52, "bull": 23}
    assert enforce_bounds(proposed, tilted) == proposed


def test_out_of_bounds_clamped_residual_to_base():
    # plan's regression case: mocked 60/30/10 vs prior 29/50/21
    tilted = {"bear": 29, "base": 50, "bull": 21}
    proposed = {"bear": 60, "base": 30, "bull": 10}
    result = enforce_bounds(proposed, tilted)
    assert sum(result.values()) == 100
    for k in result:
        assert abs(result[k] - tilted[k]) <= 10


def test_unrecoverable_falls_back_to_priors():
    tilted = {"bear": 25, "base": 50, "bull": 25}
    # both tails pinned high forces base far below its bound
    proposed = {"bear": 90, "base": 0, "bull": 90}
    assert enforce_bounds(proposed, tilted) == tilted


def test_exact_priors_unchanged():
    tilted = {"bear": 15, "base": 50, "bull": 35}
    assert enforce_bounds(dict(tilted), tilted) == tilted


class FakeOllamaResponse:
    def __init__(self, body: dict):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def test_ollama_provider_parses_and_clamps():
    llm_json = json.dumps({
        "bear": {"probability_pct": 60, "narrative": "Recession hits."},
        "base": {"probability_pct": 30, "narrative": "Muddle through."},
        "bull": {"probability_pct": 10, "narrative": "AI supercycle."},
        "advice": "hold",
        "advice_reasoning": "Valuation is stretched.",
        "key_risks": ["rates", "competition"],
    })
    fake = FakeOllamaResponse({"message": {"content": llm_json}})
    payload = {"tilted_probabilities": {"bear": 29, "base": 50, "bull": 21}}

    with patch("tricast.llm.analyst.requests.post", return_value=fake) as post:
        result = analyze(payload, provider="ollama")

    assert post.called
    body = post.call_args.kwargs["json"]
    assert "format" in body and body["stream"] is False
    probs = {k: v["probability_pct"] for k, v in result["scenarios"].items()}
    assert sum(probs.values()) == 100
    for k in probs:  # clamped to ±10pp of the tilted prior
        assert abs(probs[k] - payload["tilted_probabilities"][k]) <= 10
    assert result["advice"] == "hold"
    assert result["cost_usd"] == 0.0
    assert result["model"].startswith("ollama/")


def test_inputs_hash_deterministic_and_order_insensitive():
    a = inputs_hash({"x": 1, "y": [1, 2]})
    b = inputs_hash({"y": [1, 2], "x": 1})
    assert a == b
    assert a != inputs_hash({"x": 2, "y": [1, 2]})
