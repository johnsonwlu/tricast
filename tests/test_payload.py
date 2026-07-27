"""The payload is the model's entire view of the world. A field whose units are
ambiguous is a bug even when every number in it is correct — the local model
read `debtToEquity: 6.005` (a percent) as 6.0x leverage and reported AMD's
unusually clean balance sheet as a headline risk."""

from tricast import pipeline


def _report(**overrides):
    report = {
        "ticker": "AMD",
        "spot": 521.95,
        "scenarios": {
            "bear": {"target": 372.78, "return_pct": -28.6, "prior_prob": 25},
            "base": {"target": 614.41, "return_pct": 17.7, "prior_prob": 50},
            "bull": {"target": 1020.11, "return_pct": 95.4, "prior_prob": 25},
        },
        "tilted_probabilities": {"bear": 23, "base": 50, "bull": 27},
        "risk": {"sharpe": 0.38, "sortino": 0.51, "prob_loss_pct": 40,
                 "cvar5_pct": 62.0, "expected_return_pct": 28.0,
                 "volatility_pct": 65.0, "risk_free_pct": 4.3, "label": "weak"},
        "macro": {"regime": "Neutral", "score": 0.2, "signals": []},
        "fundamentals": {
            "shortName": "Advanced Micro Devices, Inc.",
            "profitMargins": 0.13374, "revenueGrowth": 0.378,
            "earningsGrowth": 0.912, "debtToEquity": 6.005,
            "dividendYield": None, "beta": 2.469, "targetMeanPrice": 573.15,
        },
        "history": {"close": [100.0] * 300},
    }
    report.update(overrides)
    return report


def test_debt_to_equity_is_converted_to_the_multiple_people_mean():
    """yfinance reports a percent; 6.005 means 0.06x, not 6x."""
    fund = pipeline._llm_payload(_report())["fundamentals"]
    assert fund["debt_to_equity_ratio"] == 0.06


def test_debt_to_equity_conversion_holds_for_a_genuinely_leveraged_company():
    r = _report()
    r["fundamentals"]["debtToEquity"] = 828.696          # Boeing
    fund = pipeline._llm_payload(r)["fundamentals"]
    assert fund["debt_to_equity_ratio"] == 8.287         # really is ~8x


def test_prior_probabilities_are_not_exposed_to_the_model():
    """They sit one key away from the tilted values and get quoted by mistake."""
    payload = pipeline._llm_payload(_report())
    for band in payload["quant_scenarios"].values():
        assert "prior_prob" not in band
    assert payload["tilted_probabilities"] == {"bear": 23, "base": 50, "bull": 27}
    assert "25" not in str(payload["quant_scenarios"])


def test_targets_and_returns_still_reach_the_model():
    scen = pipeline._llm_payload(_report())["quant_scenarios"]
    assert scen["bull"] == {"target": 1020.11, "return_pct": 95.4}
    assert scen["bear"]["return_pct"] == -28.6          # sign preserved


def test_fraction_fields_are_normalised_to_percent():
    fund = pipeline._llm_payload(_report())["fundamentals"]
    assert fund["profit_margin_pct"] == 13.4
    assert fund["revenue_growth_pct"] == 37.8
    assert fund["earnings_growth_pct"] == 91.2


def test_dividend_yield_is_passed_through_not_double_scaled():
    """yfinance already returns this one as a percent (KO = 2.61)."""
    r = _report()
    r["fundamentals"]["dividendYield"] = 2.61
    assert pipeline._llm_payload(r)["fundamentals"]["dividend_yield_pct"] == 2.61


def test_missing_fundamentals_stay_none_rather_than_crashing():
    r = _report()
    r["fundamentals"] = {}
    fund = pipeline._llm_payload(r)["fundamentals"]
    assert fund["debt_to_equity_ratio"] is None
    assert fund["profit_margin_pct"] is None
    assert fund["beta"] is None


def test_risk_metrics_reach_the_model():
    """The whole point of the risk-adjusted score: the advice can only weigh
    reward against risk if the risk numbers are actually in the payload."""
    risk = pipeline._llm_payload(_report())["risk_adjusted"]
    assert risk["sharpe"] == 0.38
    assert risk["prob_loss_pct"] == 40
    assert risk["cvar5_pct"] == 62.0


def test_no_raw_yfinance_field_names_survive_into_the_payload():
    """Every numeric field must state its unit in its name."""
    fund = pipeline._llm_payload(_report())["fundamentals"]
    for raw in ("debtToEquity", "profitMargins", "revenueGrowth",
                "earningsGrowth", "dividendYield", "targetMeanPrice",
                "trailingPE", "forwardPE", "marketCap"):
        assert raw not in fund


def test_payload_change_invalidates_the_analysis_cache():
    """Renaming fields must change the hash, or stale analyses written against
    the old ambiguous payload would be served forever."""
    from tricast.llm import analyst
    payload = pipeline._llm_payload(_report())
    legacy = dict(payload, fundamentals={"debtToEquity": 6.005})
    assert analyst.inputs_hash(payload) != analyst.inputs_hash(legacy)
