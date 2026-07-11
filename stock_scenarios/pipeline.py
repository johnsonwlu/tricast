"""Orchestrator: ticker -> full ScenarioReport.

Quant and macro stages are free (cached market/FRED data). The LLM stage is
paid and only runs when run_llm=True; identical inputs are served from the
analyses cache without an API call.
"""

import logging

from stock_scenarios import config, ledger, macro_regime, store
from stock_scenarios.data import macro as macro_data
from stock_scenarios.data import market
from stock_scenarios.llm import analyst
from stock_scenarios.quant import montecarlo, scenarios

log = logging.getLogger(__name__)


def get_macro_state(db_path=config.DB_PATH) -> dict:
    series = macro_data.get_all_series(db_path=db_path)
    vix = market.get_vix(db_path=db_path)
    signals = macro_regime.compute_signals(series, vix)
    score = macro_regime.composite_score(signals)
    return {
        "signals": [vars(s) for s in signals],
        "score": round(score, 2),
        "regime": macro_regime.regime_label(score),
    }


def build_report(ticker: str, run_llm: bool = False, db_path=config.DB_PATH) -> dict:
    """Quant + macro report; optionally with LLM analysis (cached by inputs hash)."""
    ticker = ticker.upper()
    prices = market.get_prices(ticker, db_path=db_path)
    fundamentals = market.get_fundamentals(ticker, db_path=db_path)

    sim = montecarlo.simulate(prices["close"], analyst_target=fundamentals.get("targetMeanPrice"))
    bands = scenarios.build_scenarios(sim["terminal"], sim["spot"])

    try:
        macro_state = get_macro_state(db_path=db_path)
    except Exception as e:
        # No FRED key / network outage: run un-tilted rather than failing the report
        log.warning("macro state unavailable (%s); using neutral regime", e)
        macro_state = {"signals": [], "score": 0.0, "regime": "Neutral (no macro data)"}
    priors = {k: v["prior_prob"] for k, v in bands.items()}
    tilted = macro_regime.tilt_probabilities(priors, macro_state["score"])

    report = {
        "ticker": ticker,
        "spot": sim["spot"],
        "mu_annual": round(sim["mu_annual"], 4),
        "horizon_days": sim["horizon"],
        "scenarios": bands,
        "tilted_probabilities": tilted,
        "macro": macro_state,
        "fundamentals": fundamentals,
        "cone": sim["cone"],
        "history": {
            "dates": [d.date().isoformat() for d in prices.index[-504:]],
            "close": [round(float(c), 2) for c in prices["close"].iloc[-504:]],
        },
        "analysis": None,
    }

    if run_llm:
        report["analysis"] = _get_or_run_analysis(ticker, report, db_path)
    else:
        cached = store.analysis_latest(ticker, db_path=db_path)
        if cached:
            report["analysis"] = cached

    ledger.record_prediction(report, db_path=db_path)
    return report


def _llm_payload(report: dict) -> dict:
    """The exact inputs the LLM sees — also the cache key material."""
    f = report["fundamentals"]
    closes = report["history"]["close"]
    spot = report["spot"]

    def momentum(days: int) -> float | None:
        if len(closes) > days and closes[-1 - days]:
            return round((closes[-1] / closes[-1 - days] - 1) * 100, 1)
        return None

    return {
        "ticker": report["ticker"],
        "spot": spot,
        "quant_scenarios": report["scenarios"],
        "tilted_probabilities": report["tilted_probabilities"],
        "macro": {
            "regime": report["macro"]["regime"],
            "score": report["macro"]["score"],
            "signals": report["macro"]["signals"],
        },
        "fundamentals": {
            k: f.get(k)
            for k in (
                "shortName", "sector", "industry", "marketCap",
                "trailingPE", "forwardPE", "profitMargins", "revenueGrowth",
                "earningsGrowth", "debtToEquity", "dividendYield", "beta",
                "targetMeanPrice", "targetHighPrice", "targetLowPrice",
                "recommendationKey", "numberOfAnalystOpinions",
            )
        },
        "momentum_pct": {"1m": momentum(21), "6m": momentum(126), "12m": momentum(252)},
    }


def _get_or_run_analysis(ticker: str, report: dict, db_path) -> dict:
    payload = _llm_payload(report)
    # cache key covers the provider/model too, so switching LLMs re-analyzes
    llm_id = (config.LLM_PROVIDER, config.OLLAMA_MODEL
              if config.LLM_PROVIDER == "ollama" else config.MODEL_ID)
    h = analyst.inputs_hash({"payload": payload, "llm": llm_id})
    cached = store.analysis_get(ticker, h, db_path=db_path)
    if cached:
        log.info("analysis cache hit: %s (%s)", ticker, h[:8])
        return cached
    log.info("running LLM analysis: %s", ticker)
    result = analyst.analyze(payload)
    store.analysis_put(ticker, h, result, db_path=db_path)
    return store.analysis_get(ticker, h, db_path=db_path)
