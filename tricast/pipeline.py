"""Orchestrator: ticker -> full ScenarioReport.

Quant and macro stages are free (cached market/FRED data). The LLM stage is
paid and only runs when run_llm=True; identical inputs are served from the
analyses cache without an API call.
"""

import logging

from tricast import config, errors, ledger, macro_regime, risk, store
from tricast.data import macro as macro_data
from tricast.data import market
from tricast.llm import analyst
from tricast.quant import montecarlo, scenarios

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

    try:
        sim = montecarlo.simulate(
            prices["close"], analyst_target=fundamentals.get("targetMeanPrice"))
    except errors.InsufficientHistory as e:
        # the simulator has no idea which ticker it was handed; name it so the
        # message the user sees says "SNDK" rather than "This stock"
        raise errors.InsufficientHistory(e.have, e.need, ticker) from None
    bands = scenarios.build_scenarios(sim["terminal"], sim["spot"])
    risk_metrics = risk.risk_metrics(sim["terminal"], sim["spot"])

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
        "vol_scale": sim["vol_scale"],
        "horizon_days": sim["horizon"],
        "scenarios": bands,
        "tilted_probabilities": tilted,
        "risk": risk_metrics,
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


def _as_pct(fraction: float | None) -> float | None:
    """yfinance returns margins and growth rates as fractions (0.378) but
    dividend yield as an already-scaled percent (1.07). Mixing both conventions
    in one unlabelled block is how a model ends up quoting 37.8% correctly and
    misreading something else — so everything is normalized and unit-suffixed."""
    return None if fraction is None else round(fraction * 100, 1)


def _llm_payload(report: dict) -> dict:
    """The exact inputs the LLM sees — also the cache key material.

    Every numeric field carries its unit in its name. This is not cosmetic:
    the local model read `debtToEquity: 6.005` (a *percent*, i.e. 0.06x) as
    6.0x leverage and reported AMD's unusually clean balance sheet as a
    headline risk. Naming the field `debt_to_equity_ratio: 0.06` removes that
    failure for any model, at no cost per call.
    """
    f = report["fundamentals"]
    closes = report["history"]["close"]
    spot = report["spot"]
    d2e = f.get("debtToEquity")

    def momentum(days: int) -> float | None:
        if len(closes) > days and closes[-1 - days]:
            return round((closes[-1] / closes[-1 - days] - 1) * 100, 1)
        return None

    return {
        "ticker": report["ticker"],
        "spot": spot,
        # `prior_prob` is deliberately dropped: it sits one key away from
        # tilted_probabilities and gets quoted by mistake (the local model
        # cited VOO's bear at the 25% prior instead of the 23% it was given).
        # The tilted values are the only probabilities the analyst should see.
        "quant_scenarios": {
            name: {"target": s["target"], "return_pct": s["return_pct"]}
            for name, s in report["scenarios"].items()
        },
        "tilted_probabilities": report["tilted_probabilities"],
        "risk_adjusted": report["risk"],
        "macro": {
            "regime": report["macro"]["regime"],
            "score": report["macro"]["score"],
            "signals": report["macro"]["signals"],
        },
        "fundamentals": {
            "name": f.get("shortName"),
            "sector": f.get("sector"),
            "industry": f.get("industry"),
            "market_cap_usd": f.get("marketCap"),
            "trailing_pe": f.get("trailingPE"),
            "forward_pe": f.get("forwardPE"),
            "profit_margin_pct": _as_pct(f.get("profitMargins")),
            "revenue_growth_pct": _as_pct(f.get("revenueGrowth")),
            "earnings_growth_pct": _as_pct(f.get("earningsGrowth")),
            # yfinance reports this as a percent (AMD 6.005 = 6%, Boeing 828.7
            # = 8.3x); convert to the multiple people actually mean by "D/E"
            "debt_to_equity_ratio": None if d2e is None else round(d2e / 100, 3),
            "dividend_yield_pct": f.get("dividendYield"),   # already a percent
            "beta": f.get("beta"),
            "analyst_target_mean": f.get("targetMeanPrice"),
            "analyst_target_high": f.get("targetHighPrice"),
            "analyst_target_low": f.get("targetLowPrice"),
            "analyst_recommendation": f.get("recommendationKey"),
            "analyst_count": f.get("numberOfAnalystOpinions"),
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
