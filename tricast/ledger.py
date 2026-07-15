"""Prediction ledger: record every forecast, score it at maturity.

Turns the app's probabilities into something falsifiable. Each report logs one
row per ticker per day (idempotent upsert). Once a prediction's 12-month
horizon passes, `score_matured` classifies the outcome (which band the actual
price landed in) and computes the multi-class Brier score — lower is better,
0 is a perfect forecast, and always-guessing-the-priors (25/50/25) scores
0.625 when base occurs / 1.125 when a tail occurs.
"""

import json
import logging
import time
from datetime import date, timedelta

from tricast import config, store

log = logging.getLogger(__name__)

SCENARIOS = ("bear", "base", "bull")


def record_prediction(report: dict, db_path=config.DB_PATH) -> None:
    """Upsert today's prediction for the report's ticker. Uses the final
    (LLM-adjusted) probabilities when present, else the macro-tilted ones."""
    analysis = report.get("analysis")
    if analysis:
        probs = {k: analysis["scenarios"][k]["probability_pct"] for k in SCENARIOS}
        model = analysis.get("model", "unknown")
    else:
        probs = report["tilted_probabilities"]
        model = "quant-only"

    today = date.today()
    cone = report["cone"]
    with store.connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO predictions "
            "(ticker, pred_date, created_at, horizon_end, spot, band_lower, band_upper, "
            " bear_target, base_target, bull_target, p_bear, p_base, p_bull, "
            " model, regime, macro_score, cone_json, outcome, outcome_price, brier) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)",
            (
                report["ticker"], today.isoformat(), time.time(),
                (today + timedelta(days=365)).isoformat(),
                report["spot"],
                cone["p25"][-1], cone["p75"][-1],
                report["scenarios"]["bear"]["target"],
                report["scenarios"]["base"]["target"],
                report["scenarios"]["bull"]["target"],
                probs["bear"], probs["base"], probs["bull"],
                model, report["macro"]["regime"], report["macro"]["score"],
                json.dumps(cone),
            ),
        )


def classify_outcome(price: float, band_lower: float, band_upper: float) -> str:
    if price < band_lower:
        return "bear"
    if price > band_upper:
        return "bull"
    return "base"


def brier_score(probs: dict[str, int], outcome: str) -> float:
    """Multi-class Brier: sum over scenarios of (p - occurred)^2, in [0, 2]."""
    return round(
        sum((probs[k] / 100 - (1.0 if k == outcome else 0.0)) ** 2 for k in SCENARIOS),
        4,
    )


def score_matured(db_path=config.DB_PATH) -> list[dict]:
    """Score every unscored prediction whose horizon has passed. Returns the
    newly scored rows. Needs market data (fetches/extends the price cache)."""
    from tricast.data import market  # deferred: network-touching

    today = date.today().isoformat()
    with store.connect(db_path) as conn:
        due = conn.execute(
            "SELECT * FROM predictions WHERE outcome IS NULL AND horizon_end <= ?",
            (today,),
        ).fetchall()

    scored = []
    for row in due:
        prices = market.get_prices(row["ticker"], db_path=db_path)
        upto = prices[prices.index <= row["horizon_end"]]
        if upto.empty:
            log.warning("no price data at maturity for %s %s", row["ticker"], row["pred_date"])
            continue
        actual = float(upto["close"].iloc[-1])
        outcome = classify_outcome(actual, row["band_lower"], row["band_upper"])
        probs = {"bear": row["p_bear"], "base": row["p_base"], "bull": row["p_bull"]}
        b = brier_score(probs, outcome)
        with store.connect(db_path) as conn:
            conn.execute(
                "UPDATE predictions SET outcome = ?, outcome_price = ?, brier = ? "
                "WHERE ticker = ? AND pred_date = ?",
                (outcome, actual, b, row["ticker"], row["pred_date"]),
            )
        scored.append({"ticker": row["ticker"], "pred_date": row["pred_date"],
                       "outcome": outcome, "price": actual, "brier": b})
    return scored


def all_predictions(db_path=config.DB_PATH) -> list[dict]:
    with store.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY pred_date DESC, ticker"
        ).fetchall()
    return [dict(r) for r in rows]


def summary(db_path=config.DB_PATH) -> dict:
    """Aggregate calibration stats over matured predictions, with the
    naive always-prior forecast as the baseline to beat."""
    matured = [p for p in all_predictions(db_path) if p["outcome"]]
    if not matured:
        return {"n_scored": 0}
    mean_brier = sum(p["brier"] for p in matured) / len(matured)
    baseline_probs = {"bear": 25, "base": 50, "bull": 25}
    baseline = sum(brier_score(baseline_probs, p["outcome"]) for p in matured) / len(matured)
    outcomes = {k: sum(1 for p in matured if p["outcome"] == k) for k in SCENARIOS}
    return {
        "n_scored": len(matured),
        "mean_brier": round(mean_brier, 4),
        "baseline_brier": round(baseline, 4),
        "beats_baseline": mean_brier < baseline,
        "outcome_counts": outcomes,
    }


def interim_position(pred: dict, current_price: float) -> str:
    """Which band the price is tracking in *today*, vs the cone at the
    matching day of the forecast horizon."""
    cone = json.loads(pred["cone_json"])
    elapsed_cal = (date.today() - date.fromisoformat(pred["pred_date"])).days
    idx = min(int(elapsed_cal * 252 / 365), len(cone["p25"]) - 1)
    if idx < 0:
        return "—"
    return classify_outcome(current_price, cone["p25"][idx], cone["p75"][idx])
