"""Claude analyst: narratives, bounded probability adjustment, and advice.

The model receives quant outputs + macro + fundamentals as one deterministic
JSON blob (sort_keys=True so the inputs hash is stable) and returns a
StockAnalysis. Probabilities are validated in code against the macro-tilted
priors; prices never pass through the model at all.
"""

import hashlib
import json
import logging

from tricast import config
from tricast.llm.client import structured_call
from tricast.llm.schemas import StockAnalysis

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an equity analyst assistant for a personal research tool.

You receive, as JSON: quantitative scenario outputs (12-month bear/base/bull price
targets with prior probabilities derived from a Monte Carlo simulation and tilted by
the current macro regime), company fundamentals, analyst consensus, price momentum,
and macro indicator values.

Your job:
1. For each scenario (bear, base, bull), write a concise 2-4 sentence narrative
   explaining what economic and company-specific conditions would have to occur for
   the stock to reach that scenario's target price.
2. Optionally adjust each scenario's probability by at most 10 percentage points from
   the provided tilted values, based on fundamentals and macro context. The three
   probabilities must sum to exactly 100. If you see no reason to adjust, return the
   provided values unchanged.
3. Give buy/hold/avoid advice with reasoning grounded ONLY in the supplied data.
4. List the 2-4 most important risks to the thesis.

Rules:
- Never state any price target other than those provided in the input.
- Do not invent data not present in the input.
- This is a personal research tool, not financial advice; keep the tone analytical.
"""


def inputs_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def analyze(payload: dict, provider: str | None = None) -> dict:
    """Run one analysis with the configured provider. Returns a dict with the
    validated/clamped analysis plus usage metadata."""
    provider = provider or config.LLM_PROVIDER
    analysis, model, cost = structured_call(provider, SYSTEM_PROMPT, payload, StockAnalysis)

    tilted = payload["tilted_probabilities"]  # {"bear": int, "base": int, "bull": int}
    probs = enforce_bounds(
        {
            "bear": analysis.bear.probability_pct,
            "base": analysis.base.probability_pct,
            "bull": analysis.bull.probability_pct,
        },
        tilted,
    )

    return {
        "scenarios": {
            name: {
                "probability_pct": probs[name],
                "narrative": getattr(analysis, name).narrative,
            }
            for name in ("bear", "base", "bull")
        },
        "advice": analysis.advice,
        "advice_reasoning": analysis.advice_reasoning,
        "key_risks": analysis.key_risks,
        "model": model,
        "cost_usd": cost,
    }


def enforce_bounds(proposed: dict[str, int], tilted: dict[str, int],
                   max_pp: int = config.LLM_ADJUST_MAX_PP) -> dict[str, int]:
    """Clamp each proposed probability to within ±max_pp of the tilted prior,
    then absorb any sum error into BASE. If the result still can't be made
    valid, fall back to the tilted priors entirely."""
    clamped = {
        k: max(tilted[k] - max_pp, min(tilted[k] + max_pp, int(proposed[k])))
        for k in ("bear", "base", "bull")
    }
    residual = 100 - sum(clamped.values())
    base_adjusted = clamped["base"] + residual
    if abs(base_adjusted - tilted["base"]) <= max_pp and base_adjusted >= 0:
        clamped["base"] = base_adjusted
    else:
        log.warning("LLM probabilities unrecoverable %s vs prior %s; using priors",
                    proposed, tilted)
        return dict(tilted)
    if clamped != proposed:
        log.info("LLM probabilities adjusted to bounds: %s -> %s", proposed, clamped)
    return clamped
