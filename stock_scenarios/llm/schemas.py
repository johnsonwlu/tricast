"""Structured output schema for the LLM analyst.

Deliberately contains NO price fields: target prices always come from the
quant layer, so the model structurally cannot invent them. Probability bounds
are enforced in code after parsing (the API strips numeric schema constraints
server-side, so ge/le here would be advisory only).
"""

from typing import Literal

from pydantic import BaseModel


class ScenarioNarrative(BaseModel):
    probability_pct: int
    narrative: str  # 2-4 sentences: what would have to happen for this case


class StockAnalysis(BaseModel):
    bear: ScenarioNarrative
    base: ScenarioNarrative
    bull: ScenarioNarrative
    advice: Literal["buy", "hold", "avoid"]
    advice_reasoning: str
    key_risks: list[str]
