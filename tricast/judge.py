"""LLM-as-judge harness: score an analyst's output against a rubric.

Turns "the local model feels worse than a frontier model" into numbers. A judge
model (ideally stronger than the analyst) scores each analysis on four
dimensions, checking the narrative's numeric claims against the ground-truth
inputs the analyst was given. Aggregating across tickers shows *where* the gap
lives (e.g. factual-accuracy lapses like reading spot-vs-target backwards), so
you know which prompt fixes to prioritize instead of guessing.

The judge should be a *different, stronger* model than the analyst — a model
grading its own output is biased. Default judge is `anthropic`; falls back to
whatever provider you point it at.
"""

import logging

from pydantic import BaseModel

from tricast import config, pipeline
from tricast.llm import analyst
from tricast.llm.client import structured_call

log = logging.getLogger(__name__)

BANDS = ("bear", "base", "bull")
DIMENSIONS = ("factual_accuracy", "grounding", "internal_consistency", "specificity")


class JudgeScore(BaseModel):
    factual_accuracy: int      # 1-5: correctly reads/uses the supplied numbers
    grounding: int             # 1-5: cites specific figures vs generic claims
    internal_consistency: int  # 1-5: narratives, probabilities, advice agree
    specificity: int           # 1-5: company-specific vs boilerplate
    overall: int               # 1-5: holistic quality
    issues: list[str]          # concrete, specific problems found


JUDGE_SYSTEM = """You are a strict senior equity-research reviewer grading a junior
analyst's write-up. You are given, as JSON:
- `inputs_given_to_analyst`: the ground-truth data (spot price, quant scenario
  targets, macro signals, fundamentals, analyst consensus, momentum).
- `analyst_output`: the narratives, probabilities, advice, and risks the analyst
  produced from those inputs.
- `probability_weighted_expected_return_pct`: computed from the analyst's own
  probabilities and the scenario targets.

Score 1 (poor) to 5 (excellent) on each dimension. Be demanding — 5 means a
senior analyst would sign off unchanged.

- factual_accuracy: does the write-up read the supplied numbers correctly?
  CHECK EXPLICITLY: is spot above or below the analyst mean target, and does the
  text state the correct direction? Are momentum signs, valuation level, and
  macro readings used correctly? A single reversed fact caps this at 2.
- grounding: does it cite the specific supplied figures (this P/E, this CPI,
  this target) rather than generic statements that fit any company?
- internal_consistency: do the narratives, the probabilities, the advice, and
  the expected return agree with each other? (e.g. "buy" while the reasoning is
  bearish, or probabilities that contradict the stated view.)
- specificity: is the analysis specific to THIS company, or boilerplate that
  would apply verbatim to any large-cap in the sector?

In `issues`, list concrete problems with evidence (quote the claim and the
number it contradicts). If there are none, return an empty list."""


def expected_return_pct(quant_scenarios: dict, probs: dict[str, int]) -> float:
    """Probability-weighted 12-month return using the scenario band returns and
    the analyst's final probabilities. A 3-point approximation, not a true EV,
    but enough to check advice consistency."""
    return round(sum(quant_scenarios[b]["return_pct"] * probs[b] / 100 for b in BANDS), 2)


def judge_analysis(payload: dict, analysis: dict, exp_ret: float,
                   provider: str = "anthropic", model: str | None = None
                   ) -> tuple[JudgeScore, str, float]:
    judge_input = {
        "inputs_given_to_analyst": payload,
        "analyst_output": {
            k: analysis[k] for k in ("scenarios", "advice", "advice_reasoning", "key_risks")
        },
        "probability_weighted_expected_return_pct": exp_ret,
    }
    return structured_call(provider, JUDGE_SYSTEM, judge_input, JudgeScore, model=model)


def run_judge_panel(tickers: list[str], analyst_provider: str = "ollama",
                    judge_provider: str = "anthropic") -> list[dict]:
    """For each ticker: build inputs, run the analyst, then have the judge score
    the result. Returns one row per ticker (skips tickers that error)."""
    rows = []
    for ticker in tickers:
        try:
            report = pipeline.build_report(ticker, run_llm=False)
            payload = pipeline._llm_payload(report)
            analysis = analyst.analyze(payload, provider=analyst_provider)
            final_probs = {b: analysis["scenarios"][b]["probability_pct"] for b in BANDS}
            exp_ret = expected_return_pct(payload["quant_scenarios"], final_probs)
            score, judge_model, cost = judge_analysis(
                payload, analysis, exp_ret, provider=judge_provider)
        except Exception as e:
            log.warning("judge panel skip %s: %s", ticker, e)
            continue
        rows.append({
            "ticker": ticker,
            "analyst_model": analysis["model"],
            "judge_model": judge_model,
            "expected_return_pct": exp_ret,
            "advice": analysis["advice"],
            **{d: getattr(score, d) for d in DIMENSIONS},
            "overall": score.overall,
            "issues": score.issues,
            "judge_cost_usd": cost,
        })
    return rows


def summarize_judge(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    means = {d: round(sum(r[d] for r in rows) / n, 2) for d in (*DIMENSIONS, "overall")}
    worst = min(DIMENSIONS, key=lambda d: means[d])
    all_issues = [f"{r['ticker']}: {i}" for r in rows for i in r["issues"]]
    return {
        "n": n,
        "mean_scores": means,
        "weakest_dimension": worst,
        "total_issues": len(all_issues),
        "issues": all_issues,
        "judge_cost_usd": round(sum(r["judge_cost_usd"] for r in rows), 4),
    }


def format_summary(summary: dict) -> str:
    if summary.get("n", 0) == 0:
        return "No judged analyses (check tickers / providers)."
    s = summary
    lines = [f"Judge panel  (n = {s['n']} analyses)", "-" * 46]
    for d in (*DIMENSIONS, "overall"):
        bar = "#" * round(s["mean_scores"][d]) + "." * (5 - round(s["mean_scores"][d]))
        lines.append(f"  {d:<22} {s['mean_scores'][d]:.2f}/5  [{bar}]")
    lines.append(f"\n  weakest dimension: {s['weakest_dimension']}")
    lines.append(f"  issues flagged: {s['total_issues']}  ·  judge cost ${s['judge_cost_usd']:.3f}")
    if s["issues"]:
        lines.append("\n  Concrete issues:")
        lines += [f"    - {i}" for i in s["issues"][:20]]
    return "\n".join(lines)
