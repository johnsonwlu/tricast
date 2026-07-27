# Local Qwen vs a frontier model: is the free option costing you anything?

**Date:** 2026-07-26 · **Local model:** `qwen3:30b-a3b-instruct-2507-q4_K_M` (Ollama,
LAN host) · **Comparison model:** Claude Opus 5

## Method

Both models received the **byte-identical payload** the app builds
(`pipeline._llm_payload`) and the same `SYSTEM_PROMPT`, for two tickers chosen
because judgment — not arithmetic — decides the call: **AMD** (high-beta,
huge simulated spread) and **VOO** (diversified index fund).

There is no `ANTHROPIC_API_KEY` on this machine, so the paid provider could not
be driven through the app. Instead the frontier analysis was produced directly
from the same payload under the same rules. That's a fair test of *judgment*,
but it is not a test of the `anthropic` code path.

The AMD/VOO run used the current branch, whose payload does **not** include the
risk-adjusted block (Sharpe, probability of loss, expected shortfall) — that
work is still open in PR #5. So this measures how each model reasons about raw
quant output, which is exactly the situation PR #5 was written to fix.

## Verdict

**The local model is good enough for the narratives and materially worse at the
advice.** It does not hallucinate prices — its numbers trace back to the input.
It fails in two specific, fixable ways: it misreads the *units* of a field, and
it lets the biggest number in the payload drive the recommendation.

| | Local Qwen 30B | Claude Opus 5 |
|---|---|---|
| Invented price targets | none | none |
| Numbers traceable to payload | 10/11 (AMD), 14/20 (VOO)¹ | all |
| Checkable factual errors | **3** | — |
| Used the tilted probabilities it was given | no (quoted the priors) | yes |
| Weighed downside against upside | no | yes |
| Cost | $0 | ~$0.02–0.03 per analysis |
| Latency | 80–153 s | ~10 s |

¹ The untraceable figures are benign context ("12-month", "S&P 500", "10y-2y"),
not fabricated data.

## The three checkable errors

**1. Read a percentage as a multiple — and inverted a strength into a risk.**

Qwen listed as a key AMD risk:

> "High leverage (debt/equity 6.0) increasing financial risk in a rate-hike environment"

yfinance reports `debtToEquity` **as a percent**. Verified across tickers:

| Ticker | `debtToEquity` | Actually means | Total debt |
|---|---|---|---|
| AMD | 6.005 | 0.06× | $3.9 B |
| NVDA | 6.555 | 0.07× | $12.8 B |
| MSFT | 30.271 | 0.30× | $125 B |
| Ford | 425.544 | 4.26× | $160 B |
| Boeing | 828.696 | 8.29× | $50 B |

AMD's 6.005 means **6%**, one of the cleanest balance sheets in the sector.
Qwen read it as 6.0× — off by ~100× — and turned a genuine strength into a
headline risk.

**This is partly the app's fault, not the model's.** The payload ships a bare
`6.005` with no unit. Any model can misread that. See "Fixes" below.

**2. Quoted the prior probability instead of the tilted one.** On VOO it wrote
"the 25% probability of a bear scenario" when the payload's
`tilted_probabilities` said **23%**. It reached back past the number it was
instructed to use and grabbed `prior_prob: 25`.

**3. Dropped a minus sign.** Same sentence: "a bear scenario (1.5% return)".
The bear return is **−1.5%**. The conclusion happened to survive, but the
statement is false.

## The judgment failure

This is the one you spotted, and it reproduces cleanly. Identical probabilities
on both (23 / 50 / 27):

| | bear | base | bull | expected return | Qwen's call |
|---|---|---|---|---|---|
| **AMD** | −28.6% | +17.7% | **+95.4%** | +28.0% | **buy** |
| **VOO** | **−1.5%** | +12.9% | +27.9% | +13.6% | hold |

Qwen's stated reason to buy AMD:

> "the 12-month target (1,020.11) implying a 95.4% return"

It calls the **bull** target "the 12-month target". That outcome has a 27%
chance. The 50% outcome is +17.7%, and there is a 23% chance of losing 28.6%.
Of AMD's +28.0% expected return, **92% comes from the bull tail alone** — and
that tail exists because beta is 2.469, i.e. it is a measure of volatility, not
of edge. Meanwhile VOO earns +13.6% expected with a worst case of −1.5%.

Two decision-relevant facts sat in the payload and neither appeared in Qwen's
reasoning:

- **Consensus is more conservative than the model.** 47 analysts' mean target
  is $573.15 — **+9.8%** vs spot, *below* the model's own base case of $614.41
  (+17.7%). Qwen cited the "strong buy" rating as support while ignoring that
  the same analysts' price target undercuts the base case.
- **The run has stalled.** Momentum is +229% over 12 months and +105.7% over 6
  — but **+0.4% over the last month**. Qwen quoted the 229% as evidence and
  never mentioned the flat month.

For contrast, the frontier analysis of the same AMD payload concluded **hold**:
the expected return is a volatility artifact, consensus sits below the base
case, momentum has flattened, and a 23% chance of −28.6% is real money — while
noting the business itself is genuinely strong (revenue +37.8%, earnings
+91.2%, and that famously low leverage). On VOO it concluded **buy**: nearly
three-quarters of AMD's base-case return for a twentieth of the downside.

## What the local model does well

Worth being fair about, because it's most of the work:

- **No invented prices.** The schema-without-price-fields design holds up.
- **Competent scenario narratives.** The AMD bear narrative (demand downturn,
  rate pressure, share loss, multiple compression) is genuinely sound.
- **Correct arithmetic** on the figures it does cite.
- **Reasonable risk lists**, apart from the leverage error.
- **Free and private**, with no rate limits.

The gap is not knowledge or fluency. It is *what to do with a number* — units,
which of two similar fields to use, and not being seduced by the largest one.

## Fixes, cheapest first

1. **Fix the payload, not the model.** Send `debt_to_equity_ratio: 0.06`
   instead of `debtToEquity: 6.005`, and label units on every ratio. This
   removes error #1 for *any* model, local or paid. Highest value, ~10 lines.
2. **Drop `prior_prob` from the payload.** It only exists to be confused with
   `tilted_probabilities`. Removing it eliminates error #2 by construction.
3. **Add prompt rules** the judge harness can then verify:
   - "State the base-case return before recommending anything."
   - "Never justify advice with the bull target; the bull case is the minority
     outcome."
   - "Compare the analyst mean target against the base case explicitly."
4. **Merge PR #5.** Putting Sharpe, probability of loss and expected shortfall
   *in the payload* is the structural fix for the judgment failure — it already
   flipped AMD to hold once.
5. **Re-run `scripts/judge.py`** after 1–3 to confirm the errors are gone.

## Verification: the payload fixes worked

Fixes 1 and 2 were applied (unit-suffixed fundamentals, `debt_to_equity_ratio`
as a real multiple, `prior_prob` withheld) and the identical tickers re-run on
the same local model.

| | Before | After |
|---|---|---|
| "High leverage (debt/equity 6.0)" as an AMD risk | present | **gone** — now cites "low debt" as a *strength* |
| Bear probability quoted (VOO) | 25% (the prior) | correct, no misquote |
| Dropped minus sign on −1.5% | present | **gone** |
| Compared analyst mean against base case | no | **yes** — "trading below its analyst consensus mean target ($573 vs. current $522)" |
| Named beta as a downside risk | no | **yes** — "High beta amplifying downside in a broad market sell-off" |
| Advice | buy AMD / hold VOO | buy AMD / hold VOO |

**Three checkable errors went to zero, and two of the reasoning gaps closed on
their own** — the model started citing the consensus target and beta once the
fields around them were unambiguous. Cost: zero per call.

What did *not* change is the advice. Qwen still recommends AMD, and on the
re-run it moved the probabilities further toward the bull case (23/50/27 →
18/50/32, inside the ±10pp bound the code enforces) despite having just
observed that consensus sits below the base case. **The bull-tail bias is not a
data-quality problem and payload hygiene will not fix it** — it is precisely
what the risk-adjusted metrics in PR #5 exist to correct.

## Bottom line

Using the local model is **not** costing you the narratives, and it is **not**
inventing numbers. It is costing you roughly one checkable factual error per
analysis and a systematic bias toward volatile stocks — and most of that is
repairable with payload and prompt changes rather than a credit card.

Reasonable split: **local for the day-to-day**, and reserve a paid call for the
handful of positions you might actually act on. Given the failures above are
concentrated in *advice* rather than *description*, the highest-value use of a
frontier model here is the final buy/hold/avoid call, not the write-ups.

**Caveats:** two tickers, one run each. The factual errors are objectively
verifiable; the judgment comparison is my own assessment of my own output, so
treat that half as an argued case rather than a measurement. The judge harness
with a strong judge model is how you'd make it a measurement.
