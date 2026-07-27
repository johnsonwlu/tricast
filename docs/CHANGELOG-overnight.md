# Overnight improvement log

Session of 2026-07-27. Constraints honoured: nothing written outside
`~/tricast`; `.env` verified gitignored, absent from all history, never staged.

---

## 1. Scenario probabilities now describe the stock, not the method

**Branch:** `feature/threshold-scenarios` · **Status:** open PR, not merged

### The defect

`scenarios.py` partitioned the simulated distribution at P25/P75, so the prior
probabilities were **always exactly 25/50/25** — for every stock, in every
market, forever. The old docstring said so plainly. That meant:

- A diversified index fund and a 2.5-beta semiconductor got identical odds.
- The probabilities carried no information about the company at all.
- The prediction ledger scored itself against a naive 25/50/25 baseline it was
  mathematically incapable of beating, because it *was* that baseline.

For an app whose entire purpose is "chances of each case", this was the
deepest flaw available to fix.

### The change

A scenario is now a plain claim about the world, and its probability is simply
how often the simulation produced that outcome:

| Scenario | Definition | Probability |
|---|---|---|
| bear | 12-month return **≤ −10%** | share of simulated paths landing there |
| base | between the two | " |
| bull | 12-month return **≥ +20%** | " |

Thresholds are config knobs (`BEAR_RETURN_PCT`, `BULL_RETURN_PCT`). Targets are
the median outcome *within* each region, so they represent the region rather
than its edge.

### Measured result

Backtest over **580 predictions** (10 diversified tickers, 2016–2025):

```
scenario Brier = 0.5313  vs naive 25/50/25 = 0.6552   -> BEATS baseline
```

**19% better than the baseline it previously tied by construction.** Interval
calibration was unaffected (P25–P75 covers 51.4%, want 50%; P10–P90 78.8%, want
80%), so this bought information without costing calibration.

### A latent bug this surfaced

`macro_regime.tilt_probabilities` never clamped: it computed `bear - shift`
with a ±10pp shift and no floor. Unreachable while every prior was 25/50/25 —
but a stable fund can now legitimately have a 4% bear probability, and a
bullish tilt would have driven it **negative**. The shift is now truncated so
neither tail crosses zero, while the sum stays exactly 100.

### Follow-on corrections

- **Ledger** now stores the *threshold prices* a forecast was made under, so a
  prediction is scored against the same definition it was issued with (it
  previously stored the cone's P25/P75, which moved with the simulation).
- **`interim_position`** compares against those fixed thresholds instead of the
  widening cone — a stock could previously appear to change scenario purely
  because time had passed.
- **Backtest** classifies outcomes on the same thresholds, carries each row's
  predicted probabilities, and reports a Brier score against the naive
  baseline.
- **Backtest reporting** — the "realized vs nominal 25/50/25" table became
  meaningless once odds vary per stock; replaced with a reliability table
  (mean predicted probability vs realized frequency).

### Tests

13 new across `tests/test_scenario_thresholds.py` — probabilities differ
between calm and volatile stocks, are integers summing to 100, match the
simulated share, targets sit inside their own region, empty regions fall back
to the threshold instead of NaN, and the tilt never goes negative. Two existing
tests were rewritten because their premise genuinely changed (band no longer
tracks PIT quartiles; it tracks the realized return). **Suite 109 → 121.**

### Honest caveats

- The −10% / +20% thresholds are defensible but not derived from anything;
  they're the obvious first choice, and worth tuning against the backtest.
- Realized frequencies over 2016–2025 are bull-heavy (48% of outcomes ≥ +20%),
  so this window flatters any model. The Brier comparison is still valid — both
  models saw the same window — but the absolute numbers are period-specific.
- Mean PIT is 0.548, i.e. mildly pessimistic. Unchanged by this work; it's a
  drift-estimation issue, and the natural next lever.

---

## Not done (next candidates, in the order I'd take them)

1. **Beta-scale the macro tilt.** The same ±10pp is applied to a utility and a
   2.5-beta semi; it should scale with the stock's sensitivity to the cycle.
2. **Drift / centering.** Mean PIT 0.548 says outcomes land slightly above the
   model's median. Currently a 50/50 blend of capped historical drift and the
   analyst target.
3. **Benchmark-relative returns** — "beats SPY?" is the question a holder
   actually has.
4. **Cut LLM output tokens.** Measured: the same payload produced 699 tokens
   (23.6 s) and 1315 tokens (44.9 s). Latency is linear in output length, so
   this is the biggest remaining speed lever now that the GPU is working.
