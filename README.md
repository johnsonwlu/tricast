# Tricast

Personal stock watchlist with **bull / base / bear scenario analysis**. For each
stock it produces a 12-month price target per scenario with a probability of
occurring, tilted by current economic conditions, plus buy/hold/avoid advice.

How a report is built:

1. **Quant layer** — bootstrap Monte Carlo on 5 years of daily log returns
   (10,000 paths, 12-month horizon). The terminal price distribution is
   partitioned at P25/P75 into bear/base/bull bands; targets are each band's
   median, priors are 25/50/25.
2. **Macro layer** — FRED indicators (yield curve, CPI, unemployment, Fed
   funds) plus VIX produce a regime score in [−1, +1] that shifts bull/bear
   probabilities by up to ±10pp.
3. **LLM layer** — an LLM writes the scenario narratives, may adjust
   probabilities within ±10pp of the tilted priors (enforced in code), and
   gives advice. The output schema contains no price fields, so the model
   cannot invent targets. Two providers (set `LLM_PROVIDER` in `.env`):
   - `ollama` (default) — local model via Ollama's structured-output API.
     Free, no key, no rate limits. Set `OLLAMA_MODEL` (e.g. `qwen3`) and
     `OLLAMA_HOST` if it runs on another machine.
   - `anthropic` — Claude API; ~$0.02–0.03 per analysis, needs
     `ANTHROPIC_API_KEY`.

   Analyses are cached by input hash (including which model produced them)
   and only run when you click "Re-run analysis".

## Setup

```sh
cd ~/stock-scenarios
cp .env.example .env   # defaults to local Ollama; optionally add:
#   FRED_API_KEY      — free, enables the macro tilt
#   ANTHROPIC_API_KEY — only if you set LLM_PROVIDER=anthropic
```

For the default local LLM path, install [Ollama](https://ollama.com) and pull
a model, e.g. `ollama pull qwen3`.

The venv is already created at `.venv` (Python 3.13). To recreate:

```sh
python3.13 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

The app degrades gracefully without keys: market data (yfinance) needs no key;
without `FRED_API_KEY` reports run with a neutral macro regime; without
`ANTHROPIC_API_KEY` you get quant scenarios but no narratives/advice.

## Run

```sh
.venv/bin/streamlit run app.py          # dashboard at http://localhost:8501
.venv/bin/python scripts/run_analysis.py NVDA            # CLI report (free)
.venv/bin/python scripts/run_analysis.py NVDA --llm      # + Claude analysis (paid)
.venv/bin/python scripts/run_analysis.py NVDA --data-only
.venv/bin/python scripts/score_predictions.py   # score matured predictions
.venv/bin/pytest                        # offline test suite
```

## Bulk analysis

Analyze the whole watchlist in one pass instead of clicking each stock. There's
an **"Analyze all"** button on the watchlist page, and a CLI:

```sh
.venv/bin/python scripts/analyze_watchlist.py            # LLM, whole watchlist
.venv/bin/python scripts/analyze_watchlist.py AAPL MSFT  # just these
.venv/bin/python scripts/analyze_watchlist.py --no-llm   # quant only (free, fast)
```

It runs sequentially (the local model serializes on one host anyway), isolates
failures so one bad ticker doesn't stop the batch, and reuses the analysis
cache — so re-running only pays for stocks whose inputs actually changed. With
the paid provider it reports total cost.

## LLM judge

The narratives and advice come from an LLM, so their quality is hard to
eyeball. The judge harness makes it measurable: a judge model scores each
analysis 1–5 on **factual accuracy, grounding, internal consistency, and
specificity**, checking the write-up's numeric claims against the ground-truth
inputs the analyst was given.

```sh
# judge should be STRONGER than the analyst (self-grading is biased):
python scripts/judge.py AAPL MSFT NVDA --judge-provider anthropic
# key-free run with a local judge (weaker signal, but factual/math errors
# are checkable regardless of judge strength):
python scripts/judge.py AAPL --analyst-provider ollama --judge-provider ollama
```

Even a local self-judge already earns its keep — on the first run it flagged
the analyst claiming a base-case return of "5.4%" when the input clearly says
8.1%, and misreading forward-vs-trailing P/E. These are exactly the fixable
lapses (a "state spot-vs-target direction before advising" prompt rule, a
few-shot example) that close the gap between a local model and a frontier one
without paying per call. Run it, see which dimension scores lowest, fix that
prompt, re-run.

## Backtest

The prediction ledger takes a year to produce its first scored point. The
backtest answers the same question *today* by replaying the quant engine on
past dates (strictly no look-ahead) and checking whether outcomes actually
landed in the bands at their nominal frequencies:

```sh
.venv/bin/python scripts/backtest.py AAPL MSFT NVDA --start 2016-01-01
```

The headline statistic is the Probability Integral Transform (PIT) — the
percentile of the simulated distribution where the realized price landed. If
the model is calibrated, PIT values are uniform: 25% of outcomes below P25,
the P25–P75 band covers 50%, the P10–P90 cone covers 80%.

**What it found, and the fix it drove:** across 1,150 predictions (10 diverse
tickers, 2016–2025), the P25–P75 band covered **69%** of outcomes (should be
50%) — the simulated cone was systematically too wide. Central tendency was
well-calibrated (mean PIT 0.51), so the defect is dispersion, not direction.

The simulator now uses a **block bootstrap** (resampling contiguous ~21-day
blocks) instead of IID daily draws, which captures the multi-day mean-reversion
that IID ignores. Measured A/B on identical inputs: P25–P75 coverage improved
68.6% → 65.1% and every tail moved toward its nominal. It helps on every metric
but closes only part of the gap — which localizes the remaining error to
volatility *level* (the 5-year lookback spans high-vol 2020/2022), making
**vol-scaling to current conditions** the next and larger lever. Run
`scripts/backtest.py --block 1` to reproduce the old IID numbers.

## Prediction ledger

Every report logs its forecast (targets, bands, probabilities, regime, model)
to a ledger — one row per ticker per day. Once a prediction's 12-month horizon
passes, `scripts/score_predictions.py` (or the button on the Prediction Ledger
page) classifies the actual outcome and computes a Brier score, so over time
you learn whether the model's probabilities are actually calibrated — and
whether they beat the naive always-25/50/25 baseline.

## Layout

```
app.py                    Watchlist page
pages/                    Stock Detail (fan chart, scenario cards), Macro Regime
tricast/                  UI-free library
  config.py               every tunable knob (horizon, percentiles, tilt, model)
  store.py                SQLite cache + watchlist + saved analyses
  data/                   yfinance + FRED with incremental/TTL caching
  quant/                  Monte Carlo + scenario band math
  macro_regime.py         indicator signals → score → probability tilt
  llm/                    Claude structured output + bounds enforcement
  pipeline.py             ticker → full report
data/cache.db             SQLite (safe to delete; refetches on next run)
```

---

> **Disclaimer:** This tool is for personal research only — it is **not
> financial advice**. Probabilities and price targets are model estimates
> derived from historical data and an LLM's interpretation; they can be badly
> wrong. Past performance does not guarantee future results.
