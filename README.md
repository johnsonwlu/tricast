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

**What it already found:** across 1,150 predictions (10 diverse tickers,
2016–2025), the P25–P75 band covered **69%** of outcomes (should be 50%) and
the cone was systematically too wide — the IID bootstrap overstates 12-month
volatility (it samples high-vol days like the 2020 crash independently, with no
mean-reversion). Central tendency was well-calibrated (mean PIT 0.51). This is
concrete evidence for the block-bootstrap / vol-scaling fixes on the roadmap.

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
