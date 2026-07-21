"""Block-bootstrap Monte Carlo on daily log returns.

Bootstrap (vs GBM) preserves the empirical fat tails and skew that define
bull/bear scenarios. Historical drift is removed and replaced with an explicit
blend of capped historical drift and the analyst-consensus implied return,
since raw past drift is a poor forward estimate.

We resample contiguous *blocks* of days (circular block bootstrap) rather than
individual days. Independent daily draws assume returns are serially
uncorrelated, so summing `horizon` of them reproduces `horizon x` the daily
variance — but real equity returns mean-revert over multi-day horizons, so
their true multi-day variance is lower. IID therefore overstates 12-month
dispersion (the backtest measured P25-P75 covering ~69% of outcomes instead of
50%). Blocks preserve the autocorrelation structure, narrowing the cone toward
reality. `block=1` recovers the old IID behavior exactly (for A/B testing).
"""

import numpy as np
import pandas as pd

from tricast import config


def _block_bootstrap(demeaned: np.ndarray, n_paths: int, horizon: int,
                     block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap: each path is built from contiguous blocks of
    `block` days sampled from random start positions (wrapping past the end).
    Returns an (n_paths, horizon) array of resampled returns."""
    if block <= 1:
        return rng.choice(demeaned, size=(n_paths, horizon), replace=True)
    n = len(demeaned)
    n_blocks = -(-horizon // block)                       # ceil division
    starts = rng.integers(0, n, size=(n_paths, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets) % n              # (n_paths, n_blocks, block)
    idx = idx.reshape(n_paths, n_blocks * block)[:, :horizon]
    return demeaned[idx]


def compute_drift(returns: pd.Series, spot: float, analyst_target: float | None) -> float:
    """Annual log drift: 50/50 blend of capped historical drift and the
    analyst-consensus implied 12-month return; historical-only if no target."""
    cap = config.DRIFT_CAP_ANNUAL
    mu_hist = float(np.clip(returns.mean() * 252, -cap, cap))
    if analyst_target and analyst_target > 0 and spot > 0:
        mu_analyst = float(np.clip(np.log(analyst_target / spot), -cap, cap))
        return 0.5 * mu_hist + 0.5 * mu_analyst
    return mu_hist


def simulate(
    closes: pd.Series,
    analyst_target: float | None = None,
    horizon: int = config.HORIZON_DAYS,
    n_paths: int = config.N_PATHS,
    seed: int = config.RNG_SEED,
    block: int | None = None,
) -> dict:
    """Run the simulation. Returns terminal prices and the percentile cone.

    closes: daily close series, oldest first. Requires MIN_HISTORY_DAYS points.
    block: block-bootstrap length in days (defaults to config.BLOCK_SIZE);
           block=1 recovers plain IID resampling.
    """
    if len(closes) < config.MIN_HISTORY_DAYS:
        raise ValueError(
            f"Need at least {config.MIN_HISTORY_DAYS} days of history, got {len(closes)}"
        )
    block = config.BLOCK_SIZE if block is None else block
    spot = float(closes.iloc[-1])
    log_returns = np.log(closes / closes.shift(1)).dropna()

    mu_annual = compute_drift(log_returns, spot, analyst_target)
    demeaned = (log_returns - log_returns.mean()).to_numpy()
    daily_drift = mu_annual / 252

    rng = np.random.default_rng(seed)
    # (n_paths, horizon) resampled daily returns with drift re-injected
    samples = _block_bootstrap(demeaned, n_paths, horizon, block, rng) + daily_drift
    cum_log = np.cumsum(samples, axis=1)
    paths = spot * np.exp(cum_log)            # (n_paths, horizon)
    terminal = paths[:, -1]

    cone = {
        f"p{p}": np.percentile(paths, p, axis=0).tolist()
        for p in config.CONE_PERCENTILES
    }
    return {
        "spot": spot,
        "mu_annual": mu_annual,
        "terminal": terminal,
        "cone": cone,
        "horizon": horizon,
    }
