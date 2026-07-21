"""Risk-adjusted metrics from the Monte Carlo terminal distribution.

The raw probability-weighted expected return rewards volatility: because price
is lognormal-ish, a high-beta name gets an enormous upper tail that inflates its
mean return, even though that upside comes with a brutal downside. So a volatile
single stock can look more "attractive" than a diversified ETF that is clearly
the better risk-adjusted holding.

These metrics divide reward by risk so the comparison is honest. Since the
horizon is one year, the terminal returns are already annual — Sharpe/Sortino
here are annual figures, no scaling needed.
"""

import numpy as np

from tricast import config


def risk_metrics(terminal: np.ndarray, spot: float,
                 rf: float = config.RISK_FREE_ANNUAL) -> dict:
    """Compute risk-adjusted metrics from the terminal price distribution.
    `rf` is the annual risk-free rate used as the Sharpe/Sortino benchmark."""
    returns = terminal / spot - 1.0                      # 12-month return per path
    exp = float(returns.mean())
    vol = float(returns.std())
    # downside deviation below the risk-free target (Sortino denominator)
    dd = float(np.sqrt(np.mean(np.minimum(returns - rf, 0.0) ** 2)))
    sharpe = round((exp - rf) / vol, 2) if vol > 0 else 0.0
    sortino = round((exp - rf) / dd, 2) if dd > 0 else 0.0

    q5 = np.percentile(returns, 5)
    cvar5 = float(-returns[returns <= q5].mean())        # avg loss in worst 5%

    return {
        "expected_return_pct": round(exp * 100, 1),
        "volatility_pct": round(vol * 100, 1),
        "sharpe": sharpe,                                 # headline risk-adjusted score
        "sortino": sortino,
        "prob_loss_pct": round(float((returns < 0).mean()) * 100),
        "cvar5_pct": round(cvar5 * 100, 1),               # expected shortfall (worst 5%)
        "risk_free_pct": round(rf * 100, 1),
        "label": score_label(sharpe),
    }


def score_label(sharpe: float) -> str:
    """Plain-language read of the annual Sharpe for a 12-month single-name bet."""
    if sharpe >= 0.75:
        return "strong"
    if sharpe >= 0.4:
        return "fair"
    if sharpe >= 0.0:
        return "weak"
    return "poor"
