"""Macro regime scoring: FRED indicators + VIX -> composite score -> regime
label -> bounded, sum-preserving tilt of scenario probabilities."""

from dataclasses import dataclass

import pandas as pd

from stock_scenarios import config


@dataclass
class IndicatorSignal:
    name: str
    value: float
    signal: int          # +1 expansionary / 0 neutral / -1 contractionary
    detail: str


def signal_yield_curve(t10y2y: float) -> int:
    if t10y2y > 0.5:
        return 1
    if t10y2y < 0:
        return -1
    return 0


def signal_cpi_yoy(cpi_yoy_pct: float) -> int:
    if cpi_yoy_pct < 2.5:
        return 1
    if cpi_yoy_pct > 4.0:
        return -1
    return 0


def signal_unemployment_trend(change_3m_pp: float) -> int:
    if change_3m_pp <= -0.1:
        return 1
    if change_3m_pp >= 0.2:
        return -1
    return 0


def signal_fed_funds_trend(change_6m_pp: float) -> int:
    if change_6m_pp <= -0.25:
        return 1
    if change_6m_pp >= 0.25:
        return -1
    return 0


def signal_vix(vix: float) -> int:
    if vix < 16:
        return 1
    if vix > 24:
        return -1
    return 0


def compute_signals(series: dict[str, pd.Series], vix: float) -> list[IndicatorSignal]:
    """series: FRED series keyed by id (T10Y2Y, CPIAUCSL, UNRATE, FEDFUNDS)."""
    # round before scoring so displayed values and signals always agree
    # (raw float subtraction can yield -0.0999... vs the -0.1 threshold)
    t10y2y = round(float(series["T10Y2Y"].iloc[-1]), 2)

    cpi = series["CPIAUCSL"]
    cpi_yoy = round(float((cpi.iloc[-1] / cpi.iloc[-13] - 1) * 100), 2)  # 12m back

    unrate = series["UNRATE"]
    un_chg = round(float(unrate.iloc[-1] - unrate.iloc[-4]), 2)          # 3-month change

    ff = series["FEDFUNDS"]
    ff_chg = round(float(ff.iloc[-1] - ff.iloc[-7]), 2)                  # 6-month change

    return [
        IndicatorSignal("Yield curve (10y-2y)", round(t10y2y, 2),
                        signal_yield_curve(t10y2y), f"{t10y2y:+.2f}pp spread"),
        IndicatorSignal("CPI YoY", round(cpi_yoy, 2),
                        signal_cpi_yoy(cpi_yoy), f"{cpi_yoy:.1f}% inflation"),
        IndicatorSignal("Unemployment 3m trend", round(un_chg, 2),
                        signal_unemployment_trend(un_chg), f"{un_chg:+.1f}pp over 3m"),
        IndicatorSignal("Fed funds 6m trend", round(ff_chg, 2),
                        signal_fed_funds_trend(ff_chg), f"{ff_chg:+.2f}pp over 6m"),
        IndicatorSignal("VIX", round(vix, 1), signal_vix(vix), f"VIX at {vix:.1f}"),
    ]


def composite_score(signals: list[IndicatorSignal]) -> float:
    return sum(s.signal for s in signals) / len(signals)


def regime_label(score: float) -> str:
    if score >= config.REGIME_EXPANSIONARY:
        return "Expansionary"
    if score <= config.REGIME_CONTRACTIONARY:
        return "Contractionary"
    return "Neutral"


def tilt_probabilities(priors: dict[str, int], score: float) -> dict[str, int]:
    """Shift bull/bear symmetrically by round(score * TILT_MAX_PP); base is
    untouched, so the sum stays exactly 100 and the tilt is bounded ±10pp."""
    shift = round(score * config.TILT_MAX_PP)
    return {
        "bear": priors["bear"] - shift,
        "base": priors["base"],
        "bull": priors["bull"] + shift,
    }
