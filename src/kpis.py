"""Layer 1: institutional performance KPIs.

All KPIs are computed from daily SIMPLE returns (they measure realised wealth)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import TRADING_DAYS
from .portfolio import capm


def wealth_index(r: pd.Series, start_value: float = 1.0) -> pd.Series:
    return start_value * (1 + r).cumprod()


def drawdown(r: pd.Series) -> pd.Series:
    w = wealth_index(r)
    return (w / w.cummax() - 1).rename("Drawdown")


def max_drawdown_details(r: pd.Series) -> dict:
    dd = drawdown(r)
    trough = dd.idxmin()
    peak = wealth_index(r).loc[:trough].idxmax()
    after = dd.loc[trough:]
    recovered = after[after >= 0]
    recovery = recovered.index[0] if len(recovered) else None
    return {"max_drawdown": float(dd.min()), "peak": peak, "trough": trough,
            "recovery": recovery,
            "days_to_recover": None if recovery is None else int((dd.loc[trough:recovery]).shape[0] - 1)}


def kpi_table(r: pd.Series, rf_daily: pd.Series, market: pd.Series | None = None,
              benchmark: pd.Series | None = None) -> dict:
    """Standard institutional KPIs (annualised where relevant)."""
    r = r.dropna()
    rf = rf_daily.reindex(r.index).fillna(0.0)
    n = len(r)
    years = n / TRADING_DAYS
    w = wealth_index(r)
    cagr = w.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    excess = r - rf
    sharpe = excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)
    downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2)) * np.sqrt(TRADING_DAYS)
    sortino = excess.mean() * TRADING_DAYS / downside if downside > 0 else np.nan
    mdd = drawdown(r).min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    out = {
        "CAGR": cagr,
        "Annualised volatility": vol,
        "Sharpe ratio": sharpe,
        "Sortino ratio": sortino,
        "Maximum drawdown": mdd,
        "Calmar ratio": calmar,
        "Omega ratio (0%)": gains / losses if losses > 0 else np.nan,
        "Skewness": stats.skew(r),
        "Excess kurtosis": stats.kurtosis(r),
        "Worst day": r.min(),
        "Best day": r.max(),
        "Positive days": (r > 0).mean(),
    }
    if market is not None:
        c = capm(r, market, rf)
        out["Beta (vs SPY)"] = c["beta"]
        out["Treynor ratio"] = excess.mean() * TRADING_DAYS / c["beta"] if c["beta"] else np.nan
        out["Jensen's alpha"] = c["alpha_annual"]
        out["Alpha t-stat"] = c["alpha_t"]
    if benchmark is not None:
        active = (r - benchmark.reindex(r.index)).dropna()
        te = active.std() * np.sqrt(TRADING_DAYS)
        out["Tracking error"] = te
        out["Information ratio"] = active.mean() * TRADING_DAYS / te if te > 0 else np.nan
    return out


KPI_FORMATS = {
    "CAGR": "pct", "Annualised volatility": "pct", "Maximum drawdown": "pct",
    "Worst day": "pct", "Best day": "pct", "Positive days": "pct", "Jensen's alpha": "pct",
    "Tracking error": "pct",
}

KPI_EXPLAIN = {
    "CAGR": "Average yearly growth rate actually achieved.",
    "Annualised volatility": "Typical size of yearly swings in value.",
    "Sharpe ratio": "Return above cash per unit of total risk. Higher is better.",
    "Sortino ratio": "Like Sharpe, but only counts downside swings as risk.",
    "Maximum drawdown": "Largest fall from a peak to a later low.",
    "Calmar ratio": "Growth rate divided by the worst drawdown.",
    "Omega ratio (0%)": "Sum of gains divided by sum of losses.",
    "Skewness": "Negative means large losses are more common than large gains.",
    "Excess kurtosis": "Above 0 means extreme days happen more than a bell curve predicts.",
    "Worst day": "Largest single-day loss in the sample.",
    "Best day": "Largest single-day gain in the sample.",
    "Positive days": "Share of days with a gain.",
    "Beta (vs SPY)": "Sensitivity to the US stock market (1 = moves with it).",
    "Treynor ratio": "Return above cash per unit of market risk (beta).",
    "Jensen's alpha": "Return beyond what CAPM predicts for this beta.",
    "Alpha t-stat": "Above ~2 means alpha is statistically distinguishable from zero.",
    "Tracking error": "Volatility of the difference versus the benchmark.",
    "Information ratio": "Excess return over the benchmark per unit of tracking error.",
}
