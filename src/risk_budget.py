"""Risk budgeting and position sizing.

Euler allocation: because VaR and ES are homogeneous of degree one in the
weights, total risk = sum_i w_i * dRisk/dw_i. Each term is that fund's
COMPONENT contribution; dRisk/dw_i is its MARGINAL contribution (extra risk from
adding a little more of the fund).

- Parametric (Normal) component VaR:  CVaR_i = w_i (Sigma w)_i / sigma_p * z
- Historical component ES:            CES_i = -w_i * E[r_i | portfolio in its worst (1-c) tail]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def component_var_normal(simple_ret: pd.DataFrame, weights: pd.Series, c: float) -> pd.DataFrame:
    x = simple_ret.dropna()
    w = weights.reindex(x.columns).values
    cov = np.cov(x.values, rowvar=False)
    mu = x.mean().values
    sigma_p = np.sqrt(w @ cov @ w)
    z = -stats.norm.ppf(1 - c)
    marginal = -mu + z * (cov @ w) / sigma_p
    comp = w * marginal
    return pd.DataFrame({"weight": w, "marginal VaR": marginal, "component VaR": comp,
                         "share of VaR": comp / comp.sum()}, index=x.columns)


def component_es_historical(simple_ret: pd.DataFrame, weights: pd.Series, c: float) -> pd.DataFrame:
    x = simple_ret.dropna()
    w = weights.reindex(x.columns).values
    port = x.values @ w
    q = np.quantile(port, 1 - c)
    tail = port <= q
    marginal = -x.values[tail].mean(axis=0)
    comp = w * marginal
    return pd.DataFrame({"weight": w, "marginal ES": marginal, "component ES": comp,
                         "share of ES": comp / comp.sum()}, index=x.columns)


def risk_budget_table(simple_ret: pd.DataFrame, weights: pd.Series, c: float) -> pd.DataFrame:
    v = component_var_normal(simple_ret, weights, c)
    e = component_es_historical(simple_ret, weights, c)
    out = pd.concat([v, e.drop(columns="weight")], axis=1)
    out["risk / capital (ES)"] = out["share of ES"] / out["weight"].replace(0, np.nan)
    return out


def scale_to_limit(es_current: float, es_limit: float) -> float:
    """Share of capital that can stay in the risky portfolio so that ES meets the
    limit, if the remainder were held in cash (ES scales linearly with exposure)."""
    if es_current <= 0:
        return 1.0
    return float(min(1.0, es_limit / es_current))
