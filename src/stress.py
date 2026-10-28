"""Scenario and stress testing.

Historical scenarios replay the actual daily fund returns of each crisis window
on today's weights (buy-and-hold from the start of the window, no rebalancing).
Hypothetical scenarios apply an instantaneous shock to each fund."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import var_es as ve


def replay(simple_ret: pd.DataFrame, weights: pd.Series, start: str, end: str,
           value: float = 1.0) -> dict | None:
    """Portfolio path through a historical window with buy-and-hold weights."""
    window = simple_ret.loc[start:end]
    if len(window) < 2:
        return None
    w = weights.reindex(window.columns).fillna(0.0)
    growth = (1 + window).cumprod()
    asset_paths = growth.mul(w, axis=1)
    path = asset_paths.sum(axis=1)
    path = pd.concat([pd.Series([1.0], index=[window.index[0] - pd.Timedelta(days=1)]), path])
    port_daily = path.pct_change().dropna()
    dd = path / path.cummax() - 1
    asset_total = growth.iloc[-1] - 1
    return {
        "path": path * value,
        "total_return": float(path.iloc[-1] - 1),
        "max_drawdown": float(dd.min()),
        "worst_day": float(port_daily.min()),
        "days": len(window),
        "asset_returns": asset_total,
        "contribution": w * asset_total,  # each fund's contribution to the total, in return points
    }


def historical_table(simple_ret: pd.DataFrame, weights: pd.Series, scenarios: dict,
                     value: float, benchmark: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    for name, sc in scenarios.items():
        res = replay(simple_ret, weights, sc["start"], sc["end"], value)
        if res is None:
            continue
        row = {"scenario": name, "start": sc["start"], "end": sc["end"], "days": res["days"],
               "portfolio return": res["total_return"], "max drawdown": res["max_drawdown"],
               "worst day": res["worst_day"], "loss (USD)": -res["total_return"] * value}
        if benchmark is not None:
            b = replay(simple_ret, benchmark, sc["start"], sc["end"])
            row["benchmark return"] = b["total_return"] if b else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("scenario")


def hypothetical(weights: pd.Series, shocks: dict, value: float) -> dict:
    s = pd.Series(shocks, dtype=float).reindex(weights.index).fillna(0.0)
    contrib = weights * s
    total = float(contrib.sum())
    return {"total_return": total, "loss_usd": -total * value, "contribution": contrib,
            "shocks": s}


def stressed_var(port_log: pd.Series, start: str, end: str, c: float) -> dict | None:
    """Basel-style stressed VaR/ES: historical simulation using only the crisis window."""
    x = port_log.loc[start:end].values
    if len(x) < 30:
        return None
    v, e = ve.historical(x, c)
    return {"var": v, "es": e, "days": len(x)}
