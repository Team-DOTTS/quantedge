"""Recommended mix versus the alternatives, and the robustness checks behind it.

Builds each candidate portfolio on the same data and scores it on the measures
the committee cares about: volatility, Sortino, maximum drawdown, CVaR and the
crisis replays, plus the mandate checks. Also holds the evidence used to choose
the mix: a grid search over round-number mixes, a walk-forward test of
"optimise on the past", a block bootstrap of 5-year drawdowns, sub-period
results and the facts quoted in the memo."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import kpis, stress
from . import portfolio as pf
from . import var_es as ve
from .config import (DRAFT_WEIGHTS, HISTORICAL_SCENARIOS, MANDATE, RECOMMENDED_WEIGHTS, TICKERS,
                     TRADING_DAYS, TRIGGER_WEIGHTS)

GFC = "2008 Global Financial Crisis"
COVID = "March 2020 COVID crash"
RATES_2022 = "2022 Rate-hiking drawdown"
SATELLITES = ["EEM", "XLE", "XLK"]
LIMITS = MANDATE["allocation_limits"]

# Constrained Markowitz: the mandate's allocation limits (AGG floor, per-fund caps,
# no shorting) with no volatility cap, so the optimiser is free to disagree.
MARKOWITZ_CONS = pf.Constraints(max_weight=LIMITS["max_weight"], min_bonds=LIMITS["min_bonds"],
                                fund_caps=tuple(LIMITS["fund_caps"].items()))
LONG_ONLY = pf.Constraints()
# The full mandate allocation limits, used for the minimum-CVaR candidate.
MANDATE_CONS = pf.Constraints(max_weight=LIMITS["max_weight"], max_equity=LIMITS["max_equity"],
                              min_bonds=LIMITS["min_bonds"], fund_caps=tuple(LIMITS["fund_caps"].items()))

# Round-number mixes searched for the drawdown floor and the walk-forward test (%).
GRID = {"AGG": range(35, 81, 5), "GLD": range(0, 31, 5), "SPY": range(0, 41, 5),
        "EFA": range(0, 16, 5), "EEM": range(0, 6, 5), "XLE": range(0, 6, 5), "XLK": range(0, 6, 5)}


def _series(d: dict) -> pd.Series:
    return pd.Series(d, dtype=float).reindex(TICKERS).fillna(0.0)


def candidate_weights(sr: pd.DataFrame, rf_annual: float, cov_method: str = "sample"
                      ) -> dict[str, pd.Series]:
    mu, cov, _ = pf.estimate_inputs(sr, cov_method)
    return {
        "Recommended": _series(RECOMMENDED_WEIGHTS),
        "Trigger mix (gold 25%)": _series(TRIGGER_WEIGHTS),
        "First draft": _series(DRAFT_WEIGHTS),
        "Constrained Markowitz": pf.max_sharpe(mu, cov, MARKOWITZ_CONS, rf_annual).weights,
        "Minimum CVaR (99%)": pf.min_cvar(sr, MANDATE_CONS, 0.99).weights,
        "Max Sharpe (long-only)": pf.max_sharpe(mu, cov, LONG_ONLY, rf_annual).weights,
        "Minimum variance": pf.min_variance(cov, LONG_ONLY).weights,
        "Equal weight": _series({t: 1 / len(TICKERS) for t in TICKERS}),
        "60/40 SPY/AGG": _series({"SPY": 0.6, "AGG": 0.4}),
    }


def score(sr: pd.DataFrame, rf_daily: pd.Series, w: pd.Series) -> dict:
    lim = MANDATE["risk_limits"]
    r = pf.portfolio_returns(sr, w)
    k = kpis.kpi_table(r, rf_daily)
    rf_annual = float(rf_daily.reindex(r.index).fillna(0.0).mean() * TRADING_DAYS)
    loss = np.log1p(r.values)
    _, es95 = ve.historical(loss, 0.95)
    _, es99 = ve.historical(loss, 0.99)
    replays = {lab: stress.replay(sr, w, HISTORICAL_SCENARIOS[sc]["start"], HISTORICAL_SCENARIOS[sc]["end"])
               for lab, sc in (("2008", GFC), ("2020", COVID), ("2022", RATES_2022))}
    excess = k["CAGR"] - rf_annual
    es99_pct = float(ve.to_loss_pct(es99))
    return {
        "CAGR": k["CAGR"],
        "Excess over T-bills": excess,
        "Volatility": k["Annualised volatility"],
        "Sharpe": k["Sharpe ratio"],
        "Sortino": k["Sortino ratio"],
        "Max drawdown": k["Maximum drawdown"],
        "CVaR 95% 1d": float(ve.to_loss_pct(es95)),
        "CVaR 99% 1d": es99_pct,
        "2008 replay": replays["2008"]["total_return"],
        "2008 max DD": replays["2008"]["max_drawdown"],
        "2020 replay": replays["2020"]["total_return"],
        "2022 replay": replays["2022"]["total_return"],
        "Satellite share": float(w.reindex(SATELLITES).sum()),
        "Meets return": excess >= MANDATE["return_excess_over_rf"],
        "Meets vol": k["Annualised volatility"] <= lim["max_volatility"],
        "Meets ES": es99_pct <= lim["max_es99_1d"],
        "Meets drawdown": abs(k["Maximum drawdown"]) <= lim["max_drawdown"],
    }


def comparison(sr: pd.DataFrame, rf_daily: pd.Series, rf_annual: float | None = None,
               cov_method: str = "sample") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (weights table, metrics table), one row per candidate."""
    if rf_annual is None:
        rf_annual = float(rf_daily.reindex(sr.index).fillna(0.0).mean() * TRADING_DAYS)
    cands = candidate_weights(sr, rf_annual, cov_method)
    weights = pd.DataFrame(cands).T[TICKERS]
    metrics = pd.DataFrame({name: score(sr, rf_daily, w) for name, w in cands.items()}).T
    return weights, metrics


def distance(a: pd.Series, b: pd.Series) -> float:
    """One-way turnover needed to move from a to b (half the sum of absolute differences)."""
    return float((a - b.reindex(a.index).fillna(0.0)).abs().sum() / 2)


# ---------------------------------------------------------------------------
# Evidence behind the choice
# ---------------------------------------------------------------------------
def _stats(r: np.ndarray, rf: np.ndarray) -> dict:
    ex = r - rf
    downside = np.sqrt(np.mean(np.minimum(ex, 0) ** 2)) * np.sqrt(TRADING_DAYS)
    wealth = np.cumprod(1 + r)
    tail = np.sort(np.log1p(r))[:max(1, int(len(r) * 0.01))]
    return {"cagr": wealth[-1] ** (TRADING_DAYS / len(r)) - 1,
            "vol": r.std() * np.sqrt(TRADING_DAYS),
            "sortino": ex.mean() * TRADING_DAYS / downside,
            "mdd": (wealth / np.maximum.accumulate(wealth) - 1).min(),
            "es99": 1 - np.exp(tail.mean())}


def grid_search(sr: pd.DataFrame, rf_daily: pd.Series, split: str | None = None) -> pd.DataFrame:
    """Score every round-number mix in GRID. With `split`, scores are computed
    separately before ("_in") and from ("_out") that date."""
    r, rf = sr[TICKERS].values, rf_daily.reindex(sr.index).fillna(0.0).values
    parts = {"": np.ones(len(sr), dtype=bool)} if split is None else \
        {"_in": sr.index < split, "_out": sr.index >= split}
    rows = []
    for combo in itertools.product(*GRID.values()):
        if sum(combo) != 100:
            continue
        d = dict(zip(GRID, combo))
        w = np.array([d[t] for t in TICKERS]) / 100
        row = {t: d[t] / 100 for t in TICKERS}
        for suffix, m in parts.items():
            row.update({k + suffix: v for k, v in _stats(r[m] @ w, rf[m]).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def walk_forward(sr: pd.DataFrame, rf_daily: pd.Series, split: str) -> dict:
    """'Optimise on the past': among mixes meeting the risk limits before `split`,
    pick the best in-sample Sortino, then see how it did afterwards."""
    lim = MANDATE["risk_limits"]
    g = grid_search(sr, rf_daily, split)
    ok = g[(g["vol_in"] <= lim["max_volatility"]) & (g["es99_in"] <= lim["max_es99_1d"])
           & (g["mdd_in"] >= -lim["max_drawdown"])]
    pick = ok.loc[ok["sortino_in"].idxmax()]
    r, rf = sr[TICKERS].values, rf_daily.reindex(sr.index).fillna(0.0).values
    after = sr.index >= split
    keys = ("cagr", "vol", "sortino", "mdd", "es99")
    return {"split": split, "feasible": len(ok), "pick": pick[TICKERS],
            "in": {k: float(pick[k + "_in"]) for k in keys},
            "out": {k: float(pick[k + "_out"]) for k in keys},
            "pick_out_percentile": float((ok["sortino_out"] < pick["sortino_out"]).mean()),
            "recommended_out": _stats(r[after] @ _series(RECOMMENDED_WEIGHTS).values, rf[after])}


def drawdown_floor(grid: pd.DataFrame) -> dict:
    """Shallowest maximum drawdown any mix in the grid achieved, and what it needed."""
    best = grid.loc[grid["mdd"].idxmax()]
    in_cap = grid["AGG"] <= LIMITS["fund_caps"]["AGG"] + 1e-9
    return {"best_mdd": float(best["mdd"]), "weights": best[TICKERS],
            "best_mdd_in_cap": float(grid.loc[in_cap, "mdd"].max()),
            "share_within_15": float((grid["mdd"] >= -0.15).mean()), "n": len(grid)}


def subperiods(sr: pd.DataFrame, rf_daily: pd.Series, weights: dict[str, pd.Series],
               cuts: list[tuple[str, str, str]]) -> pd.DataFrame:
    rows = []
    rf = rf_daily.reindex(sr.index).fillna(0.0)
    for label, start, end in cuts:
        m = (sr.index >= start) & (sr.index <= end)
        for name, w in weights.items():
            s = _stats(sr.values[m] @ w.reindex(sr.columns).values, rf.values[m])
            rows.append({"period": label, "portfolio": name, **s})
    return pd.DataFrame(rows)


def bootstrap_drawdowns(sr: pd.DataFrame, w: pd.Series, rf_annual: float, years: int = 5,
                        block: int = 21, n_paths: int = 3000, seed: int = 7) -> dict:
    """Block bootstrap of `years`-long paths built from 1-month blocks of history
    (keeps volatility clustering inside each block)."""
    r = sr.values @ w.reindex(sr.columns).values
    rng = np.random.default_rng(seed)
    n_blocks = years * TRADING_DAYS // block
    starts = rng.integers(0, len(r) - block, size=(n_paths, n_blocks))
    paths = r[starts[..., None] + np.arange(block)].reshape(n_paths, -1)
    wealth = np.cumprod(1 + paths, axis=1)
    dd = (wealth / np.maximum.accumulate(wealth, axis=1) - 1).min(axis=1)
    cagr = wealth[:, -1] ** (1 / years) - 1
    target = rf_annual + MANDATE["return_excess_over_rf"]
    return {"p_dd_15": float((dd < -0.15).mean()), "p_dd_20": float((dd < -0.20).mean()),
            "median_dd": float(np.median(dd)), "dd_5pct": float(np.quantile(dd, 0.05)),
            "p_below_target": float((cagr < target).mean()), "p_loss": float((cagr < 0).mean()),
            "median_cagr": float(np.median(cagr)), "target": target}


def fact_check(sr: pd.DataFrame) -> pd.Series:
    """Facts the memo quotes, computed from the data."""
    y22 = (1 + sr.loc["2022-01-01":"2022-12-31"]).prod() - 1
    y08 = (1 + sr.loc["2008-01-01":"2008-12-31"]).prod() - 1
    monthly = (1 + sr[["SPY", "EFA"]]).groupby(sr.index.to_period("M")).prod() - 1
    rec = _series(RECOMMENDED_WEIGHTS)
    return pd.Series({
        "AGG return, calendar 2008": y08["AGG"],
        "GLD return, calendar 2008": y08["GLD"],
        "SPY return, calendar 2008": y08["SPY"],
        "AGG return, calendar 2022": y22["AGG"],
        "GLD return, calendar 2022": y22["GLD"],
        "XLE return, calendar 2022": y22["XLE"],
        "SPY return, calendar 2022": y22["SPY"],
        "SPY-EFA correlation (daily)": sr["SPY"].corr(sr["EFA"]),
        "SPY-EFA correlation (monthly)": monthly["SPY"].corr(monthly["EFA"]),
        "SPY-AGG correlation (daily)": sr["SPY"].corr(sr["AGG"]),
        "SPY-GLD correlation (daily)": sr["SPY"].corr(sr["GLD"]),
        "Recommended, calendar 2008": stress.replay(sr, rec, "2008-01-01", "2008-12-31")["total_return"],
        "Recommended, calendar 2022": stress.replay(sr, rec, "2022-01-01", "2022-12-31")["total_return"],
    }, name="value")
