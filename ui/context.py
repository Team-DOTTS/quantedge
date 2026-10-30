"""Sidebar controls and the cached computation pipeline shared by every page.

Pages never compute heavy results directly: they call the cached
functions here, so switching pages is instant and results are consistent."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from src import backtest as bt
from src import compare, data, evt, kpis, stress
from src import portfolio as pf
from src import risk_budget as rb
from src import var_es as ve
from src.config import (BOND_TICKERS, DATA_START, EQUITY_TICKERS, HISTORICAL_SCENARIOS,
                        MANDATE, MARKET_PROXY, TICKERS)
from src.vol import regime_labels

LIMITS = MANDATE["allocation_limits"]
FUND_CAPS = tuple(LIMITS["fund_caps"].items())
BENCHMARKS = {
    "60/40 (SPY / AGG)": {"SPY": 0.6, "AGG": 0.4},
    "40/60 (SPY / AGG)": {"SPY": 0.4, "AGG": 0.6},
    "SPY (US equities)": {"SPY": 1.0},
}


@dataclass(frozen=True)
class Settings:
    source: str
    start: str
    end: str
    method: str
    cov_method: str
    max_weight: float
    max_equity: float
    min_bonds: float
    max_vol: float | None
    custom: tuple
    no_lookahead: bool
    confidence: float
    var_method: str
    horizon: int
    value: float
    bt_start: str
    bt_end: str
    window: int
    evt_rule: str
    manual_pct: float
    benchmark: str


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> Settings:
    sb = st.sidebar
    sb.header("Risk settings")
    conf = sb.radio("Confidence level", [0.90, 0.95, 0.99], index=2, horizontal=True,
                    format_func=lambda c: f"{c:.0%}")
    var_method = sb.selectbox("VaR methodology", bt.BACKTEST_METHODS,
                              index=bt.BACKTEST_METHODS.index("Filtered HS (EWMA)"),
                              help="Used for the headline numbers and the backtest.")
    horizon = sb.radio("Holding period (days)", [1, 10], horizontal=True,
                       help="10-day figures use square-root-of-time scaling.")

    sb.header("Backtest")
    # Default start: the first date with 500 days of history behind it, so 2008 is in the test
    rng = sb.date_input("Backtest period", value=(date(2007, 1, 3), date.today()),
                        min_value=date(2007, 1, 1), max_value=date.today())
    bt_start, bt_end = (rng if isinstance(rng, (tuple, list)) and len(rng) == 2
                        else (date(2007, 1, 3), date.today()))
    window = sb.select_slider("Estimation window (trading days)", [250, 500, 750, 1000], value=500)

    with sb.expander("Portfolio construction", expanded=False):
        method = st.selectbox("Construction method", pf.METHODS, index=pf.METHODS.index(pf.RECOMMENDED))
        cov_method = st.radio("Covariance estimate", ["sample", "ledoit-wolf"], horizontal=True,
                              format_func=lambda m: "Sample" if m == "sample" else "Ledoit-Wolf shrinkage")
        max_w = st.slider("Max weight per fund (other funds)", 0.10, 1.0, float(LIMITS["max_weight"]), 0.05)
        max_eq = st.slider("Max total equity", 0.0, 1.0, float(LIMITS["max_equity"]), 0.05)
        min_bd = st.slider("Min bonds (AGG)", 0.0, 1.0, float(LIMITS["min_bonds"]), 0.05)
        st.caption("Per-fund caps (these replace the slider for these funds): " + ", ".join(f"{t} ≤ {c:.0%}" for t, c in FUND_CAPS))
        use_vol = st.checkbox("Cap volatility", value=False,
                              help="Applies to variance-based methods.")
        max_vol = st.slider("Volatility cap (annual)", 0.04, 0.20,
                            float(MANDATE["risk_limits"]["max_volatility"]), 0.01) if use_vol else None
        custom = ()
        if method == "Custom weights":
            st.caption("Weights are normalised to 100%.")
            custom = tuple((t, st.number_input(t, 0.0, 100.0, round(100 / len(TICKERS), 1), 1.0))
                           for t in TICKERS)
        no_look = st.checkbox("Estimate weights only on data before the backtest start",
                              value=False, help="Removes look-ahead bias from the weights.")
        benchmark = st.selectbox("Benchmark", list(BENCHMARKS), index=0)

    with sb.expander("Extreme Value Theory", expanded=False):
        evt_rule = st.selectbox("Threshold rule", evt.THRESHOLD_RULES, index=0)
        manual = st.slider("Manual threshold percentile", 0.85, 0.99, 0.95, 0.005,
                           disabled=evt_rule != "Manual percentile")

    with sb.expander("Data", expanded=False):
        source = st.radio("Data source", ["snapshot", "live"], horizontal=True,
                          format_func=lambda s: "Frozen snapshot" if s == "snapshot" else "Live download",
                          help="The snapshot makes results exactly reproducible.")
        start = st.date_input("Sample start", pd.Timestamp(DATA_START).date(),
                              min_value=pd.Timestamp(DATA_START).date(), max_value=date.today())
        end = st.date_input("Sample end", date.today(), max_value=date.today())
        value = st.number_input("Portfolio value (USD)", 1_000_000, 100_000_000,
                                int(MANDATE["portfolio_value"]), 1_000_000)

    return Settings(source, str(start), str(end), method, cov_method, max_w, max_eq, min_bd,
                    max_vol, custom, no_look, float(conf), var_method, int(horizon), float(value),
                    str(bt_start), str(bt_end), int(window), evt_rule, float(manual), benchmark)


# ---------------------------------------------------------------------------
# Cached computations
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading market data...")
def load_data(source: str, start: str, end: str):
    prices, rf, notes = data.load_market_data(source, start, end)
    res = data.clean_prices(prices)
    lr = data.log_returns(res.prices)
    sr = data.simple_returns(res.prices)
    rf_daily = data.daily_risk_free(rf, sr.index)
    return {"prices": res.prices, "raw": prices, "flags": res.flags, "log": res.log,
            "notes": notes, "lr": lr, "sr": sr, "rf_annual": rf, "rf_daily": rf_daily}


@st.cache_data(show_spinner="Optimising portfolio...")
def optimise(method, sr, cov_method, cons_tuple, rf_annual, cvar_conf, custom):
    cons = pf.Constraints(*cons_tuple)
    mu, cov, shrink = pf.estimate_inputs(sr, cov_method)
    res = pf.build_portfolio(method, sr, mu, cov, cons, rf_annual, cvar_conf, dict(custom))
    frontier = pf.efficient_frontier(mu, cov, cons)
    return {"result": res, "mu": mu, "cov": cov, "shrink": shrink, "frontier": frontier,
            "issues": pf.check_constraints(res.weights, cons, cov)}


@st.cache_data(show_spinner="Computing VaR and ES...")
def risk_table(port_log: pd.Series, asset_log: pd.DataFrame, weights: tuple, evt_rule: str,
               manual_pct: float):
    w = np.array(weights)
    r = port_log.values
    tab = ve.compute_all(r, asset_log.values, w, confidences=(0.90, 0.95, 0.99))
    extra = []
    for c in (0.90, 0.95, 0.99):
        e = evt.fit_evt(r, c, evt_rule, manual_pct, with_diagnostics=False)
        g = evt.garch_evt(r, c, evt_rule, manual_pct)
        extra += [{"method": "EVT (GPD-POT)", "confidence": c, "VaR": e.var, "ES": e.es},
                  {"method": "GARCH-EVT (conditional)", "confidence": c, "VaR": g["var"], "ES": g["es"]}]
    return pd.concat([tab, pd.DataFrame(extra)], ignore_index=True)


@st.cache_data(show_spinner="Fitting the extreme-value tail...")
def evt_analysis(port_log: pd.Series, c: float, rule: str, manual_pct: float):
    r = port_log.values
    res = evt.fit_evt(r, c, rule, manual_pct)
    rules = []
    for rl in evt.THRESHOLD_RULES:
        e = evt.fit_evt(r, c, rl, manual_pct, with_diagnostics=False)
        e999 = evt.fit_evt(r, 0.999, rl, manual_pct, with_diagnostics=False)
        rules.append({"rule": rl, "threshold percentile": e.threshold_pct, "exceedances": e.fit.n_exc,
                      "xi": e.fit.xi, "VaR": e.var, "ES": e.es, "VaR 99.9%": e999.var})
    g = evt.garch_evt(r, c, rule, manual_pct)
    return {"res": res, "rules": pd.DataFrame(rules).set_index("rule"),
            "tail": evt.tail_curve(r, res), "qq": evt.qq_data(r, res), "garch": g}


@st.cache_data(show_spinner="Running the out-of-sample backtest...")
def run_backtest(port_log, asset_log, weights, method, c, window, start, end, evt_rule):
    f = bt.rolling_forecast(port_log, asset_log, np.array(weights), method, c, window, start,
                            end, 20, evt_rule)
    return f, bt.summarise(f, c)


@st.cache_data(show_spinner="Backtesting every method (this takes a little while)...")
def compare_backtests(port_log, asset_log, weights, c, window, start, end, evt_rule):
    return bt.compare_methods(port_log, asset_log, np.array(weights), c, window, start, end,
                              evt_rule)


@st.cache_data(show_spinner="Comparing the alternatives...")
def alternatives(sr: pd.DataFrame, rf_daily: pd.Series, cov_method: str):
    weights, metrics = compare.comparison(sr, rf_daily, cov_method=cov_method)
    rf_annual = float(rf_daily.mean() * 252)
    rec = weights.loc["Recommended"]
    return {"weights": weights, "metrics": metrics,
            "floor": compare.drawdown_floor(compare.grid_search(sr, rf_daily)),
            "walk_forward": [compare.walk_forward(sr, rf_daily, d) for d in ("2012-01-01", "2016-01-01")],
            "bootstrap": {n: compare.bootstrap_drawdowns(sr, weights.loc[n], rf_annual)
                          for n in ("Recommended", "First draft", "60/40 SPY/AGG")},
            "facts": compare.fact_check(sr), "turnover_to_markowitz":
                compare.distance(rec, weights.loc["Constrained Markowitz"])}


@st.cache_data
def regimes(market_log: pd.Series):
    return regime_labels(market_log)


# ---------------------------------------------------------------------------
# Context object handed to pages
# ---------------------------------------------------------------------------
@dataclass
class Context:
    s: Settings
    d: dict
    opt: dict
    weights: pd.Series
    port: pd.Series          # simple daily returns
    port_log: pd.Series      # log daily returns
    bench: pd.Series
    bench_weights: pd.Series
    cons: pf.Constraints

    @property
    def value(self) -> float:
        return self.s.value

    def loss_usd(self, log_loss: float) -> float:
        """Log-loss over the chosen horizon -> USD loss on the portfolio."""
        return float(ve.to_loss_pct(ve.scale_horizon(log_loss, self.s.horizon)) * self.value)

    def loss_pct(self, log_loss: float) -> float:
        return float(ve.to_loss_pct(ve.scale_horizon(log_loss, self.s.horizon)))


def get_context() -> Context:
    s: Settings = st.session_state["settings"]
    d = load_data(s.source, s.start, s.end)
    sr = d["sr"]
    est = sr.loc[: pd.Timestamp(s.bt_start) - pd.Timedelta(days=1)] if s.no_lookahead else sr
    if len(est) < 250:
        st.warning("Too little data before the backtest start to estimate weights; "
                   "using the full sample instead.")
        est = sr
    rf_ann = float(d["rf_daily"].loc[est.index].mean() * 252)
    cons_tuple = (0.0, s.max_weight, s.max_equity, s.min_bonds, s.max_vol, FUND_CAPS)
    opt = optimise(s.method, est, s.cov_method, cons_tuple, rf_ann, 0.95, s.custom)
    w = opt["result"].weights
    port = pf.portfolio_returns(sr, w)
    bench_w = pd.Series(BENCHMARKS[s.benchmark]).reindex(TICKERS).fillna(0.0)
    bench = pf.portfolio_returns(sr, bench_w).rename("Benchmark")
    return Context(s, d, opt, w, port, np.log1p(port).rename("Portfolio"), bench, bench_w,
                   pf.Constraints(*cons_tuple))


# ---------------------------------------------------------------------------
# Results used by more than one page
# ---------------------------------------------------------------------------
def headline(ctx: Context) -> dict:
    """Headline risk numbers at the sidebar confidence and at 99% (the mandate limit)."""
    tab = risk_table(ctx.port_log, ctx.d["lr"], tuple(ctx.weights.values), ctx.s.evt_rule,
                     ctx.s.manual_pct)

    def get(method, c):
        row = tab[(tab["method"] == method) & np.isclose(tab["confidence"], c)].iloc[0]
        return float(row["VaR"]), float(row["ES"])

    sel = get(ctx.s.var_method, ctx.s.confidence)
    ev = get("EVT (GPD-POT)", ctx.s.confidence)
    sel99 = get(ctx.s.var_method, 0.99)
    ev99 = get("EVT (GPD-POT)", 0.99)
    hs99 = get("Historical simulation", 0.99)
    # The ES held against the limit is the largest of the selected model, EVT and history
    return {"table": tab, "var": sel[0], "es": sel[1], "evt_var": ev[0], "evt_es": ev[1],
            "var99": sel99[0], "es99_conservative": max(sel99[1], ev99[1], hs99[1])}


def stress_results(ctx: Context) -> pd.DataFrame:
    return stress.historical_table(ctx.d["sr"], ctx.weights, HISTORICAL_SCENARIOS, ctx.value,
                                   ctx.bench_weights)


def group_weights(w: pd.Series) -> dict:
    return {"Equity": float(w[EQUITY_TICKERS].sum()), "Bonds": float(w[BOND_TICKERS].sum()),
            "Gold": float(w.get("GLD", 0.0))}


def kpi_frame(ctx: Context) -> pd.DataFrame:
    sr, rf = ctx.d["sr"], ctx.d["rf_daily"]
    mkt = sr[MARKET_PROXY]
    cols = {"Portfolio": kpis.kpi_table(ctx.port, rf, mkt, ctx.bench),
            "Benchmark": kpis.kpi_table(ctx.bench, rf, mkt, None),
            "SPY": kpis.kpi_table(mkt, rf, mkt, None)}
    return pd.DataFrame(cols)


def risk_budget(ctx: Context) -> pd.DataFrame:
    return rb.risk_budget_table(ctx.d["sr"], ctx.weights, ctx.s.confidence)


@st.cache_data(show_spinner=False)
def bootstrap(port_log: pd.Series, c: float):
    return ve.bootstrap_ci(port_log.values, c, n_boot=1000)


def hypothetical_worst(ctx: Context) -> tuple[str, dict]:
    from src.config import HYPOTHETICAL_SCENARIOS
    res = {k: stress.hypothetical(ctx.weights, v["shocks"], ctx.value)
           for k, v in HYPOTHETICAL_SCENARIOS.items()}
    name = min(res, key=lambda k: res[k]["total_return"])
    return name, res[name]
