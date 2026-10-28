"""Alternatives page: the recommended mix against the portfolios the memo rejects,
and the robustness evidence behind the choice."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import ASSET_COLORS, MANDATE, TICKERS
from ui import context as cx
from ui.theme import BAD_TEXT, GOOD_TEXT, note, pct, show_chart, show_table, style

PCT_ROWS = ["CAGR", "Excess over T-bills", "Volatility", "Max drawdown", "CVaR 95% 1d", "CVaR 99% 1d",
            "2008 replay", "2008 max DD", "2020 replay", "2022 replay", "Satellite share"]
CHECK_ROWS = ["Meets return", "Meets vol", "Meets ES", "Meets drawdown"]


def _fmt_metrics(m: pd.DataFrame) -> pd.DataFrame:
    out = m.T.astype(object)
    for r in PCT_ROWS:
        out.loc[r] = [pct(v, 1) for v in m[r]]
    for r in ("Sharpe", "Sortino"):
        out.loc[r] = [f"{v:.2f}" for v in m[r]]
    for r in CHECK_ROWS:
        out.loc[r] = ["Pass" if v else "Breach" for v in m[r]]
    return out


def _colour(v):
    return f"color:{GOOD_TEXT if v == 'Pass' else BAD_TEXT};font-weight:600" if v in ("Pass", "Breach") else ""


def render() -> None:
    ctx = cx.get_context()
    sr, lim = ctx.d["sr"], MANDATE["risk_limits"]
    a = cx.alternatives(sr, ctx.d["rf_daily"], ctx.s.cov_method)
    weights, metrics = a["weights"], a["metrics"]

    st.title("Recommended mix versus the alternatives")
    st.caption(f"Full sample {sr.index[0].date()} to {sr.index[-1].date()} · limits: T-bills + "
               f"{pct(MANDATE['return_excess_over_rf'], 1)}, volatility ≤ {pct(lim['max_volatility'], 0)}, "
               f"99% 1-day ES ≤ {pct(lim['max_es99_1d'], 0)}, max drawdown ≤ {pct(lim['max_drawdown'], 0)}")

    st.subheader("Weights")
    fig = go.Figure()
    for t in TICKERS:
        fig.add_trace(go.Bar(y=weights.index, x=weights[t], name=t, orientation="h",
                             marker_color=ASSET_COLORS[t]))
    fig.update_layout(barmode="stack", yaxis=dict(autorange="reversed"))
    show_chart(style(fig, 360, xfmt=".0%"))
    note("Constrained Markowitz maximises the Sharpe ratio under the mandate's allocation limits ("
         + ", ".join(f"{t} ≤ {c:.0%}" for t, c in MANDATE["allocation_limits"]["fund_caps"].items())
         + f", AGG ≥ {pct(MANDATE['allocation_limits']['min_bonds'], 0)}, no shorting). Moving from the "
         f"recommended mix to its optimum takes {pct(a['turnover_to_markowitz'], 1)} one-way turnover. "
         "Optimised weights use full-sample historical means, so they carry look-ahead bias.")

    st.subheader("Risk and return")
    show_table(_fmt_metrics(metrics).style.map(_colour), height=640)
    note("CVaR is the historical 1-day Expected Shortfall. Crisis replays hold the starting weights "
         "through the stress windows in src/config.py (market peak to trough, not calendar years).")

    st.subheader("Why not just optimise on the past?")
    rows = []
    for wf in a["walk_forward"]:
        p = wf["pick"]
        rows.append({"chosen on data before": wf["split"][:4],
                     "pick": " / ".join(f"{t} {p[t]:.0%}" for t in TICKERS if p[t] > 0),
                     "Sortino before": f"{wf['in']['sortino']:.2f}", "Sortino after": f"{wf['out']['sortino']:.2f}",
                     "max DD after": pct(wf["out"]["mdd"], 1),
                     "recommended: Sortino after": f"{wf['recommended_out']['sortino']:.2f}"})
    show_table(pd.DataFrame(rows).set_index("chosen on data before"))
    note("Each row picks the best-Sortino round-number mix that met the risk limits using only data "
         "before the split, then scores it afterwards. Chasing the best past mix is unstable, which is why "
         "the recommendation is a judgement inside the limits rather than an optimiser output.")

    st.subheader("How likely is a large fall over 5 years?")
    b = pd.DataFrame(a["bootstrap"]).T
    show_table(pd.DataFrame({
        "chance of a fall > 15%": b["p_dd_15"].map(lambda v: pct(v, 1)),
        "chance of a fall > 20%": b["p_dd_20"].map(lambda v: pct(v, 1)),
        "median worst fall": b["median_dd"].map(lambda v: pct(v, 1)),
        "chance of a 5-year loss": b["p_loss"].map(lambda v: pct(v, 1)),
        "chance of missing the target": b["p_below_target"].map(lambda v: pct(v, 1)),
    }))
    f = a["floor"]
    note(f"3,000 simulated 5-year paths built from 1-month blocks of real history. Across {f['n']} "
         f"round-number mixes the shallowest full-sample drawdown was {pct(f['best_mdd'], 1)}, and only "
         f"{pct(f['share_within_15'], 1)} of mixes stayed within 15%: that is why the drawdown limit is "
         f"{pct(lim['max_drawdown'], 0)}.")

    st.subheader("Facts quoted in the memo")
    facts = a["facts"]
    show_table(pd.DataFrame({"value": [f"{v:.2f}" if "correlation" in k else pct(v, 1)
                                       for k, v in facts.items()]}, index=facts.index))
