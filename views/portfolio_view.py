"""Portfolio panel: weights, efficient frontier, cumulative return, drawdown, KPIs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import kpis
from src.config import ASSET_COLORS, MARKET_PROXY, UNIVERSE
from src.kpis import KPI_EXPLAIN, KPI_FORMATS
from src.portfolio import capm
from ui import context as cx
from ui.theme import (GARNET, SAPPHIRE, SLATE, note, num, pct, show_chart, show_table, style)


def render() -> None:
    ctx = cx.get_context()
    s, sr = ctx.s, ctx.d["sr"]
    res = ctx.opt["result"]
    st.title("Portfolio")
    st.caption(f"{res.method} · covariance: {'Ledoit-Wolf shrinkage' if s.cov_method != 'sample' else 'sample'}"
               f" · sample {sr.index[0].date()} to {sr.index[-1].date()}")
    if not res.success:
        st.warning(f"The optimiser did not fully converge: {res.message}")
    for issue in ctx.opt["issues"]:
        st.warning(issue)

    # --- weights and frontier ------------------------------------------------
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.subheader("Weights")
        w = ctx.weights
        fig = go.Figure(go.Pie(labels=list(w.index), values=w.values, hole=0.55, sort=False,
                               marker=dict(colors=[ASSET_COLORS[t] for t in w.index]),
                               textinfo="label+percent"))
        fig.update_traces(hovertemplate="%{label}: %{percent}<extra></extra>")
        show_chart(style(fig, 300).update_layout(showlegend=False))
        tbl = pd.DataFrame({"Fund": [UNIVERSE[t]["name"] for t in w.index], "Weight": w.values,
                            "USD": w.values * ctx.value}, index=w.index)
        show_table(tbl.style.format({"Weight": "{:.1%}", "USD": "${:,.0f}"}))
    with c2:
        st.subheader("Efficient frontier")
        mu, cov = ctx.opt["mu"], ctx.opt["cov"]
        fr = ctx.opt["frontier"]
        fig = go.Figure()
        if not fr.empty:
            fig.add_trace(go.Scatter(x=fr["volatility"], y=fr["return"], mode="lines",
                                     name="Frontier (within limits)", line=dict(color=SLATE, width=2)))
        vols = np.sqrt(np.diag(cov.values))
        fig.add_trace(go.Scatter(x=vols, y=mu.values, mode="markers+text", text=list(mu.index),
                                 textposition="top center", name="Single funds",
                                 marker=dict(size=9, color=[ASSET_COLORS[t] for t in mu.index])))
        for label, wt, sym, col in [("Proposed", ctx.weights, "star", GARNET),
                                    ("Benchmark", ctx.bench_weights, "diamond", SAPPHIRE)]:
            v = float(np.sqrt(wt.values @ cov.values @ wt.values))
            fig.add_trace(go.Scatter(x=[v], y=[float(wt.values @ mu.values)], mode="markers",
                                     name=label, marker=dict(size=16, symbol=sym, color=col)))
        if s.max_vol:
            fig.add_vline(x=s.max_vol, line_dash="dot", line_color=GARNET,
                          annotation_text="vol cap", annotation_position="top")
        fig.update_xaxes(title="Annualised volatility")
        fig.update_yaxes(title="Expected annual return (historical mean)")
        show_chart(style(fig, 380, yfmt=".0%", xfmt=".0%"))
        avg_vol = float(ctx.weights.values @ vols)
        port_vol = float(np.sqrt(ctx.weights.values @ cov.values @ ctx.weights.values))
        note(f"Diversification ratio {avg_vol / port_vol:.2f}: the funds' weighted-average volatility "
             f"({pct(avg_vol, 1)}) is {avg_vol / port_vol:.2f}× the portfolio's ({pct(port_vol, 1)}), "
             "because the funds do not move together. Expected returns are historical averages and "
             "are the least reliable input.")
        if ctx.opt["shrink"]:
            note(f"Ledoit-Wolf shrinkage intensity: {ctx.opt['shrink']:.2f}.")

    # --- cumulative return and drawdown -------------------------------------
    st.subheader(f"Growth of USD {ctx.value:,.0f}")
    series = {"Portfolio": ctx.port, "Benchmark": ctx.bench, "SPY": sr[MARKET_PROXY]}
    colors = {"Portfolio": GARNET, "Benchmark": SAPPHIRE, "SPY": SLATE}
    fig = go.Figure()
    for name, r in series.items():
        fig.add_trace(go.Scatter(x=r.index, y=kpis.wealth_index(r, ctx.value), name=name,
                                 line=dict(color=colors[name], width=2.4 if name == "Portfolio" else 1.4)))
    fig.update_yaxes(type="log", title="Value (USD, log scale)")
    show_chart(style(fig, 380))

    st.subheader("Drawdown")
    fig = go.Figure()
    for name in ("Portfolio", "Benchmark"):
        dd = kpis.drawdown(series[name])
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name=name, fill="tozeroy" if name == "Portfolio" else None,
                                 line=dict(color=colors[name], width=1.6)))
    show_chart(style(fig, 300, yfmt=".0%"))
    md = kpis.max_drawdown_details(ctx.port)
    rec = f"recovered on {md['recovery'].date()} ({md['days_to_recover']} trading days later)" \
        if md["recovery"] is not None else "not yet recovered"
    note(f"Worst drawdown {pct(md['max_drawdown'], 1)}: peak {md['peak'].date()}, "
         f"trough {md['trough'].date()}, {rec}.")

    # --- KPIs ------------------------------------------------------------------
    st.subheader("Performance KPIs")
    kf = cx.kpi_frame(ctx)
    disp = kf.copy().astype(object)
    for idx in kf.index:
        fmt = pct if KPI_FORMATS.get(idx) == "pct" else num
        disp.loc[idx] = [fmt(v) if pd.notna(v) else "n/a" for v in kf.loc[idx]]
    disp["What it means"] = [KPI_EXPLAIN.get(i, "") for i in kf.index]
    show_table(disp, height=600)
    note(f"Risk-free rate: 3-month T-bill (FRED DTB3). Benchmark: {s.benchmark}. "
         "Information ratio and tracking error are measured against the benchmark.")

    c = capm(ctx.port, sr[MARKET_PROXY], ctx.d["rf_daily"])
    st.subheader("CAPM")
    a, b, r2, t = st.columns(4)
    a.metric("Beta vs SPY", num(c["beta"]))
    b.metric("Jensen's alpha (annual)", pct(c["alpha_annual"]))
    r2.metric("R²", num(c["r_squared"]))
    t.metric("Alpha t-statistic", num(c["alpha_t"]))
    note("A beta well below 1 means the portfolio captures only part of US stock-market moves. "
         "An alpha t-statistic below about 2 means the alpha could be luck.")

    # --- correlations ------------------------------------------------------------
    st.subheader("Correlations: normal times versus stressed markets")
    reg = cx.regimes(ctx.d["lr"][MARKET_PROXY])
    lr = ctx.d["lr"]
    stressed = lr.loc[reg.index[reg["regime"] == "Stressed"]]
    cc1, cc2 = st.columns(2)
    for col, data_, title in [(cc1, lr, "All days"), (cc2, stressed, "Stressed days (high market volatility)")]:
        corr = data_.corr()
        fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                                   colorscale=[[0, SAPPHIRE], [0.5, "#FFFFFF"], [1, GARNET]],
                                   text=np.round(corr.values, 2), texttemplate="%{text}"))
        with col:
            show_chart(style(fig, 360, title=title))
    note("Equity correlations rise in stressed markets, which is when diversification is needed most. "
         "Bonds (AGG) and gold (GLD) are the funds that keep diversifying.")
