"""Stress testing: historical crisis replays and the scenarios we invented."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import stress
from src import var_es as ve
from src.config import HISTORICAL_SCENARIOS, HYPOTHETICAL_SCENARIOS, MANDATE, TICKERS
from ui import context as cx
from ui.theme import (GARNET, MOSS, SAPPHIRE, SLATE, note, pct, show_chart, show_table, style, usd,
                      usdm)


def _contribution_chart(contrib: pd.Series, title: str) -> go.Figure:
    c = contrib.sort_values()
    fig = go.Figure(go.Bar(x=c.values, y=c.index, orientation="h",
                           marker_color=[GARNET if v < 0 else MOSS for v in c.values],
                           text=[pct(v, 1) for v in c.values], textposition="outside"))
    return style(fig, 300, title=title, xfmt=".0%")


def render() -> None:
    ctx = cx.get_context()
    s, sr, val = ctx.s, ctx.d["sr"], ctx.value
    limit = MANDATE["risk_limits"]["max_stress_loss"]
    st.title("Stress testing")
    st.caption("Today's weights replayed through real crises (buy-and-hold from the start of each "
               "window), plus scenarios we designed ourselves")

    # --- historical -----------------------------------------------------------
    tab = cx.stress_results(ctx)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=tab.index, x=tab["portfolio return"], name="Proposed portfolio",
                         orientation="h", marker_color=GARNET))
    fig.add_trace(go.Bar(y=tab.index, x=tab["benchmark return"], name=s.benchmark, orientation="h",
                         marker_color="#B8C2D9"))
    fig.add_vline(x=-limit, line_dash="dot", line_color=GARNET, annotation_text="stress limit")
    fig.update_layout(barmode="group")
    show_chart(style(fig, 380, title="Total return through each crisis", xfmt=".0%"))
    show_table(tab.style.format({"portfolio return": "{:.1%}", "max drawdown": "{:.1%}", "worst day": "{:.2%}",
                                 "loss (USD)": "${:,.0f}", "benchmark return": "{:.1%}"}))

    st.subheader("Inside one crisis")
    pick = st.selectbox("Scenario", list(HISTORICAL_SCENARIOS), index=0)
    sc = HISTORICAL_SCENARIOS[pick]
    res = stress.replay(sr, ctx.weights, sc["start"], sc["end"], val)
    resb = stress.replay(sr, ctx.bench_weights, sc["start"], sc["end"], val)
    st.markdown(f"*{sc['story']}* ({sc['start']} to {sc['end']})")
    if res is None:
        st.info("This window is outside the data sample.")
    else:
        c1, c2 = st.columns([1.5, 1])
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res["path"].index, y=res["path"].values, name="Proposed portfolio",
                                     line=dict(color=GARNET, width=2.4)))
            fig.add_trace(go.Scatter(x=resb["path"].index, y=resb["path"].values, name=s.benchmark,
                                     line=dict(color=SAPPHIRE, width=1.4)))
            fig.add_hline(y=val, line_dash="dot", line_color=SLATE)
            show_chart(style(fig, 320, title="Portfolio value (USD)", yfmt="$,.0f"))
        with c2:
            show_chart(_contribution_chart(res["contribution"], "Contribution by fund"))
        m1, m2, m3 = st.columns(3)
        m1.metric("Total return", pct(res["total_return"], 1), usd(res["total_return"] * val), delta_color="off")
        m2.metric("Deepest drawdown", pct(res["max_drawdown"], 1))
        m3.metric("Worst single day", pct(res["worst_day"], 2))

    # --- stressed VaR ---------------------------------------------------------
    st.subheader("Stressed VaR")
    gfc = HISTORICAL_SCENARIOS["2008 Global Financial Crisis"]
    sv = stress.stressed_var(ctx.port_log, gfc["start"], gfc["end"], s.confidence)
    nv = ve.historical(ctx.port_log.values, s.confidence)
    if sv:
        a, b, c = st.columns(3)
        a.metric(f"Full-sample ES ({s.confidence:.0%}, {s.horizon}-day)", usd(ctx.loss_usd(nv[1])))
        b.metric("Stressed ES (2008 crisis data only)", usd(ctx.loss_usd(sv["es"])))
        c.metric("Stress multiple", f"{sv['es'] / nv[1]:.1f}×")
        note("Basel-style stressed risk: the same historical method, estimated only on the 2008 crisis. "
             "It shows how much larger daily risk becomes if crisis conditions return.")

    # --- hypothetical -----------------------------------------------------------
    st.subheader("Hypothetical scenario")
    name = st.selectbox("Scenario template", list(HYPOTHETICAL_SCENARIOS) + ["Custom"], index=0)
    base = HYPOTHETICAL_SCENARIOS.get(name, {"story": "Set your own shocks below.",
                                            "shocks": {t: 0.0 for t in TICKERS}})
    st.markdown(f"*{base['story']}*")
    editor = pd.DataFrame({"Shock (%)": [base["shocks"][t] * 100 for t in TICKERS]}, index=TICKERS)
    edited = st.data_editor(editor, key=f"shock_{name}",
                            column_config={"Shock (%)": st.column_config.NumberColumn(
                                format="%.1f", min_value=-100.0, max_value=200.0, step=1.0)})
    shocks = (edited["Shock (%)"] / 100).to_dict()
    hy = stress.hypothetical(ctx.weights, shocks, val)
    h1, h2 = st.columns([1, 1.4])
    with h1:
        st.metric("Instant portfolio impact", pct(hy["total_return"], 1), usd(-hy["loss_usd"]),
                  delta_color="off")
        status = "within" if abs(min(hy["total_return"], 0)) <= limit else "outside"
        st.markdown(f"The loss is **{status}** the {pct(limit, 0)} stress limit "
                    f"({usdm(limit * val)}).")
        note("Shocks are instantaneous, one-off moves in each fund. Edit the table to test other views; "
             "we designed the scenarios and justify them in our report.")
    with h2:
        show_chart(_contribution_chart(hy["contribution"], "Contribution by fund"))
