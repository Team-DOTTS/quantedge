"""Extreme Value Theory: threshold choice, fit quality and how bad the tail can get."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src import var_es as ve
from src.evt import THRESHOLD_RULE_NOTES
from ui import context as cx
from ui.theme import (GARNET, LINE, SAPPHIRE, SLATE, note, num, pct, show_chart, show_table, style,
                      usd, usdm)


def render() -> None:
    ctx = cx.get_context()
    s = ctx.s
    a = cx.evt_analysis(ctx.port_log, s.confidence, s.evt_rule, s.manual_pct)
    res, diag = a["res"], a["res"].diagnostics
    st.title("Extreme Value Theory")
    st.caption("Peaks-over-Threshold with a Generalised Pareto tail, fitted by maximum likelihood")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Threshold", pct(ve.to_loss_pct(res.threshold)), f"{res.threshold_pct:.1%} percentile of losses",
              delta_color="off")
    c2.metric("Tail observations", f"{res.fit.n_exc}", f"of {res.n} days", delta_color="off")
    c3.metric("Shape ξ", f"{res.fit.xi:.3f}", f"± {1.96 * res.fit.se_xi:.3f} (95%)", delta_color="off")
    c4.metric(f"EVT VaR / ES at {s.confidence:.0%}", f"{usd(ctx.loss_usd(res.var))} / {usd(ctx.loss_usd(res.es))}")
    st.markdown(f"**Threshold rule: {s.evt_rule}.** {THRESHOLD_RULE_NOTES[s.evt_rule]}")
    if res.fit.xi > 0:
        st.markdown(f"ξ > 0 means a **heavy, power-law tail**: losses decay more slowly than a bell curve "
                    f"predicts, so very large losses are more likely than Normal models suggest. "
                    f"With ξ = {res.fit.xi:.2f}, moments above order {1 / res.fit.xi:.1f} do not exist.")
    else:
        st.markdown("ξ ≤ 0 means a thin or bounded tail at this threshold; the tail is no heavier than a "
                    "bell curve's. Check the stability plot before relying on this.")
    if res.inside_body:
        st.warning("At this confidence level the requested quantile lies below the threshold, so the "
                   "empirical estimate is shown instead of the GPD tail. Pick a higher confidence or a "
                   "lower threshold.")

    # --- threshold diagnostics ----------------------------------------------
    st.subheader("Choosing the threshold")
    d1, d2 = st.columns(2)
    with d1:
        fig = go.Figure(go.Scatter(x=diag["threshold"].map(lambda u: float(ve.to_loss_pct(u))),
                                   y=diag["mean_excess"].map(lambda m: float(ve.to_loss_pct(m))),
                                   mode="lines+markers", line=dict(color=LINE), marker=dict(size=5),
                                   name="Mean excess"))
        fig.add_vline(x=float(ve.to_loss_pct(res.threshold)), line_color=GARNET, line_width=2,
                      annotation_text="chosen")
        fig.update_xaxes(title="Threshold (daily loss)")
        fig.update_yaxes(title="Average loss beyond threshold")
        show_chart(style(fig, 330, title="Mean excess plot", yfmt=".2%", xfmt=".2%"))
        note("For a GPD tail the mean excess is roughly a straight line in the threshold; an upward "
             "slope signals ξ > 0. Look for where the line becomes linear.")
    with d2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=diag["percentile"], y=diag["xi_hi"], line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=diag["percentile"], y=diag["xi_lo"], fill="tonexty",
                                 fillcolor="rgba(36,64,142,0.15)", line=dict(width=0),
                                 name="95% confidence band"))
        fig.add_trace(go.Scatter(x=diag["percentile"], y=diag["xi"], mode="lines+markers",
                                 line=dict(color=SAPPHIRE), marker=dict(size=5), name="ξ estimate"))
        fig.add_vline(x=res.threshold_pct, line_color=GARNET, line_width=2, annotation_text="chosen")
        fig.update_xaxes(title="Threshold percentile")
        fig.update_yaxes(title="Shape ξ")
        show_chart(style(fig, 330, title="Parameter stability plot", xfmt=".0%"))
        note("Too low a threshold mixes in ordinary days and biases ξ; too high leaves too few points "
             "and widens the band. A good threshold sits where ξ settles into a plateau.")

    # --- goodness of fit ------------------------------------------------------
    g1, g2 = st.columns(2)
    with g1:
        qq = a["qq"]
        lim = float(max(qq["fitted"].max(), qq["observed"].max()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", line=dict(color=SLATE, dash="dot"),
                                 name="Perfect fit"))
        fig.add_trace(go.Scatter(x=qq["fitted"], y=qq["observed"], mode="markers",
                                 marker=dict(color=GARNET, size=6), name="Tail losses"))
        fig.update_xaxes(title="GPD quantile (excess over threshold)")
        fig.update_yaxes(title="Observed excess")
        show_chart(style(fig, 340, title="Quantile-quantile plot", yfmt=".1%", xfmt=".1%"))
        note("Points close to the dotted line mean the fitted GPD describes the observed tail well.")
    with g2:
        t = a["tail"]
        fig = go.Figure()
        for col, color, dash in [("Empirical", SLATE, "dot"), ("Normal", SAPPHIRE, "dash"),
                                 ("EVT (GPD)", GARNET, "solid")]:
            fig.add_trace(go.Scatter(x=t["once every (trading days)"],
                                     y=[ctx.loss_usd(v) for v in t[col]], name=col,
                                     mode="lines+markers", line=dict(color=color, dash=dash, width=2)))
        fig.update_xaxes(type="log", title="Happens once every … trading days (log scale)")
        fig.update_yaxes(title=f"{s.horizon}-day loss (USD)")
        show_chart(style(fig, 340, title="How bad can a bad day get?", yfmt="$,.0f"))
        rare = t.iloc[-1]
        note(f"A 1-in-{rare['once every (trading days)']:.0f}-day loss (about once a decade) is "
             f"{usdm(ctx.loss_usd(rare['EVT (GPD)']))} under EVT but only "
             f"{usdm(ctx.loss_usd(rare['Normal']))} under a Normal model.")

    tt = t.copy()
    tt["years (approx.)"] = tt["once every (trading days)"] / 252
    for col in ("Empirical", "Normal", "EVT (GPD)"):
        tt[col] = [ctx.loss_usd(v) for v in tt[col]]
    show_table(tt.set_index("confidence").style.format(
        {"once every (trading days)": "{:,.0f}", "years (approx.)": "{:.1f}", "Empirical": "${:,.0f}",
         "Normal": "${:,.0f}", "EVT (GPD)": "${:,.0f}"}))

    # --- sensitivity to the threshold rule ---------------------------------------
    st.subheader("Sensitivity to the threshold rule")
    rules = a["rules"].copy()
    for col in ("VaR", "ES", "VaR 99.9%"):
        rules[col] = [ctx.loss_usd(v) for v in rules[col]]
    show_table(rules.style.format({"threshold percentile": "{:.1%}", "xi": "{:.3f}",
                                   "VaR": "${:,.0f}", "ES": "${:,.0f}", "VaR 99.9%": "${:,.0f}"}))
    spread = rules["VaR 99.9%"].max() / rules["VaR 99.9%"].min() - 1
    note(f"Across five defensible threshold rules, the 99.9% VaR varies by {pct(spread, 0)}. "
         "Estimates near the data's edge are stable across rules; far-tail extrapolations are more "
         "sensitive to the choice and should be quoted as ranges.")

    # --- conditional EVT -------------------------------------------------------
    st.subheader("Conditional EVT (GARCH filter)")
    g = a["garch"]["garch"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("GARCH α (reaction)", num(g["alpha"], 3))
    k2.metric("GARCH β (persistence)", num(g["beta"], 3))
    k3.metric("Long-run volatility", pct(g["long_run_vol_annual"], 1))
    k4.metric(f"GARCH-EVT VaR / ES at {s.confidence:.0%}",
              f"{usd(ctx.loss_usd(a['garch']['var']))} / {usd(ctx.loss_usd(a['garch']['es']))}")
    tomorrow = g["sigma_next"] * np.sqrt(252)
    note(f"Tomorrow's forecast volatility is {pct(tomorrow, 1)} a year against a long-run "
         f"{pct(g['long_run_vol_annual'], 1)}, so conditional tail risk is "
         f"{'above' if tomorrow > g['long_run_vol_annual'] else 'below'} its unconditional level. "
         "Unconditional EVT answers 'how bad can it get over the cycle'; GARCH-EVT answers "
         "'how bad can it get tomorrow'.")
