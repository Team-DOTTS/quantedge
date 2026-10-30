"""Recommendation page: the decision first, then the evidence behind it."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import var_es as ve
from src.config import ASSET_COLORS, MANDATE, UNIVERSE
from ui import context as cx
from ui.theme import (BAD_TEXT, GARNET, GOOD_TEXT, MOSS, SAFFRON, WARN_TEXT, note, pct, show_chart,
                      style, usd, usdm, verdict_box)


def _status(ok: bool | None) -> str:
    return {True: "Pass", False: "Breach", None: "Caution"}[ok]


def render() -> None:
    ctx = cx.get_context()
    s, lim = ctx.s, MANDATE["risk_limits"]
    k = cx.kpi_frame(ctx)["Portfolio"]
    h = cx.headline(ctx)
    stress_tab = cx.stress_results(ctx)
    hyp_name, hyp = cx.hypothetical_worst(ctx)
    f, summ = cx.run_backtest(ctx.port_log, ctx.d["lr"], tuple(ctx.weights.values), s.var_method,
                              s.confidence, s.window, s.bt_start, s.bt_end, s.evt_rule)
    zone = summ["traffic"]["zone"]
    es99_pct = float(ve.to_loss_pct(h["es99_conservative"]))
    worst_hist = stress_tab["portfolio return"].min()
    worst_hist_name = stress_tab["portfolio return"].idxmin()
    # The stress limit is tested on the deepest fall inside each crisis window, which
    # can be much larger than the end-to-end return (2008: the portfolio recovered
    # most of its loss by March 2009).
    worst_dd = stress_tab["max drawdown"].min()
    worst_dd_name = stress_tab["max drawdown"].idxmin()
    rf_mean = float(ctx.d["rf_daily"].mean() * 252)
    return_target = rf_mean + MANDATE["return_excess_over_rf"]
    rf_today = float((1 + ctx.d["rf_daily"].iloc[-1]) ** 252 - 1)
    target_today = rf_today + MANDATE["return_excess_over_rf"]

    checks = [
        (f"Return (T-bills + {pct(MANDATE['return_excess_over_rf'], 1)})", pct(k["CAGR"]),
         f"≥ {pct(return_target, 1)}", k["CAGR"] >= return_target, "soft"),
        ("Return against today's target", pct(k["CAGR"]), f"≥ {pct(target_today, 1)}",
         k["CAGR"] >= target_today, "soft"),
        ("Volatility", pct(k["Annualised volatility"]), f"≤ {pct(lim['max_volatility'], 1)}",
         k["Annualised volatility"] <= lim["max_volatility"], "hard"),
        ("99% 1-day ES", pct(es99_pct), f"≤ {pct(lim['max_es99_1d'], 1)}",
         es99_pct <= lim["max_es99_1d"], "hard"),
        ("Max drawdown", pct(k["Maximum drawdown"]), f"≥ -{pct(lim['max_drawdown'], 0)}",
         abs(k["Maximum drawdown"]) <= lim["max_drawdown"], "hard"),
        ("Deepest crisis fall", pct(worst_dd),
         f"≥ -{pct(lim['max_stress_loss'], 0)}", abs(min(worst_dd, 0)) <= lim["max_stress_loss"], "hard"),
        ("Invented scenario", pct(hyp["total_return"]),
         f"≥ -{pct(lim['max_stress_loss'], 0)}", abs(min(hyp["total_return"], 0)) <= lim["max_stress_loss"],
         "hard"),
        (f"VaR backtest ({s.confidence:.0%})", zone, "Green",
         True if zone == "Green" else (None if zone == "Yellow" else False), "hard"),
        ("Allocation limits", "Met" if not ctx.opt["issues"] else "; ".join(ctx.opt["issues"]),
         "All met", not ctx.opt["issues"], "hard"),
    ]
    hard_fail = sum(1 for c in checks if c[4] == "hard" and c[3] is False)
    caution = sum(1 for c in checks if c[3] is None) + sum(1 for c in checks if c[4] == "soft" and not c[3])

    st.title("Recommendation to the Investment Committee")
    st.caption(f"{MANDATE['client']} · USD {ctx.value:,.0f} · Construction method: {ctx.opt['result'].method}")

    if hard_fail >= 2 or zone == "Red":
        verdict_box("Reject or redesign", "The portfolio breaches more than one risk limit or its risk "
                    "model fails validation. It should not be accepted in its current form.", GARNET)
    elif hard_fail == 1 or caution:
        verdict_box("Accept with adjusted risk limits", "The portfolio is broadly suitable, but at least "
                    "one limit or check needs attention. Adjust the allocation or the limit, and record why.",
                    SAFFRON)
    else:
        verdict_box("Accept as proposed", "The portfolio meets every risk limit, survives the stress "
                    "scenarios within tolerance, and its risk model passes out-of-sample validation.", MOSS)
    note("This verdict is computed from the limits in src/config.py, which already include our one "
         f"adjustment: the drawdown limit is {pct(lim['max_drawdown'], 0)} instead of the 15% first "
         "proposed. So \"accept as proposed\" here matches the report's recommendation to accept "
         "the mandate with a 20% tolerance. The report gives the reasoning.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Historical return (CAGR)", pct(k["CAGR"]))
    c2.metric("Volatility", pct(k["Annualised volatility"]))
    c3.metric("99% 1-day ES (held against limit)", usd(es99_pct * ctx.value),
              help=f"Largest of {s.var_method}, EVT and historical simulation")
    c4.metric("Maximum drawdown", pct(k["Maximum drawdown"]))
    c5.metric("Deepest crisis fall", pct(worst_dd), help=f"{worst_dd_name}; tested against the stress limit")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Proposed allocation")
        w = ctx.weights.sort_values()
        fig = go.Figure(go.Bar(x=w.values, y=[f"{t}  {UNIVERSE[t]['role']}" for t in w.index],
                               orientation="h", marker_color=[ASSET_COLORS[t] for t in w.index],
                               text=[pct(x, 1) for x in w.values], textposition="outside"))
        fig.update_traces(cliponaxis=False)  # never cut off the % labels at the chart edge
        fig.update_xaxes(range=[0, float(w.max()) * 1.25])
        show_chart(style(fig, 320, xfmt=".0%"))
        g = cx.group_weights(ctx.weights)
        note(f"Equity {pct(g['Equity'], 0)} · Bonds {pct(g['Bonds'], 0)} · Gold {pct(g['Gold'], 0)}")
    with right:
        st.subheader("Mandate checks")
        tbl = pd.DataFrame([{"Check": c[0], "Portfolio": c[1], "Limit": c[2], "Status": _status(c[3])}
                            for c in checks]).set_index("Check")
        status_colour = {"Pass": GOOD_TEXT, "Breach": BAD_TEXT, "Caution": WARN_TEXT}
        # st.table wraps long check names, so all four columns stay visible
        st.table(tbl.style.map(lambda v: f"color:{status_colour[v]};font-weight:600"
                               if v in status_colour else "", subset=["Status"]))
        note(f"ES is the largest of {s.var_method}, EVT and historical simulation ({usd(es99_pct * ctx.value)}). "
             f"Deepest crisis fall: {worst_dd_name}. Invented scenario: {hyp_name}. "
             f"VaR backtest uses {s.var_method}.")

    st.subheader("The committee's three questions")
    boot = cx.bootstrap(ctx.port_log, 0.99)
    q1, q2, q3 = st.columns(3)
    with q1:
        st.markdown("**What can go wrong?**")
        top = cx.risk_budget(ctx)["share of ES"].sort_values(ascending=False)
        st.markdown(
            f"Most tail risk comes from **{top.index[0]}** ({pct(top.iloc[0], 0)} of ES) and "
            f"**{top.index[1]}** ({pct(top.iloc[1], 0)}). The deepest fall inside a crisis was "
            f"{pct(worst_dd, 1)} ({worst_dd_name}); the worst end-to-end replay was the "
            f"{worst_hist_name} ({pct(worst_hist, 1)}); our invented "
            f"\"{hyp_name}\" scenario costs {pct(hyp['total_return'], 1)}.")
    with q2:
        st.markdown("**How bad could losses become?**")
        st.markdown(
            f"On a bad day (1 in 100) we expect to lose more than "
            f"**{usdm(float(ve.to_loss_pct(h['var99'])) * ctx.value)}**; when that happens the average loss is "
            f"**{usdm(es99_pct * ctx.value)}** (the larger of our model, EVT and history). Extreme Value Theory puts the tail average at "
            f"**{usdm(ctx.loss_usd(h['evt_es']))}**. Over the full sample the deepest fall was "
            f"{pct(k['Maximum drawdown'], 1)} ({usdm(-k['Maximum drawdown'] * ctx.value)}).")
    with q3:
        st.markdown("**How confident are we?**")
        st.markdown(
            f"The {s.var_method} model had {summ['kupiec']['breaches']} breaches against "
            f"{summ['kupiec']['expected']:.0f} expected out-of-sample (Kupiec p = "
            f"{summ['kupiec']['p_value']:.2f}); traffic light **{zone}**. A bootstrap gives a 90% "
            f"range for 99% 1-day historical ES of {pct(ve.to_loss_pct(boot['es_lo']))} to "
            f"{pct(ve.to_loss_pct(boot['es_hi']))}.")

    summary = pd.DataFrame({
        "metric": ["CAGR", "Volatility", f"VaR {s.confidence:.0%} {s.horizon}d (USD)",
                   f"ES {s.confidence:.0%} {s.horizon}d (USD)", "EVT ES (USD)", "Max drawdown",
                   "Backtest zone"] + [f"weight {t}" for t in ctx.weights.index],
        "value": [k["CAGR"], k["Annualised volatility"], ctx.loss_usd(h["var"]), ctx.loss_usd(h["es"]),
                  ctx.loss_usd(h["evt_es"]), k["Maximum drawdown"], zone] + list(ctx.weights.values)})
    st.download_button("Download tear-sheet numbers (CSV)", summary.to_csv(index=False),
                       "dotts_tear_sheet_numbers.csv", "text/csv")
