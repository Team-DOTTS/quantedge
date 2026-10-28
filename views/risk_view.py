"""Risk dashboard: VaR, ES and the EVT tail estimate side by side."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import var_es as ve
from src.config import MARKET_PROXY
from ui import context as cx
from ui.theme import (GARNET, GARNET_LIGHT, INK, SAFFRON, SAPPHIRE, note, pct, show_chart,
                      show_table, style, usd)


def render() -> None:
    ctx = cx.get_context()
    s = ctx.s
    h = cx.headline(ctx)
    tab = h["table"]
    st.title("Risk dashboard")
    st.caption(f"{s.confidence:.0%} confidence · {s.horizon}-day holding period · "
               f"headline model: {s.var_method} · USD {ctx.value:,.0f}")

    # --- the three numbers --------------------------------------------------
    trio = [("Value-at-Risk", h["var"], SAPPHIRE,
             f"Loss exceeded on only {1 - s.confidence:.0%} of days ({s.var_method})."),
            ("Expected Shortfall", h["es"], SAFFRON,
             "Average loss on those worst days."),
            ("EVT tail ES", h["evt_es"], GARNET,
             f"Average tail loss from the fitted extreme-value tail (EVT VaR {usd(ctx.loss_usd(h['evt_var']))}).")]
    cols = st.columns(3)
    for col, (label, val, _, helptext) in zip(cols, trio):
        col.metric(label, usd(ctx.loss_usd(val)), f"{pct(ctx.loss_pct(val))} of capital",
                   delta_color="off", help=helptext)

    # --- tail ladder: the loss distribution with the three markers ----------
    loss_usd = ve.to_loss_pct(-ctx.port_log.values) * ctx.value  # daily, 1-day
    scale = np.sqrt(s.horizon)
    fig = go.Figure()
    cut = np.quantile(loss_usd, 0.80)
    body, tail = loss_usd[loss_usd <= cut], loss_usd[loss_usd > cut]
    bins = dict(start=float(loss_usd.min()), end=float(loss_usd.max()), size=float(np.ptp(loss_usd) / 140))
    fig.add_trace(go.Histogram(x=body, xbins=bins, marker_color="#C9CFDB", name="Daily outcomes",
                               hovertemplate="%{x:$,.0f}<extra></extra>"))
    fig.add_trace(go.Histogram(x=tail, xbins=bins, marker_color=GARNET_LIGHT, name="Worst 20% of days",
                               hovertemplate="%{x:$,.0f}<extra></extra>"))
    ymax = 1.0
    for label, val, color, _ in trio:
        x = ctx.loss_usd(val) / scale if s.horizon > 1 else ctx.loss_usd(val)
        fig.add_vline(x=x, line_color=color, line_width=3)
        fig.add_annotation(x=x, y=ymax, yref="paper", text=f"{label}<br>{usd(x)}", showarrow=False,
                           xanchor="left", yanchor="top", font=dict(color=color, size=12), xshift=4,
                           bgcolor="rgba(255,255,255,0.85)")
        ymax -= 0.17
    fig.update_layout(barmode="overlay", bargap=0.02)
    fig.update_xaxes(title="Daily loss in USD (gains to the left of zero)", tickformat="$,.0f")
    fig.update_yaxes(title="Number of days", type="log")
    show_chart(style(fig, 420, title="Where the three measures sit in the distribution of daily outcomes"))
    note("Each bar counts historical days with that profit or loss on today's portfolio (1-day, "
         "log count scale so rare tail days stay visible). The gap between the blue VaR line and the "
         "red EVT line is the extra loss a VaR-only report would hide."
         + (" Markers show 1-day equivalents of the 10-day figures." if s.horizon > 1 else ""))

    # --- every method side by side -----------------------------------------
    st.subheader("All methods at the selected confidence level")
    sel = tab[np.isclose(tab["confidence"], s.confidence)].copy()
    sel["VaR USD"] = [ctx.loss_usd(v) for v in sel["VaR"]]
    sel["ES USD"] = [ctx.loss_usd(v) for v in sel["ES"]]
    sel = sel.sort_values("ES USD")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=sel["method"], x=sel["VaR USD"], name="VaR", orientation="h",
                         marker_color=SAPPHIRE))
    fig.add_trace(go.Bar(y=sel["method"], x=sel["ES USD"], name="Expected Shortfall", orientation="h",
                         marker_color=GARNET))
    fig.update_layout(barmode="group")
    show_chart(style(fig, 440, xfmt="$,.0f"))

    st.subheader("VaR and ES at 95% and 99%")
    full = tab[tab["confidence"].isin([0.95, 0.99])].copy()
    full["VaR %"] = [ctx.loss_pct(v) for v in full["VaR"]]
    full["ES %"] = [ctx.loss_pct(v) for v in full["ES"]]
    full["VaR USD"] = full["VaR %"] * ctx.value
    full["ES USD"] = full["ES %"] * ctx.value
    full["ES / VaR"] = full["ES"] / full["VaR"]
    pv = full.pivot(index="method", columns="confidence",
                    values=["VaR %", "ES %", "VaR USD", "ES USD", "ES / VaR"])
    pv.columns = [f"{a} @{b:.0%}" for a, b in pv.columns]
    order = ve.METHODS + ["EVT (GPD-POT)", "GARCH-EVT (conditional)"]
    pv = pv.reindex(order)
    fmt = {c: ("{:.2%}" if "%" in c else "${:,.0f}" if "USD" in c else "{:.2f}") for c in pv.columns}
    show_table(pv.style.format(fmt))

    # --- why they disagree --------------------------------------------------
    st.subheader("Why the methods disagree")
    dd = ve.disagreement_drivers(ctx.port_log.values)
    v99 = tab[np.isclose(tab["confidence"], 0.99)].set_index("method")["VaR"]
    st.markdown(
        f"- **Fat tails.** Portfolio returns have excess kurtosis of **{dd['excess_kurtosis']:.1f}** "
        f"(0 for a bell curve) and the Jarque-Bera test rejects normality (p = {dd['jb_pvalue']:.3g}). "
        f"A fitted Student-t needs only **{dd['t_df']:.1f}** degrees of freedom. So the Normal models "
        f"understate extreme losses: at 99%, Normal VaR is {pct(ve.to_loss_pct(v99['Parametric (Normal)']))} "
        f"against {pct(ve.to_loss_pct(v99['Historical simulation']))} historically and "
        f"{pct(ve.to_loss_pct(v99['EVT (GPD-POT)']))} from EVT.\n"
        f"- **Asymmetry.** Skewness is **{dd['skewness']:.2f}**; Cornish-Fisher adjusts for it, the "
        f"Normal and Monte Carlo (Normal) models cannot.\n"
        f"- **Today versus the average.** Current EWMA volatility is **{dd['current_vs_avg_vol']:.2f}×** "
        f"the long-run average, so Filtered HS and GARCH-EVT, which scale to today's conditions, sit "
        f"{'above' if dd['current_vs_avg_vol'] > 1 else 'below'} the unconditional models.\n"
        f"- **Sampling noise.** Historical simulation at 99% rests on about "
        f"{int(len(ctx.port_log) * 0.01)} worst days; Monte Carlo adds simulation error (below).")
    with st.expander("Method notes"):
        for m, txt in ve.METHOD_NOTES.items():
            st.markdown(f"**{m}.** {txt}")
        st.markdown("**EVT (GPD-POT).** Fits a Generalised Pareto distribution to losses beyond a "
                    "justified threshold, so the tail is modelled directly rather than assumed.")
        st.markdown("**GARCH-EVT.** GARCH(1,1) removes volatility clustering, EVT models the "
                    "standardised tail, and the result is scaled to tomorrow's forecast volatility.")

    # --- confidence in the numbers -------------------------------------------
    st.subheader("How confident are we in these numbers?")
    boot = cx.bootstrap(ctx.port_log, s.confidence)
    mce = ve.monte_carlo_error(ctx.d["lr"].values, ctx.weights.values, s.confidence)
    b1, b2, b3 = st.columns(3)
    b1.metric("Historical VaR, 90% bootstrap range",
              f"{usd(ctx.loss_usd(boot['var_lo']))} – {usd(ctx.loss_usd(boot['var_hi']))}")
    b2.metric("Historical ES, 90% bootstrap range",
              f"{usd(ctx.loss_usd(boot['es_lo']))} – {usd(ctx.loss_usd(boot['es_hi']))}")
    b3.metric("Monte Carlo VaR standard error", usd(mce["var_se"] * ctx.value * np.sqrt(s.horizon)))
    note("The bootstrap resamples history 500 times and recomputes the estimate; a wide range means "
         "the number should be quoted with caution. ES ranges are wider than VaR ranges because ES "
         "depends on the few most extreme days.")

    # --- regime-conditioned risk -----------------------------------------------
    st.subheader("Regime-conditioned risk")
    reg = cx.regimes(ctx.d["lr"][MARKET_PROXY])
    joined = ctx.port_log.to_frame("r").join(reg, how="inner")
    rows = []
    for name, grp in joined.groupby("regime"):
        v, e = ve.historical(grp["r"].values, s.confidence)
        rows.append({"regime": name, "share of days": len(grp) / len(joined),
                     "VaR USD": ctx.loss_usd(v), "ES USD": ctx.loss_usd(e)})
    rt = pd.DataFrame(rows).set_index("regime")
    current = reg["regime"].iloc[-1]
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Current regime", current, f"SPY EWMA vol {pct(reg['market_vol'].iloc[-1], 1)}",
                  delta_color="off")
        show_table(rt.style.format({"share of days": "{:.0%}", "VaR USD": "${:,.0f}", "ES USD": "${:,.0f}"}))
    with r2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=reg.index, y=reg["market_vol"], name="SPY EWMA volatility",
                                 line=dict(color=INK, width=1.2)))
        fig.add_hline(y=float(reg["threshold"].iloc[0]), line_dash="dot", line_color=GARNET,
                      annotation_text="stressed above this line")
        show_chart(style(fig, 280, yfmt=".0%"))
    note("Days are split into calm and stressed by the market's EWMA volatility (top quarter = stressed). "
         "Risk in stressed regimes is a multiple of calm-regime risk: a single unconditional number "
         "averages the two and understates risk precisely when markets are turbulent.")

    with st.expander("Why report Expected Shortfall as well as VaR?"):
        st.markdown(
            "- VaR says where the bad zone **starts**; ES says how deep it **goes**. Two portfolios can "
            "share a VaR while one has far worse tail losses.\n"
            "- ES is **sub-additive** (a coherent risk measure): combining positions never increases "
            "measured risk, so it rewards diversification consistently. VaR can violate this.\n"
            "- Regulators moved the market-risk capital standard (Basel FRTB) from 99% VaR to 97.5% ES "
            "for these reasons.\n"
            "- For a client who says *avoiding large losses matters more than maximising gains*, the "
            "average size of the large losses is the quantity that matters.")
