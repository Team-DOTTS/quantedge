"""Risk budgeting and position sizing using marginal and component VaR / ES."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src import risk_budget as rb
from src import var_es as ve
from src.config import MANDATE
from ui import context as cx
from ui.theme import GARNET, SAPPHIRE, note, pct, show_chart, show_table, style, usd, usdm


def render() -> None:
    ctx = cx.get_context()
    s = ctx.s
    t = cx.risk_budget(ctx)
    st.title("Risk budget")
    st.caption(f"Who contributes the risk? Euler allocation at {s.confidence:.0%} confidence (1-day)")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=t.index, y=t["weight"], name="Share of capital", marker_color=SAPPHIRE))
    fig.add_trace(go.Bar(x=t.index, y=t["share of ES"], name="Share of Expected Shortfall",
                         marker_color=GARNET))
    fig.update_layout(barmode="group")
    show_chart(style(fig, 360, title="Capital versus tail risk", yfmt=".0%"))
    biggest = t["risk / capital (ES)"].dropna().idxmax()
    calmest = t["risk / capital (ES)"].dropna().idxmin()
    note(f"{biggest} uses {t.loc[biggest, 'risk / capital (ES)']:.1f}× more of the tail-risk budget than "
         f"its share of capital; {calmest} uses only {t.loc[calmest, 'risk / capital (ES)']:.2f}×. "
         "A fund whose bar of risk is much taller than its bar of capital is the first candidate to trim "
         "if the risk limit is tightened.")

    disp = t.copy()
    for col in ("marginal VaR", "component VaR", "marginal ES", "component ES"):
        disp[col + " (USD)"] = disp[col] * ctx.value
    disp = disp[["weight", "component VaR (USD)", "share of VaR", "component ES (USD)", "share of ES",
                 "marginal ES", "risk / capital (ES)"]]
    show_table(disp.style.format({"weight": "{:.1%}", "component VaR (USD)": "${:,.0f}",
                                  "share of VaR": "{:.1%}", "component ES (USD)": "${:,.0f}",
                                  "share of ES": "{:.1%}", "marginal ES": "{:.2%}",
                                  "risk / capital (ES)": "{:.2f}"}))
    st.markdown(
        "- **Component VaR** (parametric) and **component ES** (historical) add up exactly to the "
        "portfolio's total, so each fund's share is a clean slice of the risk.\n"
        "- **Marginal ES** is the tail loss of one extra dollar in that fund: the price, in risk, of "
        "adding to it.")

    st.subheader("Sizing to the risk limit")
    limit = MANDATE["risk_limits"]["max_es99_1d"]
    h = cx.headline(ctx)
    es_now = float(ve.to_loss_pct(h["es99_conservative"]))
    keep = rb.scale_to_limit(es_now, limit)
    a, b, c = st.columns(3)
    a.metric("99% 1-day ES (conservative)", pct(es_now), usd(es_now * ctx.value), delta_color="off")
    b.metric("Mandate limit", pct(limit), usd(limit * ctx.value), delta_color="off")
    c.metric("Limit used", pct(es_now / limit, 0))
    if keep < 1:
        st.markdown(f"The portfolio exceeds the limit. Holding **{pct(keep, 0)}** in this mix and the rest "
                    f"in cash (or re-optimising with a tighter bond floor) would bring ES to the limit.")
    else:
        st.markdown(f"The portfolio uses {pct(es_now / limit, 0)} of the ES limit, leaving "
                    f"{usdm((limit - es_now) * ctx.value)} of daily tail-risk headroom.")
