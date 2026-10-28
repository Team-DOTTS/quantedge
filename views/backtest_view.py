"""Backtesting panel: Kupiec, Christoffersen, Basel traffic light, ES check."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.backtest import zone_for
from ui import context as cx
from ui.theme import (GARNET, INK, MOSS, SAFFRON, SAPPHIRE, ZONE_COLORS, note, num, pct,
                      show_chart, show_table, style, traffic_light_html)


def _verdict(p: float) -> str:
    return "Reject (model fails)" if p < 0.05 else "Do not reject (model passes)"


def render() -> None:
    ctx = cx.get_context()
    s = ctx.s
    st.title("Backtesting")
    st.caption(f"{s.var_method} · {s.confidence:.0%} one-day VaR · rolling {s.window}-day estimation "
               f"window · out-of-sample {s.bt_start} to {s.bt_end}")
    try:
        f, summ = cx.run_backtest(ctx.port_log, ctx.d["lr"], tuple(ctx.weights.values), s.var_method,
                                  s.confidence, s.window, s.bt_start, s.bt_end, s.evt_rule)
    except ValueError as exc:
        st.error(f"{exc} Choose a later backtest start or a shorter estimation window.")
        return
    k, ch, tl, es = summ["kupiec"], summ["christoffersen"], summ["traffic"], summ["es"]

    left, right = st.columns([1, 2])
    with left:
        st.markdown("**Basel traffic light (last 250 days)**")
        st.markdown(traffic_light_html(tl["zone"]), unsafe_allow_html=True)
        st.markdown(f"{tl['breaches_last']} breaches in the last {tl['window']} days; "
                    f"green allows up to {tl['green_max']}.")
        if "basel_multiplier" in tl:
            st.markdown(f"Basel capital multiplier: **{tl['basel_multiplier']:.2f}** (3.00 in green).")
    with right:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Days tested", f"{k['n']:,}")
        m2.metric("Breaches", f"{k['breaches']}", f"{k['expected']:.1f} expected", delta_color="off")
        m3.metric("Breach rate", pct(k["breach_rate"]), f"target {pct(1 - s.confidence, 1)}", delta_color="off")
        m4.metric("Time spent in red zone", pct(tl["share_red"], 0))

    # --- formal tests ---------------------------------------------------------
    st.subheader("Statistical tests")
    tests = pd.DataFrame([
        {"Test": "Kupiec proportion of failures", "Question": "Is the number of breaches right?",
         "Statistic": k["LR_uc"], "p-value": k["p_value"], "Result": _verdict(k["p_value"])},
        {"Test": "Christoffersen independence", "Question": "Do breaches cluster together?",
         "Statistic": ch["LR_ind"], "p-value": ch["p_ind"], "Result": _verdict(ch["p_ind"])},
        {"Test": "Christoffersen conditional coverage", "Question": "Both at once",
         "Statistic": ch["LR_cc"], "p-value": ch["p_cc"], "Result": _verdict(ch["p_cc"])},
    ]).set_index("Test")
    show_table(tests.style.format({"Statistic": "{:.2f}", "p-value": "{:.3f}"}).map(
        lambda v: f"color:{GARNET if 'Reject' in str(v) and 'not' not in str(v) else MOSS};font-weight:600",
        subset=["Result"]))
    note(f"Tests use a 5% significance level. A breach followed by another breach happened with "
         f"probability {pct(ch['pi11'], 1)} versus {pct(ch['pi01'], 1)} after a normal day; "
         "a large gap means the model reacts too slowly to turbulence.")

    # --- breach chart ------------------------------------------------------------
    st.subheader("Forecast versus reality")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f.index, y=f["realized"], mode="markers", name="Daily return",
                             marker=dict(size=3, color="#9AA3B5")))
    fig.add_trace(go.Scatter(x=f.index, y=-f["VaR"], mode="lines", name=f"−VaR ({s.confidence:.0%})",
                             line=dict(color=SAPPHIRE, width=1.6)))
    br = f[f["breach"]]
    fig.add_trace(go.Scatter(x=br.index, y=br["realized"], mode="markers", name="Breach",
                             marker=dict(size=7, color=GARNET, symbol="x")))
    show_chart(style(fig, 400, yfmt=".1%"))
    note("A breach is a day when the actual loss was larger than the VaR forecast made the evening before.")

    st.subheader("Rolling 250-day breach count")
    rb = tl["rolling_breaches"]
    fig = go.Figure()
    green_max = tl["green_max"]
    yellow_max = max(i for i in range(251) if zone_for(i, 250, s.confidence) != "Red")
    top = max(float(rb.max()) + 2, yellow_max + 3)
    for lo, hi, z in [(0, green_max + 0.5, "Green"), (green_max + 0.5, yellow_max + 0.5, "Yellow"),
                      (yellow_max + 0.5, top, "Red")]:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=ZONE_COLORS[z], opacity=0.12, line_width=0)
    fig.add_trace(go.Scatter(x=rb.index, y=rb.values, mode="lines", line=dict(color=INK, width=1.5),
                             name="Breaches in trailing 250 days"))
    fig.add_hline(y=250 * (1 - s.confidence), line_dash="dot", line_color=SAPPHIRE,
                  annotation_text="expected")
    show_chart(style(fig, 300).update_yaxes(range=[0, top]))

    # --- ES check -------------------------------------------------------------------
    st.subheader("Does Expected Shortfall hold up?")
    e1, e2 = st.columns(2)
    e1.metric("Acerbi-Szekely Z₂", num(es["Z2"], 2), help="Near 0 = ES about right; clearly above 0 = ES too low.")
    e2.metric("Realised loss ÷ predicted ES on breach days", num(es["realised_over_predicted"], 2),
              help="1.00 means the ES forecast matched the average breach loss.")
    note("VaR backtests only check how often losses exceed VaR. This check asks whether the average "
         "breach loss matches the ES forecast, which is what the committee cares about.")

    # --- comparison ------------------------------------------------------------------
    st.subheader("Compare every method")
    if st.toggle("Run the backtest for all nine methods", value=False,
                 help="Takes 10–30 seconds the first time; results are cached afterwards."):
        comp = cx.compare_backtests(ctx.port_log, ctx.d["lr"], tuple(ctx.weights.values), s.confidence,
                                    s.window, s.bt_start, s.bt_end, s.evt_rule)
        zc = {"Green": MOSS, "Yellow": SAFFRON, "Red": GARNET}
        show_table(comp.style.format({"expected": "{:.1f}", "breach rate": "{:.2%}", "Kupiec p": "{:.3f}",
                                      "Christoffersen ind. p": "{:.3f}", "Cond. coverage p": "{:.3f}",
                                      "time in red": "{:.0%}", "ES Z2": "{:.2f}", "avg VaR": "{:.4f}"})
                   .map(lambda v: f"background-color:{zc.get(v, '')};color:white;font-weight:600"
                        if v in zc else "", subset=["zone (last 250d)"]))
        passing = comp[(comp["Kupiec p"] >= 0.05) & (comp["Cond. coverage p"] >= 0.05)]
        if len(passing):
            best = passing["avg VaR"].idxmin()
            note(f"{len(passing)} of {len(comp)} methods pass both Kupiec and conditional coverage. "
                 f"Among them, {best} does so with the lowest average VaR, i.e. it is accurate without "
                 "being needlessly conservative.")
        else:
            note("No method passes both tests over this period; report this openly and prefer the most "
                 "conservative model.")
