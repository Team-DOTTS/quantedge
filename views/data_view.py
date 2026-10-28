"""Data sources, cleaning log, flagged observations and statistical diagnostics."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src import data
from src.config import ASSET_COLORS
from ui import context as cx
from ui.theme import LINE, note, show_chart, show_table, style


@st.cache_data(show_spinner="Running stationarity tests...")
def _adf(frame):
    return data.adf_table(frame)


def render() -> None:
    ctx = cx.get_context()
    d = ctx.d
    st.title("Data and diagnostics")

    st.subheader("Sources")
    st.markdown(
        "- **Prices:** Yahoo Finance via `yfinance`, adjusted close (dividends and splits included).\n"
        "- **Risk-free rate:** FRED series DTB3, 3-month US Treasury bill, converted to a daily rate.\n"
        "- No other data source is used.")
    for n in d["notes"]:
        st.info(n)

    st.subheader("Cleaning log")
    for line in d["log"]:
        st.markdown(f"- {line}")
    note("Return conventions: log returns for statistical work (distribution tests, VaR, ES, EVT, "
         "backtests); simple returns wherever funds are combined (portfolio returns, optimisation, "
         "stress tests, risk contributions), because only simple returns add across weights.")

    st.subheader("Flagged observations")
    flags = d["flags"]
    if flags.empty:
        st.success("No observations were flagged.")
    else:
        kinds = st.multiselect("Issue type", sorted(flags["issue"].unique()),
                               default=sorted(flags["issue"].unique()))
        show_table(flags[flags["issue"].isin(kinds)], hide_index=True, height=320)
        note("Flagged points are kept unless a log return cannot be computed. Extreme moves in "
             "crises are genuine and are exactly what tail-risk models must see.")

    st.subheader("Prices (rebased to 100)")
    px = d["prices"]
    fig = go.Figure()
    for t in px.columns:
        fig.add_trace(go.Scatter(x=px.index, y=100 * px[t] / px[t].iloc[0], name=t,
                                 line=dict(color=ASSET_COLORS[t], width=1.3)))
    fig.update_yaxes(type="log")
    show_chart(style(fig, 360))

    st.subheader("Return distributions")
    frame = d["lr"].copy()
    frame["Portfolio"] = ctx.port_log
    desc = data.describe_returns(frame)
    show_table(desc.style.format({"ann. mean (log)": "{:.2%}", "ann. volatility": "{:.2%}",
                                  "skewness": "{:.2f}", "excess kurtosis": "{:.2f}",
                                  "worst day": "{:.2%}", "best day": "{:.2%}",
                                  "Jarque-Bera p-value": "{:.3g}"}))
    note("Every series has excess kurtosis well above 0 and fails the Jarque-Bera normality test: "
         "the evidence for using fat-tailed and non-parametric risk models.")

    st.subheader("Stationarity (Augmented Dickey-Fuller)")
    try:
        show_table(_adf(frame).style.format({"ADF statistic": "{:.2f}", "p-value": "{:.4f}",
                                             "5% critical value": "{:.2f}"}))
        note("H0: the series has a unit root. Rejection means returns are stationary, so estimating "
             "risk from their history is meaningful. Prices themselves are not stationary, which is why "
             "all models use returns.")
    except ImportError:
        st.warning("Install statsmodels to run the ADF test (it is listed in requirements.txt).")

    st.subheader("Risk-free rate")
    rf = d["rf_annual"]
    if rf is not None and not rf.dropna().empty:
        fig = go.Figure(go.Scatter(x=rf.index, y=rf.values / 100, line=dict(color=LINE, width=1.3),
                                   name="3-month T-bill"))
        show_chart(style(fig, 260, yfmt=".1%"))
    else:
        st.warning("FRED data unavailable; a constant 2% annual risk-free rate is used.")
