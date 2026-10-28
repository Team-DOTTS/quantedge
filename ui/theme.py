"""Visual identity, number formatting and small reusable display helpers.

Palette ("Ceylon sapphire"): a deep sapphire for the portfolio, garnet for losses
and tail risk, moss for pass, saffron for caution. Losses are always garnet so a
committee member can find the downside on any chart at a glance."""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SAPPHIRE = "#24408E"
GARNET = "#9B2335"
GARNET_LIGHT = "#D98A96"
MOSS = "#4F7942"
SAFFRON = "#C9971C"
INK = "#1D2433"
SLATE = "#6B7280"
MIST = "#E9ECF2"

ZONE_COLORS = {"Green": MOSS, "Yellow": SAFFRON, "Red": GARNET}

CSS = f"""
<style>
.block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
h1 {{ font-weight: 700; letter-spacing: -0.01em; color: {INK}; }}
h2, h3 {{ color: {INK}; }}
[data-testid="stMetric"] {{
    background: #FFFFFF; border: 1px solid {MIST}; border-radius: 6px;
    padding: 0.7rem 0.9rem;
}}
[data-testid="stMetricLabel"] p {{ color: {SLATE}; }}
.verdict {{ border-radius: 6px; padding: 1rem 1.2rem; margin: 0.4rem 0 1.2rem 0;
            border-left: 6px solid; background: #FFFFFF; }}
.verdict h3 {{ margin: 0 0 0.25rem 0; }}
.verdict p {{ margin: 0; color: {INK}; }}
.light {{ display: flex; gap: 0.5rem; align-items: center; }}
.light span {{ width: 34px; height: 34px; border-radius: 50%; display: inline-block;
               opacity: 0.18; border: 1px solid rgba(0,0,0,0.1); }}
.light span.on {{ opacity: 1; box-shadow: 0 0 0 3px rgba(0,0,0,0.06); }}
.note {{ color: {SLATE}; font-size: 0.92rem; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# --- formatting -------------------------------------------------------------
def pct(x, d: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def usd(x, d: int = 0) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"${x:,.{d}f}"


def usdm(x, d: int = 0) -> str:
    """USD for st.markdown text (escapes $ so it is not read as LaTeX)."""
    return usd(x, d).replace("$", "\\$")


def num(x, d: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{d}f}"


# --- charts -----------------------------------------------------------------
def style(fig: go.Figure, height: int = 380, title: str | None = None,
          yfmt: str | None = None, xfmt: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height,
        title=dict(text=title, x=0, font=dict(size=15, color=INK)) if title else None,
        margin=dict(l=10, r=10, t=45 if title else 15, b=10),
        font=dict(family="Source Sans Pro, Helvetica, Arial, sans-serif", size=13, color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="white"),
    )
    if yfmt:
        fig.update_yaxes(tickformat=yfmt)
    if xfmt:
        fig.update_xaxes(tickformat=xfmt)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=MIST)
    return fig


def _full_width(func) -> dict:
    """Newer Streamlit versions use width="stretch"; older ones use use_container_width."""
    if "width" in inspect.signature(func).parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}


def show_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, config={"displaylogo": False}, **_full_width(st.plotly_chart))


def show_table(df: pd.DataFrame, **kwargs) -> None:
    st.dataframe(df, **_full_width(st.dataframe), **kwargs)


def verdict_box(title: str, body: str, color: str) -> None:
    st.markdown(f'<div class="verdict" style="border-left-color:{color}">'
                f'<h3 style="color:{color}">{title}</h3><p>{body}</p></div>',
                unsafe_allow_html=True)


def traffic_light_html(zone: str) -> str:
    spans = "".join(
        f'<span class="{"on" if z == zone else ""}" style="background:{c}"></span>'
        for z, c in ZONE_COLORS.items())
    return f'<div class="light">{spans}<strong style="margin-left:0.6rem;font-size:1.3rem;color:{ZONE_COLORS[zone]}">{zone}</strong></div>'


def note(text: str) -> None:
    """Small grey explanatory text. Dollar signs become HTML entities so they are
    never read as LaTeX delimiters inside the HTML block."""
    text = text.replace("\\$", "&#36;").replace("$", "&#36;")
    st.markdown(f'<p class="note">{text}</p>', unsafe_allow_html=True)
