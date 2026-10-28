"""QuantEdge 1.0 · Team DOTTS · Serendib University Endowment risk application.

Run with:  streamlit run app.py
"""
import streamlit as st

from ui.context import render_sidebar
from ui.theme import inject_css
from views import (alternatives_view, assumptions_view, backtest_view, budget_view, data_view,
                   evt_view, overview, portfolio_view, risk_view, stress_view)

st.set_page_config(page_title="DOTTS · QuantEdge", page_icon="🛡️", layout="wide")
inject_css()

st.session_state["settings"] = render_sidebar()

pages = {
    "Decision": [
        st.Page(overview.render, title="Recommendation", icon=":material/gavel:",
                url_path="recommendation", default=True),
        st.Page(alternatives_view.render, title="Alternatives", icon=":material/compare_arrows:",
                url_path="alternatives"),
    ],
    "Evidence": [
        st.Page(portfolio_view.render, title="Portfolio", icon=":material/pie_chart:", url_path="portfolio"),
        st.Page(risk_view.render, title="Risk dashboard", icon=":material/monitoring:", url_path="risk"),
        st.Page(evt_view.render, title="Extreme Value Theory", icon=":material/show_chart:", url_path="evt"),
        st.Page(backtest_view.render, title="Backtesting", icon=":material/fact_check:", url_path="backtest"),
        st.Page(stress_view.render, title="Stress testing", icon=":material/thunderstorm:", url_path="stress"),
        st.Page(budget_view.render, title="Risk budget", icon=":material/balance:", url_path="risk-budget"),
    ],
    "Reference": [
        st.Page(data_view.render, title="Data & diagnostics", icon=":material/database:", url_path="data"),
        st.Page(assumptions_view.render, title="Assumptions & Limitations", icon=":material/info:",
                url_path="assumptions"),
    ],
}

st.navigation(pages).run()

st.sidebar.caption("Team DOTTS · QuantEdge 1.0 Final Challenge")
