"""Central configuration: investable universe, client mandate and constants.

Every number marked TEAM DECISION is our own choice; the report explains each
one. The app reads these as defaults, and most can be changed from the sidebar.
"""
from __future__ import annotations

TRADING_DAYS = 252
DATA_START = "2004-11-18"  # GLD's first trading day; the binding start date for the universe

# ---------------------------------------------------------------------------
# Investable universe (fixed by the competition brief)
# ---------------------------------------------------------------------------
UNIVERSE: dict[str, dict[str, str]] = {
    "SPY": {"name": "S&P 500 ETF", "role": "US large-cap equities", "group": "Equity"},
    "EFA": {"name": "iShares MSCI EAFE", "role": "International developed equities", "group": "Equity"},
    "EEM": {"name": "iShares MSCI Emerging Markets", "role": "Emerging market equities", "group": "Equity"},
    "AGG": {"name": "iShares US Aggregate Bond", "role": "US investment-grade bonds", "group": "Bonds"},
    "GLD": {"name": "SPDR Gold Shares", "role": "Gold / real asset", "group": "Gold"},
    "XLE": {"name": "Energy Select Sector SPDR", "role": "Energy / commodities proxy", "group": "Equity"},
    "XLK": {"name": "Technology Select Sector SPDR", "role": "Technology sector", "group": "Equity"},
}
TICKERS: list[str] = list(UNIVERSE)
EQUITY_TICKERS: list[str] = [t for t, v in UNIVERSE.items() if v["group"] == "Equity"]
BOND_TICKERS: list[str] = [t for t, v in UNIVERSE.items() if v["group"] == "Bonds"]
MARKET_PROXY = "SPY"  # market portfolio for CAPM (beta, Treynor, Jensen's alpha)

# Fixed colour per asset so every chart in the app reads the same way
ASSET_COLORS: dict[str, str] = {
    "SPY": "#24408E", "EFA": "#5B7BC9", "EEM": "#8E6CB8", "AGG": "#4F7942",
    "GLD": "#C9971C", "XLE": "#8C5A3C", "XLK": "#2E8A99",
}

# ---------------------------------------------------------------------------
# Client mandate: Serendib University Endowment Fund (TEAM DECISION values)
# ---------------------------------------------------------------------------
MANDATE = {
    "client": "Serendib University Endowment Fund",
    "portfolio_value": 10_000_000,              # USD, fixed by the brief
    # TEAM DECISION: an endowment is long-lived, but a conservative client wants a
    # measurable review point. Risk is still monitored daily and reviewed quarterly.
    "horizon": "5 years, with risk monitored daily and reviewed quarterly",
    "horizon_years": 5,
    # TEAM DECISION: risk-free rate plus 2.5% a year, enough to fund scholarships
    # without chasing risk. Checked against the sample's average T-bill rate.
    "return_excess_over_rf": 0.025,
    "risk_limits": {
        "max_volatility": 0.08,                 # TEAM DECISION: annualised volatility ceiling
        "max_es99_1d": 0.020,                   # TEAM DECISION: 99% 1-day ES as a share of capital
        # TEAM DECISION: tolerable peak-to-trough loss. 15% was the first proposal, but
        # only 0.6% of 1,601 round-number mixes stayed inside it over 2004-2026, all with
        # no broad equity at all (minimum variance fell 16.8%). See src/compare.py.
        "max_drawdown": 0.20,
        "max_stress_loss": 0.25,                # TEAM DECISION: worst acceptable crisis-scenario loss
    },
    "allocation_limits": {
        "min_weight": 0.0,                      # long-only, no leverage (brief: ETFs only)
        "max_weight": 0.50,                     # TEAM DECISION: single-fund concentration cap
        "max_equity": 0.60,                     # TEAM DECISION: total equity cap
        "min_bonds": 0.35,                      # TEAM DECISION: defensive core floor in AGG
        # TEAM DECISION: per-fund caps that replace max_weight for these funds.
        # AGG may exceed the general cap because it is itself thousands of
        # investment-grade bonds; gold is capped because it is a single commodity
        # that can go years without gaining; the volatile satellites are kept small
        # so none can dominate risk.
        "fund_caps": {"AGG": 0.60, "GLD": 0.20, "EEM": 0.05, "XLE": 0.05, "XLK": 0.05},
    },
}

# ---------------------------------------------------------------------------
# Recommended portfolio (TEAM DECISION; see the Alternatives page and the report)
# ---------------------------------------------------------------------------
# Defensive core: AGG 55% + GLD 20% = 75%. Bonds cushion equity crashes (2008);
# gold cushions the case where bonds fail too (2022).
# Growth: SPY 15% + XLK 5%. XLE 5% hedges inflation and oil shocks (+ in 2022).
# EFA and EEM are left out: EFA's daily correlation with SPY is 0.88 and adding
# it made every sub-period slightly worse; EEM only helped in the 2004-11 boom.
RECOMMENDED_WEIGHTS: dict[str, float] = {
    "AGG": 0.55, "GLD": 0.20, "SPY": 0.15, "EFA": 0.0, "EEM": 0.0, "XLE": 0.05, "XLK": 0.05,
}

# First draft, kept for comparison: it breaches the volatility, ES and drawdown limits.
DRAFT_WEIGHTS: dict[str, float] = {
    "AGG": 0.45, "SPY": 0.20, "GLD": 0.15, "EFA": 0.10, "EEM": 0.04, "XLE": 0.03, "XLK": 0.03,
}

# ---------------------------------------------------------------------------
# Historical stress windows (peak-to-trough of the relevant market episode)
# ---------------------------------------------------------------------------
HISTORICAL_SCENARIOS: dict[str, dict[str, str]] = {
    "2008 Global Financial Crisis": {
        "start": "2007-10-09", "end": "2009-03-09",
        "story": "S&P 500 peak to trough; credit freeze, Lehman failure, global recession.",
    },
    "2011 US downgrade & euro crisis": {
        "start": "2011-07-22", "end": "2011-10-03",
        "story": "US credit rating downgrade and euro-area sovereign debt fears.",
    },
    "2013 Taper tantrum": {
        "start": "2013-05-22", "end": "2013-06-24",
        "story": "Bond yields jumped after the Fed signalled slower asset purchases.",
    },
    "2018 Q4 sell-off": {
        "start": "2018-09-20", "end": "2018-12-24",
        "story": "Rate-hike and trade-war fears drove a sharp equity drawdown.",
    },
    "March 2020 COVID crash": {
        "start": "2020-02-19", "end": "2020-03-23",
        "story": "Fastest bear market on record as economies shut down.",
    },
    "2022 Rate-hiking drawdown": {
        "start": "2022-01-03", "end": "2022-10-12",
        "story": "Inflation shock and rapid Fed hikes; stocks and bonds fell together.",
    },
}

# Hypothetical scenario: instantaneous shocks (simple returns). TEAM DECISION:
# the brief asks us to design and justify our own scenario.
HYPOTHETICAL_SCENARIOS: dict[str, dict] = {
    "Stagflation with EM debt stress": {
        "story": (
            "Inflation re-accelerates, the Fed hikes again, long yields jump, a strong dollar "
            "triggers emerging-market debt stress, oil spikes on a supply shock."
        ),
        "shocks": {"SPY": -0.22, "EFA": -0.25, "EEM": -0.35, "AGG": -0.10,
                   "GLD": 0.08, "XLE": 0.05, "XLK": -0.32},
    },
    "Stock-bond correlation breakdown": {
        "story": (
            "A 2022-style episode but twice as severe: equities and bonds fall together, "
            "so the usual diversification from bonds fails."
        ),
        "shocks": {"SPY": -0.30, "EFA": -0.32, "EEM": -0.38, "AGG": -0.15,
                   "GLD": -0.05, "XLE": -0.20, "XLK": -0.40},
    },
}

CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]
