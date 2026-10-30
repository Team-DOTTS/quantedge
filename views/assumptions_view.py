"""Assumptions & Limitations: plain-language statement of what the model does not capture.

Kept in sync with Section 12 of the report."""
from __future__ import annotations

import streamlit as st

from src.config import MANDATE


def render() -> None:
    lim = MANDATE["risk_limits"]
    st.title("Assumptions & Limitations")
    st.markdown("Every model simplifies reality. This page states, in plain language, what our analysis "
                "assumes and what it cannot see, so the committee can judge how far to trust each number.")

    st.header("What we assume")
    st.markdown(f"""
**The past is a guide to risk, not a promise.** Every estimate is built from daily data since November
2004 (the youngest fund, GLD, started then). That window contains the 2008 crisis, the 2020 crash and the
2022 rate shock, but future crises may look different from all three.

**Weights stay fixed.** Risk numbers assume the portfolio is held at the proposed weights every day
(daily rebalancing). Crisis replays instead hold the starting weights and let them drift, as a real
portfolio would during a fast sell-off.

**Expected returns are historical averages.** They are the least reliable input to portfolio
construction. We lean on risk-based methods (minimum CVaR, minimum variance, risk parity) partly because
they do not depend on return forecasts.

**Ten-day risk is scaled from one-day risk** using the square root of time. That assumes days are
independent; in a crisis, losses cluster and the true 10-day risk can be larger.

**The risk-free rate is the 3-month US Treasury bill.**

**The mandate limits are our own proposal.** The horizon ({MANDATE['horizon_years']} years), return
objective (T-bill rate plus {MANDATE['return_excess_over_rf']:.1%} a year), volatility ceiling ({lim['max_volatility']:.0%}), 99% one-day ES limit ({lim['max_es99_1d']:.1%} of
capital), drawdown tolerance ({lim['max_drawdown']:.0%}) and stress-loss limit ({lim['max_stress_loss']:.0%})
are judgements about a conservative endowment, explained in our report. Different, reasonable limits
would change the verdict.
""")

    st.header("What the model does not capture")
    st.markdown("""
**Currency risk.** All funds are priced in US dollars. If the endowment's spending is in Sri Lankan
rupees, movements in USD/LKR add risk (or return) that none of these numbers include.

**Inflation.** Returns are nominal. A 5% return with 4% inflation preserves far less purchasing power
than the headline suggests.

**Trading costs, taxes and fund fees beyond those in prices.** Adjusted prices already reflect each ETF's
expense ratio, but not brokerage costs, bid-ask spreads, withholding taxes or the cost of rebalancing.

**Liquidity in a panic.** We assume every fund can be traded at its closing price. In severe stress,
spreads widen and prices can gap, so realised losses on forced sales could exceed the model's.

**Events with no precedent in the data.** Historical simulation cannot produce a loss worse than the worst
day in the sample. EVT extrapolates beyond it, but that extrapolation is itself an estimate with wide
uncertainty, as the sensitivity table on the EVT page shows.

**Correlations that break.** Monte Carlo and parametric models use average correlations. In crises,
correlations between risky assets tend to rise, as the stressed-days heatmap shows. Our hypothetical
"correlation breakdown" scenario tests this, but only one version of it.

**Spending and cash-flow needs.** We model a fixed USD 10 million pool. Withdrawals for scholarships and
projects during a downturn would lock in losses; this is not modelled.

**Fund-level risks.** Each ETF is treated as a clean exposure to its index. Tracking error, securities
lending, counterparty and custody risk, and the possibility of a fund closing are not modelled.
""")

    st.header("Limitations of specific methods")
    st.markdown("""
- **Parametric (Normal) VaR** assumes a bell curve; our data has much fatter tails, so it understates
  extreme losses. We show it as a reference point, not as a recommendation.
- **Cornish-Fisher** can misbehave when skewness and kurtosis are very large: its quantiles may stop
  increasing with the confidence level.
- **Monte Carlo** results carry simulation error (shown on the dashboard) and inherit every assumption of
  the distribution they simulate from.
- **EVT** results depend on the threshold. We justify our rule and show how much the answer changes under
  four alternatives. The shape estimate has a confidence interval, also shown.
- **GARCH-EVT** assumes the GARCH(1,1) volatility model is right. Its parameters are re-estimated only
  every 60 days in the backtest.
- **The calm/stressed regime split** uses a threshold computed from the whole sample, so it describes
  history; it is not a real-time forecast.
- **Backtests** cover one historical path. Passing them shows the model was not wrong on this path; it does
  not guarantee the next crisis. With about 50 expected breaches at 99% over 2007-2026, the tests have limited power
  to tell a good model from a slightly wrong one.
- **Our own choice of weights saw the whole sample.** We picked the mix after looking at 2004-2026, so its
  backtest is in-sample. The sub-period and walk-forward tests on the Alternatives page limit this bias but
  cannot remove it; that is why we chose round weights by judgement rather than the best-looking optimum.
- **Optimised weights** are estimated on the same history used to evaluate them unless the "no look-ahead"
  option is switched on in the sidebar, so in-sample performance flatters the optimiser.
""")

    st.header("How confident we are")
    st.markdown("""
We are most confident in **one-day VaR and ES at 95% and 99%**: they sit inside the data, the methods
roughly agree once fat tails are allowed for, and they pass out-of-sample validation. We are **less
confident in very rare losses** (1-in-10-year and beyond), where estimates rely on extrapolation and vary
across methods. We therefore quote those as ranges and set limits on the more conservative figure.
""")

    st.header("Use of AI tools")
    st.markdown("""
- **Which tools.** Claude (Anthropic). We used no other AI tool.
- **For what.** Brainstorming: understanding the brief and the client, identifying the key problem
  (measuring and defending the downside), and suggesting a first-draft portfolio, which we tested against the
  data and rejected. Help writing and debugging the Python code. Help drafting and editing the report.
- **How we checked the output.** Tests in `tests/` compare the maths with answers known in advance; every
  claim in the first-draft portfolio was tested against the data; every number in the report is produced by
  our code from the frozen data, not typed by hand; each module has an owner on our team who reviewed it.
- **What is ours.** The mandate limits, the final portfolio, the invented scenarios, the interpretation and
  the recommendation are our own decisions, and every one of us can explain them.
""")
