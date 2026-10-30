# QuantEdge 1.0 · Team DOTTS

**Risk-aware portfolio construction and tail-risk dashboard for a conservative endowment.**

The Serendib University Endowment Fund has USD 10 million to invest across seven ETFs (SPY, EFA, EEM,
AGG, GLD, XLE, XLK). The client cares more about avoiding big losses than about chasing returns, so
this project does not look for the highest-return portfolio. It picks a sensible one and then measures,
tests and defends its worst case: Value-at-Risk, Expected Shortfall, Extreme Value Theory, out-of-sample
backtesting, stress testing and risk budgeting. Everything is presented in a Streamlit app that a
non-technical committee member can click through.

## Recommendation in one paragraph

**Accept the mandate with one risk limit adjusted.** Invest AGG 55%, GLD 20%, SPY 15%, XLE 5% and
XLK 5%, rebalanced quarterly. Over 2004–2026 this mix returned 7.3% a year with 7.3% volatility. Its
worst peak-to-trough fall was −18.7%, against −34.7% for a 60/40 fund. Its conservative 99% one-day
Expected Shortfall is 1.79% of capital, inside our 2% limit. The maximum drawdown tolerance moves from
the 15% we first proposed to 20%: only 0.6% of 1,601 round-number mixes stayed within 15%, and none of
those held meaningful equity. Three monitoring triggers are attached: clustered VaR breaches, a
conditional ES limit, and a stock–bond correlation trigger. That correlation has been positive in
every year since 2020, which weakens the bond hedge the portfolio relies on.

Full reasoning: [`docs/report/report.pdf`](docs/report/report.pdf).

---

## Quick start

```bash
git clone https://github.com/Team-DOTTS/quantedge
cd quantedge
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>. By default it reads the frozen data snapshot in `data/`, so it
needs no internet connection and reproduces the report's numbers exactly. Tested with Python 3.9 and later.

## Repository layout

```
quantedge/
├── app.py                    Streamlit entry point (navigation and sidebar)
├── requirements.txt
├── data/
│   ├── prices.csv            Frozen adjusted closes (Yahoo Finance)
│   ├── rf.csv                Frozen 3-month T-bill rate (FRED DTB3)
│   └── README.md
├── src/                      All analytics; no Streamlit code
│   ├── config.py             Universe, mandate, risk limits, weights, scenarios (every TEAM DECISION)
│   ├── data.py               Download, cleaning and flagging, returns, ADF and Jarque–Bera
│   ├── portfolio.py          Layer 1: Markowitz, max Sharpe, risk parity, min CVaR, Ledoit–Wolf, CAPM
│   ├── kpis.py               Sharpe, Sortino, drawdown, Calmar, Omega, beta, alpha, Treynor, IR
│   ├── compare.py            Recommended mix vs alternatives; grid search, walk-forward, bootstrap
│   ├── var_es.py             Layers 2–3: nine VaR/ES methods, bootstrap CI, Monte Carlo error
│   ├── vol.py                EWMA and GARCH(1,1) volatility, calm/stressed regimes
│   ├── evt.py                Layer 4: GPD peaks-over-threshold, threshold rules, GARCH-EVT
│   ├── backtest.py           Layer 5: rolling forecasts, Kupiec, Christoffersen, Basel, Acerbi–Székely
│   ├── stress.py             Historical replays, invented scenarios, stressed VaR
│   └── risk_budget.py        Euler component VaR and ES, position scaling
├── ui/
│   ├── context.py            Sidebar, cached computations shared by pages
│   └── theme.py              Styling (light and dark themes), chart helpers
├── views/                    One module per app page
├── scripts/
│   ├── fetch_data.py         Refresh and freeze the data snapshot
│   ├── compare_portfolios.py Print the alternatives table and robustness checks
├── docs/report/
│   └── report.pdf
└── tests/
    ├── synthetic.py          Synthetic returns with known properties
    └── test_analytics.py     22 tests of the maths
```

## The app

**Sidebar.** Confidence level (90 / 95 / 99%), VaR method (all nine), holding period (1 or 10 days),
backtest dates and estimation window (250 to 1,000 days). It also has these settings:

- portfolio construction method and covariance estimate (sample or Ledoit–Wolf)
- allocation limits and an optional volatility cap
- a "no look-ahead" option that estimates weights only on data before the backtest start
- benchmark: 60/40, 40/60 or SPY
- EVT threshold rule
- data source (frozen snapshot or live download) and sample dates

| Section | Page | What it shows |
|---|---|---|
| Decision | **Recommendation** | Verdict (accept / accept with adjusted limits / reject), mandate checks, the committee's three questions, downloadable tear-sheet numbers |
| | **Alternatives** | Recommended mix vs six alternatives, walk-forward test, five-year drawdown bootstrap |
| Evidence | **Portfolio** | Weights, cumulative return, drawdown, efficient frontier, correlations on calm vs stressed days |
| | **Risk dashboard** | VaR, ES and EVT side by side so the gap is obvious; Monte Carlo error; risk by regime |
| | **Extreme Value Theory** | Mean-excess and parameter-stability plots, threshold sensitivity, QQ plot, tail curve |
| | **Backtesting** | Out-of-sample breaches, Kupiec and Christoffersen tests, Basel traffic light, all-method comparison |
| | **Stress testing** | 2008, 2011, 2013, 2018, March 2020 and 2022 replays plus two invented scenarios |
| | **Risk budget** | Each fund's share of VaR and ES, and position scaling to the ES limit |
| Reference | **Data & diagnostics** | Cleaning log, flagged data points, distribution statistics, ADF tests |
| | **Assumptions & Limitations** | What the models assume and cannot see, in plain language; AI-use disclosure |

## Data

| Source | Series | Use |
|---|---|---|
| Yahoo Finance (`yfinance`, `auto_adjust=True`) | Adjusted daily closes for the 7 ETFs | All returns |
| FRED | DTB3, 3-month Treasury bill rate | Risk-free rate, return target |

- **Sample:** 18 Nov 2004 (GLD's first trading day) to 26 Oct 2026, giving 5,517 daily returns.
- **Return conventions:**
  - Log returns are used for statistics: tests, VaR, ES, EVT and backtests.
  - Simple returns are used wherever funds are combined into a portfolio, because only simple returns add up across weights.
- **Cleaning** (`src/data.py`): every step is in code and logged.
  - Checks for duplicated dates, non-positive prices, missing values (gaps of up to 2 days are
    forward-filled), stale prices and implausible jumps.
  - Odd data points are **flagged, not deleted**. The 55 extreme moves found are almost all genuine
    2008 and 2020 crisis days, and they are kept.
- **Stationarity:** ADF rejects a unit root for every fund (p < 0.001). Jarque–Bera rejects normality
  for every fund.

Refresh the snapshot (needs internet):

```bash
python -m scripts.fetch_data --end 2026-10-30
```

## Methods

| Layer | Question | Methods |
|---|---|---|
| 1. Portfolio | How should we split the money? | Markowitz minimum variance and maximum Sharpe, risk parity, minimum CVaR (Rockafellar–Uryasev linear programme), equal weight; Ledoit–Wolf shrinkage; CAPM; KPI scorecard. Checked by a 1,601-mix grid search, walk-forward tests, sub-periods and a block bootstrap of five-year paths |
| 2. VaR | How much could we lose on a bad day? | Historical simulation; parametric Normal and Student-t; Cornish–Fisher; Monte Carlo (Normal and Student-t, 50,000 draws); filtered historical simulation with EWMA volatility |
| 3. ES | When a bad day happens, how bad is it on average? | ES for every VaR method; bootstrap confidence interval; square-root-of-time scaling; stressed ES |
| 4. EVT | What about once-in-a-decade disasters? | Peaks-over-threshold with a Generalised Pareto fit (maximum likelihood written by hand). Threshold chosen by a reproducible parameter-stability rule and compared with three alternatives. GARCH-EVT (McNeil–Frey) |
| 5. Backtesting | Was the risk model right? | Rolling 500-day out-of-sample forecasts from 2012; Kupiec, Christoffersen independence and conditional coverage; Basel traffic light; Acerbi–Székely Z2 test for ES |
| Decision | What should the board do? | Euler risk budgeting, historical and invented stress tests, mandate checks, a three-way verdict rule and monitoring triggers |

## Mandate (team decisions in `src/config.py`)

| Item | Value |
|---|---|
| Horizon | 5 years, risk monitored daily, reviewed quarterly |
| Return target | 3-month T-bill rate + 2.5% a year |
| Volatility ceiling | 8% a year |
| 99% one-day ES | ≤ 2% of capital (USD 200,000) |
| Maximum drawdown | ≤ 20% (first proposed as 15%) |
| Worst stress scenario | ≤ 25% loss |
| Allocation | Long only; AGG ≥ 35%; equity ≤ 60%; AGG ≤ 60%; GLD ≤ 20%; EEM, XLE, XLK ≤ 5% each; other funds ≤ 50% |

The report explains the reason for every value. Most can be changed from the app's sidebar.

## Key results

| Measure | Recommended | 60/40 |
|---|---|---|
| CAGR (2004–2026) | 7.3% | 8.1% |
| Volatility | 7.3% | 11.5% |
| Sortino ratio | 1.07 | 0.83 |
| Maximum drawdown | −18.7% | −34.7% |
| 99% one-day ES (historical) | 1.8% | 3.1% |
| 2008 crisis replay | −4.2% | −30.0% |
| 2022 rate shock replay | −12.8% | −20.4% |
| Chance of a fall > 20% within 5 years (bootstrap) | 3.5% | 33.0% |

**Out-of-sample backtest** (99% one-day VaR, filtered historical simulation, 3,725 days):
- 40 breaches against 37 expected (Kupiec p = 0.65), in the Basel green zone.
- Every method tested fails the Christoffersen independence test, because breaches cluster in crises.

The 22 tests run on synthetic data with known answers. They check that:

- Normal VaR and ES match the closed form, historical VaR converges to the true quantile, and ES is never below VaR
- the GPD fit recovers a known shape
- Kupiec, Christoffersen and the Basel zones behave as specified
- the backtest has no look-ahead
- the optimisers respect every constraint, and minimum CVaR has the lowest ES
- Euler risk contributions add up to the total
- GARCH recovers known parameters
- cleaning flags extreme moves but keeps them
- the walk-forward test uses only past data

## Assumptions and limitations (summary)

- **One historical path (2004–2026).** The recommended weights were chosen after seeing it. Walk-forward
  and sub-period tests limit this bias but cannot remove it.
- **Modelling simplifications:**
  - Risk models assume constant (daily rebalanced) weights; crisis replays hold the starting weights.
  - Ten-day risk is scaled by the square root of time, which ignores the clustering the backtest finds.
  - EVT estimates beyond the 99.9% level are extrapolations.
- **Not modelled:**
  - Returns are nominal USD, with no USD/LKR currency risk, inflation, trading costs, taxes, liquidity gaps or spending withdrawals.
  - The stock–bond correlation has turned positive since 2020, so full-sample diversification may overstate today's protection.

The full list is on the app's **Assumptions & Limitations** page and in Section 10 of the report.

## Use of AI tools

We used Claude (Anthropic) and no other AI tool. It helped with understanding the brief, a first-draft
portfolio (which we tested against the data and rejected), writing and debugging code, and drafting the
report. We checked its output with the test suite. Every reported number is generated by our code from
the frozen data, and every claim about the funds was tested against the data. The mandate limits, final
portfolio, invented scenarios and recommendation are the team's own decisions.

The full list is in the report.

## Team Members
* Dinera Susiriwardana
* Thinula Harischandra
* Thisumi Hettiarachchige
* Omindu Ranthika

