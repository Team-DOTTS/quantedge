"""Checks of the risk maths against cases with known answers.

Run with:  pytest -q
"""
import numpy as np
import pandas as pd
from scipy import stats

from src import backtest as bt
from src import compare, data, evt, kpis, stress
from src import portfolio as pf
from src import risk_budget as rb
from src import var_es as ve
from src.config import MANDATE, TICKERS
from src.vol import garch11_fit
from tests.synthetic import synthetic_prices, synthetic_returns

RNG = np.random.default_rng(123)


# --- VaR / ES -------------------------------------------------------------
def test_parametric_normal_matches_closed_form():
    r = RNG.normal(0.0, 0.01, 200_000)
    v, e = ve.parametric_normal(r, 0.99)
    assert abs(v - 0.01 * 2.3263) < 2e-4
    assert abs(e - 0.01 * 2.6652) < 2e-4


def test_historical_converges_to_true_quantile():
    r = RNG.normal(0.0, 0.01, 400_000)
    v, e = ve.historical(r, 0.99)
    assert abs(v - 0.023263) < 3e-4
    assert abs(e - 0.026652) < 3e-4


def test_student_t_es_matches_simulation():
    x = stats.t.rvs(4, size=1_000_000, random_state=1) * 0.01
    v_sim, e_sim = ve.historical(x, 0.99)
    v, e = ve.parametric_t(x, 0.99, params=(4.0, 0.0, 0.01))
    assert abs(v - v_sim) / v_sim < 0.02
    assert abs(e - e_sim) / e_sim < 0.03


def test_es_is_never_below_var():
    lr = synthetic_returns(3000, df=5)
    w = np.full(7, 1 / 7)
    p = np.log1p(np.expm1(lr.values) @ w)
    tab = ve.compute_all(p, lr.values, w, n_sims=20_000)
    assert (tab["ES"] >= tab["VaR"] - 1e-12).all()


# --- EVT ------------------------------------------------------------------
def test_gpd_fit_recovers_shape():
    y = stats.genpareto.rvs(0.25, scale=0.01, size=20_000, random_state=3)
    f = evt.gpd_fit(y)
    assert abs(f.xi - 0.25) < 0.04
    assert abs(f.beta - 0.01) / 0.01 < 0.05


def test_evt_var_close_to_true_extreme_quantile():
    x = stats.t.rvs(3, size=20_000, random_state=5) * 0.01
    res = evt.fit_evt(x, 0.999, rule="95th percentile")
    true = -stats.t.ppf(0.001, 3) * 0.01
    assert abs(res.var - true) / true < 0.15


# --- Backtesting ----------------------------------------------------------
def test_kupiec_exact_rate_is_not_rejected():
    b = np.zeros(1000, dtype=bool)
    b[::100] = True  # exactly 1%
    k = bt.kupiec(b, 0.99)
    assert k["breaches"] == 10 and k["p_value"] > 0.99


def test_christoffersen_detects_clustering():
    b = np.zeros(1000, dtype=bool)
    b[500:510] = True  # 10 breaches in a row
    assert bt.christoffersen(b, 0.99)["reject_ind"]


def test_basel_zones_match_regulation():
    assert bt.zone_for(4, 250, 0.99) == "Green"
    assert bt.zone_for(5, 250, 0.99) == "Yellow"
    assert bt.zone_for(9, 250, 0.99) == "Yellow"
    assert bt.zone_for(10, 250, 0.99) == "Red"


def test_backtest_has_no_look_ahead():
    lr = synthetic_returns(1200)
    w = np.full(7, 1 / 7)
    p = pd.Series(np.log1p(np.expm1(lr.values) @ w), index=lr.index)
    f1 = bt.rolling_forecast(p, lr, w, "Historical simulation", 0.99, 500)
    p2 = p.copy()
    p2.iloc[900:] = -0.5  # wreck the future
    f2 = bt.rolling_forecast(p2, lr, w, "Historical simulation", 0.99, 500)
    day = p.index[900]
    assert np.isclose(f1.loc[day, "VaR"], f2.loc[day, "VaR"])


# --- Portfolio and risk budgeting ----------------------------------------
def test_optimisers_respect_constraints():
    sr = data.simple_returns(synthetic_prices(2500))
    mu, cov, _ = pf.estimate_inputs(sr)
    cons = pf.Constraints(0.0, 0.5, 0.6, 0.3, None)
    for m in pf.METHODS[:4]:
        w = pf.build_portfolio(m, sr, mu, cov, cons, 0.02).weights
        assert pf.check_constraints(w, cons) == [], m


def test_min_cvar_has_lowest_es():
    sr = data.simple_returns(synthetic_prices(2500, df=4))
    mu, cov, _ = pf.estimate_inputs(sr)
    cons = pf.Constraints(0.0, 0.5, 0.6, 0.3, None)
    es = {}
    for m in pf.METHODS[:4]:
        w = pf.build_portfolio(m, sr, mu, cov, cons, 0.02, cvar_conf=0.95).weights
        es[m] = ve.historical(sr.values @ w.values, 0.95)[1]
    assert es["Minimum CVaR (Expected Shortfall)"] <= min(es.values()) + 1e-6


def test_min_cvar_return_floor_is_respected_and_costs_es():
    sr = data.simple_returns(synthetic_prices(2500, df=4))
    cons = pf.Constraints(0.0, 0.5, 0.6, 0.3, None)
    free = pf.min_cvar(sr, cons, 0.95).weights
    floor = float(sr.values.mean(axis=0) @ free.values) + 0.00001
    res = pf.min_cvar(sr, cons, 0.95, min_return=floor)
    assert res.success
    tied = res.weights
    assert float(sr.values.mean(axis=0) @ tied.values) >= floor - 1e-9
    es = lambda w: ve.historical(sr.values @ w.values, 0.95)[1]  # noqa: E731
    assert es(tied) >= es(free) - 1e-9


def test_qlr_finds_a_known_break():
    from src.regime import QLR_CRITICAL, qlr_break
    rng = np.random.default_rng(5)
    n = 3000
    x = rng.standard_normal(n) * 0.01
    beta = np.where(np.arange(n) < 1800, -0.2, 0.3)
    y = beta * x + rng.standard_normal(n) * 0.004
    idx = pd.bdate_range("2010-01-01", periods=n)
    res = qlr_break(pd.Series(y, idx), pd.Series(x, idx), step=10)
    assert res["F"] > QLR_CRITICAL[0.05]
    assert abs(idx.get_loc(res["date"]) - 1800) <= 30
    assert res["beta_before"] < 0 < res["beta_after"]


def test_fund_caps_are_respected():
    sr = data.simple_returns(synthetic_prices(2500))
    mu, cov, _ = pf.estimate_inputs(sr)
    cons = pf.Constraints(min_bonds=0.35, fund_caps=(("EEM", 0.05), ("XLE", 0.05), ("XLK", 0.05)))
    for m in pf.METHODS[:4]:
        w = pf.build_portfolio(m, sr, mu, cov, cons, 0.02).weights
        assert pf.check_constraints(w, cons) == [], m
        assert w[["EEM", "XLE", "XLK"]].max() <= 0.05 + 1e-6, m


def test_recommended_mix_is_fixed_and_within_mandate():
    sr = data.simple_returns(synthetic_prices(500))
    mu, cov, _ = pf.estimate_inputs(sr)
    lim = MANDATE["allocation_limits"]
    cons = pf.Constraints(0.0, lim["max_weight"], lim["max_equity"], lim["min_bonds"], None,
                          tuple(lim["fund_caps"].items()))
    w = pf.build_portfolio(pf.RECOMMENDED, sr, mu, cov, cons).weights
    assert np.isclose(w.sum(), 1.0)
    assert np.isclose(w["AGG"], 0.55) and np.isclose(w["GLD"], 0.20) and w["EFA"] == 0
    assert pf.check_constraints(w, cons) == []


def test_euler_components_add_up():
    sr = data.simple_returns(synthetic_prices(3000, df=5))
    w = pd.Series([.1, .1, .05, .45, .2, .05, .05], index=sr.columns)
    v = rb.component_var_normal(sr, w, 0.99)
    total_var = ve.parametric_normal(sr.values @ w.values, 0.99)[0]
    assert abs(v["component VaR"].sum() - total_var) < 1e-6
    e = rb.component_es_historical(sr, w, 0.99)
    assert abs(e["component ES"].sum() - ve.historical(sr.values @ w.values, 0.99)[1]) < 1e-9


def test_ledoit_wolf_is_valid():
    x = synthetic_returns(300).values
    c, s = pf.ledoit_wolf(x)
    assert 0 <= s <= 1 and np.all(np.linalg.eigvalsh(c) > 0)


# --- Data, KPIs, volatility ---------------------------------------------
def test_cleaning_flags_but_keeps_extreme_moves():
    px = synthetic_prices(1000)
    px.iloc[500, 0] *= 0.6  # a crash day
    px.iloc[100, 2] = np.nan
    res = data.clean_prices(px)
    assert len(res.prices) == 1000
    assert (res.flags["issue"] == "Extreme move").any()
    assert (res.flags["issue"] == "Missing price").any()


def test_drawdown_on_known_path():
    r = pd.Series([0.10, -0.20, 0.05, 0.10])
    assert abs(kpis.drawdown(r).min() - (-0.20)) < 1e-12


def test_garch_recovers_parameters():
    rng = np.random.default_rng(9)
    n, omega, alpha, beta = 8000, 2e-6, 0.08, 0.90
    h, r = omega / (1 - alpha - beta), np.empty(n)
    for t in range(n):
        r[t] = np.sqrt(h) * rng.standard_normal()
        h = omega + alpha * r[t] ** 2 + beta * h
    g = garch11_fit(r)
    assert abs(g["alpha"] - alpha) < 0.03 and abs(g["beta"] - beta) < 0.04


# --- Stress testing and portfolio choice ----------------------------------
def test_stress_replay_on_known_path():
    idx = pd.bdate_range("2020-01-01", periods=2)
    sr = pd.DataFrame(0.0, index=idx, columns=TICKERS)
    sr["SPY"] = [0.10, -0.50]
    w = pd.Series({"SPY": 0.5, "AGG": 0.5}).reindex(TICKERS).fillna(0.0)
    res = stress.replay(sr, w, "2020-01-01", "2020-01-02")
    # buy and hold: 0.5 * 1.10 * 0.50 + 0.5 = 0.775 at the end, 1.05 at the peak
    assert abs(res["total_return"] - (-0.225)) < 1e-12
    assert abs(res["max_drawdown"] - (0.775 / 1.05 - 1)) < 1e-12


def test_grid_and_bootstrap_are_well_formed():
    sr = data.simple_returns(synthetic_prices(800))
    rf = pd.Series(0.0, index=sr.index)
    g = compare.grid_search(sr, rf)
    assert np.allclose(g[TICKERS].sum(axis=1), 1.0) and (g[TICKERS] >= 0).all().all()
    b = compare.bootstrap_drawdowns(sr, g.loc[0, TICKERS], 0.02, n_paths=200)
    assert 0 <= b["p_dd_20"] <= b["p_dd_15"] <= 1 and b["median_dd"] <= 0


def test_walk_forward_pick_uses_only_the_past():
    sr = data.simple_returns(synthetic_prices(1500))
    rf = pd.Series(0.0, index=sr.index)
    split = str(sr.index[1000].date())
    a = compare.walk_forward(sr, rf, split)
    wrecked = sr.copy()
    wrecked.iloc[1000:] = -0.01  # change the future only
    b = compare.walk_forward(wrecked, rf, split)
    assert a["pick"].equals(b["pick"])
