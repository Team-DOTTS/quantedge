"""Layer 5: out-of-sample VaR backtesting.

Out-of-sample design: the forecast for day t is estimated ONLY from the
`window` days before t (t-window ... t-1). It is then compared with the
realised return on day t. Nothing from day t or later is used.
Slow models are re-estimated every `refit` days (GARCH parameters every 60 days,
while the GARCH volatility itself is updated every day).

Tests
- Kupiec (1995) proportion of failures: is the breach RATE right?
- Christoffersen (1998) independence: do breaches CLUSTER?
- Christoffersen conditional coverage: both at once (chi-square, 2 d.o.f.)
- Basel traffic light: zone from the number of breaches in the last 250 days
- Acerbi-Szekely (2014) Z2 statistic as a check on Expected Shortfall
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from scipy.special import xlogy

from . import evt as evt_mod
from . import var_es as ve
from .vol import ewma_sigma, garch11_fit

BACKTEST_METHODS = ve.METHODS + ["EVT (GPD-POT)", "GARCH-EVT (conditional)"]
VECTORISED = {"Historical simulation", "Parametric (Normal)", "Cornish-Fisher", "Filtered HS (EWMA)"}

BASEL_MULTIPLIER = {0: 3.00, 1: 3.00, 2: 3.00, 3: 3.00, 4: 3.00, 5: 3.40, 6: 3.50,
                    7: 3.65, 8: 3.75, 9: 3.85}


# ---------------------------------------------------------------------------
# Rolling forecasts
# ---------------------------------------------------------------------------
def rolling_forecast(r: pd.Series, asset_log: pd.DataFrame, weights: np.ndarray, method: str,
                     c: float, window: int = 500, start=None, end=None, refit: int = 20,
                     evt_rule: str = "95th percentile", n_sims: int = 10_000) -> pd.DataFrame:
    """One-day-ahead VaR and ES forecasts for each day in [start, end]."""
    vals = r.values.astype(float)
    dates = r.index
    mask = np.ones(len(r), dtype=bool)
    if start is not None:
        mask &= dates >= pd.Timestamp(start)
    if end is not None:
        mask &= dates <= pd.Timestamp(end)
    idx = np.where(mask)[0]
    idx = idx[idx >= window]
    if len(idx) == 0:
        raise ValueError("Backtest range is too short for the estimation window.")
    var = np.empty(len(idx))
    es = np.empty(len(idx))

    if method in VECTORISED:
        if method == "Filtered HS (EWMA)":
            sigma, _ = ewma_sigma(vals)
            series = vals / sigma
        else:
            series = vals
        w = sliding_window_view(series, window)[idx - window]
        if method in ("Historical simulation", "Filtered HS (EWMA)"):
            q = np.quantile(w, 1 - c, axis=1)
            tail = w <= q[:, None]
            tail_mean = (w * tail).sum(axis=1) / tail.sum(axis=1)
            scale = sigma[idx] if method == "Filtered HS (EWMA)" else 1.0
            var, es = -q * scale, -tail_mean * scale
        elif method == "Parametric (Normal)":
            mu, sd = w.mean(axis=1), w.std(axis=1, ddof=1)
            z = stats.norm.ppf(1 - c)
            var = -(mu + sd * z)
            es = -(mu - sd * stats.norm.pdf(z) / (1 - c))
        else:  # Cornish-Fisher
            mu, sd = w.mean(axis=1), w.std(axis=1, ddof=1)
            s, k = stats.skew(w, axis=1), stats.kurtosis(w, axis=1)
            var = -(mu + sd * ve._cf_z(1 - c, s, k))
            ps = (np.arange(1, 201) - 0.5) / 200 * (1 - c)
            zgrid = ve._cf_z(ps[None, :], s[:, None], k[:, None])
            es = -(mu + sd * zgrid.mean(axis=1))
    else:
        a = asset_log.values.astype(float)
        state = {}
        every = max(refit, 60) if method == "GARCH-EVT (conditional)" else refit
        for j, i in enumerate(idx):
            if j % every == 0:
                sample = vals[i - window:i]
                if method == "Parametric (Student-t)":
                    state["v"] = ve.parametric_t(sample, c)
                elif method.startswith("Monte Carlo"):
                    dist = "t" if "Student" in method else "normal"
                    df = float(np.clip(ve.fit_student_t(sample)[0], 3, 30)) if dist == "t" else None
                    state["v"] = ve.monte_carlo(a[i - window:i], weights, c, dist, n_sims,
                                                seed=42 + j, df=df)
                elif method == "EVT (GPD-POT)":
                    ev = evt_mod.fit_evt(sample, c, evt_rule, with_diagnostics=False, diag_step=0.01)
                    state["v"] = (ev.var, ev.es)
                elif method == "GARCH-EVT (conditional)":
                    g = garch11_fit(sample)
                    z = (sample - g["mu"]) / g["sigma"]
                    ev = evt_mod.fit_evt(z, c, evt_rule, with_diagnostics=False, diag_step=0.01)
                    state.update(g=g, vz=ev.var, ez=ev.es, h=g["sigma_next"] ** 2)
                else:
                    raise ValueError(f"Unknown method: {method}")
            elif method == "GARCH-EVT (conditional)":
                g = state["g"]
                e_prev = vals[i - 1] - g["mu"]
                state["h"] = g["omega"] + g["alpha"] * e_prev ** 2 + g["beta"] * state["h"]
            if method == "GARCH-EVT (conditional)":
                g, sd = state["g"], np.sqrt(state["h"])
                var[j], es[j] = -g["mu"] + sd * state["vz"], -g["mu"] + sd * state["ez"]
            else:
                var[j], es[j] = state["v"]

    out = pd.DataFrame({"realized": vals[idx], "VaR": var, "ES": es}, index=dates[idx])
    out["breach"] = out["realized"] < -out["VaR"]
    return out


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def kupiec(breaches: np.ndarray, c: float) -> dict:
    b = np.asarray(breaches, dtype=bool)
    n, x = len(b), int(b.sum())
    p = 1 - c
    phat = x / n
    ll0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(n - x, 1 - phat) + xlogy(x, phat)
    lr = -2 * (ll0 - ll1)
    pval = 1 - stats.chi2.cdf(lr, 1)
    return {"n": n, "breaches": x, "expected": n * p, "breach_rate": phat, "LR_uc": lr,
            "p_value": pval, "reject": pval < 0.05}


def christoffersen(breaches: np.ndarray, c: float) -> dict:
    b = np.asarray(breaches, dtype=int)
    prev, cur = b[:-1], b[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr_ind = -2 * (ll0 - ll1)
    p_ind = 1 - stats.chi2.cdf(lr_ind, 1)
    lr_cc = kupiec(b.astype(bool), c)["LR_uc"] + lr_ind
    p_cc = 1 - stats.chi2.cdf(lr_cc, 2)
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11, "pi01": pi01, "pi11": pi11,
            "LR_ind": lr_ind, "p_ind": p_ind, "reject_ind": p_ind < 0.05,
            "LR_cc": lr_cc, "p_cc": p_cc, "reject_cc": p_cc < 0.05}


def zone_for(x: int, n: int, c: float) -> str:
    """Basel zone from the cumulative binomial probability of seeing <= x breaches.
    At 99% and n = 250 this reproduces the Basel table exactly
    (green 0-4, yellow 5-9, red 10+); other levels use the same probability cut-offs."""
    cdf = stats.binom.cdf(x, n, 1 - c)
    if cdf < 0.95:
        return "Green"
    if cdf < 0.9999:
        return "Yellow"
    return "Red"


def traffic_light(bt: pd.DataFrame, c: float, window: int = 250) -> dict:
    b = bt["breach"].astype(int)
    last = b.iloc[-window:]
    x = int(last.sum())
    zone = zone_for(x, len(last), c)
    rolling = b.rolling(window).sum().dropna()
    zones = rolling.apply(lambda k: {"Green": 0, "Yellow": 1, "Red": 2}[zone_for(int(k), window, c)])
    out = {"zone": zone, "breaches_last": x, "window": len(last),
           "green_max": int(max(k for k in range(window + 1) if zone_for(k, window, c) == "Green")),
           "rolling_breaches": rolling, "rolling_zone": zones,
           "share_green": float((zones == 0).mean()) if len(zones) else np.nan,
           "share_red": float((zones == 2).mean()) if len(zones) else np.nan}
    if abs(c - 0.99) < 1e-9 and window == 250:
        out["basel_multiplier"] = BASEL_MULTIPLIER.get(x, 4.0)
    return out


def es_backtest(bt: pd.DataFrame, c: float) -> dict:
    """Acerbi-Szekely Z2 = sum(L_t * I_t / ES_t) / (T * (1-c)) - 1 with losses L_t > 0.
    Near 0 means ES is about right; clearly positive means ES underestimates tail losses.
    Also reports realised average loss on breach days against average predicted ES."""
    loss = -bt["realized"].values
    br = bt["breach"].values
    t = len(bt)
    z2 = float(np.sum(loss * br / bt["ES"].values) / (t * (1 - c)) - 1)
    if br.any():
        ratio = float(loss[br].mean() / bt["ES"].values[br].mean())
    else:
        ratio = np.nan
    return {"Z2": z2, "realised_over_predicted": ratio}


def summarise(bt: pd.DataFrame, c: float) -> dict:
    k = kupiec(bt["breach"].values, c)
    ch = christoffersen(bt["breach"].values, c)
    tl = traffic_light(bt, c)
    es = es_backtest(bt, c)
    return {"kupiec": k, "christoffersen": ch, "traffic": tl, "es": es}


def compare_methods(r: pd.Series, asset_log: pd.DataFrame, weights: np.ndarray, c: float,
                    window: int, start, end, evt_rule: str, methods=None, refit: int = 20,
                    n_sims: int = 5_000) -> pd.DataFrame:
    rows = []
    for m in methods or BACKTEST_METHODS:
        bt = rolling_forecast(r, asset_log, weights, m, c, window, start, end, refit,
                              evt_rule, n_sims)
        s = summarise(bt, c)
        rows.append({"method": m, "days": s["kupiec"]["n"], "breaches": s["kupiec"]["breaches"],
                     "expected": s["kupiec"]["expected"],
                     "breach rate": s["kupiec"]["breach_rate"],
                     "Kupiec p": s["kupiec"]["p_value"], "Christoffersen ind. p": s["christoffersen"]["p_ind"],
                     "Cond. coverage p": s["christoffersen"]["p_cc"],
                     "zone (last 250d)": s["traffic"]["zone"],
                     "time in red": s["traffic"]["share_red"],
                     "ES Z2": s["es"]["Z2"],
                     "avg VaR": bt["VaR"].mean()})
    return pd.DataFrame(rows).set_index("method")
