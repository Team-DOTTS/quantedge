"""Layer 4: Extreme Value Theory (Peaks-over-Threshold with a Generalised Pareto tail).

Theory in one paragraph: for a high enough threshold u, losses in excess of u
are approximately Generalised Pareto (Pickands-Balkema-de Haan theorem) with
shape xi and scale beta. xi > 0 means a heavy (power-law) tail, the usual case
for financial losses. The fitted tail gives VaR and ES beyond the range of the
data, which is exactly where historical simulation runs out of observations.

Tail formulas (McNeil, Frey & Embrechts, Quantitative Risk Management):
    VaR_c = u + beta/xi * [ ((n/N_u) * (1-c))^(-xi) - 1 ]
    ES_c  = (VaR_c + beta - xi*u) / (1 - xi)        valid for xi < 1
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .vol import garch11_fit

THRESHOLD_RULES = [
    "Parameter stability (automatic)",
    "90th percentile",
    "95th percentile",
    "Square-root rule (k = √n)",
    "Manual percentile",
]

THRESHOLD_RULE_NOTES = {
    "Parameter stability (automatic)": (
        "Lowest threshold (from the 85th percentile up) whose fitted shape xi is within "
        "one standard error of the median xi of all higher thresholds, i.e. where the "
        "parameter-stability plot flattens into a plateau. Makes the visual rule reproducible."),
    "90th percentile": "Common rule of thumb: the worst 10% of days form the tail.",
    "95th percentile": "Stricter rule of thumb: the worst 5% of days form the tail.",
    "Square-root rule (k = √n)": (
        "Uses the k = √n largest losses (Ferreira, de Haan & Peng, 2003); balances bias "
        "from too low a threshold against variance from too few exceedances."),
    "Manual percentile": "Threshold chosen by the user from the diagnostic plots.",
}


@dataclass
class GPDFit:
    xi: float
    beta: float
    n_exc: int
    se_xi: float
    se_beta: float
    loglik: float


@dataclass
class EVTResult:
    threshold: float
    threshold_pct: float
    rule: str
    fit: GPDFit
    n: int
    var: float
    es: float
    inside_body: bool
    diagnostics: pd.DataFrame = field(repr=False)


def _gpd_nll(p, y):
    xi, log_beta = p
    beta = np.exp(log_beta)
    if abs(xi) < 1e-8:  # exponential limit
        return len(y) * log_beta + y.sum() / beta
    t = 1 + xi * y / beta
    if np.any(t <= 0):
        return 1e10
    return len(y) * log_beta + (1 + 1 / xi) * np.log(t).sum()


def _gpd_grad(p, y):
    xi, log_beta = p
    beta = np.exp(log_beta)
    t = 1 + xi * y / beta
    if np.any(t <= 0) or abs(xi) < 1e-8:
        return np.zeros(2)
    g_xi = -np.log(t).sum() / xi ** 2 + (1 + 1 / xi) * np.sum((y / beta) / t)
    g_lb = len(y) - (1 + 1 / xi) * np.sum((xi * y / beta) / t)
    return np.array([g_xi, g_lb])


def gpd_fit(excess: np.ndarray) -> GPDFit:
    """Hand-rolled maximum-likelihood GPD fit to threshold excesses y > 0.

    Log-likelihood: l(xi, beta) = -n log(beta) - (1 + 1/xi) * sum log(1 + xi y / beta).
    Optimised over (xi, log beta) with analytic gradients, started from the
    method-of-moments estimate. Falls back to scipy's generic fit if needed.
    Standard errors from the asymptotic covariance (valid for xi > -0.5):
    Var(xi) = (1+xi)^2/n, Var(beta) = 2 beta^2 (1+xi)/n."""
    y = np.asarray(excess, dtype=float)
    n = len(y)
    m, v = y.mean(), y.var()
    xi0 = float(np.clip(0.5 * (1 - m * m / v), -0.4, 0.8)) if v > 0 else 0.1
    xi0 = xi0 if abs(xi0) > 1e-3 else 0.01
    beta0 = max(m * (1 - xi0), 1e-12)
    res = optimize.minimize(_gpd_nll, [xi0, np.log(beta0)], args=(y,), jac=_gpd_grad,
                            method="L-BFGS-B", bounds=[(-0.49, 1.5), (None, None)])
    if res.success and res.fun < 1e9:
        xi, beta = float(res.x[0]), float(np.exp(res.x[1]))
    else:
        xi, _, beta = stats.genpareto.fit(y, floc=0)
    ll = float(np.sum(stats.genpareto.logpdf(y, xi, 0, beta)))
    se_xi = (1 + xi) / np.sqrt(n) if xi > -0.5 else np.nan
    se_beta = beta * np.sqrt(2 * (1 + xi) / n) if xi > -0.5 else np.nan
    return GPDFit(float(xi), float(beta), n, float(se_xi), float(se_beta), ll)


def threshold_diagnostics(losses: np.ndarray, q_lo: float = 0.80, q_hi: float = 0.985,
                          step: float = 0.005, min_exc: int = 30) -> pd.DataFrame:
    """Mean-excess function and GPD parameters across candidate thresholds."""
    losses = np.asarray(losses, dtype=float)
    rows = []
    for q in np.arange(q_lo, q_hi + 1e-9, step):
        u = np.quantile(losses, q)
        exc = losses[losses > u] - u
        if len(exc) < min_exc:
            continue
        f = gpd_fit(exc)
        rows.append({"percentile": q, "threshold": u, "n_exc": len(exc),
                     "mean_excess": exc.mean(), "xi": f.xi, "se_xi": f.se_xi,
                     "xi_lo": f.xi - 1.96 * f.se_xi, "xi_hi": f.xi + 1.96 * f.se_xi,
                     "modified_scale": f.beta - f.xi * u})
    return pd.DataFrame(rows)


def select_threshold(losses: np.ndarray, rule: str, manual_pct: float = 0.95,
                     diag: pd.DataFrame | None = None, min_exc_select: int = 50) -> tuple[float, float]:
    """Return (threshold, its percentile) according to the chosen rule."""
    losses = np.asarray(losses, dtype=float)
    n = len(losses)
    if rule == "90th percentile":
        return float(np.quantile(losses, 0.90)), 0.90
    if rule == "95th percentile":
        return float(np.quantile(losses, 0.95)), 0.95
    if rule == "Square-root rule (k = √n)":
        k = max(int(np.sqrt(n)), 30)
        u = float(np.sort(losses)[-k - 1])
        return u, float(1 - k / n)
    if rule == "Manual percentile":
        return float(np.quantile(losses, manual_pct)), manual_pct
    # Parameter stability (automatic): lowest threshold whose shape estimate sits
    # within one standard error of the plateau formed by all higher thresholds.
    d = diag if diag is not None else threshold_diagnostics(losses)
    d = d[d["percentile"] >= 0.85 - 1e-9].reset_index(drop=True)
    for j in range(len(d)):
        if d.loc[j, "n_exc"] < min_exc_select:
            break
        higher = d.iloc[j + 1:]
        if len(higher) < 3:
            break
        if abs(np.median(higher["xi"]) - d.loc[j, "xi"]) <= d.loc[j, "se_xi"]:
            return float(d.loc[j, "threshold"]), float(d.loc[j, "percentile"])
    return float(np.quantile(losses, 0.95)), 0.95


def gpd_var_es(u: float, fit: GPDFit, n: int, c: float) -> tuple[float, float]:
    ratio = (1 - c) * n / fit.n_exc
    if abs(fit.xi) < 1e-6:
        var = u - fit.beta * np.log(ratio)
    else:
        var = u + fit.beta / fit.xi * (ratio ** (-fit.xi) - 1)
    es = (var + fit.beta - fit.xi * u) / (1 - fit.xi) if fit.xi < 1 else np.inf
    return float(var), float(es)


def fit_evt(r: np.ndarray, c: float, rule: str = "Parameter stability (automatic)",
            manual_pct: float = 0.95, with_diagnostics: bool = True,
            diag_step: float = 0.005) -> EVTResult:
    """EVT VaR/ES for portfolio log returns r (losses = -r).
    `diag_step` is the spacing of candidate thresholds (coarser = faster, used in backtests)."""
    losses = -np.asarray(r, dtype=float)
    diag = threshold_diagnostics(losses, step=diag_step) \
        if with_diagnostics or rule.startswith("Parameter") else pd.DataFrame()
    u, pct = select_threshold(losses, rule, manual_pct, diag)
    exc = losses[losses > u] - u
    fit = gpd_fit(exc)
    n = len(losses)
    inside = (1 - c) >= fit.n_exc / n  # requested quantile is not in the tail region
    if inside:
        q = np.quantile(losses, c)
        var, es = float(q), float(losses[losses >= q].mean())
    else:
        var, es = gpd_var_es(u, fit, n, c)
    return EVTResult(u, pct, rule, fit, n, var, es, bool(inside), diag)


def garch_evt(r: np.ndarray, c: float, rule: str = "Parameter stability (automatic)",
              manual_pct: float = 0.95) -> dict:
    """Conditional EVT (McNeil & Frey, 2000): GARCH(1,1) filters volatility
    clustering, EVT models the tail of the standardised residuals, and the result
    is rescaled by tomorrow's forecast volatility."""
    g = garch11_fit(r)
    z = (np.asarray(r) - g["mu"]) / g["sigma"]
    ev = fit_evt(z, c, rule, manual_pct, with_diagnostics=False)
    var = -g["mu"] + g["sigma_next"] * ev.var
    es = -g["mu"] + g["sigma_next"] * ev.es
    return {"var": float(var), "es": float(es), "garch": g, "evt_z": ev}


def tail_curve(r: np.ndarray, res: EVTResult, confidences=None) -> pd.DataFrame:
    """Loss at increasingly extreme confidence levels: empirical, Normal and EVT.
    The 'one bad day every N trading days' column makes the numbers intuitive."""
    r = np.asarray(r, dtype=float)
    losses = -r
    if confidences is None:
        confidences = np.array([0.95, 0.975, 0.99, 0.995, 0.998, 0.999, 0.9996])
    mu, sd = r.mean(), r.std(ddof=1)
    rows = []
    for c in confidences:
        evt_var = gpd_var_es(res.threshold, res.fit, res.n, c)[0] \
            if (1 - c) < res.fit.n_exc / res.n else np.quantile(losses, c)
        rows.append({"confidence": c, "once every (trading days)": 1 / (1 - c),
                     "Empirical": float(np.quantile(losses, c)),
                     "Normal": float(-(mu + sd * stats.norm.ppf(1 - c))),
                     "EVT (GPD)": float(evt_var)})
    return pd.DataFrame(rows)


def qq_data(r: np.ndarray, res: EVTResult) -> pd.DataFrame:
    """Observed vs fitted GPD quantiles of the excesses (goodness-of-fit check)."""
    losses = -np.asarray(r, dtype=float)
    exc = np.sort(losses[losses > res.threshold] - res.threshold)
    p = (np.arange(1, len(exc) + 1) - 0.5) / len(exc)
    theo = stats.genpareto.ppf(p, res.fit.xi, 0, res.fit.beta)
    return pd.DataFrame({"fitted": theo, "observed": exc})
