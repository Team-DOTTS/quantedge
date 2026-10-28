"""Layers 2 and 3: Value-at-Risk and Expected Shortfall.

Conventions
- Inputs are daily portfolio LOG returns r.
- VaR and ES are reported as POSITIVE numbers meaning a loss in log-return units.
  Use `to_loss_pct` to convert to a percentage loss of capital: 1 - exp(-x).
- `confidence` c is e.g. 0.99; the tail probability is 1 - c.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .vol import ewma_sigma

METHODS = [
    "Historical simulation",
    "Parametric (Normal)",
    "Parametric (Student-t)",
    "Cornish-Fisher",
    "Monte Carlo (Normal)",
    "Monte Carlo (Student-t)",
    "Filtered HS (EWMA)",
]

METHOD_NOTES = {
    "Historical simulation": "Uses the actual past returns; no distribution assumed, "
                             "but blind to anything worse than the sample contains.",
    "Parametric (Normal)": "Assumes a bell curve; fast and smooth, but understates fat tails.",
    "Parametric (Student-t)": "Bell-shaped but with fatter tails, fitted to the data.",
    "Cornish-Fisher": "Adjusts the Normal quantile for observed skewness and kurtosis.",
    "Monte Carlo (Normal)": "Simulates correlated fund returns from a multivariate Normal.",
    "Monte Carlo (Student-t)": "Simulates correlated fund returns with fat tails (multivariate t).",
    "Filtered HS (EWMA)": "Historical shapes rescaled to today's volatility, so it reacts to "
                          "calm and stressed markets.",
}


def to_loss_pct(x):
    """Log-loss -> percentage loss of capital."""
    return 1.0 - np.exp(-np.asarray(x, dtype=float))


def scale_horizon(x, days: int):
    """Square-root-of-time scaling (assumes i.i.d. returns; stated as a limitation)."""
    return np.asarray(x, dtype=float) * np.sqrt(days)


# ---------------------------------------------------------------------------
# Individual methods: each returns (VaR, ES)
# ---------------------------------------------------------------------------
def historical(r: np.ndarray, c: float) -> tuple[float, float]:
    r = np.asarray(r, dtype=float)
    q = np.quantile(r, 1 - c)
    return float(-q), float(-r[r <= q].mean())


def parametric_normal(r: np.ndarray, c: float) -> tuple[float, float]:
    mu, sd = np.mean(r), np.std(r, ddof=1)
    z = stats.norm.ppf(1 - c)
    var = -(mu + sd * z)
    es = -(mu - sd * stats.norm.pdf(z) / (1 - c))
    return float(var), float(es)


def fit_student_t(r: np.ndarray) -> tuple[float, float, float]:
    """MLE Student-t fit. Data are standardised first because the optimiser is
    unreliable on raw daily returns of size ~0.001. Degrees of freedom are kept
    above 2.5 so that the variance and ES stay finite and stable."""
    r = np.asarray(r, dtype=float)
    m, s = np.median(r), np.std(r, ddof=1)
    df, loc, scale = stats.t.fit((r - m) / s)
    return float(max(df, 2.5)), float(m + s * loc), float(s * scale)


def parametric_t(r: np.ndarray, c: float, params=None) -> tuple[float, float]:
    df, loc, scale = params or fit_student_t(r)
    q = stats.t.ppf(1 - c, df)
    var = -(loc + scale * q)
    es = -(loc - scale * stats.t.pdf(q, df) / (1 - c) * (df + q ** 2) / (df - 1))
    return float(var), float(es)


def _cf_z(p, skew, exkurt):
    z = stats.norm.ppf(p)
    return (z + (z ** 2 - 1) * skew / 6 + (z ** 3 - 3 * z) * exkurt / 24
            - (2 * z ** 3 - 5 * z) * skew ** 2 / 36)


def cornish_fisher(r: np.ndarray, c: float) -> tuple[float, float]:
    mu, sd = np.mean(r), np.std(r, ddof=1)
    s, k = stats.skew(r), stats.kurtosis(r)
    var = -(mu + sd * _cf_z(1 - c, s, k))
    ps = (np.arange(1, 201) - 0.5) / 200 * (1 - c)  # tail grid for ES
    es = -np.mean(mu + sd * _cf_z(ps, s, k))
    return float(var), float(es)


def simulate_portfolio(asset_log: np.ndarray, weights: np.ndarray, dist: str = "normal",
                       n_sims: int = 50_000, seed: int = 42, df: float | None = None) -> np.ndarray:
    """Simulate one-day portfolio log returns.

    Fund log returns are drawn from a multivariate Normal or Student-t with the
    sample mean and covariance; they are converted to simple returns, combined
    with the weights, then converted back to a portfolio log return."""
    rng = np.random.default_rng(seed)
    m = asset_log.mean(axis=0)
    cov = np.cov(asset_log, rowvar=False)
    chol = np.linalg.cholesky(cov + 1e-12 * np.eye(len(m)))
    z = rng.standard_normal((n_sims, len(m))) @ chol.T
    if dist == "t":
        nu = df or 5.0
        g = rng.chisquare(nu, size=(n_sims, 1))
        z = z * np.sqrt((nu - 2) / g)  # scaled so the covariance equals the sample covariance
    x = m + z
    port_simple = np.expm1(x) @ weights
    return np.log1p(port_simple)


def monte_carlo(asset_log: np.ndarray, weights: np.ndarray, c: float, dist: str = "normal",
                n_sims: int = 50_000, seed: int = 42, df: float | None = None) -> tuple[float, float]:
    sims = simulate_portfolio(asset_log, weights, dist, n_sims, seed, df)
    return historical(sims, c)


def monte_carlo_error(asset_log, weights, c, dist="normal", n_sims=50_000, seed=42,
                      df=None, batches: int = 10) -> dict:
    """Monte Carlo standard error of VaR/ES from independent batches."""
    sims = simulate_portfolio(asset_log, weights, dist, n_sims, seed, df)
    parts = np.array_split(sims, batches)
    est = np.array([historical(p, c) for p in parts])
    return {"var_se": float(est[:, 0].std(ddof=1) / np.sqrt(batches)),
            "es_se": float(est[:, 1].std(ddof=1) / np.sqrt(batches))}


def filtered_hs(r: np.ndarray, c: float, lam: float = 0.94) -> tuple[float, float]:
    """Filtered historical simulation: standardise returns by EWMA volatility, take
    the empirical tail of the standardised returns, rescale by tomorrow's volatility."""
    r = np.asarray(r, dtype=float)
    sigma, sigma_next = ewma_sigma(r, lam)
    z = r / sigma
    q = np.quantile(z, 1 - c)
    return float(-sigma_next * q), float(-sigma_next * z[z <= q].mean())


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------
def var_es(method: str, r: np.ndarray, c: float, asset_log: np.ndarray | None = None,
           weights: np.ndarray | None = None, n_sims: int = 50_000, seed: int = 42,
           t_params=None) -> tuple[float, float]:
    if method == "Historical simulation":
        return historical(r, c)
    if method == "Parametric (Normal)":
        return parametric_normal(r, c)
    if method == "Parametric (Student-t)":
        return parametric_t(r, c, t_params)
    if method == "Cornish-Fisher":
        return cornish_fisher(r, c)
    if method == "Monte Carlo (Normal)":
        return monte_carlo(asset_log, weights, c, "normal", n_sims, seed)
    if method == "Monte Carlo (Student-t)":
        df = (t_params or fit_student_t(r))[0]
        return monte_carlo(asset_log, weights, c, "t", n_sims, seed, df=float(np.clip(df, 3, 30)))
    if method == "Filtered HS (EWMA)":
        return filtered_hs(r, c)
    raise ValueError(f"Unknown method: {method}")


def compute_all(r: np.ndarray, asset_log: np.ndarray, weights: np.ndarray,
                confidences=(0.95, 0.99), n_sims: int = 50_000) -> pd.DataFrame:
    """Table of VaR and ES for every method and confidence level (log-loss units)."""
    t_params = fit_student_t(r)
    rows = []
    for c in confidences:
        for m in METHODS:
            v, e = var_es(m, r, c, asset_log, weights, n_sims=n_sims, t_params=t_params)
            rows.append({"method": m, "confidence": c, "VaR": v, "ES": e})
    return pd.DataFrame(rows)


def bootstrap_ci(r: np.ndarray, c: float, n_boot: int = 1000, seed: int = 7,
                 level: float = 0.90) -> dict:
    """Bootstrap confidence interval for historical VaR and ES: answers
    'how confident are we in this number?'"""
    rng = np.random.default_rng(seed)
    r = np.asarray(r, dtype=float)
    est = np.empty((n_boot, 2))
    for b in range(n_boot):
        est[b] = historical(rng.choice(r, size=len(r), replace=True), c)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return {"var_lo": float(np.quantile(est[:, 0], lo)), "var_hi": float(np.quantile(est[:, 0], hi)),
            "es_lo": float(np.quantile(est[:, 1], lo)), "es_hi": float(np.quantile(est[:, 1], hi)),
            "level": level}


def disagreement_drivers(r: np.ndarray) -> dict:
    """Statistics that explain why the methods disagree."""
    jb = stats.jarque_bera(r)
    sigma, _ = ewma_sigma(r)
    return {"skewness": float(stats.skew(r)), "excess_kurtosis": float(stats.kurtosis(r)),
            "jb_pvalue": float(jb.pvalue),
            "t_df": fit_student_t(r)[0],
            "current_vs_avg_vol": float(sigma[-1] / np.std(r, ddof=1))}
