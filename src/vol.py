"""Volatility models used for filtering and regime conditioning.

- EWMA (RiskMetrics, lambda = 0.94)
- GARCH(1,1), fitted by Gaussian quasi-maximum likelihood (hand-rolled, so every
  step is explainable)
- Calm / stressed regime labels from market volatility

All conditional volatilities are one-step-ahead: sigma[t] uses information up to
day t-1 only, so they can be used in out-of-sample forecasts without look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, signal

from .config import TRADING_DAYS


def ewma_sigma(r: np.ndarray, lam: float = 0.94, init_window: int = 30) -> tuple[np.ndarray, float]:
    """One-step-ahead EWMA volatility: var[t] = lam * var[t-1] + (1 - lam) * r[t-1]^2.

    Returns (sigma, sigma_next): sigma[t] is the forecast for day t made at t-1,
    sigma_next is the forecast for the day after the sample ends. The recursion
    starts from the variance of the first `init_window` returns."""
    r = np.asarray(r, dtype=float)
    var0 = np.var(r[:init_window]) if len(r) >= init_window else np.var(r)
    var = np.empty(len(r))
    var[0] = var0
    if len(r) > 1:
        rest, _ = signal.lfilter([1.0], [1.0, -lam], (1 - lam) * r[:-1] ** 2, zi=[lam * var0])
        var[1:] = rest
    var_next = lam * var[-1] + (1 - lam) * r[-1] ** 2
    return np.sqrt(var), float(np.sqrt(var_next))


def _garch_var(e: np.ndarray, omega: float, alpha: float, beta: float, var0: float) -> np.ndarray:
    """Conditional variance h[t] = omega + alpha e[t-1]^2 + beta h[t-1], h[0] = var0."""
    x = omega + alpha * e[:-1] ** 2
    rest, _ = signal.lfilter([1.0], [1.0, -beta], x, zi=[beta * var0])
    h = np.empty(len(e))
    h[0] = var0
    h[1:] = rest
    return np.maximum(h, 1e-12)


def garch11_fit(r: np.ndarray) -> dict:
    """Fit a constant-mean GARCH(1,1) by Gaussian QMLE.

    Parameters are bounded so that omega > 0, alpha, beta >= 0, alpha + beta < 1
    (covariance stationarity)."""
    r = np.asarray(r, dtype=float)
    mu = r.mean()
    e = r - mu
    var0 = e.var()

    def nll(p):
        omega, alpha, beta = p
        if alpha + beta >= 0.9999:
            return 1e10
        h = _garch_var(e, omega, alpha, beta, var0)
        return 0.5 * np.sum(np.log(h) + e ** 2 / h)

    best = None
    for a0, b0 in [(0.08, 0.90), (0.05, 0.93), (0.12, 0.85)]:
        p0 = [var0 * (1 - a0 - b0), a0, b0]
        res = optimize.minimize(nll, p0, method="L-BFGS-B",
                                bounds=[(1e-12, var0), (1e-6, 0.5), (0.0, 0.9999)])
        if best is None or res.fun < best.fun:
            best = res
    omega, alpha, beta = best.x
    h = _garch_var(e, omega, alpha, beta, var0)
    h_next = omega + alpha * e[-1] ** 2 + beta * h[-1]
    return {"mu": mu, "omega": omega, "alpha": alpha, "beta": beta,
            "persistence": alpha + beta,
            "long_run_vol_annual": float(np.sqrt(omega / max(1 - alpha - beta, 1e-6) * TRADING_DAYS)),
            "sigma": np.sqrt(h), "sigma_next": float(np.sqrt(h_next)),
            "loglik": -best.fun - 0.5 * len(e) * np.log(2 * np.pi), "success": bool(best.success)}


def garch_filter(e: np.ndarray, params: dict, var0: float) -> np.ndarray:
    """Apply fitted GARCH parameters to a residual series (for rolling backtests)."""
    return np.sqrt(_garch_var(e, params["omega"], params["alpha"], params["beta"], var0))


def regime_labels(market_log_ret: pd.Series, lam: float = 0.94, quantile: float = 0.75) -> pd.DataFrame:
    """Label each day calm or stressed using the market's EWMA volatility.

    A day is 'Stressed' when the market's annualised EWMA volatility is above its
    full-sample `quantile`. This is a descriptive split of history; the threshold
    uses the whole sample, which is stated as a limitation."""
    sigma, _ = ewma_sigma(market_log_ret.values, lam)
    vol = pd.Series(sigma * np.sqrt(TRADING_DAYS), index=market_log_ret.index, name="market_vol")
    thr = vol.quantile(quantile)
    regime = np.where(vol > thr, "Stressed", "Calm")
    return pd.DataFrame({"market_vol": vol, "regime": regime, "threshold": thr})
