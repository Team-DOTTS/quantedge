"""Layer 1: portfolio construction (Markowitz, risk parity, minimum CVaR) and CAPM.

Optimisation inputs use SIMPLE returns, because a portfolio's simple return is
exactly the weighted sum of its assets' simple returns (log returns are not).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, sparse

from .config import BOND_TICKERS, EQUITY_TICKERS, RECOMMENDED_WEIGHTS, TRADING_DAYS

RECOMMENDED = "Recommended mix (team proposal)"

METHODS = [
    "Minimum CVaR (Expected Shortfall)",
    "Minimum variance",
    "Maximum Sharpe ratio",
    "Risk parity (equal risk contribution)",
    "Equal weight",
    "Custom weights",
    RECOMMENDED,
]


@dataclass(frozen=True)
class Constraints:
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_equity: float = 1.0
    min_bonds: float = 0.0
    max_vol: float | None = None  # annualised; applied to variance-based optimisers
    fund_caps: tuple = ()         # ((ticker, cap), ...) per-fund upper limits that replace max_weight


@dataclass
class OptResult:
    weights: pd.Series
    method: str
    success: bool
    message: str


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def ledoit_wolf(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage of the sample covariance toward a scaled identity.

    Returns (covariance, shrinkage intensity in [0, 1])."""
    t, n = x.shape
    x = x - x.mean(axis=0)
    s = x.T @ x / t
    mu = np.trace(s) / n
    delta = np.sum((s - mu * np.eye(n)) ** 2)
    beta_bar = (np.sum(np.sum(x ** 2, axis=1) ** 2) / t - np.sum(s ** 2)) / t
    shrink = 0.0 if delta == 0 else float(np.clip(beta_bar / delta, 0.0, 1.0))
    return shrink * mu * np.eye(n) + (1 - shrink) * s, shrink


def estimate_inputs(simple_ret: pd.DataFrame, cov_method: str = "sample"
                    ) -> tuple[pd.Series, pd.DataFrame, float]:
    """Annualised expected returns and covariance matrix.

    Returns (mu, cov, shrinkage) where shrinkage is 0 for the sample covariance."""
    x = simple_ret.dropna()
    mu = x.mean() * TRADING_DAYS
    if cov_method == "ledoit-wolf":
        c, shrink = ledoit_wolf(x.values)
    else:
        c, shrink = np.cov(x.values, rowvar=False), 0.0
    cov = pd.DataFrame(c * TRADING_DAYS, index=x.columns, columns=x.columns)
    return mu, cov, shrink


# ---------------------------------------------------------------------------
# Constraint helpers
# ---------------------------------------------------------------------------
def _group_vectors(tickers: list[str]) -> tuple[np.ndarray, np.ndarray]:
    eq = np.array([1.0 if t in EQUITY_TICKERS else 0.0 for t in tickers])
    bd = np.array([1.0 if t in BOND_TICKERS else 0.0 for t in tickers])
    return eq, bd


def _slsqp_constraints(tickers, cons: Constraints, cov: np.ndarray | None = None):
    eq, bd = _group_vectors(tickers)
    out = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if cons.max_equity < 1.0:
        out.append({"type": "ineq", "fun": lambda w: cons.max_equity - eq @ w})
    if cons.min_bonds > 0.0:
        out.append({"type": "ineq", "fun": lambda w: bd @ w - cons.min_bonds})
    if cons.max_vol is not None and cov is not None:
        out.append({"type": "ineq",
                    "fun": lambda w: cons.max_vol ** 2 - w @ cov @ w})
    return out


def _bounds(tickers: list[str], cons: Constraints) -> list[tuple[float, float]]:
    caps = dict(cons.fund_caps)
    return [(cons.min_weight, caps.get(t, cons.max_weight)) for t in tickers]


def _start_point(tickers: list[str], cons: Constraints) -> np.ndarray:
    lo, hi = np.array(_bounds(tickers, cons)).T
    w = np.clip(np.full(len(tickers), 1.0 / len(tickers)), lo, hi)
    return w / w.sum()


def check_constraints(w: pd.Series, cons: Constraints, cov: pd.DataFrame | None = None,
                      tol: float = 1e-4) -> list[str]:
    """Plain-language list of constraint violations (empty list = all satisfied)."""
    issues = []
    eq, bd = _group_vectors(list(w.index))
    if abs(w.sum() - 1) > tol:
        issues.append(f"Weights sum to {w.sum():.2%}, not 100%.")
    if (w < cons.min_weight - tol).any():
        issues.append(f"A weight is below the {cons.min_weight:.0%} minimum.")
    caps = dict(cons.fund_caps)
    general = w.drop([t for t in caps if t in w.index])
    if (general > cons.max_weight + tol).any():
        issues.append(f"{general.idxmax()} exceeds the {cons.max_weight:.0%} single-fund cap.")
    for t, cap in cons.fund_caps:
        if w.get(t, 0.0) > cap + tol:
            issues.append(f"{t} is {w[t]:.1%}, above its {cap:.0%} cap.")
    if eq @ w.values > cons.max_equity + tol:
        issues.append(f"Equity is {eq @ w.values:.0%}, above the {cons.max_equity:.0%} cap.")
    if bd @ w.values < cons.min_bonds - tol:
        issues.append(f"Bonds are {bd @ w.values:.0%}, below the {cons.min_bonds:.0%} floor.")
    if cons.max_vol is not None and cov is not None:
        vol = float(np.sqrt(w.values @ cov.values @ w.values))
        if vol > cons.max_vol + tol:
            issues.append(f"Volatility is {vol:.1%}, above the {cons.max_vol:.1%} ceiling.")
    return issues


# ---------------------------------------------------------------------------
# Optimisers
# ---------------------------------------------------------------------------
def _solve(objective, n, tickers, cons, cov=None, jac=None):
    bounds = _bounds(tickers, cons)
    res = optimize.minimize(objective, _start_point(tickers, cons), method="SLSQP", jac=jac,
                            bounds=bounds, constraints=_slsqp_constraints(tickers, cons, cov),
                            options={"maxiter": 1000, "ftol": 1e-12})
    w = np.clip(res.x, 0.0, None)
    return w / w.sum(), bool(res.success), str(res.message)


def min_variance(cov: pd.DataFrame, cons: Constraints) -> OptResult:
    c = cov.values
    w, ok, msg = _solve(lambda w: w @ c @ w, len(c), list(cov.index), cons, c,
                        jac=lambda w: 2 * c @ w)
    return OptResult(pd.Series(w, cov.index), "Minimum variance", ok, msg)


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, cons: Constraints, rf: float = 0.0) -> OptResult:
    c, m = cov.values, mu.values

    def neg_sharpe(w):
        return -(w @ m - rf) / np.sqrt(w @ c @ w)

    w, ok, msg = _solve(neg_sharpe, len(c), list(cov.index), cons, c)
    return OptResult(pd.Series(w, cov.index), "Maximum Sharpe ratio", ok, msg)


def risk_parity(cov: pd.DataFrame, cons: Constraints) -> OptResult:
    """Equal risk contribution: each fund contributes the same share of variance.
    If the allocation limits make exact parity impossible, the closest feasible
    portfolio is returned."""
    c = cov.values
    n = len(c)

    def obj(w):
        rc = w * (c @ w)
        return np.sum((rc - (w @ c @ w) / n) ** 2) * 1e4

    w, ok, msg = _solve(obj, n, list(cov.index), cons, c)
    return OptResult(pd.Series(w, cov.index), "Risk parity (equal risk contribution)", ok, msg)


def min_cvar(simple_ret: pd.DataFrame, cons: Constraints, confidence: float = 0.95) -> OptResult:
    """Minimise historical Expected Shortfall via the Rockafellar-Uryasev (2000)
    linear programme.  Variables: weights w, VaR level a, tail excesses u_t >= 0.

        min  a + 1 / ((1 - c) T) * sum(u_t)
        s.t. u_t >= -r_t.w - a,   sum(w) = 1,  group and bound limits

    The volatility ceiling is non-linear and is not imposed here (checked afterwards)."""
    r = simple_ret.dropna().values
    t, n = r.shape
    tickers = list(simple_ret.columns)
    eq, bd = _group_vectors(tickers)
    cost = np.concatenate([np.zeros(n), [1.0], np.full(t, 1.0 / ((1 - confidence) * t))])
    a_tail = sparse.hstack([sparse.csr_matrix(-r), -np.ones((t, 1)), -sparse.eye(t)])
    rows, rhs = [a_tail], [np.zeros(t)]
    if cons.max_equity < 1.0:
        rows.append(sparse.csr_matrix(np.concatenate([eq, [0.0], np.zeros(t)])))
        rhs.append([cons.max_equity])
    if cons.min_bonds > 0.0:
        rows.append(sparse.csr_matrix(np.concatenate([-bd, [0.0], np.zeros(t)])))
        rhs.append([-cons.min_bonds])
    a_ub = sparse.vstack(rows).tocsr()
    b_ub = np.concatenate([np.atleast_1d(x) for x in rhs])
    a_eq = sparse.csr_matrix(np.concatenate([np.ones(n), [0.0], np.zeros(t)]))
    bounds = _bounds(tickers, cons) + [(None, None)] + [(0, None)] * t
    res = optimize.linprog(cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[1.0],
                           bounds=bounds, method="highs")
    if not res.success:
        return OptResult(pd.Series(np.full(n, 1 / n), tickers),
                         "Minimum CVaR (Expected Shortfall)", False, res.message)
    w = np.clip(res.x[:n], 0, None)
    return OptResult(pd.Series(w / w.sum(), tickers), "Minimum CVaR (Expected Shortfall)",
                     True, "Optimal")


def build_portfolio(method: str, simple_ret: pd.DataFrame, mu: pd.Series, cov: pd.DataFrame,
                    cons: Constraints, rf: float = 0.0, cvar_conf: float = 0.95,
                    custom: dict | None = None) -> OptResult:
    tickers = list(simple_ret.columns)
    if method == "Minimum variance":
        return min_variance(cov, cons)
    if method == "Maximum Sharpe ratio":
        return max_sharpe(mu, cov, cons, rf)
    if method == "Risk parity (equal risk contribution)":
        return risk_parity(cov, cons)
    if method == "Minimum CVaR (Expected Shortfall)":
        return min_cvar(simple_ret, cons, cvar_conf)
    if method == RECOMMENDED:
        w = pd.Series(RECOMMENDED_WEIGHTS, dtype=float).reindex(tickers).fillna(0.0)
        return OptResult(w / w.sum(), method, True, "Fixed team weights (src/config.py)")
    if method == "Custom weights":
        w = pd.Series(custom or {}, dtype=float).reindex(tickers).fillna(0.0)
        if w.sum() <= 0:
            w[:] = 1.0
        return OptResult(w / w.sum(), method, True, "User-defined weights (normalised to 100%)")
    return OptResult(pd.Series(1.0 / len(tickers), tickers), "Equal weight", True, "1/N")


# ---------------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------------
def efficient_frontier(mu: pd.Series, cov: pd.DataFrame, cons: Constraints,
                       n_points: int = 40) -> pd.DataFrame:
    """Minimum-variance frontier under the allocation limits (volatility cap not applied)."""
    tickers = list(mu.index)
    n = len(tickers)
    c, m = cov.values, mu.values
    base = Constraints(cons.min_weight, cons.max_weight, cons.max_equity, cons.min_bonds, None,
                       cons.fund_caps)
    w_min = min_variance(cov, base).weights.values
    # highest achievable return under the linear limits
    eq, bd = _group_vectors(tickers)
    a_ub, b_ub = [], []
    if base.max_equity < 1:
        a_ub.append(eq); b_ub.append(base.max_equity)
    if base.min_bonds > 0:
        a_ub.append(-bd); b_ub.append(-base.min_bonds)
    lp = optimize.linprog(-m, A_ub=np.array(a_ub) if a_ub else None,
                          b_ub=b_ub if b_ub else None, A_eq=np.ones((1, n)), b_eq=[1.0],
                          bounds=_bounds(tickers, base), method="highs")
    r_lo, r_hi = float(w_min @ m), float(-lp.fun) if lp.success else float(m.max())
    rows = []
    for target in np.linspace(r_lo, r_hi, n_points):
        cs = _slsqp_constraints(tickers, base) + [
            {"type": "eq", "fun": lambda w, tr=target: w @ m - tr}]
        res = optimize.minimize(lambda w: w @ c @ w, w_min, method="SLSQP",
                                jac=lambda w: 2 * c @ w,
                                bounds=_bounds(tickers, base),
                                constraints=cs, options={"maxiter": 500, "ftol": 1e-12})
        if res.success:
            rows.append({"return": target, "volatility": float(np.sqrt(res.x @ c @ res.x))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Portfolio returns and CAPM
# ---------------------------------------------------------------------------
def portfolio_returns(simple_ret: pd.DataFrame, weights: pd.Series,
                      rebalance: str = "daily") -> pd.Series:
    """Portfolio simple returns.

    rebalance="daily": constant weights (the standard assumption in VaR models).
    rebalance="none": buy-and-hold, weights drift with prices."""
    w = weights.reindex(simple_ret.columns).fillna(0.0)
    if rebalance == "none":
        wealth = (1 + simple_ret).cumprod().mul(w, axis=1).sum(axis=1)
        return wealth.pct_change().fillna(wealth.iloc[0] - 1).rename("Portfolio")
    return simple_ret.mul(w, axis=1).sum(axis=1).rename("Portfolio")


def capm(port: pd.Series, market: pd.Series, rf_daily: pd.Series) -> dict:
    """OLS of portfolio excess returns on market excess returns:
    (R_p - R_f) = alpha + beta (R_m - R_f) + e."""
    df = pd.concat([port, market, rf_daily], axis=1, keys=["p", "m", "rf"]).dropna()
    y = (df["p"] - df["rf"]).values
    x = (df["m"] - df["rf"]).values
    xmat = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(xmat, y, rcond=None)
    resid = y - xmat @ coef
    dof = len(y) - 2
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(xmat.T @ xmat)))
    r2 = 1 - (resid @ resid) / np.sum((y - y.mean()) ** 2)
    return {"alpha_daily": coef[0], "alpha_annual": coef[0] * TRADING_DAYS,
            "beta": coef[1], "alpha_t": coef[0] / se[0], "beta_t": coef[1] / se[1],
            "r_squared": r2, "residual_vol_annual": np.sqrt(s2 * TRADING_DAYS)}
