"""Has the stock-bond relationship changed? A test for an unknown break date.

Campbell, Pflueger and Viceira (2020) explain the sign of the stock-bond
correlation by the kind of shock that dominates: growth shocks make bonds a
hedge, inflation shocks make bonds fall with stocks. They date the change with a
Quandt likelihood ratio (QLR) test on daily returns (their eq. 34):

    r_bond,t = a + b I(t >= tau) + c r_stock,t + d I(t >= tau) r_stock,t + e_t

For each candidate break date tau in the middle 70% of the sample, the F-statistic
for d = 0 is computed with Newey-West standard errors (one lag). The QLR statistic
is the largest F; the break date is where it occurs (Andrews, 1993). Returns are
winsorised at 0.5% / 99.5% so a handful of crash days cannot drive the result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Critical values of the QLR statistic for one restriction and 15% trimming
# (Andrews, 1993; as tabulated in Stock and Watson, Introduction to Econometrics).
QLR_CRITICAL = {0.10: 7.12, 0.05: 8.68, 0.01: 12.16}


def _winsorise(s: pd.Series, p: float) -> pd.Series:
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)


def qlr_break(y: pd.Series, x: pd.Series, trim: float = 0.15, step: int = 5,
              winsor: float = 0.005, maxlags: int = 1) -> dict:
    """QLR test for a break in the slope of y on x. Returns the break date, the
    maximum F-statistic, the F path, and the slope and correlation either side."""
    df = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
    yw, xw = _winsorise(df["y"], winsor).values, _winsorise(df["x"], winsor).values
    n = len(df)
    pos = np.arange(n)
    f_path = {}
    for t in range(int(trim * n), int((1 - trim) * n), step):
        d = (pos >= t).astype(float)
        exog = sm.add_constant(np.column_stack([d, xw, d * xw]))
        fit = sm.OLS(yw, exog).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        f_path[df.index[t]] = float(fit.tvalues[3] ** 2)
    f = pd.Series(f_path, name="F")
    tau = f.idxmax()
    before, after = df.loc[df.index < tau], df.loc[df.index >= tau]

    def slope(part):
        return float(np.cov(part["y"], part["x"])[0, 1] / part["x"].var())

    return {"date": tau, "F": float(f.max()), "F_path": f,
            "beta_before": slope(before), "beta_after": slope(after),
            "corr_before": float(before["y"].corr(before["x"])),
            "corr_after": float(after["y"].corr(after["x"])),
            "significant_5pct": bool(f.max() > QLR_CRITICAL[0.05])}
