"""Synthetic data with known properties, used to check the maths."""
import numpy as np
import pandas as pd

from src.config import TICKERS


def synthetic_returns(n=3000, seed=0, df=None):
    """Correlated daily log returns for the 7 funds (Normal, or Student-t if df given)."""
    rng = np.random.default_rng(seed)
    vols = np.array([0.18, 0.20, 0.25, 0.05, 0.16, 0.28, 0.24]) / np.sqrt(252)
    corr = np.full((7, 7), 0.6)
    corr[3, :] = corr[:, 3] = -0.1
    corr[4, :] = corr[:, 4] = 0.05
    corr[3, 4] = corr[4, 3] = 0.2
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vols, vols) * corr
    z = rng.multivariate_normal(np.zeros(7), cov, size=n)
    if df:
        g = rng.chisquare(df, size=(n, 1))
        z = z * np.sqrt((df - 2) / g)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame(z + 0.0003, index=idx, columns=TICKERS)


def synthetic_prices(n=3000, seed=0, df=None):
    lr = synthetic_returns(n, seed, df)
    return 100 * np.exp(lr.cumsum())
