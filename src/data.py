"""Data layer: download, cleaning, return construction and statistical diagnostics.

Return conventions used throughout the project
- Log returns  ln(P_t / P_{t-1}) are used for statistical work: distribution tests,
  ADF tests, VaR/ES/EVT estimation and backtesting (they are time-additive and
  closer to symmetric).
- Simple returns  P_t / P_{t-1} - 1 are used wherever returns are combined across
  assets (portfolio returns, optimisation, stress tests, risk contributions),
  because only simple returns add up linearly across portfolio weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import DATA_START, TICKERS, TRADING_DAYS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICES_SNAPSHOT = DATA_DIR / "prices.csv"
RF_SNAPSHOT = DATA_DIR / "rf.csv"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
RF_SERIES = "DTB3"  # 3-month US Treasury bill, secondary market rate (annualised %)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_prices(tickers: list[str] = TICKERS, start: str = DATA_START,
                    end: str | None = None) -> pd.DataFrame:
    """Adjusted close prices from Yahoo Finance (auto_adjust=True adjusts for
    dividends and splits)."""
    import yfinance as yf  # imported lazily so the maths modules work without it

    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True,
                      progress=False, group_by="column", threads=True)
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no data. Check your internet connection.")
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        px = raw["Close"] if "Close" in level0 else raw.xs("Close", axis=1, level=1)
    else:  # single ticker
        px = raw[["Close"]].rename(columns={"Close": tickers[0]})
    px = px.reindex(columns=list(tickers))
    px.index = pd.to_datetime(px.index)
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    px.index.name = "Date"
    return px.sort_index()


def download_rf(start: str = DATA_START, end: str | None = None,
                series: str = RF_SERIES) -> pd.Series:
    """Risk-free rate from FRED (public CSV endpoint, no API key). Annualised percent."""
    df = pd.read_csv(FRED_URL.format(series=series))
    date_col, value_col = df.columns[0], df.columns[-1]
    s = pd.Series(pd.to_numeric(df[value_col], errors="coerce").values,
                  index=pd.to_datetime(df[date_col]), name=series)
    s = s.loc[start:] if end is None else s.loc[start:end]
    s.index.name = "Date"
    return s


def save_snapshot(prices: pd.DataFrame, rf: pd.Series) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    prices.to_csv(PRICES_SNAPSHOT)
    rf.to_csv(RF_SNAPSHOT)


def load_snapshot() -> tuple[pd.DataFrame, pd.Series]:
    prices = pd.read_csv(PRICES_SNAPSHOT, index_col=0, parse_dates=True)
    rf = pd.read_csv(RF_SNAPSHOT, index_col=0, parse_dates=True).iloc[:, 0]
    return prices, rf


def load_market_data(source: str = "snapshot", start: str = DATA_START,
                     end: str | None = None) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load prices and the risk-free rate, falling back gracefully.

    source="snapshot" uses the frozen CSVs in data/ (exactly reproducible);
    source="live" downloads fresh data. Every fallback is reported in `notes`.
    """
    notes: list[str] = []
    prices = rf = None
    if source == "snapshot" and PRICES_SNAPSHOT.exists() and RF_SNAPSHOT.exists():
        prices, rf = load_snapshot()
        notes.append(f"Loaded frozen data snapshot from {PRICES_SNAPSHOT.parent.name}/.")
    else:
        if source == "snapshot":
            notes.append("No snapshot found in data/, so data was downloaded live.")
        try:
            prices = download_prices(start=start, end=end)
            notes.append("Prices downloaded live from Yahoo Finance (yfinance).")
        except Exception as exc:  # network or API failure
            if PRICES_SNAPSHOT.exists():
                prices, rf = load_snapshot()
                notes.append(f"Live download failed ({exc}); using the snapshot instead.")
            else:
                raise
        if rf is None:
            try:
                rf = download_rf(start=start, end=end)
                notes.append(f"Risk-free rate downloaded live from FRED ({RF_SERIES}).")
            except Exception as exc:
                rf = pd.Series(dtype=float, name=RF_SERIES)
                notes.append(f"FRED download failed ({exc}); a constant 2% risk-free rate is used.")
    prices = prices.loc[start:] if end is None else prices.loc[start:end]
    rf = rf.loc[start:] if end is None else rf.loc[start:end]
    return prices, rf, notes


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
@dataclass
class CleaningResult:
    prices: pd.DataFrame
    flags: pd.DataFrame
    log: list[str] = field(default_factory=list)


def _flag(rows: list, date, ticker, issue, detail, action):
    rows.append({"date": pd.Timestamp(date).date(), "ticker": ticker, "issue": issue,
                 "detail": detail, "action": action})


def clean_prices(raw: pd.DataFrame, max_ffill: int = 2, stale_days: int = 5,
                 jump_mad_z: float = 10.0) -> CleaningResult:
    """Clean adjusted prices. Suspected errors are FLAGGED, not silently removed.

    Steps (each is logged):
    1. Sort, remove duplicated dates.
    2. Non-positive prices are set to missing (a log return cannot be computed).
    3. Start the sample on the first date where every fund has a price.
    4. Calendar alignment: all seven ETFs trade on US exchanges, so they share the
       NYSE calendar. Isolated missing values (up to `max_ffill` days) are
       forward-filled (equivalent to a zero return on that day); dates still
       missing a price after that are dropped.
    5. Stale prices: runs of `stale_days`+ identical closes are flagged (kept).
    6. Implausible jumps: daily log returns more than `jump_mad_z` robust
       standard deviations (median absolute deviation) from the median are
       flagged and kept, because real crashes are exactly what tail-risk models
       must see.
    """
    log: list[str] = []
    rows: list[dict] = []
    px = raw.copy().sort_index()

    dup = px.index.duplicated(keep="last")
    if dup.any():
        log.append(f"Removed {dup.sum()} duplicated date rows (kept the last value).")
        px = px[~dup]

    nonpos = px <= 0
    for d, t in zip(*np.where(nonpos.values)):
        _flag(rows, px.index[d], px.columns[t], "Non-positive price",
              f"{px.iat[d, t]:.4f}", "Set to missing")
    px = px.mask(nonpos)

    first_valid = px.apply(lambda s: s.first_valid_index())
    common_start = first_valid.max()
    youngest = [t for t in px.columns if first_valid[t] == common_start]
    if len(youngest) == len(px.columns):
        log.append(f"Common sample starts {common_start.date()}: every fund has a price from this date, "
                   f"because the data begins on {DATA_START}, the first trading day of GLD, the youngest fund.")
    else:
        log.append(f"Common sample starts {common_start.date()} (limited by {', '.join(youngest)}, "
                   f"the youngest fund).")
    px = px.loc[common_start:]

    missing = px.isna()
    n_missing = int(missing.values.sum())
    for d, t in zip(*np.where(missing.values)):
        _flag(rows, px.index[d], px.columns[t], "Missing price", "No close on this date",
              f"Forward-filled (max {max_ffill} days)")
    px = px.ffill(limit=max_ffill)
    still = px.isna().any(axis=1)
    if still.any():
        log.append(f"Dropped {still.sum()} dates that still had missing prices after forward-fill.")
        px = px[~still]
    log.append(f"{n_missing} missing price points found; isolated gaps of up to "
               f"{max_ffill} days were forward-filled.")

    for t in px.columns:
        same = px[t].diff().eq(0)
        run_id = (~same).cumsum()
        run_len = same.groupby(run_id).transform("sum")
        starts = same & ~same.shift(1, fill_value=False) & (run_len >= stale_days - 1)
        for d in px.index[starts]:
            _flag(rows, d, t, "Stale price",
                  f"Unchanged for {int(run_len.loc[d]) + 1} days", "Kept, flagged")

    lr = np.log(px).diff()
    for t in px.columns:
        x = lr[t].dropna()
        mad = stats.median_abs_deviation(x, scale="normal")
        z = (x - x.median()) / mad
        for d in z.index[np.abs(z) > jump_mad_z]:
            _flag(rows, d, t, "Extreme move",
                  f"Daily log return {x.loc[d]:+.2%} ({z.loc[d]:+.1f} robust s.d.)",
                  "Kept, flagged (genuine crisis moves are retained)")

    flags = pd.DataFrame(rows, columns=["date", "ticker", "issue", "detail", "action"])
    log.append(f"Final sample: {len(px)} price dates ({len(px) - 1} daily returns), {px.index[0].date()} to "
               f"{px.index[-1].date()}; {len(flags)} data points flagged for review.")
    return CleaningResult(prices=px, flags=flags, log=log)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff().dropna(how="all")


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def daily_risk_free(rf_annual_pct: pd.Series, index: pd.DatetimeIndex,
                    fallback_annual: float = 0.02) -> pd.Series:
    """Convert an annualised % T-bill yield to a daily rate on the trading calendar."""
    if rf_annual_pct is None or rf_annual_pct.dropna().empty:
        annual = pd.Series(fallback_annual, index=index)
    else:
        annual = (rf_annual_pct.reindex(index.union(rf_annual_pct.index)).ffill()
                  .reindex(index).bfill() / 100.0)
    return ((1.0 + annual) ** (1.0 / TRADING_DAYS) - 1.0).rename("rf")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def adf_table(returns: pd.DataFrame) -> pd.DataFrame:
    """Augmented Dickey-Fuller test (H0: unit root, i.e. non-stationary)."""
    from statsmodels.tsa.stattools import adfuller

    out = []
    for col in returns.columns:
        x = returns[col].dropna().values
        stat, p, lags, nobs, crit, _ = adfuller(x, autolag="AIC")
        out.append({"series": col, "ADF statistic": stat, "p-value": p, "lags used": lags,
                    "5% critical value": crit["5%"],
                    "result": "Stationary (unit root rejected)" if p < 0.05
                    else "Cannot reject unit root"})
    return pd.DataFrame(out).set_index("series")


def describe_returns(log_ret: pd.DataFrame) -> pd.DataFrame:
    """Distribution summary. Jarque-Bera tests normality (H0: skew 0, excess kurtosis 0)."""
    out = []
    for col in log_ret.columns:
        x = log_ret[col].dropna()
        jb = stats.jarque_bera(x)
        out.append({
            "series": col,
            "ann. mean (log)": x.mean() * TRADING_DAYS,
            "ann. volatility": x.std() * np.sqrt(TRADING_DAYS),
            "skewness": stats.skew(x),
            "excess kurtosis": stats.kurtosis(x),
            "worst day": x.min(),
            "best day": x.max(),
            "Jarque-Bera p-value": jb.pvalue,
            "normal?": "No" if jb.pvalue < 0.05 else "Cannot reject",
        })
    return pd.DataFrame(out).set_index("series")
