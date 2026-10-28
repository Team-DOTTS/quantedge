"""Print the recommended mix against the alternatives, and the robustness
evidence behind it.

Usage (from the repository root):
    python -m scripts.compare_portfolios                  # frozen snapshot in data/
    python -m scripts.compare_portfolios --source live    # fresh download
    python -m scripts.compare_portfolios --out comparison.csv
"""
import argparse

import pandas as pd

from src import compare, data
from src.config import DATA_START, MANDATE

PCT_COLS = ["CAGR", "Excess over T-bills", "Volatility", "Max drawdown", "CVaR 95% 1d",
            "CVaR 99% 1d", "2008 replay", "2008 max DD", "2020 replay", "2022 replay",
            "Satellite share"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["snapshot", "live"], default="snapshot")
    ap.add_argument("--start", default=DATA_START)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=None, help="Also write the metrics table to this CSV file")
    args = ap.parse_args()

    prices, rf, notes = data.load_market_data(args.source, args.start, args.end)
    sr = data.simple_returns(data.clean_prices(prices).prices)
    rf_daily = data.daily_risk_free(rf, sr.index)
    weights, metrics = compare.comparison(sr, rf_daily)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n".join(notes))
    print(f"Sample {sr.index[0].date()} to {sr.index[-1].date()} ({len(sr)} days)\n")
    print("Weights")
    print(weights.map(lambda x: f"{x:.1%}").to_string(), "\n")

    shown = metrics.copy().astype(object)
    for c in PCT_COLS:
        shown[c] = metrics[c].map(lambda x: f"{x:.1%}")
    for c in ("Sharpe", "Sortino"):
        shown[c] = metrics[c].map(lambda x: f"{x:.2f}")
    lim = MANDATE["risk_limits"]
    print(f"Metrics (limits: excess >= {MANDATE['return_excess_over_rf']:.1%}, "
          f"vol <= {lim['max_volatility']:.0%}, max DD <= {lim['max_drawdown']:.0%})")
    print(shown.T.to_string(), "\n")

    rec = weights.loc["Recommended"]
    print(f"Turnover from Recommended to Constrained Markowitz: "
          f"{compare.distance(rec, weights.loc['Constrained Markowitz']):.1%}\n")

    rf_annual = float(rf_daily.mean() * 252)
    boot = pd.DataFrame({n: compare.bootstrap_drawdowns(sr, weights.loc[n], rf_annual)
                         for n in ("Recommended", "First draft", "60/40 SPY/AGG")})
    print("5-year block bootstrap")
    print(boot.map(lambda x: f"{x:.1%}").to_string(), "\n")

    for split in ("2012-01-01", "2016-01-01"):
        wf = compare.walk_forward(sr, rf_daily, split)
        pick = ", ".join(f"{t} {v:.0%}" for t, v in wf["pick"].items() if v > 0)
        print(f"Walk-forward, chosen before {split}: {pick}; Sortino {wf['in']['sortino']:.2f} -> "
              f"{wf['out']['sortino']:.2f} (recommended after: {wf['recommended_out']['sortino']:.2f})")
    print()

    facts = compare.fact_check(sr)
    print("Facts quoted in the memo")
    print("\n".join(f"  {k}: {v:.2f}" if "correlation" in k else f"  {k}: {v:.1%}" for k, v in facts.items()))

    if args.out:
        pd.concat([weights.add_prefix("w_"), metrics], axis=1).to_csv(args.out)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
