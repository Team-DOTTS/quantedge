"""Download the data once and freeze it in data/ so every run (and anyone checking our work)
sees exactly the same numbers.

Usage (from the repository root):
    python -m scripts.fetch_data              # up to today
    python -m scripts.fetch_data --end 2026-10-30
"""
import argparse

from src import data
from src.config import DATA_START


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=DATA_START)
    ap.add_argument("--end", default=None, help="Last date to include (YYYY-MM-DD)")
    args = ap.parse_args()
    prices = data.download_prices(start=args.start, end=args.end)
    rf = data.download_rf(start=args.start, end=args.end)
    data.save_snapshot(prices, rf)
    print(f"Saved {len(prices)} days of prices ({prices.index[0].date()} to {prices.index[-1].date()}) "
          f"and {rf.notna().sum()} risk-free observations to {data.DATA_DIR}/")
    print("Missing values per ticker:\n" + prices.isna().sum().to_string())


if __name__ == "__main__":
    main()
