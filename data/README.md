# Data snapshot

`prices.csv` (adjusted close, Yahoo Finance) and `rf.csv` (FRED DTB3) are created by:

```bash
python -m scripts.fetch_data --end 2026-10-30
```

Commit both files. The app's default "Frozen snapshot" mode reads them, so anyone can reproduce
exactly the numbers in the report. Switch to "Live download" in the sidebar for fresh data.
