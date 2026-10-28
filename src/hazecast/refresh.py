"""One command to bring all data up to now: fetch live data, then run the
whole pipeline (bronze, silver, gold) and the data quality report.

    python -m hazecast.refresh            # fetch the last 30 days, rebuild everything
    python -m hazecast.refresh --days 7
    python -m hazecast.refresh --no-fetch # rebuild from the files already in data/raw/

At the end it prints the newest hour in the gold table, which is the "now"
the hourly forecast starts from.
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from hazecast.config import GOLD_DIR, SILVER_DIR
from hazecast.ingest import pm25, weather, firms, live
from hazecast import bronze_to_silver as b2s, silver_to_gold as s2g, data_quality

log = logging.getLogger("hazecast.refresh")


def run(days: int = 30, fetch: bool = True) -> pd.DataFrame:
    if fetch:
        live.fetch_all(days)
    pm25.write_bronze(pm25.load_pm25_bronze())
    weather.write_bronze(weather.load_weather_bronze())
    firms.write_bronze(firms.load_firms_bronze())
    pm, we, fi = b2s.pm25_silver(), b2s.weather_silver(), b2s.firms_daily_silver()
    fh = b2s.firms_hourly_silver()
    b2s.write_silver(pm, "pm25"); b2s.write_silver(we, "weather"); b2s.write_silver(fi, "firms_daily")
    b2s.write_silver(fh, "firms_hourly")
    s2g.write_gold(s2g.build_gold(pm, we, fi))
    gold = s2g.build_gold_hourly(pm, we, fi, fh)
    s2g.write_gold_hourly(gold)
    data_quality.write_reports()
    return gold


def latest_complete_hour(gold: pd.DataFrame) -> pd.Timestamp:
    """Newest hour whose features are all filled in (the forecast's starting point)."""
    from hazecast.hourly_model import _feature_columns
    cols = _feature_columns(gold)
    ok = gold[cols].notna().all(axis=1)
    return gold.loc[ok, "date"].max()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Fetch live data and rebuild the pipeline.")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--no-fetch", action="store_true", help="skip downloading; rebuild from data/raw only")
    a = ap.parse_args()
    gold = run(a.days, fetch=not a.no_fetch)
    newest = latest_complete_hour(gold)
    print(f"\nDone. Gold table has {len(gold):,} hours. Newest hour ready to forecast from: {newest}")
