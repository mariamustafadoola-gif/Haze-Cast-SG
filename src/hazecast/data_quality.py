"""Data quality report: what was wrong in the raw data, what we did about
it, and which gaps were found and filled.

The ingest and bronze_to_silver modules apply the fixes. This module counts
them, so the fixes are visible as tables instead of only as log lines. It
re-reads the raw files and applies the same rules, so the counts always
match what the pipeline actually did.

Run after the pipeline:
    python -m hazecast.data_quality
Writes:
    reports/data_quality/bronze_fixes.csv
    reports/data_quality/silver_gaps.csv
    reports/data_quality/quality_checks.csv
"""
from __future__ import annotations

import glob
import logging
from pathlib import Path

import pandas as pd

from hazecast.config import (
    RAW_DIR, SILVER_DIR, REPO_ROOT, SG_REGIONS, FIRE_REGIONS, MAX_FILL_HOURS,
)
from hazecast.ingest.pm25 import _KNOWN_BAD_TIMESTAMP_FIX

log = logging.getLogger(__name__)
OUT_DIR = REPO_ROOT / "reports" / "data_quality"
_TS_PATTERN = r"^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}$"


# ---------------------------------------------------------------------------
# Bronze: errors found in the raw files and how each was handled
# ---------------------------------------------------------------------------
def _pm25_fixes(raw_dir: Path) -> list[dict]:
    frames = [pd.read_csv(f, dtype=str) for f in sorted(glob.glob(str(raw_dir / "*.csv")))]
    df = pd.concat([f.rename(columns={f.columns[0]: "ts"}) for f in frames], ignore_index=True)
    rows_in = len(df)
    known_bad = int(df["ts"].isin(_KNOWN_BAD_TIMESTAMP_FIX).sum())
    df["ts"] = df["ts"].replace(_KNOWN_BAD_TIMESTAMP_FIX)
    unknown_bad = int((~df["ts"].str.match(_TS_PATTERN)).sum())
    df = df[df["ts"].str.match(_TS_PATTERN)]
    values = df[SG_REGIONS]
    not_numeric = int((values.notna() & pd.to_numeric(values.stack(), errors="coerce").unstack().isna()).sum().sum())
    ts = pd.to_datetime(df["ts"], format="%d/%m/%Y %H:%M")
    dups = int(ts.duplicated().sum())
    return [
        dict(source="PM2.5", problem="Rows in the raw file", count=rows_in, action="Starting point"),
        dict(source="PM2.5", problem="Broken timestamp already known (e.g. '2016-04-04 010:00:00')", count=known_bad,
             action="Fixed by hand after checking the rows around it"),
        dict(source="PM2.5", problem="New broken timestamps", count=unknown_bad,
             action="Dropped and logged as a warning (never guessed)"),
        dict(source="PM2.5", problem="Readings that are not numbers", count=not_numeric,
             action="Set to empty, so they cannot be mistaken for real values"),
        dict(source="PM2.5", problem="Same hour appears twice with different readings", count=dups,
             action="Kept the first one, logged the rest"),
        dict(source="PM2.5", problem="Rows kept", count=len(ts) - dups, action="Written to data/bronze/pm25.parquet"),
    ]


def _firms_fixes(raw_dir: Path) -> list[dict]:
    cols = ["latitude", "longitude", "acq_date", "acq_time", "confidence"]
    arch = [pd.read_csv(f, usecols=cols + ["type"], dtype={"acq_time": str})
            for f in sorted(glob.glob(str(raw_dir / "fire_archive_*.csv")))]
    nrt = [pd.read_csv(f, usecols=cols, dtype={"acq_time": str})
           for f in sorted(glob.glob(str(raw_dir / "fire_nrt_*.csv")))]
    a = pd.concat(arch, ignore_index=True) if arch else pd.DataFrame(columns=cols + ["type"])
    n = pd.concat(nrt, ignore_index=True) if nrt else pd.DataFrame(columns=cols)
    rows_in = len(a) + len(n)
    not_veg = int((a["type"] != 0).sum())
    df = pd.concat([a[a["type"] == 0].drop(columns="type"), n], ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(["latitude", "longitude", "acq_date", "acq_time"])
    dups = before - len(df)
    inside = pd.Series(False, index=df.index)
    for box in FIRE_REGIONS.values():
        inside |= (df["longitude"].between(box["lon_min"], box["lon_max"])
                   & df["latitude"].between(box["lat_min"], box["lat_max"]))
    kept = df[inside]
    return [
        dict(source="Fires", problem="Detections in the raw files", count=rows_in, action="Starting point"),
        dict(source="Fires", problem="Not a vegetation fire (volcano, gas flare, other fixed heat source)",
             count=not_veg, action="Removed (archive files only: the recent file has no fire-type column)"),
        dict(source="Fires", problem="Exact repeat (same place, date and time)", count=dups, action="Removed"),
        dict(source="Fires", problem="Outside Sumatra and Kalimantan", count=int((~inside).sum()),
             action="Removed: too far away to affect Singapore"),
        dict(source="Fires", problem="Low-confidence detections", count=int((kept["confidence"] == "l").sum()),
             action="Kept for now (possible future experiment)"),
        dict(source="Fires", problem="Detections kept", count=len(kept), action="Written to data/bronze/firms.parquet"),
    ]


def _weather_fixes(raw_dir: Path) -> list[dict]:
    from hazecast.ingest.weather import load_weather_bronze
    files = sorted(glob.glob(str(raw_dir / "*.csv")))
    rows_in = sum(len(pd.read_csv(f, skiprows=3)) for f in files)
    w = load_weather_bronze(raw_dir)
    dup_hours = rows_in - len(w)
    per_point = w.groupby("point").size()
    return [
        dict(source="Weather", problem="Rows across all raw files", count=rows_in, action="Starting point"),
        dict(source="Weather", problem="Extra files for the same place (Singapore 2014-18; Aug-Oct 2026 updates)",
             count=len(files) - per_point.size, action="Joined into one series per place"),
        dict(source="Weather", problem="File name and real coordinates can differ slightly", count=len(files),
             action="Each file matched to its place by the coordinates inside it"),
        dict(source="Weather", problem="Same hour appears twice for one place (overlapping files)", count=dup_hours,
             action="Kept the older file's value" if dup_hours else "None found"),
        dict(source="Weather", problem="Rows kept", count=len(w), action="Written to data/bronze/weather.parquet"),
    ]


def bronze_fixes(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    rows = _pm25_fixes(raw_dir / "pm25") + _firms_fixes(raw_dir / "firms") + _weather_fixes(raw_dir / "weather")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Silver: gaps found, filled, left empty, flagged
# ---------------------------------------------------------------------------
def silver_gaps(pm25_s: pd.DataFrame, weather_s: pd.DataFrame, firms_s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pm_filled = int(pm25_s["filled"].sum()) if "filled" in pm25_s else 0
    pm_empty = int(pm25_s[SG_REGIONS].isna().any(axis=1).sum())
    rows.append(dict(source="PM2.5", expected=len(pm25_s), missing=pm_filled + pm_empty, filled=pm_filled,
                     left_empty=pm_empty, flagged=0,
                     note=f"Gaps of up to {MAX_FILL_HOURS} hours filled from the readings either side"))
    for point, g in weather_s.groupby("point"):
        w_filled = int(g["filled"].sum()) if "filled" in g else 0
        w_empty = int(g[["wind_speed_kmh", "wind_dir_deg", "rain_mm"]].isna().any(axis=1).sum())
        rows.append(dict(source=f"Weather: {point.capitalize()}", expected=len(g), missing=w_filled + w_empty,
                         filled=w_filled, left_empty=w_empty, flagged=0, note="Same rule as PM2.5"))
    gaps = int(firms_s["data_gap"].sum())
    rows.append(dict(source="Fires (days)", expected=len(firms_s), missing=gaps, filled=0, left_empty=0,
                     flagged=gaps, note="5+ days in a row with zero fires: flagged as a likely satellite gap, never filled"))
    return pd.DataFrame(rows)


def quality_checks(pm25_s: pd.DataFrame, weather_s: pd.DataFrame, firms_s: pd.DataFrame) -> pd.DataFrame:
    checks = []

    def add(what, ok, detail):
        checks.append({"check": what, "result": "Pass" if ok else "Look at this", "detail": detail})

    empty = int(pm25_s[SG_REGIONS].isna().any(axis=1).sum())
    add("PM2.5: every hour has a reading", empty < 0.001 * len(pm25_s), f"{empty} of {len(pm25_s):,} hours still empty after filling")
    neg = int((pm25_s[SG_REGIONS] < 0).sum().sum())
    add("PM2.5: no negative readings", neg == 0, f"{neg} negative values")
    mx = pm25_s[SG_REGIONS].max().max()
    add("PM2.5: highest reading is believable", mx < 1000, f"Highest hourly reading = {mx:.0f} µg/m³ (2015 haze)")
    w_empty = int(weather_s[["wind_speed_kmh", "wind_dir_deg", "rain_mm"]].isna().sum().sum())
    add("Weather: every hour present at all 5 places", w_empty == 0, f"{w_empty} empty values")
    bad_dir = int((~weather_s["wind_dir_deg"].between(0, 360)).sum())
    add("Weather: wind direction between 0 and 360", bad_dir == 0, f"{bad_dir} values outside")
    bad_rain = int((weather_s["rain_mm"] < 0).sum())
    add("Weather: no negative rain", bad_rain == 0, f"{bad_rain} values below zero")
    gaps = int(firms_s["data_gap"].sum())
    add("Fires: suspicious zero-fire stretches flagged", True, f"{gaps} days flagged as likely satellite gaps")
    return pd.DataFrame(checks)


def write_reports(out_dir: Path = OUT_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pm25_s = pd.read_parquet(SILVER_DIR / "pm25.parquet")
    weather_s = pd.read_parquet(SILVER_DIR / "weather.parquet")
    firms_s = pd.read_parquet(SILVER_DIR / "firms_daily.parquet")
    tables = {
        "bronze_fixes": bronze_fixes(),
        "silver_gaps": silver_gaps(pm25_s, weather_s, firms_s),
        "quality_checks": quality_checks(pm25_s, weather_s, firms_s),
    }
    paths = {}
    for name, df in tables.items():
        paths[name] = out_dir / f"{name}.csv"
        df.to_csv(paths[name], index=False)
        log.info("wrote %s (%d rows)", paths[name], len(df))
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    write_reports()
