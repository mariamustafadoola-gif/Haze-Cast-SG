"""Bronze -> silver: align to hourly Singapore-time grids, flag gaps, and
reshape into tidy long frames. No feature engineering happens here -- that
is silver -> gold's job. Silver's job is just "clean and trustworthy".
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

import pandas as pd

from hazecast.config import (BRONZE_DIR, RAW_DIR, SILVER_DIR, SG_REGIONS, MAX_FILL_HOURS,
                             FIRE_KNOWN_HOURS_AFTER_LAST_DETECTION, FIRE_LATENCY_HOURS)

log = logging.getLogger(__name__)


def fill_short_gaps(df: pd.DataFrame, cols: list[str], max_hours: int = MAX_FILL_HOURS,
                    carry_forward: tuple[str, ...] = ()) -> tuple[pd.DataFrame, pd.Series]:
    """Fill runs of up to `max_hours` missing hours; leave longer runs empty.

    df must have an hourly DatetimeIndex with no missing timestamps (i.e.
    already reindexed). Returns the filled frame and a boolean Series that
    is True for every hour where at least one column was filled.
    """
    df = df.copy()
    missing = df[cols].isna().any(axis=1)
    run_id = (~missing).cumsum()
    run_len = missing.groupby(run_id).transform("sum")
    # a run touching the start or end of the series has no reading on one side
    inside = missing & (run_id > 0) & (run_id < run_id.max())
    fillable = missing & (run_len <= max_hours) & inside
    for c in cols:
        if c in carry_forward:
            filled_values = df[c].ffill()
        else:
            filled_values = df[c].interpolate(method="time", limit_area="inside")
        df.loc[fillable, c] = filled_values[fillable]
    return df, fillable


def pm25_silver(bronze_path: Path = BRONZE_DIR / "pm25.parquet") -> pd.DataFrame:
    df = pd.read_parquet(bronze_path).set_index("ts").sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    gap_hours = full.difference(df.index)
    if len(gap_hours):
        log.warning("PM2.5: %d missing hours (e.g. %s)", len(gap_hours), list(gap_hours[:5]))
    df = df.reindex(full)
    df, filled = fill_short_gaps(df, SG_REGIONS)
    if filled.any():
        log.info("PM2.5: filled %d short gap hour(s) (<= %d h) by interpolation", int(filled.sum()), MAX_FILL_HOURS)
    df["filled"] = filled
    df["national_mean"] = df[SG_REGIONS].mean(axis=1)
    df.index.name = "ts"
    return df.reset_index()


def weather_silver(bronze_path: Path = BRONZE_DIR / "weather.parquet") -> pd.DataFrame:
    df = pd.read_parquet(bronze_path)
    out = []
    for point, g in df.groupby("point"):
        g = g.set_index("ts").sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq="h")
        gap_hours = full.difference(g.index)
        if len(gap_hours):
            log.warning("weather[%s]: %d missing hours", point, len(gap_hours))
        g = g.reindex(full)
        g, filled = fill_short_gaps(g, ["wind_speed_kmh", "wind_dir_deg", "rain_mm"],
                                    carry_forward=("wind_dir_deg",))
        if filled.any():
            log.info("weather[%s]: filled %d short gap hour(s)", point, int(filled.sum()))
        g["filled"] = filled
        g["point"] = point
        g.index.name = "ts"
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def firms_daily_silver(bronze_path: Path = BRONZE_DIR / "firms.parquet") -> pd.DataFrame:
    """Detection-level FIRMS bronze -> daily detection counts per region.

    Also flags days with zero detections in BOTH regions as `data_gap`
    (rather than silently trusting them as "no fires"), based on this
    project's own profiling, which found month-long stretches of exactly
    zero detections that neighbouring days show is implausible (e.g. all
    of May 2026 in the Indonesia VIIRS export used for this project).
    A day is flagged as a likely gap, not dropped, so downstream code can
    choose to impute or exclude it explicitly.
    """
    df = pd.read_parquet(bronze_path)
    counts = df.groupby(["date", "region"]).size().unstack(fill_value=0)
    full_days = pd.date_range(
        pd.Timestamp(counts.index.min()), pd.Timestamp(counts.index.max()), freq="D"
    )
    counts.index = pd.to_datetime(counts.index)
    counts = counts.reindex(full_days, fill_value=0)
    counts.index.name = "date"
    for region in ["sumatra", "kalimantan"]:
        if region not in counts.columns:
            counts[region] = 0
    counts["total"] = counts[["sumatra", "kalimantan"]].sum(axis=1)

    # A run of 5+ consecutive zero-total days is treated as a likely data
    # gap rather than a genuinely fire-free week (threshold is a documented
    # assumption; tune once 2014-18 data confirms typical dry-spell length).
    is_zero = counts["total"] == 0
    run_id = (~is_zero).cumsum()
    run_len = is_zero.groupby(run_id).transform("sum")
    counts["data_gap"] = is_zero & (run_len >= 5)

    return counts.reset_index()


def firms_processed_until(availability_path: Path = RAW_DIR / "firms" / "availability.json") -> pd.Timestamp | None:
    """Up to which hour (Singapore time) FIRMS has certainly delivered, from the availability check
    the live fetch saves (hazecast.ingest.live.fetch_firms_availability): the end of the newest
    processed UTC day, but no later than FIRE_LATENCY_HOURS + 1 before the check. So "today is
    being processed" means passes older than 4 h are in; a day FIRMS has not listed (28 Oct 2026:
    still the previous day hours after the 13:00 SGT pass) is not trusted at all. Residual risk: a
    second pass of a listed UTC day arriving more than 4 h late is read as zero until it arrives
    (the next rebuild corrects it; training never sees it). None if no check has been saved."""
    path = Path(availability_path)
    if not path.exists():
        return None
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
        checked = pd.Timestamp(info["checked_at"])
        day_end = pd.Timestamp(info["max_date"]).tz_localize("UTC") + pd.Timedelta(days=1)
    except (ValueError, KeyError, TypeError) as exc:
        log.warning("Ignoring unreadable FIRMS availability file %s: %s", path, exc)
        return None
    until = min(day_end, checked - pd.Timedelta(hours=FIRE_LATENCY_HOURS + 1))
    return until.tz_convert("Asia/Singapore").tz_localize(None).floor("h")


def firms_hourly_silver(bronze_path: Path = BRONZE_DIR / "firms.parquet",
                        availability_path: Path = RAW_DIR / "firms" / "availability.json") -> pd.DataFrame:
    """Detection-level FIRMS bronze -> detections per hour (Singapore time) per region.

    Hours without a detection are 0 (no satellite pass, or no fire). Where the grid ends
    decides which counts are known:
      - FIRE_KNOWN_HOURS_AFTER_LAST_DETECTION after the latest detection (passes are ~11 h
        apart, so a longer silence may mean a missing pass or a stalled feed), or
      - later, up to the hour FIRMS reports as processed (firms_processed_until): an empty
        processed period is a true "no fires", e.g. several quiet days in the wet season,
        which the 15 h rule alone would have read as an outage and paused forecasts on.
    Counts after the end are unknown (left out: the gold features are carried forward up to
    24 h, then empty, and no forecast is issued). Longer gaps inside the history stay as zeros
    and are covered by the daily data_gap flag, as in firms_daily_silver.
    """
    df = pd.read_parquet(bronze_path)
    if "ts" not in df.columns:
        raise ValueError(f"{bronze_path} has no detection timestamps; re-run hazecast.ingest.firms")
    counts = df.groupby([df["ts"].dt.floor("h"), "region"]).size().unstack(fill_value=0)
    end = counts.index.max() + pd.Timedelta(hours=FIRE_KNOWN_HOURS_AFTER_LAST_DETECTION)
    processed = firms_processed_until(availability_path)
    if processed is not None and processed > end:
        log.info("FIRMS processed through %s: fire counts known to then (15 h rule alone: %s)", processed, end)
        end = processed
    counts = counts.reindex(pd.date_range(counts.index.min(), end, freq="h"), fill_value=0)
    for region in ["sumatra", "kalimantan"]:
        if region not in counts.columns:
            counts[region] = 0
    counts = counts[["sumatra", "kalimantan"]].copy()
    counts["total"] = counts.sum(axis=1)
    counts.index.name = "ts"
    return counts.reset_index()


def write_silver(df: pd.DataFrame, name: str, silver_dir: Path = SILVER_DIR) -> Path:
    silver_dir.mkdir(parents=True, exist_ok=True)
    out_path = silver_dir / f"{name}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pm25 = pm25_silver()
    write_silver(pm25, "pm25")
    log.info("silver/pm25: %d rows", len(pm25))

    weather = weather_silver()
    write_silver(weather, "weather")
    log.info("silver/weather: %d rows", len(weather))

    firms = firms_daily_silver()
    write_silver(firms, "firms_daily")
    log.info(
        "silver/firms_daily: %d days, %d flagged as likely data gaps",
        len(firms), int(firms["data_gap"].sum()),
    )
