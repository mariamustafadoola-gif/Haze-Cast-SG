"""Silver -> gold: build the daily feature + label table used for
training and scoring.

One row per calendar day (Singapore time). Features use only information
available by end-of-day "today" (23:30 per the team's proposal); labels
are each region's next-day daily-mean PM2.5, per hazecast.config's
FORECAST_HORIZON_DAYS. This keeps the framing leak-free: nothing from the
forecast day itself is ever a feature.
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from hazecast.config import (
    SILVER_DIR, GOLD_DIR, SG_REGIONS, WEATHER_POINTS, FIRE_REGIONS,
    HOTSPOT_LAGS, ROLLING_WINDOWS, FORECAST_HORIZON_DAYS,
    FORECAST_HORIZON_HOURS, HOURLY_RAIN_ROLLING_WINDOWS, FIRE_DAY_AVAILABLE_HOUR_SGT,
    FIRE_LATENCY_HOURS, FIRE_CARRY_FORWARD_HOURS,
)

log = logging.getLogger(__name__)


def _daily_pm25(pm25_silver: pd.DataFrame) -> pd.DataFrame:
    df = pm25_silver.set_index("ts")
    daily = df[SG_REGIONS + ["national_mean"]].resample("D").mean()
    # a day built from fewer than 20 of its 24 hourly readings is
    # unreliable as a label or persistence feature -- flag it.
    coverage = df[SG_REGIONS].resample("D").count().min(axis=1)
    daily["low_coverage"] = coverage < 20
    daily.index.name = "date"
    return daily.reset_index()


def _fire_features(firms_daily: pd.DataFrame) -> pd.DataFrame:
    df = firms_daily.set_index("date").sort_index()
    out = pd.DataFrame(index=df.index)
    for region in ["sumatra", "kalimantan", "total"]:
        for lag in HOTSPOT_LAGS:
            out[f"fire_{region}_lag{lag}"] = df[region].shift(lag)
        for w in ROLLING_WINDOWS:
            out[f"fire_{region}_roll{w}"] = df[region].rolling(w, min_periods=1).sum()
        out[f"fire_{region}_log_lag0"] = np.log1p(out[f"fire_{region}_lag0"])
    # Kept as float (0.0/1.0), not bool: this column is merged against
    # years with no FIRMS coverage yet (all-NaN), and a bool/NaN mix would
    # silently become an "object" dtype that breaks XGBoost downstream.
    out["fire_data_gap_recent"] = df["data_gap"].astype(float).rolling(
        max(HOTSPOT_LAGS) + 1, min_periods=1
    ).max()
    out.index.name = "date"
    return out.reset_index()


def _weather_features(weather_silver: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for point in WEATHER_POINTS:
        g = weather_silver[weather_silver["point"] == point].set_index("ts").sort_index()
        if g.empty:
            log.warning("No weather rows for point '%s'; its features will be all-NaN", point)
            continue
        daily = pd.DataFrame(index=g.index.normalize().unique())
        daily.index.name = "date"
        daily[f"wind_speed_{point}"] = g["wind_speed_kmh"].resample("D").mean()
        # circular mean of wind direction (degrees) via unit vectors
        rad = np.deg2rad(g["wind_dir_deg"])
        sin_d = np.sin(rad).resample("D").mean()
        cos_d = np.cos(rad).resample("D").mean()
        daily[f"wind_dir_{point}"] = (np.degrees(np.arctan2(sin_d, cos_d)) + 360) % 360
        daily[f"rain_{point}_mm"] = g["rain_mm"].resample("D").sum()
        for w in ROLLING_WINDOWS:
            daily[f"rain_{point}_roll{w}"] = daily[f"rain_{point}_mm"].rolling(w, min_periods=1).sum()
        frames.append(daily)
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="outer")
    out.index.name = "date"
    return out.reset_index()


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing (degrees, 0=N, clockwise) from point 1 to point 2."""
    lat1r, lon1r, lat2r, lon2r = (np.radians(v) for v in (lat1, lon1, lat2, lon2))
    dlon = lon2r - lon1r
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def _upwind_features(gold: pd.DataFrame) -> pd.DataFrame:
    """Does today's wind at Singapore actually blow FROM each fire source
    region? Fire counts (fire_<region>_roll7) and wind direction
    (wind_dir_singapore) are otherwise two unrelated columns -- a tree
    split on raw wind-direction degrees can't recover "which compass
    direction Sumatra/Kalimantan is from Singapore" on its own, since
    that's a geographic fact, not something learnable from the labels.
    This makes the alignment explicit instead of leaving it to be
    (incompletely) rediscovered from raw features.

    alignment = cos(wind_dir_singapore - bearing_to_region_centroid),
    clipped to >=0 (favourable transport only -- a negative value just
    means "wind blowing away from this source", not a different kind of
    exposure), scaled by that region's existing 7-day fire count to get
    an "upwind exposure dose".
    """
    sg_lat, sg_lon = WEATHER_POINTS["singapore"]
    out = pd.DataFrame(index=gold.index)
    for region, box in FIRE_REGIONS.items():
        centroid_lat = (box["lat_min"] + box["lat_max"]) / 2
        centroid_lon = (box["lon_min"] + box["lon_max"]) / 2
        bearing = _bearing_deg(sg_lat, sg_lon, centroid_lat, centroid_lon)
        alignment = np.cos(np.radians(gold["wind_dir_singapore"] - bearing)).clip(lower=0)
        out[f"upwind_{region}"] = alignment * gold[f"fire_{region}_roll7"]
    out["upwind_total"] = out.sum(axis=1)
    return out


def build_gold(
    pm25_silver: pd.DataFrame,
    weather_silver_df: pd.DataFrame,
    firms_daily: pd.DataFrame,
) -> pd.DataFrame:
    daily_pm25 = _daily_pm25(pm25_silver)
    fire = _fire_features(firms_daily)
    weather = _weather_features(weather_silver_df)

    daily_pm25["date"] = pd.to_datetime(daily_pm25["date"])
    fire["date"] = pd.to_datetime(fire["date"])
    weather["date"] = pd.to_datetime(weather["date"])

    gold = daily_pm25.merge(fire, on="date", how="left").merge(weather, on="date", how="left")
    gold = pd.concat([gold, _upwind_features(gold)], axis=1)

    # Persistence features: today's own reading, renamed so it's obviously
    # a feature and not accidentally used as the label.
    for r in SG_REGIONS + ["national_mean"]:
        gold[f"today_{r}"] = gold[r]

    # Labels: each region's PM2.5 exactly FORECAST_HORIZON_DAYS ahead.
    # Built with shift(-h) on the *label source* column, then the
    # unshifted "today_*" columns above remain leak-free features.
    h = FORECAST_HORIZON_DAYS
    for r in SG_REGIONS + ["national_mean"]:
        gold[f"label_{r}"] = gold[r].shift(-h)
    gold["label_low_coverage"] = gold["low_coverage"].shift(-h)

    gold = gold.drop(columns=SG_REGIONS + ["national_mean", "low_coverage"])
    gold = gold.rename(columns={"label_national_mean": "label_national", "today_national_mean": "today_national"})
    return gold.sort_values("date").reset_index(drop=True)


def write_gold(df: pd.DataFrame, gold_dir: Path = GOLD_DIR) -> Path:
    gold_dir.mkdir(parents=True, exist_ok=True)
    out_path = gold_dir / "daily_features.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Hourly pipeline (revised core target). Additive: build_gold() above and
# daily_features.parquet are untouched and still what the shipped daily
# model uses. Everything below writes to a separate hourly_features.parquet.
# ---------------------------------------------------------------------------


def _hourly_pm25(pm25_silver: pd.DataFrame) -> pd.DataFrame:
    """Unlike _daily_pm25, nothing is resampled away here -- bronze_to_silver
    already reindexes PM2.5 onto a full hourly grid, so this is the native
    grain, not an aggregate of it."""
    return pm25_silver[["ts"] + SG_REGIONS + ["national_mean"]].copy()


def _fire_features_hourly(firms_hourly: pd.DataFrame, firms_daily: pd.DataFrame) -> pd.DataFrame:
    """Hourly fire features from detection timestamps, using only detections delivered
    by each hour (see config.FIRE_LATENCY_HOURS):

      fire_<r>_lag0      detections in the 24 h ending FIRE_LATENCY_HOURS full hours before the hour
      fire_<r>_lag1..3   the 24 h blocks before that
      fire_<r>_roll3/7   the 72 h / 168 h ending at the same point
      fire_<r>_log_lag0  log1p(lag0)
      fire_data_gap_recent  the daily satellite-gap flag, from the latest FIRMS day
                            complete by the hour (fire_day_for_hour)

    Names match the daily table, so SELECTED_FEATURES_HOURLY, the models and the
    monitors are unchanged; "lag0" now means "the latest 24 h available".

    History: the first version gave every hour of Singapore day D the features of FIRMS
    (UTC) day D-1. Live, day D's rows only appeared after D's first detection (around
    midday), so no forecast could be issued from midnight until then; and in training,
    hours 00:00-10:59 used counts including the night pass at 01:00-04:59 SGT, i.e.
    detections made after the hour. A day-based as-of version fixed both but made a
    column's age depend on the hour and lost accuracy (holdout F2 0.594 vs 0.641); these
    rolling windows matched the old recall with fewer false alarms (F2 0.659).
    """
    c = firms_hourly.set_index("ts").sort_index()
    gap = FIRE_LATENCY_HOURS + 1          # whole hourly buckets fully delivered by the hour
    out = pd.DataFrame(index=c.index)
    for region in ["sumatra", "kalimantan", "total"]:
        day = c[region].rolling(24, min_periods=24).sum()
        for lag in HOTSPOT_LAGS:
            out[f"fire_{region}_lag{lag}"] = day.shift(gap + 24 * lag)
        for w in ROLLING_WINDOWS:
            out[f"fire_{region}_roll{w}"] = c[region].rolling(24 * w, min_periods=24 * w).sum().shift(gap)
        out[f"fire_{region}_log_lag0"] = np.log1p(out[f"fire_{region}_lag0"])
    out.index.name = "ts"
    out = out.reset_index()
    daily = _fire_features(firms_daily)[["date", "fire_data_gap_recent"]].copy()
    daily["date"] = pd.to_datetime(daily["date"])
    out["fire_date"] = fire_day_for_hour(out["ts"])
    out = out.merge(daily.rename(columns={"date": "fire_date"}), on="fire_date", how="left")
    return out.drop(columns="fire_date")


def _carry_fire_forward(gold: pd.DataFrame, fire_cols: list[str],
                        limit_hours: float = FIRE_CARRY_FORWARD_HOURS) -> pd.DataFrame:
    """Late FIRMS delivery: give the hours after the last hour with known fire features that
    hour's values, for up to `limit_hours`, and record the carry in `fire_carried_hours`
    (0 when the features are current). Rows further out keep empty fire features. Only the
    live edge is affected: earlier hours all have known features once a later detection arrives.
    """
    gold = gold.sort_values("ts").reset_index(drop=True)
    gold["fire_carried_hours"] = 0.0
    known = gold[fire_cols].notna().all(axis=1)
    if not known.any():
        return gold
    last = gold.loc[known, "ts"].max()
    carried = (gold["ts"] - last) / pd.Timedelta(hours=1)
    tail = carried > 0
    fill = tail & (carried <= limit_hours)
    gold.loc[fill, fire_cols] = gold.loc[gold["ts"] == last, fire_cols].to_numpy()
    gold.loc[tail, "fire_carried_hours"] = carried[tail]
    if fill.any():
        log.warning("FIRMS late: fire features carried forward from %s for up to %.0f h", last, carried[fill].max())
    return gold


def fire_day_for_hour(ts: pd.Series) -> pd.Series:
    """The latest FIRMS (UTC) day whose detections are complete at each Singapore hour."""
    return (ts - pd.Timedelta(hours=FIRE_DAY_AVAILABLE_HOUR_SGT)).dt.floor("D") - pd.Timedelta(days=1)


def _weather_features_hourly(weather_silver: pd.DataFrame) -> pd.DataFrame:
    """Weather is genuinely hourly upstream already -- no resampling, just
    a per-point pivot, plus short hour-scale rolling rain sums in place of
    the daily pipeline's 3-day/7-day ones (HOURLY_RAIN_ROLLING_WINDOWS)."""
    frames = []
    for point in WEATHER_POINTS:
        g = weather_silver[weather_silver["point"] == point].set_index("ts").sort_index()
        if g.empty:
            log.warning("No weather rows for point '%s'; its features will be all-NaN", point)
            continue
        hourly = pd.DataFrame(index=g.index)
        hourly.index.name = "ts"
        hourly[f"wind_speed_{point}"] = g["wind_speed_kmh"]
        hourly[f"wind_dir_{point}"] = g["wind_dir_deg"]
        hourly[f"rain_{point}_mm"] = g["rain_mm"]
        for w in HOURLY_RAIN_ROLLING_WINDOWS:
            hourly[f"rain_{point}_roll{w}h"] = hourly[f"rain_{point}_mm"].rolling(w, min_periods=1).sum()
        frames.append(hourly)
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="outer")
    out.index.name = "ts"
    return out.reset_index()


def _upwind_features_hourly(gold: pd.DataFrame) -> pd.DataFrame:
    """Same geometry as _upwind_features. Wind now genuinely changes hour
    to hour while the fire term it's scaled by only updates once a day
    (_fire_features_hourly) -- the alignment term is what makes this
    feature actually move within a day."""
    sg_lat, sg_lon = WEATHER_POINTS["singapore"]
    out = pd.DataFrame(index=gold.index)
    for region, box in FIRE_REGIONS.items():
        centroid_lat = (box["lat_min"] + box["lat_max"]) / 2
        centroid_lon = (box["lon_min"] + box["lon_max"]) / 2
        bearing = _bearing_deg(sg_lat, sg_lon, centroid_lat, centroid_lon)
        alignment = np.cos(np.radians(gold["wind_dir_singapore"] - bearing)).clip(lower=0)
        out[f"upwind_{region}"] = alignment * gold[f"fire_{region}_roll7"]
    out["upwind_total"] = out.sum(axis=1)
    return out


def build_gold_hourly(
    pm25_silver: pd.DataFrame,
    weather_silver_df: pd.DataFrame,
    firms_daily: pd.DataFrame,
    firms_hourly: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Hourly counterpart to build_gold(). One row per hour (Singapore
    time). Labels are FORECAST_HORIZON_HOURS ahead, not FORECAST_HORIZON_DAYS.
    """
    if firms_hourly is None:
        firms_hourly = pd.read_parquet(SILVER_DIR / "firms_hourly.parquet")
    pm25 = _hourly_pm25(pm25_silver)
    fire = _fire_features_hourly(firms_hourly, firms_daily)
    weather = _weather_features_hourly(weather_silver_df)

    gold = pm25.merge(fire, on="ts", how="left").merge(weather, on="ts", how="left")
    gold = _carry_fire_forward(gold, [c for c in fire if c != "ts"])
    gold = pd.concat([gold, _upwind_features_hourly(gold)], axis=1)

    # Persistence features: this hour's own reading.
    for r in SG_REGIONS + ["national_mean"]:
        gold[f"now_{r}"] = gold[r]

    # Labels: each region's PM2.5 exactly FORECAST_HORIZON_HOURS ahead.
    h = FORECAST_HORIZON_HOURS
    for r in SG_REGIONS + ["national_mean"]:
        gold[f"label_{r}"] = gold[r].shift(-h)

    gold = gold.drop(columns=SG_REGIONS + ["national_mean"])
    gold = gold.rename(columns={
        "label_national_mean": "label_national", "now_national_mean": "now_national", "ts": "date",
    })
    return gold.sort_values("date").reset_index(drop=True)


def write_gold_hourly(df: pd.DataFrame, gold_dir: Path = GOLD_DIR) -> Path:
    gold_dir.mkdir(parents=True, exist_ok=True)
    out_path = gold_dir / "hourly_features.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pm25 = pd.read_parquet(SILVER_DIR / "pm25.parquet")
    weather = pd.read_parquet(SILVER_DIR / "weather.parquet")
    firms = pd.read_parquet(SILVER_DIR / "firms_daily.parquet")

    gold = build_gold(pm25, weather, firms)
    path = write_gold(gold)
    log.info(
        "gold: %d rows, %d columns, %s to %s -> %s",
        len(gold), gold.shape[1], gold.date.min(), gold.date.max(), path,
    )

    # Also build the hourly table (the team's main forecast target).
    gold_h = build_gold_hourly(pm25, weather, firms, pd.read_parquet(SILVER_DIR / "firms_hourly.parquet"))
    path_h = write_gold_hourly(gold_h)
    log.info(
        "gold (hourly): %d rows, %d columns, %s to %s -> %s",
        len(gold_h), gold_h.shape[1], gold_h.date.min(), gold_h.date.max(), path_h,
    )
