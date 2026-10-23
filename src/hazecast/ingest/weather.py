"""Ingest Open-Meteo historical weather (Singapore + 4 fire-region points)
into the bronze layer.

Manual path (used for this proof of concept): download one CSV per point
from the Open-Meteo Historical Weather API, e.g.:

  https://archive-api.open-meteo.com/v1/archive?latitude=1.37&longitude=103.80
    &start_date=2019-01-01&end_date=2026-07-31
    &hourly=wind_speed_10m,wind_direction_10m,precipitation
    &timezone=Asia%2FSingapore&format=csv

and place each file under data/raw/weather/. Point names are inferred from
the filename's lat/lon prefix and matched to the nearest entry in
hazecast.config.WEATHER_POINTS (within 0.5 degrees) -- this is a
convenience matcher for this proof of concept's specific downloads, not a
general geocoder.

Live path: hazecast.ingest.live downloads the most recent hours for every
point from the Open-Meteo forecast API (past_days) into data/raw/weather/,
in this same file format. Hours that appear in more than one file are
de-duplicated below, keeping the older file's value.
"""
from __future__ import annotations
import glob
import logging
import re
from pathlib import Path

import pandas as pd

from hazecast.config import RAW_DIR, BRONZE_DIR, WEATHER_POINTS

log = logging.getLogger(__name__)

_FNAME_LATLON = re.compile(r"(-?\d+\.?\d*)([NS])(-?\d+\.?\d*)([EW])")


def _nearest_point_name(lat: float, lon: float, tol: float = 0.5) -> str | None:
    best, best_d = None, tol
    for name, (plat, plon) in WEATHER_POINTS.items():
        d = ((lat - plat) ** 2 + (lon - plon) ** 2) ** 0.5
        if d < best_d:
            best, best_d = name, d
    return best


def load_weather_bronze(raw_dir: Path = RAW_DIR / "weather") -> pd.DataFrame:
    """Load every Open-Meteo CSV under raw_dir into one long bronze frame.

    Returns columns: ts (datetime, SGT), point, wind_speed_kmh,
    wind_dir_deg, rain_mm.
    """
    files = sorted(glob.glob(str(raw_dir / "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No weather CSV found under {raw_dir}. Download Open-Meteo "
            f"historical weather CSVs for each point in WEATHER_POINTS "
            f"and place them there (see module docstring)."
        )

    frames = []
    for f in files:
        # Open-Meteo CSVs start with a 3-line metadata header (incl. the
        # exact lat/lon actually served, which can differ slightly from
        # the requested one) then the data header row.
        meta = pd.read_csv(f, nrows=1)
        lat, lon = float(meta["latitude"].iloc[0]), float(meta["longitude"].iloc[0])
        point = _nearest_point_name(lat, lon)
        if point is None:
            m = _FNAME_LATLON.search(Path(f).name)
            hint = m.group(0) if m else Path(f).name
            raise ValueError(
                f"{f}: served coordinates ({lat},{lon}) don't match any "
                f"WEATHER_POINTS entry within tolerance (filename hint: {hint}). "
                f"Add it to config.WEATHER_POINTS or check the download."
            )

        df = pd.read_csv(f, skiprows=3)
        df.columns = ["ts", "wind_speed_kmh", "wind_dir_deg", "rain_mm"]
        df["ts"] = pd.to_datetime(df["ts"])
        df["point"] = point
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    # Files for the same place can overlap (historical export + update files +
    # live downloads). Files are read in name order, so the older file's value
    # is kept for any hour that appears twice.
    before = len(out)
    out = out.drop_duplicates(["point", "ts"], keep="first")
    if before != len(out):
        log.info("weather: dropped %d hours that appeared in more than one file", before - len(out))
    missing = set(WEATHER_POINTS) - set(out["point"].unique())
    if missing:
        log.warning("No weather file found for expected point(s): %s", missing)
    return out.sort_values(["point", "ts"]).reset_index(drop=True)


def write_bronze(df: pd.DataFrame, bronze_dir: Path = BRONZE_DIR) -> Path:
    bronze_dir.mkdir(parents=True, exist_ok=True)
    out_path = bronze_dir / "weather.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    frame = load_weather_bronze()
    path = write_bronze(frame)
    for pt, g in frame.groupby("point"):
        log.info("%s: %d rows, %s to %s", pt, len(g), g.ts.min(), g.ts.max())
    log.info("Wrote %d total rows to %s", len(frame), path)
