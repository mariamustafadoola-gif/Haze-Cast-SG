"""Ingest NASA FIRMS active-fire detections into the bronze layer.

Manual path (used for this proof of concept): request a CSV export from
https://firms.modaps.eosdis.nasa.gov/download/ (Suomi-NPP VIIRS, area =
Indonesia, format = CSV) and place the downloaded .zip (or its extracted
.csv files) under data/raw/firms/. FIRMS emails a zip containing an
"archive" file (finalised, science-quality) and, for the most recent ~2
months, an "nrt" (near-real-time) file with a slightly different, smaller
column set.

Live path: hazecast.ingest.live calls the FIRMS area API
(/api/area/csv/<MAP_KEY>/VIIRS_SNPP_NRT/<AREA>/<DAY_RANGE>/<DATE>, up to
10 days per call) and saves the result as data/raw/firms/fire_nrt_zlive.csv.
Requires a free MAP_KEY, read from FIRMS_MAP_KEY in .env.

We keep only detections inside the two source-region bounding boxes
defined in hazecast.config.FIRE_REGIONS (Sumatra, Kalimantan); everything
else (Java, Papua, etc.) is dropped at bronze already, since gold only
ever needs the two regions used for upwind features. The archive file's
`type` column (0 = presumed vegetation fire) is used to drop static heat
sources (industrial flares, volcanoes) where available; the NRT file has
no such column, so it is kept as-is (a small, known source of noise in
the most recent ~2 months of data).
"""
from __future__ import annotations
import glob
import logging
import zipfile
from pathlib import Path

import pandas as pd

from hazecast.config import RAW_DIR, BRONZE_DIR, FIRE_REGIONS

log = logging.getLogger(__name__)

_COLS = ["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp", "daynight"]


def _tag_region(df: pd.DataFrame) -> pd.Series:
    region = pd.Series("other", index=df.index)
    for name, box in FIRE_REGIONS.items():
        m = (
            df["longitude"].between(box["lon_min"], box["lon_max"])
            & df["latitude"].between(box["lat_min"], box["lat_max"])
        )
        region.loc[m] = name
    return region


def _extract_zips(raw_dir: Path) -> None:
    for z in glob.glob(str(raw_dir / "*.zip")):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(raw_dir)
        log.info("Extracted %s", z)


def load_firms_bronze(raw_dir: Path = RAW_DIR / "firms") -> pd.DataFrame:
    """Load every FIRMS CSV (archive + nrt, any number of requests) into
    one long bronze frame, filtered to the Sumatra/Kalimantan boxes.

    Returns columns: date (datetime.date), latitude, longitude,
    confidence, frp, daynight, region.
    """
    _extract_zips(raw_dir)
    archive_files = sorted(glob.glob(str(raw_dir / "fire_archive_*.csv")))
    nrt_files = sorted(glob.glob(str(raw_dir / "fire_nrt_*.csv")))
    if not archive_files and not nrt_files:
        raise FileNotFoundError(
            f"No FIRMS fire_archive_*.csv or fire_nrt_*.csv found under "
            f"{raw_dir} (zips are auto-extracted). See module docstring."
        )

    frames = []
    for f in archive_files:
        df = pd.read_csv(f, usecols=_COLS + ["type"], dtype={"acq_time": str})
        df = df[df["type"] == 0].drop(columns="type")  # vegetation fires only
        frames.append(df)
    for f in nrt_files:
        df = pd.read_csv(f, usecols=_COLS, dtype={"acq_time": str})
        frames.append(df)  # no `type` column in NRT; kept as-is, see docstring

    out = pd.concat(frames, ignore_index=True)
    before = len(out)
    out = out.drop_duplicates(["latitude", "longitude", "acq_date", "acq_time"])
    if before != len(out):
        log.info("Dropped %d exact-duplicate detections", before - len(out))

    out["date"] = pd.to_datetime(out["acq_date"]).dt.date
    out["region"] = _tag_region(out)
    out = out[out["region"] != "other"]
    # lat/lon are kept for potential future wind-weighting; only the raw
    # acquisition-date/time strings are dropped now that `date` exists.
    out = out.drop(columns=["acq_date", "acq_time"])
    return out.reset_index(drop=True)


def write_bronze(df: pd.DataFrame, bronze_dir: Path = BRONZE_DIR) -> Path:
    bronze_dir.mkdir(parents=True, exist_ok=True)
    out_path = bronze_dir / "firms.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    frame = load_firms_bronze()
    path = write_bronze(frame)
    log.info(
        "Wrote %d detections (%s to %s), by region: %s",
        len(frame), frame.date.min(), frame.date.max(),
        frame.region.value_counts().to_dict(),
    )
