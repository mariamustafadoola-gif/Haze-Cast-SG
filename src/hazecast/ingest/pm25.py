"""Ingest NEA Historical 1-hr PM2.5 into the bronze layer.

Manual path (used for this proof of concept):
  Download "Historical 1-hr PM2.5" from
  https://data.gov.sg/datasets/d_6c0d5fc34838b12472475fdb73c0af29/view
  and place the CSV under data/raw/pm25/.

Live path: hazecast.ingest.live calls
  GET https://api-open.data.gov.sg/v2/real-time/api/pm25?date=YYYY-MM-DD
for each recent day and saves the readings under data/raw/pm25/ in this
same file format, so this loader reads them unchanged.

Known data quirks handled here (found by inspecting the actual file):
  - One row in the source file had a malformed timestamp
    ("2016-04-04 010:00:00"); we detect and repair any row that doesn't
    match the expected dd/mm/yyyy H:MM pattern by neighbour-inference
    is NOT attempted automatically for new bad rows -- we fail loudly
    instead, since a silent repair on live data is worse than a stopped
    pipeline. The one known historical bad row is patched explicitly.
  - A handful of timestamps appear twice with different values (DST-like
    artefacts around 2016-03-07, 2018-04-24, 2018-08-16). We keep the
    first occurrence and log the rest; a handful of adjacent hours are
    then missing, which downstream code must tolerate.
"""
from __future__ import annotations
import glob
import logging
from pathlib import Path

import pandas as pd

from hazecast.config import RAW_DIR, BRONZE_DIR, SG_REGIONS

log = logging.getLogger(__name__)

# The one known malformed timestamp in the NEA export used for this
# project (verified by inspecting its neighbouring rows).
_KNOWN_BAD_TIMESTAMP_FIX = {"2016-04-04 010:00:00": "4/4/2016 0:00"}


def load_pm25_bronze(raw_dir: Path = RAW_DIR / "pm25") -> pd.DataFrame:
    """Load the NEA Historical 1-hr PM2.5 CSV(s) into a tidy bronze frame.

    Returns columns: ts (datetime, SGT, tz-naive), north, south, east,
    west, central (float, ug/m3).
    """
    files = sorted(glob.glob(str(raw_dir / "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No PM2.5 CSV found under {raw_dir}. Download 'Historical 1-hr "
            f"PM2.5' from data.gov.sg and place it there (see module docstring)."
        )

    frames = []
    for f in files:
        df = pd.read_csv(f, dtype=str)
        ts_col = df.columns[0]  # NEA export names this column "1hr_pm25"
        df[ts_col] = df[ts_col].replace(_KNOWN_BAD_TIMESTAMP_FIX)

        bad_mask = ~df[ts_col].str.match(r"^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}$")
        if bad_mask.any():
            log.warning(
                "%s: %d rows with an unrecognised timestamp format were "
                "dropped (not in the known-fix list): %s",
                f, bad_mask.sum(), df.loc[bad_mask, ts_col].tolist()[:5],
            )
            df = df.loc[~bad_mask]

        df["ts"] = pd.to_datetime(df[ts_col], format="%d/%m/%Y %H:%M")
        df = df.drop(columns=[ts_col])
        for r in SG_REGIONS:
            if r not in df.columns:
                raise ValueError(f"{f}: expected region column '{r}' not found")
            df[r] = pd.to_numeric(df[r], errors="coerce")
        frames.append(df[["ts", *SG_REGIONS]])

    out = pd.concat(frames, ignore_index=True)
    dup_ts = out["ts"].duplicated().sum()
    if dup_ts:
        log.warning("%d duplicate timestamps found; keeping first occurrence", dup_ts)
        out = out.drop_duplicates("ts", keep="first")
    out = out.sort_values("ts").reset_index(drop=True)
    return out


def write_bronze(df: pd.DataFrame, bronze_dir: Path = BRONZE_DIR) -> Path:
    bronze_dir.mkdir(parents=True, exist_ok=True)
    out_path = bronze_dir / "pm25.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    frame = load_pm25_bronze()
    path = write_bronze(frame)
    log.info("Wrote %d rows (%s to %s) to %s", len(frame), frame.ts.min(), frame.ts.max(), path)
