"""Download historical PM2.5, weather and fire data for model training.

Fills data/raw/ with everything between --start and --end that is not
already there, in the same file formats as the manual downloads described
in data/README.md, so ingest -> bronze -> silver -> gold works unchanged.

    python -m hazecast.ingest.historical                         # HISTORY_START to yesterday
    python -m hazecast.ingest.historical --start 2019-01-01 --end 2019-12-31
    python -m hazecast.ingest.historical --sources pm25,weather  # skip FIRMS
    python -m hazecast.ingest.historical --dry-run               # show what is missing, download nothing

Coverage-driven, so it is safe to re-run and cheap when nothing is missing:
each source first works out which days/hours already exist in data/raw/
(ignoring the *live* files, which hazecast.ingest.live overwrites every run)
and only requests the rest. The same command therefore does both jobs:
  - first-time backfill on a fresh clone (no LFS files): downloads ~12 years;
  - daily top-up (Airflow): turns the last few days into permanent history,
    so a missed hourly run never leaves a hole in the training data.

Sources (see data/README.md):
  PM2.5   1. data.gov.sg bulk download of dataset d_6c0d5fc34838b12472475fdb73c0af29
             ("Historical 1-hr PM2.5", 2014-04 onwards, updated roughly monthly)
          2. https://api-open.data.gov.sg/v2/real-time/api/pm25?date=YYYY-MM-DD
             for the days after the bulk file ends (served from mid-2016)
  Weather https://archive-api.open-meteo.com/v1/archive  (Open-Meteo's default model blend, no key),
          one file per point per year; the last WEATHER_REFETCH_DAYS are re-fetched every run
  Fires   https://firms.modaps.eosdis.nasa.gov/api/area/csv/<KEY>/<SOURCE>/<area>/<days>/<date>
          VIIRS_SNPP_SP (science quality) where available, VIIRS_SNPP_NRT after that;
          needs FIRMS_MAP_KEY in env or .env

Files written:
  data/raw/pm25/Historical1hrPM2.5_datagov.csv
  data/raw/pm25/Historical1hrPM2.5_rtapi_<YYYYMM>.csv
  data/raw/weather/open-meteo-<lat>N<lon>E_hist_<point>_<YYYY>.csv
  data/raw/firms/fire_archive_api_<YYYYMM>.csv   (SP)  /  fire_nrt_api_<YYYYMM>.csv  (NRT)
  data/raw/firms/fetch_log.json                  (days already requested, incl. zero-fire days)
File names sort after the original manual downloads, so the ingest step's
"older file wins on overlaps" rule keeps the originals where both exist.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from hazecast.config import RAW_DIR, SG_REGIONS, WEATHER_POINTS, HISTORY_START, WEATHER_REFETCH_DAYS
from hazecast.ingest import live
from hazecast.ingest.weather import _nearest_point_name

log = logging.getLogger(__name__)
SOURCES = ("pm25", "weather", "firms")


def _yesterday_sgt() -> date:
    return (live.SGT_NOW() - timedelta(days=1)).date()


def _persistent_files(raw_dir: Path, pattern: str = "*.csv") -> list[str]:
    """Raw files that stay put. *live* files are overwritten every run, so they don't count as history."""
    return [f for f in sorted(glob.glob(str(raw_dir / pattern))) if "live" not in Path(f).name]


def _days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _month_key(d) -> str:
    return f"{d.year}{d.month:02d}"


# ---------------------------------------------------------------------------
# PM2.5
# ---------------------------------------------------------------------------
PM25_DATASET_ID = "d_6c0d5fc34838b12472475fdb73c0af29"
PM25_DOWNLOAD_URL = "https://api-open.data.gov.sg/v1/public/api/datasets/{id}/{step}"
PM25_REALTIME_FIRST_DAY = date(2016, 6, 1)  # earliest date the real-time API served when checked (2026-10)
PM25_BULK_MIN_MISSING_DAYS = 30  # below this, per-day API calls are cheaper than the ~3 MB bulk file


def pm25_covered_days(raw_dir: Path = RAW_DIR / "pm25") -> set[date]:
    """Days with (nearly) all hourly readings present in the persistent raw files.

    22, not 24: a few historical days have one or two hours NEA never published
    (see ingest.pm25); requiring 24 would re-request them on every run."""
    stamps = []
    for f in _persistent_files(raw_dir):
        col = pd.read_csv(f, usecols=[0], dtype=str).iloc[:, 0]
        stamps.append(pd.to_datetime(col, format="%d/%m/%Y %H:%M", errors="coerce").dropna())
    if not stamps:
        return set()
    ts = pd.concat(stamps).drop_duplicates()
    per_day = ts.dt.date.value_counts()
    return set(per_day[per_day >= 22].index)


def download_pm25_bulk(raw_dir: Path = RAW_DIR / "pm25", poll_seconds: int = 3, max_polls: int = 40) -> Path:
    """data.gov.sg dataset download: initiate, poll until a signed S3 URL is ready, save the CSV."""
    headers = {"x-api-key": k} if (k := live._env("DATA_GOV_SG_API_KEY")) else None
    live._get(PM25_DOWNLOAD_URL.format(id=PM25_DATASET_ID, step="initiate-download"), headers=headers)
    for _ in range(max_polls):
        body = live._get(PM25_DOWNLOAD_URL.format(id=PM25_DATASET_ID, step="poll-download"), headers=headers).json()
        data = body.get("data") or {}
        if data.get("status") == "DOWNLOAD_SUCCESS" and data.get("url"):
            break
        time.sleep(poll_seconds)
    else:
        raise RuntimeError(f"data.gov.sg download of {PM25_DATASET_ID} never became ready: {body}")
    text = live._get(data["url"]).text
    header = text.split("\n", 1)[0].strip().split(",")
    if header[1:] != SG_REGIONS and set(header[1:]) != set(SG_REGIONS):
        raise ValueError(f"Unexpected PM2.5 bulk file header {header}; expected 1hr_pm25 + {SG_REGIONS}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "Historical1hrPM2.5_datagov.csv"
    path.write_text(text, encoding="utf-8")
    log.info("PM2.5 bulk file: %d rows -> %s", text.count("\n"), path)
    return path


def _merge_csv(path: Path, new: pd.DataFrame, key: list[str]) -> int:
    """Append rows to a CSV, de-duplicating on `key` (existing rows win). Returns rows written."""
    if path.exists():
        old = pd.read_csv(path, dtype=str)
        new = pd.concat([old, new.astype(str)], ignore_index=True).drop_duplicates(key, keep="first")
    new.to_csv(path, index=False)
    return len(new)


def fetch_pm25_history(start: date, end: date, raw_dir: Path = RAW_DIR / "pm25",
                       dry_run: bool = False, delay: float | None = None) -> dict:
    wanted = set(_days(start, end))
    missing = sorted(wanted - pm25_covered_days(raw_dir))
    log.info("PM2.5: %d of %d days missing between %s and %s", len(missing), len(wanted), start, end)
    out = {"missing_days": len(missing), "bulk": None, "api_days": 0}
    if not missing or dry_run:
        return out
    if len(missing) >= PM25_BULK_MIN_MISSING_DAYS or missing[0] < PM25_REALTIME_FIRST_DAY:
        out["bulk"] = str(download_pm25_bulk(raw_dir))
        missing = sorted(wanted - pm25_covered_days(raw_dir))
        log.info("PM2.5: %d days still missing after the bulk file", len(missing))
    # Per-day API for whatever the bulk file doesn't reach (usually its last few weeks).
    api_days = [d for d in missing if d >= PM25_REALTIME_FIRST_DAY]
    if len(api_days) < len(missing):
        log.warning("PM2.5: %d day(s) before %s cannot be fetched from the real-time API and are left missing",
                    len(missing) - len(api_days), PM25_REALTIME_FIRST_DAY)
    # Without a key data.gov.sg rate-limits quickly; pace the requests.
    delay = (0.3 if live._env("DATA_GOV_SG_API_KEY") else 1.5) if delay is None else delay
    by_month: dict[str, list[pd.DataFrame]] = {}
    for d in api_days:
        df = live.fetch_pm25(d, d)
        if not df.empty:
            by_month.setdefault(_month_key(d), []).append(df)
        time.sleep(delay)
    for month, frames in by_month.items():
        df = pd.concat(frames, ignore_index=True).drop_duplicates("ts").sort_values("ts")
        df.insert(0, "1hr_pm25", [f"{t.day}/{t.month}/{t.year} {t.hour}:{t.minute:02d}" for t in df.pop("ts")])
        _merge_csv(raw_dir / f"Historical1hrPM2.5_rtapi_{month}.csv", df, ["1hr_pm25"])
    out["api_days"] = len(api_days)
    return out


# ---------------------------------------------------------------------------
# Weather (Open-Meteo archive)
# ---------------------------------------------------------------------------
METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_VARS = "wind_speed_10m,wind_direction_10m,precipitation"  # same columns/order as the manual files


def weather_covered_hours(raw_dir: Path = RAW_DIR / "weather") -> dict[str, set[pd.Timestamp]]:
    """Point name -> hours with a complete reading in the persistent raw files."""
    covered: dict[str, set[pd.Timestamp]] = {p: set() for p in WEATHER_POINTS}
    for f in _persistent_files(raw_dir):
        meta = pd.read_csv(f, nrows=1)
        point = _nearest_point_name(float(meta["latitude"].iloc[0]), float(meta["longitude"].iloc[0]))
        if point is None:
            continue
        df = pd.read_csv(f, skiprows=3).dropna()
        covered[point].update(pd.to_datetime(df.iloc[:, 0]))
    return covered


def fetch_weather_year(lat: float, lon: float, start: date, end: date) -> str:
    params = {"latitude": lat, "longitude": lon, "start_date": start.isoformat(), "end_date": end.isoformat(),
              "hourly": WEATHER_VARS, "timezone": "Asia/Singapore", "format": "csv"}
    return live._get(METEO_ARCHIVE_URL, params=params).text


def _clean_weather_csv(text: str, now: datetime) -> tuple[list[str], list[str]]:
    """Split Open-Meteo CSV into its 4 header lines and data lines; drop empty rows and future hours."""
    lines = text.strip().splitlines()
    head, body = lines[:4], lines[4:]
    keep = [ln for ln in body if ln and ",," not in ln and not ln.endswith(",")
            and pd.Timestamp(ln.split(",")[0]) <= pd.Timestamp(now)]
    return head, keep


def _merge_weather_rows(old: list[str], new: list[str]) -> list[str]:
    """Combine CSV data lines by timestamp; a re-fetched hour replaces the stored one."""
    rows = {ln.split(",", 1)[0]: ln for ln in old}
    rows.update({ln.split(",", 1)[0]: ln for ln in new})
    return [rows[ts] for ts in sorted(rows)]


def fetch_weather_history(start: date, end: date, raw_dir: Path = RAW_DIR / "weather",
                          dry_run: bool = False, delay: float = 1.0,
                          refetch_days: int = WEATHER_REFETCH_DAYS) -> dict:
    now = live.SGT_NOW()
    covered = weather_covered_hours(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    # Open-Meteo still revises the most recent hours for about a day after they are first
    # served (checked 2026-10-26: only the last ~4 hours of a download changed later), so
    # the last `refetch_days` before `end` are always requested again, even if stored.
    refresh_from = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(days=refetch_days)
    for point, (lat, lon) in WEATHER_POINTS.items():
        hours = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
        hours = hours[hours <= pd.Timestamp(now)]
        missing = hours[~hours.isin(list(covered[point])) | (hours >= refresh_from)]
        years = sorted({t.year for t in missing})
        out[point] = {"missing_hours": len(missing), "years": years}
        log.info("weather[%s]: %d hours to fetch (years %s; last %d day(s) always re-fetched)",
                 point, len(missing), years or "none", refetch_days)
        if dry_run:
            continue
        for year in years:
            in_year = missing[missing.year == year]
            text = fetch_weather_year(lat, lon, in_year.min().date(), in_year.max().date())
            head, body = _clean_weather_csv(text, now)
            ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
            path = raw_dir / f"open-meteo-{abs(lat):.2f}{ns}{abs(lon):.2f}{ew}_hist_{point}_{year}.csv"
            if path.exists():  # keep this year's earlier rows; re-fetched hours replace stored ones
                _, old = _clean_weather_csv(path.read_text(encoding="utf-8"), now)
                body = _merge_weather_rows(old, body)
            path.write_text("\n".join(head + body) + "\n", encoding="utf-8")
            log.info("weather[%s] %d: %d hours -> %s", point, year, len(body), path.name)
            time.sleep(delay)
    return out


# ---------------------------------------------------------------------------
# Fires (NASA FIRMS)
# ---------------------------------------------------------------------------
FIRMS_AVAILABILITY_URL = "https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{key}/ALL"
FIRMS_SP, FIRMS_NRT = "VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"
FIRMS_DAYS_PER_CALL = 5


def firms_sp_last_day(key: str) -> date | None:
    """Last day served by the science-quality (SP) archive; NRT is used after it."""
    try:
        text = live._get(FIRMS_AVAILABILITY_URL.format(key=key)).text
        df = pd.read_csv(io.StringIO(text))
        row = df.loc[df["data_id"] == FIRMS_SP]
        return pd.Timestamp(row["max_date"].iloc[0]).date() if len(row) else None
    except (requests.RequestException, KeyError, ValueError, pd.errors.ParserError) as exc:
        log.warning("FIRMS data_availability check failed (%s); using NRT for the last 90 days only", exc)
        return None


def _firms_log_path(raw_dir: Path) -> Path:
    return raw_dir / "fetch_log.json"


def firms_covered_days(raw_dir: Path = RAW_DIR / "firms") -> set[date]:
    """Days with at least one detection in the persistent raw files, plus days already requested
    (zero-fire days produce no rows, so the fetch log is what stops them being re-requested)."""
    days: set[date] = set()
    for f in _persistent_files(raw_dir, "fire_*.csv"):
        days.update(pd.to_datetime(pd.read_csv(f, usecols=["acq_date"])["acq_date"]).dt.date.unique())
    log_path = _firms_log_path(raw_dir)
    if log_path.exists():
        days.update(date.fromisoformat(d) for d in json.loads(log_path.read_text()).get("fetched_days", []))
    return days


def _record_firms_days(raw_dir: Path, new_days: list[date]) -> None:
    log_path = _firms_log_path(raw_dir)
    logged = set(json.loads(log_path.read_text()).get("fetched_days", [])) if log_path.exists() else set()
    logged.update(d.isoformat() for d in new_days)
    log_path.write_text(json.dumps({"fetched_days": sorted(logged)}, indent=0))


def _runs(days: list[date], max_len: int) -> list[tuple[date, int]]:
    """Group sorted days into (first_day, n_days) runs of consecutive days, at most max_len long."""
    runs: list[tuple[date, int]] = []
    for d in days:
        if runs and runs[-1][0] + timedelta(days=runs[-1][1]) == d and runs[-1][1] < max_len:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((d, 1))
    return runs


def fetch_firms_history(start: date, end: date, raw_dir: Path = RAW_DIR / "firms",
                        dry_run: bool = False, delay: float = 1.0) -> dict:
    missing = sorted(set(_days(start, end)) - firms_covered_days(raw_dir))
    log.info("fires: %d of %d days not yet requested", len(missing), (end - start).days + 1)
    out = {"missing_days": len(missing), "requests": 0}
    if not missing or dry_run:
        return out
    key = live._env("FIRMS_MAP_KEY")
    if not key:
        raise RuntimeError("No FIRMS_MAP_KEY. Get a free key from NASA FIRMS and put FIRMS_MAP_KEY=... in .env")
    sp_last = firms_sp_last_day(key) or (live.SGT_NOW().date() - timedelta(days=90))
    raw_dir.mkdir(parents=True, exist_ok=True)
    for first, n in _runs(missing, FIRMS_DAYS_PER_CALL):
        # Never let one request straddle the SP/NRT boundary.
        if first <= sp_last < first + timedelta(days=n - 1):
            n = (sp_last - first).days + 1
        source = FIRMS_SP if first <= sp_last else FIRMS_NRT
        url = live.FIRMS_URL.format(key=key, source=source, area=live.firms_area(), days=n, day=first.isoformat())
        text = live._get(url).text
        if text.lower().startswith(("invalid", "error")):
            raise RuntimeError(f"FIRMS refused the request: {text[:200]}")
        df = live.normalise_firms(text)
        if not df.empty:
            prefix = "fire_archive_api" if source == FIRMS_SP else "fire_nrt_api"
            for month, g in df.groupby(pd.to_datetime(df["acq_date"]).map(_month_key)):
                _merge_csv(raw_dir / f"{prefix}_{month}.csv", g, ["latitude", "longitude", "acq_date", "acq_time"])
        _record_firms_days(raw_dir, _days(first, first + timedelta(days=n - 1)))
        out["requests"] += 1
        time.sleep(delay)
    # The split at the SP boundary can leave a remainder; one more pass picks it up.
    if set(missing) - firms_covered_days(raw_dir):
        return {**out, **{"second_pass": fetch_firms_history(start, end, raw_dir, dry_run, delay)}}
    return out


# ---------------------------------------------------------------------------
def fetch_history(start: date | None = None, end: date | None = None,
                  sources: tuple[str, ...] = SOURCES, dry_run: bool = False) -> dict:
    start = start or date.fromisoformat(HISTORY_START)
    end = end or _yesterday_sgt()
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    unknown = set(sources) - set(SOURCES)
    if unknown:
        raise ValueError(f"Unknown source(s) {sorted(unknown)}; choose from {SOURCES}")
    fetchers = {"pm25": fetch_pm25_history, "weather": fetch_weather_history, "firms": fetch_firms_history}
    return {s: fetchers[s](start, end, dry_run=dry_run) for s in sources}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", type=date.fromisoformat, help=f"first day (default {HISTORY_START})")
    ap.add_argument("--end", type=date.fromisoformat, help="last day (default yesterday, Singapore time)")
    ap.add_argument("--sources", default=",".join(SOURCES), help="comma-separated subset of pm25,weather,firms")
    ap.add_argument("--dry-run", action="store_true", help="report what is missing without downloading")
    a = ap.parse_args()
    result = fetch_history(a.start, a.end, tuple(s.strip() for s in a.sources.split(",") if s.strip()), a.dry_run)
    print(json.dumps(result, indent=2, default=str))
