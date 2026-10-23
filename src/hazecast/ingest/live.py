"""Fetch the latest PM2.5, weather and fire data from the live APIs.

The rest of the pipeline only reads files under data/raw/. This module
downloads recent data and saves it there in exactly the same format as the
historical files, so ingest -> bronze -> silver -> gold works unchanged.

    python -m hazecast.ingest.live              # last 30 days up to now
    python -m hazecast.ingest.live --days 10

Sources:
  PM2.5   https://api-open.data.gov.sg/v2/real-time/api/pm25?date=YYYY-MM-DD
          (optional key in env DATA_GOV_SG_API_KEY, sent as x-api-key)
  Weather https://api.open-meteo.com/v1/forecast  (past_days, no key)
  Fires   https://firms.modaps.eosdis.nasa.gov/api/area/csv/<KEY>/VIIRS_SNPP_NRT/<area>/<days>/<date>
          (needs FIRMS_MAP_KEY in env or .env, free from NASA FIRMS)

Files written (overwritten each run, so re-running is safe):
  data/raw/pm25/Historical1hrPM2.5_zlive.csv
  data/raw/weather/open-meteo-<lat>N<lon>E_live_<point>.csv  (one per place)
  data/raw/firms/fire_nrt_zlive.csv
Overlaps with older files are removed later by the normal ingest step
(duplicate hours and duplicate fire detections are dropped there).
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from hazecast.config import RAW_DIR, REPO_ROOT, SG_REGIONS, WEATHER_POINTS, FIRE_REGIONS

log = logging.getLogger(__name__)
TIMEOUT = 60
SGT_NOW = lambda: datetime.utcnow() + timedelta(hours=8)  # noqa: E731  Singapore time, tz-naive


def _env(name: str) -> str | None:
    """Read a setting from the environment, or from the project's .env file."""
    if os.environ.get(name):
        return os.environ[name]
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _get(url: str, params: dict | None = None, headers: dict | None = None, tries: int = 3) -> requests.Response:
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
            if r.status_code == 429 and attempt < tries:  # rate limited: wait and retry
                time.sleep(10 * attempt)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == tries:
                raise
            time.sleep(3 * attempt)
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# PM2.5 (data.gov.sg)
# ---------------------------------------------------------------------------
PM25_URL = "https://api-open.data.gov.sg/v2/real-time/api/pm25"


def parse_pm25_items(items: list[dict]) -> pd.DataFrame:
    """API items -> rows of ts (SGT, tz-naive) + the 5 regions."""
    rows = []
    for it in items:
        ts = pd.Timestamp(it["timestamp"])
        if ts.tzinfo is not None:
            ts = ts.tz_convert("Asia/Singapore").tz_localize(None)
        vals = it["readings"]["pm25_one_hourly"]
        rows.append({"ts": ts, **{r: vals.get(r) for r in SG_REGIONS}})
    return pd.DataFrame(rows, columns=["ts", *SG_REGIONS])


def fetch_pm25(start: date, end: date) -> pd.DataFrame:
    key = _env("DATA_GOV_SG_API_KEY")
    headers = {"x-api-key": key} if key else None
    frames = []
    d = start
    while d <= end:
        params = {"date": d.isoformat()}
        while True:
            body = _get(PM25_URL, params=params, headers=headers).json()
            if body.get("code") not in (0, None):
                raise RuntimeError(f"data.gov.sg error for {d}: {body.get('errorMsg')}")
            data = body.get("data") or {}
            frames.append(parse_pm25_items(data.get("items", [])))
            token = data.get("paginationToken")
            if not token:
                break
            params = {"date": d.isoformat(), "paginationToken": token}
        d += timedelta(days=1)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ts", *SG_REGIONS])
    df = df.dropna(subset=["ts"]).drop_duplicates("ts").sort_values("ts")
    log.info("PM2.5: %d hourly readings, %s to %s", len(df), df["ts"].min(), df["ts"].max())
    return df


def write_pm25_raw(df: pd.DataFrame, raw_dir: Path = RAW_DIR / "pm25") -> Path:
    """Same layout as NEA's historical export: '1hr_pm25' = 'd/m/YYYY H:MM'."""
    out = df.copy()
    out.insert(0, "1hr_pm25", [f"{t.day}/{t.month}/{t.year} {t.hour}:{t.minute:02d}" for t in out.pop("ts")])
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "Historical1hrPM2.5_zlive.csv"  # sorts last, so older files win on overlaps
    out.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Weather (Open-Meteo)
# ---------------------------------------------------------------------------
METEO_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_weather_csv(lat: float, lon: float, past_days: int) -> str:
    params = {
        "latitude": lat, "longitude": lon,
        # same three columns, same order, as the historical files
        "hourly": "wind_speed_10m,wind_direction_10m,precipitation",
        "past_days": min(past_days, 92), "forecast_days": 1,
        "timezone": "Asia/Singapore", "format": "csv",
    }
    return _get(METEO_URL, params=params).text


def trim_future_hours(csv_text: str, now: datetime) -> str:
    """Keep Open-Meteo's 3-line header, drop hours after `now` (those are forecasts, not observations)."""
    lines = csv_text.strip().splitlines()
    head, body = lines[:4], lines[4:]
    keep = [ln for ln in body if ln and pd.Timestamp(ln.split(",")[0]) <= pd.Timestamp(now)]
    return "\n".join(head + keep) + "\n"


def write_weather_raw(past_days: int, raw_dir: Path = RAW_DIR / "weather", now: datetime | None = None) -> list[Path]:
    now = now or SGT_NOW()
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for point, (lat, lon) in WEATHER_POINTS.items():
        text = trim_future_hours(fetch_weather_csv(lat, lon, past_days), now)
        ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
        path = raw_dir / f"open-meteo-{abs(lat):.2f}{ns}{abs(lon):.2f}{ew}_live_{point}.csv"
        path.write_text(text)
        paths.append(path)
        log.info("weather[%s]: %d hours", point, max(len(text.splitlines()) - 4, 0))
    return paths


# ---------------------------------------------------------------------------
# Fires (NASA FIRMS)
# ---------------------------------------------------------------------------
FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{area}/{days}/{day}"
FIRMS_SOURCE = "VIIRS_SNPP_NRT"  # same satellite as the historical nrt files


def firms_area() -> str:
    west = min(b["lon_min"] for b in FIRE_REGIONS.values())
    east = max(b["lon_max"] for b in FIRE_REGIONS.values())
    south = min(b["lat_min"] for b in FIRE_REGIONS.values())
    north = max(b["lat_max"] for b in FIRE_REGIONS.values())
    return f"{west},{south},{east},{north}"


def normalise_firms(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(csv_text), dtype={"acq_time": str})
    if df.empty:
        return df
    df["acq_time"] = df["acq_time"].str.zfill(4)  # "402" -> "0402", matches the historical files
    return df


def fetch_firms(start: date, end: date) -> pd.DataFrame:
    key = _env("FIRMS_MAP_KEY")
    if not key:
        raise RuntimeError("No FIRMS_MAP_KEY. Get a free key from NASA FIRMS and put FIRMS_MAP_KEY=... in .env")
    frames, d = [], start
    while d <= end:
        days = min(10, (end - d).days + 1)  # the API allows 1 to 10 days per call
        url = FIRMS_URL.format(key=key, source=FIRMS_SOURCE, area=firms_area(), days=days, day=d.isoformat())
        text = _get(url).text
        if text.lower().startswith(("invalid", "error")):
            raise RuntimeError(f"FIRMS refused the request: {text[:200]}")
        frames.append(normalise_firms(text))
        d += timedelta(days=days)
    frames = [f for f in frames if not f.empty]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log.info("fires: %d detections", len(df))
    return df


def write_firms_raw(df: pd.DataFrame, raw_dir: Path = RAW_DIR / "firms") -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "fire_nrt_zlive.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
def fetch_all(days: int = 30, now: datetime | None = None) -> dict[str, object]:
    now = now or SGT_NOW()
    start = (now - timedelta(days=days)).date()
    out = {}
    out["pm25"] = write_pm25_raw(fetch_pm25(start, now.date()))
    out["weather"] = write_weather_raw(days + 1, now=now)
    out["firms"] = write_firms_raw(fetch_firms(start, now.date()))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=30, help="how many days back to fetch (default 30)")
    fetch_all(ap.parse_args().days)
