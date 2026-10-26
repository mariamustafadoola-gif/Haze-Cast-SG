"""Historical backfill: coverage detection and request planning, offline.

The real APIs were exercised by hand (see README "End-to-end pipeline");
these tests pin the logic that decides what to download.
"""
from __future__ import annotations
import json
from datetime import date

import pandas as pd

from hazecast.ingest import historical as h
from hazecast.ingest.firms import load_firms_bronze


def _pm25_csv(path, start, hours):
    ts = pd.date_range(start, periods=hours, freq="h")
    pd.DataFrame({"1hr_pm25": [f"{t.day}/{t.month}/{t.year} {t.hour}:{t.minute:02d}" for t in ts],
                  **{r: 10 for r in ("north", "south", "east", "west", "central")}}).to_csv(path, index=False)


def test_pm25_coverage_ignores_live_files_and_partial_days(tmp_path):
    _pm25_csv(tmp_path / "Historical1hrPM2.5.csv", "2026-01-01 00:00", 24 * 2 + 5)  # 2 full days + 5 h
    _pm25_csv(tmp_path / "Historical1hrPM2.5_zlive.csv", "2026-01-04 00:00", 24)  # live: not history
    covered = h.pm25_covered_days(tmp_path)
    assert covered == {date(2026, 1, 1), date(2026, 1, 2)}


def test_pm25_small_gap_uses_daily_api_not_bulk(tmp_path, monkeypatch):
    _pm25_csv(tmp_path / "Historical1hrPM2.5.csv", "2026-01-01 00:00", 24 * 3)
    fetched = []

    def fake_fetch(start, end):
        fetched.append(start)
        ts = pd.date_range(pd.Timestamp(start), periods=24, freq="h")
        return pd.DataFrame({"ts": ts, **{r: 20.0 for r in ("north", "south", "east", "west", "central")}})

    monkeypatch.setattr(h.live, "fetch_pm25", fake_fetch)
    monkeypatch.setattr(h, "download_pm25_bulk", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no bulk")))
    out = h.fetch_pm25_history(date(2026, 1, 1), date(2026, 1, 5), raw_dir=tmp_path, delay=0)
    assert fetched == [date(2026, 1, 4), date(2026, 1, 5)]
    assert out["api_days"] == 2
    assert h.pm25_covered_days(tmp_path) >= {date(2026, 1, d) for d in range(1, 6)}


def test_runs_group_consecutive_days_up_to_limit():
    days = [date(2026, 1, d) for d in (1, 2, 3, 4, 5, 6, 7, 10, 11)]
    assert h._runs(days, 5) == [(date(2026, 1, 1), 5), (date(2026, 1, 6), 2), (date(2026, 1, 10), 2)]


FIRMS_HEADER = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"


def test_firms_history_splits_at_sp_boundary_and_logs_zero_fire_days(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, params=None, headers=None, tries=3):
        parts = url.split("/")
        source, days, day = parts[-4], int(parts[-2]), parts[-1]
        calls.append((source, day, days))
        body = FIRMS_HEADER
        if day == "2026-01-01":  # one detection on the first day, none afterwards
            body += "0.5,101.0,330,0.4,0.4,2026-01-01,402,N,VIIRS,n,2,290,5.0,D\n"
        return type("R", (), {"text": body})()

    monkeypatch.setattr(h.live, "_get", fake_get)
    monkeypatch.setattr(h.live, "_env", lambda name: "KEY" if name == "FIRMS_MAP_KEY" else None)
    monkeypatch.setattr(h, "firms_sp_last_day", lambda key: date(2026, 1, 3))
    h.fetch_firms_history(date(2026, 1, 1), date(2026, 1, 8), raw_dir=tmp_path, delay=0)

    assert calls[0] == ("VIIRS_SNPP_SP", "2026-01-01", 3)           # stops at the SP boundary
    assert all(c[0] == "VIIRS_SNPP_NRT" for c in calls[1:])          # NRT afterwards
    assert sum(c[2] for c in calls) == 8                             # every day requested exactly once
    logged = json.loads((tmp_path / "fetch_log.json").read_text())["fetched_days"]
    assert len(logged) == 8                                          # zero-fire days remembered too
    # Re-running requests nothing.
    calls.clear()
    h.fetch_firms_history(date(2026, 1, 1), date(2026, 1, 8), raw_dir=tmp_path, delay=0)
    assert calls == []
    # API archive files have no `type` column; the loader keeps them (and pads acq_time).
    bronze = load_firms_bronze(tmp_path)
    assert len(bronze) == 1 and bronze.region.iloc[0] == "sumatra"


def test_weather_clean_drops_empty_and_future_rows():
    text = ("latitude,longitude,elevation,utc_offset_seconds,timezone,timezone_abbreviation\n"
            "1.37,103.8,62.0,28800,Asia/Singapore,GMT+8\n\n"
            "time,wind_speed_10m (km/h),wind_direction_10m (°),precipitation (mm)\n"
            "2026-01-01T00:00,4.7,94,0.00\n2026-01-01T01:00,,,\n2026-01-01T02:00,3.0,90,0.1\n")
    head, body = h._clean_weather_csv(text, pd.Timestamp("2026-01-01 01:30").to_pydatetime())
    assert len(head) == 4 and body == ["2026-01-01T00:00,4.7,94,0.00"]


def _meteo_csv(start, hours, wind):
    ts = pd.date_range(start, periods=hours, freq="h")
    head = ("latitude,longitude,elevation,utc_offset_seconds,timezone,timezone_abbreviation\n"
            "1.37,103.8,62.0,28800,Asia/Singapore,GMT+8\n\n"
            "time,wind_speed_10m (km/h),wind_direction_10m (°),precipitation (mm)\n")
    return head + "".join(f"{t:%Y-%m-%dT%H:%M},{wind},90,0.00\n" for t in ts)


def test_weather_top_up_refetches_recent_days_and_new_values_win(tmp_path, monkeypatch):
    from hazecast.config import WEATHER_POINTS
    monkeypatch.setattr(h, "WEATHER_POINTS", {"singapore": WEATHER_POINTS["singapore"]})
    monkeypatch.setattr(h.live, "SGT_NOW", lambda: pd.Timestamp("2026-01-10 05:00").to_pydatetime())
    stored = tmp_path / "open-meteo-1.37N103.80E_hist_singapore_2026.csv"
    stored.write_text(_meteo_csv("2026-01-01 00:00", 24 * 9, wind=1.0), encoding="utf-8")  # 1-9 Jan, preliminary
    requests = []

    def fake_fetch(lat, lon, start, end):
        requests.append((start, end))
        return _meteo_csv(f"{start} 00:00", ((end - start).days + 1) * 24, wind=7.0)  # revised values

    monkeypatch.setattr(h, "fetch_weather_year", fake_fetch)
    out = h.fetch_weather_history(date(2026, 1, 1), date(2026, 1, 9), raw_dir=tmp_path, delay=0, refetch_days=2)

    assert requests == [(date(2026, 1, 8), date(2026, 1, 9))]          # only the last 2 days, though all were stored
    assert out["singapore"]["missing_hours"] == 48
    rows = pd.read_csv(stored, skiprows=3)
    rows.columns = ["ts", "wind", "dir", "rain"]
    assert rows.ts.is_unique and len(rows) == 24 * 9                   # replaced in place, no duplicate hours
    assert (rows.set_index("ts").wind.loc["2026-01-08T00:00":] == 7.0).all()
    assert (rows.set_index("ts").wind.loc[:"2026-01-07T23:00"] == 1.0).all()
