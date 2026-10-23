"""Live ingestion: parsing and file formats, checked against mocked API
responses so the tests run offline. To try the real APIs:
python -m hazecast.ingest.live --days 3
"""
from __future__ import annotations
from datetime import date, datetime

import pandas as pd

from hazecast.ingest import live
from hazecast.ingest.pm25 import load_pm25_bronze
from hazecast.ingest.weather import load_weather_bronze
from hazecast.ingest.firms import load_firms_bronze


class FakeResponse:
    def __init__(self, json_body=None, text=""):
        self._json, self.text, self.status_code = json_body, text, 200

    def json(self):
        return self._json

    def raise_for_status(self):
        pass


def _pm25_body(day: str, hours, token=None):
    items = [{"date": day, "updatedTimestamp": f"{day}T{h:02d}:05:00+08:00", "timestamp": f"{day}T{h:02d}:00:00+08:00",
              "readings": {"pm25_one_hourly": {"national": 20, "north": 10 + h, "south": 11, "east": 12, "west": 13, "central": 14}}}
             for h in hours]
    data = {"regionMetadata": [], "items": items}
    if token:
        data["paginationToken"] = token
    return {"code": 0, "errorMsg": "", "data": data}


def test_pm25_fetch_follows_pages_and_writes_nea_format(monkeypatch, tmp_path):
    calls = []

    def fake_get(url, params=None, headers=None, tries=3):
        calls.append(params)
        if "paginationToken" in params:
            return FakeResponse(_pm25_body(params["date"], [12, 13]))
        return FakeResponse(_pm25_body(params["date"], [0, 1], token="next"))

    monkeypatch.setattr(live, "_get", fake_get)
    df = live.fetch_pm25(date(2026, 11, 2), date(2026, 11, 3))
    assert len(calls) == 4 and len(df) == 8
    assert df["ts"].min() == pd.Timestamp("2026-11-02 00:00")
    path = live.write_pm25_raw(df, tmp_path)
    assert path.read_text().splitlines()[1].startswith("2/11/2026 0:00,10,11,12,13,14")
    bronze = load_pm25_bronze(tmp_path)  # the normal loader reads it unchanged
    assert len(bronze) == 8 and bronze["north"].max() == 23


def test_weather_csv_trims_future_and_loads(monkeypatch, tmp_path):
    def fake_csv(lat, lon, past_days):
        head = ("latitude,longitude,elevation,utc_offset_seconds,timezone,timezone_abbreviation\n"
                f"{lat},{lon},10.0,28800,Asia/Singapore,GMT+8\n\n"
                "time,wind_speed_10m (km/h),wind_direction_10m (°),precipitation (mm)\n")
        rows = "".join(f"2026-11-03T{h:02d}:00,5.0,200,0.10\n" for h in range(24))
        return head + rows

    monkeypatch.setattr(live, "fetch_weather_csv", fake_csv)
    live.write_weather_raw(2, tmp_path, now=datetime(2026, 11, 3, 9, 30))
    w = load_weather_bronze(tmp_path)
    assert set(w["point"]) == set(live.WEATHER_POINTS)
    assert w.groupby("point")["ts"].max().eq(pd.Timestamp("2026-11-03 09:00")).all()


def test_firms_pads_time_and_loads(monkeypatch, tmp_path):
    csv = ("latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
           "0.5,101.0,330.0,0.4,0.4,2026-11-02,402,N,VIIRS,n,2.0NRT,290.0,5.0,D\n"
           "0.0,112.0,330.0,0.4,0.4,2026-11-03,1730,N,VIIRS,h,2.0NRT,290.0,9.0,N\n")
    monkeypatch.setattr(live, "_get", lambda url, **k: FakeResponse(text=csv))
    monkeypatch.setattr(live, "_env", lambda name: "TESTKEY")
    df = live.fetch_firms(date(2026, 11, 2), date(2026, 11, 3))
    assert df["acq_time"].tolist()[:2] == ["0402", "1730"]
    live.write_firms_raw(df, tmp_path)
    f = load_firms_bronze(tmp_path)
    assert set(f["region"]) == {"sumatra", "kalimantan"}


def test_firms_area_covers_both_fire_regions():
    w, s, e, n = map(float, live.firms_area().split(","))
    assert w <= 95.0 and e >= 119.5 and s <= -6.2 and n >= 6.0
