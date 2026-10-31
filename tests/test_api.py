"""API + dashboard: routes, the hourly forecast contract the dashboard relies on, and the
page itself (it must only call endpoints that exist, and must not show raw floats)."""
from __future__ import annotations
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import serving.app as api

STATIC = Path(api.__file__).parent / "static" / "index.html"


@pytest.fixture
def client(tmp_path, monkeypatch):
    rows = [dict(region=r, horizon_hours=h, forecast_pm25=20.123456 + h, as_of_time="2026-10-26T14:00:00+00:00",
                 forecast_time="2026-10-26T20:00:00+00:00", elevated_flag=0, threshold=56.0)
            for r in ("north", "south", "east", "west", "central", "national") for h in (6, 12, 24)]
    path = tmp_path / "latest_hourly_forecast.json"
    path.write_text(json.dumps(rows))
    monkeypatch.setattr(api, "HOURLY_FORECAST_PATH", path)
    return TestClient(api.app)


def test_health_and_root_redirects_to_dashboard(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/ui/"


def test_dashboard_is_served(client):
    r = client.get("/ui/")
    assert r.status_code == 200 and "HazeCast SG" in r.text


def test_hourly_forecast_has_the_fields_the_dashboard_reads(client):
    rows = client.get("/forecast/hourly").json()
    assert len(rows) == 18
    assert {"region", "horizon_hours", "forecast_pm25", "as_of_time"} <= set(rows[0])
    assert len(client.get("/forecast/hourly", params={"region": "west", "horizon": 6}).json()) == 1


def test_missing_forecast_is_503_so_the_dashboard_falls_back_to_demo(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "HOURLY_FORECAST_PATH", tmp_path / "nothing.json")
    assert TestClient(api.app).get("/forecast/hourly").status_code == 503


def test_dashboard_only_calls_endpoints_that_exist():
    html = STATIC.read_text(encoding="utf-8")
    called = set(re.findall(r'fetch(?:JSON)?\("(/[^"]*)"\)', html))
    routes = {r.path for r in api.app.routes}
    assert called, "no API calls found in the dashboard"
    assert called <= routes, f"dashboard calls missing endpoints: {called - routes}"
    assert "/forecast/hourly" in called          # /forecast is the next-day daily product here


def test_dashboard_rounds_values_and_flags_stale_forecasts():
    html = STATIC.read_text(encoding="utf-8")
    assert "fmtPm(" in html and not re.search(r"\$\{main\.forecast_pm25\}", html)
    assert "STALE_HOURS" in html
    docs_copy = STATIC.parents[2] / "docs" / "index.html"
    assert docs_copy.read_text(encoding="utf-8") == html  # GitHub Pages copy stays identical


def test_source_status_names_each_upstream_source(client, monkeypatch):
    import hazecast.model_monitor as mm
    monkeypatch.setattr(api, "_SOURCES_CACHE", {})
    monkeypatch.setattr(mm, "source_health", lambda now: {"pm25": {"age_hours": 0.5, "status": "ok"},
                                                          "fires": {"age_hours": 30.0, "status": "alert"}})
    body = client.get("/health/sources").json()
    assert body["sources"]["fires"]["status"] == "alert" and "checked_at" in body


def test_dashboard_greys_out_target_times_that_have_passed():
    html = STATIC.read_text(encoding="utf-8")
    assert "function isPassed(row)" in html and "dataMode === MODE_LIVE" in html   # never applied to demo data
    assert html.count("isPassed(") >= 3      # forecast values per tab, the notice, and its definition
    assert ".slot.passed" in html and "waiting for " in html


def test_context_gives_latest_readings_and_accuracy(tmp_path, monkeypatch):
    import pandas as pd
    hours = pd.date_range("2026-10-16 20:00", periods=4, freq="h")
    gold = pd.DataFrame({"date": hours, **{f"now_{r}": 60.0 for r in ("north", "south", "east", "west", "central")},
                         "now_national": 60.0})
    gold.loc[3, "now_national"] = None   # newest hour not complete yet, so the one before is used
    path = tmp_path / "hourly_features.parquet"
    gold.to_parquet(path)
    monkeypatch.setattr(api, "GOLD_HOURLY_PATH", path)
    monkeypatch.setattr(api, "MONITOR_PATH", tmp_path / "none.json")
    c = TestClient(api.app).get("/context").json()
    assert c["now"]["west"] == 60.0 and c["as_of_time"].startswith("2026-10-16T22:00")
    assert set(c) == {"as_of_time", "now", "accuracy", "monitoring"}
    assert c["monitoring"] is None


def test_missing_gold_makes_context_503(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "GOLD_HOURLY_PATH", tmp_path / "nothing.parquet")
    assert TestClient(api.app).get("/context").status_code == 503
