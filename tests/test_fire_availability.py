"""FIRMS availability: telling "no fires" from "not delivered yet" at the live edge."""
from __future__ import annotations

import json

import pandas as pd

from hazecast import bronze_to_silver as b2s
from hazecast.ingest import live


def _bronze(tmp_path, last_detection: str):
    path = tmp_path / "firms.parquet"
    ts = pd.date_range(pd.Timestamp(last_detection) - pd.Timedelta(days=2), last_detection, freq="12h")
    pd.DataFrame({"ts": ts, "region": "kalimantan"}).to_parquet(path)
    return path


def _availability(tmp_path, max_date: str, checked_at: str):
    path = tmp_path / "availability.json"
    path.write_text(json.dumps(dict(source="VIIRS_SNPP_NRT", max_date=max_date, checked_at=checked_at)))
    return path


def test_no_check_saved_keeps_the_15_hour_rule(tmp_path):
    grid = b2s.firms_hourly_silver(_bronze(tmp_path, "2026-10-28 01:00"), tmp_path / "missing.json")
    assert grid.ts.max() == pd.Timestamp("2026-10-28 16:00")


def test_a_late_pass_on_an_unlisted_day_stays_unknown(tmp_path):
    # 28 Oct 2026, 19:38 SGT: FIRMS still listed only 27 Oct; the 13:00 SGT pass had not arrived
    avail = _availability(tmp_path, "2026-10-27", "2026-10-28T11:38:00+00:00")
    assert b2s.firms_processed_until(avail) == pd.Timestamp("2026-10-28 08:00")   # end of 27 Oct UTC, in SGT
    grid = b2s.firms_hourly_silver(_bronze(tmp_path, "2026-10-28 01:00"), avail)
    assert grid.ts.max() == pd.Timestamp("2026-10-28 16:00")                     # unchanged: the pass is unknown


def test_quiet_days_without_fires_are_true_zeros(tmp_path):
    # last detection three days ago, but FIRMS has processed today: an empty period means no fires
    avail = _availability(tmp_path, "2026-11-12", "2026-11-12T12:00:00+00:00")
    until = b2s.firms_processed_until(avail)
    assert until == pd.Timestamp("2026-11-12 16:00")                             # 4 h before the check, SGT
    grid = b2s.firms_hourly_silver(_bronze(tmp_path, "2026-11-09 02:00"), avail)
    assert grid.ts.max() == until
    assert (grid.loc[grid.ts > "2026-11-09 03:00", ["sumatra", "kalimantan", "total"]] == 0).all().all()


def test_an_unreadable_check_is_ignored(tmp_path):
    bad = tmp_path / "availability.json"
    bad.write_text("{not json")
    assert b2s.firms_processed_until(bad) is None


def test_availability_is_read_from_the_firms_service(monkeypatch):
    class Response:
        text = "data_id,min_date,max_date\nVIIRS_NOAA20_NRT,2026-08-01,2026-10-29\nVIIRS_SNPP_NRT,2026-08-01,2026-10-30\n"
    monkeypatch.setattr(live, "_env", lambda name: "key")
    monkeypatch.setattr(live, "_get", lambda url: Response())
    info = live.fetch_firms_availability()
    assert info["source"] == "VIIRS_SNPP_NRT" and info["max_date"] == "2026-10-30" and "checked_at" in info
