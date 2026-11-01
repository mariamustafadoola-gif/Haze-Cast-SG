"""Forecast-vs-actual chart: live forecasts over the holdout replay, which is display only."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from hazecast import dashboard as d
from hazecast.pipeline_store import TARGETS


def _frame(start, hours, step=1, seed=0):
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, periods=hours, freq=f"{step}h", tz="Asia/Singapore")
    rows = []
    for t in TARGETS:
        for h in (6, 12, 24):
            rows.append(pd.DataFrame(dict(forecast_time=times, region=t, horizon_hours=h,
                                          forecast_pm25=30 + rng.random(len(times)), persistence_pm25=28.0,
                                          elevated_flag=0, actual_pm25=29.0)))
    return pd.concat(rows, ignore_index=True)


def test_live_over_replay_with_go_live_marker():
    replay = _frame("2026-01-01", 24 * 300)
    live = _frame("2026-10-26 22:00", 12, step=7)          # sparse: one forecast every 7 hours
    fig = d.forecast_vs_actual(live, replay)
    names = {tr.name for tr in fig.data}
    assert {"Actual", "Replay (model trained before the holdout)", "Live forecast"} <= names
    live_tr = next(tr for tr in fig.data if tr.name == "Live forecast")
    assert "markers" in live_tr.mode                      # isolated hours between gaps stay visible
    assert int(np.isfinite(np.asarray(live_tr.y, dtype=float)).sum()) == 12
    assert any("live since" in (a.text or "") for a in fig.layout.annotations)
    assert "365 days" in fig.layout.title.text


def test_live_only_keeps_its_actuals_and_no_marker():
    fig = d.forecast_vs_actual(_frame("2026-10-26", 48))
    names = [tr.name for tr in fig.data if tr.visible]
    assert names[0] == "Actual" and "Live forecast" in names
    assert not any("live since" in (a.text or "") for a in fig.layout.annotations)


def test_dashboard_reads_the_replay_without_touching_the_live_ledger(tmp_path):
    from hazecast.pipeline_store import connect
    live_db, replay_db = tmp_path / "pipeline.sqlite", tmp_path / "backtest.sqlite"
    for path, frame in ((live_db, _frame("2026-10-26", 24)), (replay_db, _frame("2026-06-01", 24 * 30))):
        db = connect(path)
        with db:
            for i, r in enumerate(frame.itertuples()):
                db.execute("INSERT INTO forecasts (forecast_id, issued_at, as_of_time, forecast_time, region, horizon_hours,"
                           " model, manifest_sha256, forecast_pm25, threshold, elevated_flag, feature_age_hours, persistence_pm25)"
                           " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (f"{path.stem}{i}", r.forecast_time.isoformat(), r.forecast_time.isoformat(), r.forecast_time.isoformat(),
                            r.region, r.horizon_hours, "m", "x", r.forecast_pm25, 56.0, 0, 0.0, r.persistence_pm25))
        db.close()
    before = sqlite3.connect(live_db).execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
    out = d.build_dashboard(live_db, tmp_path / "dash.html", replay_path=replay_db)
    assert "Replay (model trained before the holdout)" in out.read_text(encoding="utf-8")
    assert sqlite3.connect(live_db).execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == before
