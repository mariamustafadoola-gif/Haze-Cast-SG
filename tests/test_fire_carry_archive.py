"""The forecast archive records how long the fire features were carried forward (late FIRMS)."""
from __future__ import annotations

import sqlite3

from hazecast.pipeline_store import connect

OLD_SCHEMA = """
CREATE TABLE forecasts (
    forecast_id TEXT PRIMARY KEY,
    issued_at TEXT NOT NULL, as_of_time TEXT NOT NULL,
    forecast_time TEXT NOT NULL, region TEXT NOT NULL,
    horizon_hours INTEGER NOT NULL, model TEXT NOT NULL,
    model_run_id TEXT, manifest_sha256 TEXT NOT NULL,
    forecast_pm25 REAL NOT NULL, threshold REAL NOT NULL,
    elevated_flag INTEGER NOT NULL, feature_age_hours REAL NOT NULL,
    persistence_pm25 REAL NOT NULL,
    UNIQUE(as_of_time, region, horizon_hours)
);
INSERT INTO forecasts VALUES ('a', 't0', 't0', 't6', 'west', 6, 'xgboost_q85', NULL, 'x', 30, 56, 0, 0.2, 28);
"""


def test_existing_archive_gains_the_column_and_old_rows_read_as_current(tmp_path):
    path = tmp_path / "pipeline.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(OLD_SCHEMA)
    db = connect(path)
    try:
        assert db.execute("SELECT fire_carried_hours FROM forecasts WHERE forecast_id='a'").fetchone()[0] == 0
        db.execute("INSERT INTO forecasts (forecast_id, issued_at, as_of_time, forecast_time, region, horizon_hours,"
                   " model, manifest_sha256, forecast_pm25, threshold, elevated_flag, feature_age_hours,"
                   " persistence_pm25, fire_carried_hours) VALUES ('b','t1','t1','t7','west',6,'m','x',30,56,0,0.2,28,5)")
        assert db.execute("SELECT fire_carried_hours FROM forecasts WHERE forecast_id='b'").fetchone()[0] == 5
    finally:
        db.close()
    connect(path).close()                       # reconnecting does not try to add it again
