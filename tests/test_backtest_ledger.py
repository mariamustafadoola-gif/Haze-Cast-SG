"""The holdout replay writes its own ledger with the live forecasts table, so it must keep up
with that table's columns (a positional insert broke when fire_carried_hours was added)."""
from __future__ import annotations

import pandas as pd

from hazecast.monitor_backtest import write_ledger
from hazecast.pipeline_store import SGT, TARGETS


def test_replay_ledger_writes_with_the_current_forecasts_table(tmp_path):
    origin = pd.Timestamp("2025-01-01 00:00", tz=SGT)
    holdout = pd.DataFrame([dict(as_of_time=origin, forecast_time=origin + pd.Timedelta(hours=6), region=t,
                                 horizon_hours=6, model="xgboost_q85", predicted_pm25=20.0, threshold=56.0,
                                 persistence_pm25=18.0) for t in TARGETS])
    gold = pd.DataFrame({"date": [origin], **{f"now_{t}": [18.0] for t in TARGETS}})
    db = write_ledger(tmp_path / "backtest.sqlite", holdout, gold, "sha")
    try:
        rows = db.execute("SELECT region, forecast_pm25, fire_carried_hours FROM forecasts").fetchall()
        assert len(rows) == len(TARGETS) and all(r["fire_carried_hours"] == 0 for r in rows)
    finally:
        db.close()
