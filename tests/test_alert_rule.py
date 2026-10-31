"""Model-or-persistence alert rule: at +12/+24 h an elevated reading now also raises the flag."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from hazecast.inference import elevated, predict_final
from hazecast.pipeline_store import HORIZONS, TARGETS


def test_rule():
    assert elevated(60, 10, 6, 56) and elevated(60, 10, 24, 56)      # the model alone
    assert not elevated(40, 70, 6, 56)                               # +6 h: the model decides
    assert elevated(40, 70, 12, 56) and elevated(40, 70, 24, 56)     # hazy now -> flagged ahead
    assert not elevated(40, 50, 24, 56)                              # neither reaches 56


def _setup(tmp_path, now_pm25, forecast):
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    dates = pd.date_range(end=now, periods=48, freq="h")
    gold = pd.DataFrame({"date": dates, "x": np.random.default_rng(0).normal(size=len(dates))})
    for t in TARGETS:
        gold[f"now_{t}"] = float(now_pm25)
    gold.to_parquet(tmp_path / "hourly.parquet")
    cols = ["x"] + [f"now_{t}" for t in TARGETS]
    joblib.dump(DummyRegressor(strategy="constant", constant=forecast).fit(gold[cols], gold["now_north"]),
                tmp_path / "m.joblib")
    entry = dict(model="xgboost_q85", artifact="m.joblib", training_label_end="2026-01-01T00:00:00+08:00")
    (tmp_path / "model_manifest.json").write_text(json.dumps(dict(
        schema_version=1, pipeline="hourly", horizons_hours=list(HORIZONS), feature_columns=cols, threshold=56.0,
        models={str(h): {t: entry for t in TARGETS} for h in HORIZONS})))
    return predict_final(tmp_path / "hourly.parquet", tmp_path, now=datetime.now(timezone.utc) + timedelta(minutes=5))


@pytest.mark.parametrize("now_pm25,forecast,flags", [
    (70, 40, {6: 0, 12: 1, 24: 1}),    # haze now, model expects it to clear
    (30, 40, {6: 0, 12: 0, 24: 0}),    # clear now, clear forecast
    (30, 60, {6: 1, 12: 1, 24: 1}),    # model sees haze coming
])
def test_archived_flags_follow_the_rule(tmp_path, now_pm25, forecast, flags):
    records = _setup(tmp_path, now_pm25, forecast)
    assert {r["horizon_hours"]: r["elevated_flag"] for r in records} == flags
    assert all(r["forecast_pm25"] == forecast and r["persistence_pm25"] == now_pm25 for r in records)
