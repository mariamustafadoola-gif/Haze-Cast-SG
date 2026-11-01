"""Quantile range: three quantile models per region/horizon, one primary per horizon."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from hazecast import governance as g
from hazecast.hourly_model import BAND, QUANTILE_FAMILY, _choose_quantile, incumbent_quantiles
from hazecast.inference import _band, archive_forecasts, predict_final
from hazecast.pipeline_store import BAND_COLUMNS, HORIZONS, TARGETS, connect
from hazecast.train import _make_model, quantile_alpha, quantile_name


def test_band_names_and_models():
    assert BAND == ("xgboost_q85", "xgboost_q90", "xgboost_q95")
    assert [quantile_alpha(n) for n in BAND] == [0.85, 0.90, 0.95]
    assert quantile_name(0.9) == "xgboost_q90" and quantile_alpha("ridge") is None
    assert _make_model("xgboost_q95").get_params()["quantile_alpha"] == 0.95
    with pytest.raises(ValueError):
        _make_model("xgboost_qx")


def _haze(seed=0, n=24 * 7 * 30):
    rng = np.random.default_rng(seed)
    y = np.clip(30 + np.cumsum(rng.normal(size=n)) * 0.8, 0, None)
    y[rng.random(n) < 0.08] += 60                     # haze hours
    dates = pd.date_range("2019-01-01", periods=n, freq="h", tz="Asia/Singapore")
    return y, dates, rng


def test_a_clearly_better_quantile_replaces_the_incumbent():
    y, dates, rng = _haze()
    preds = {"xgboost_q85": y - 25 + rng.normal(size=len(y)),        # misses most haze
             "xgboost_q90": y - 5 + rng.normal(size=len(y)),
             "xgboost_q95": y + 2 + rng.normal(size=len(y))}          # catches it
    out = _choose_quantile(y, dates, preds, "xgboost_q85")
    assert out["proposed"] == "xgboost_q95" and out["chosen"] == "xgboost_q95" and out["confidence"] >= 0.9


def test_a_thin_edge_keeps_the_incumbent_and_quiet_data_changes_nothing():
    y, dates, rng = _haze(1)
    noise = rng.normal(size=len(y)) * 6
    preds = {n: y + noise + rng.normal(size=len(y)) * 0.3 for n in BAND}
    out = _choose_quantile(y, dates, preds, "xgboost_q90")
    assert out["chosen"] == "xgboost_q90"
    quiet = np.full(len(y), 10.0)
    out = _choose_quantile(quiet, dates, {n: quiet + i for i, n in enumerate(BAND)}, "xgboost_q95")
    assert out["chosen"] == "xgboost_q95" and out["confidence"] is None
    assert _choose_quantile(quiet, dates, {n: quiet for n in BAND}, "ridge")["chosen"] == QUANTILE_FAMILY


def test_incumbent_is_read_from_the_published_manifest(tmp_path):
    path = tmp_path / "model_manifest.json"
    assert incumbent_quantiles(path) == {}
    path.write_text(json.dumps({"quantile_selection": {"6": {"chosen": "xgboost_q90"}, "24": {"chosen": "bogus"}}}))
    assert incumbent_quantiles(path) == {6: "xgboost_q90"}


def test_band_is_sorted_and_empty_without_range_models():
    entry = {"band": {"xgboost_q85": "a", "xgboost_q90": "b", "xgboost_q95": "c"}}
    crossing = {"a": 50.0, "b": 41.0, "c": 47.0}
    assert _band(entry, crossing.get) == {"xgboost_q85": 41.0, "xgboost_q90": 47.0, "xgboost_q95": 50.0}
    assert _band({"artifact": "x"}, crossing.get) == {}


def _range_dir(path, gold, cols, levels=(40.0, 45.0, 52.0), primary="xgboost_q95"):
    path.mkdir(parents=True, exist_ok=True)
    models = {}
    for h in HORIZONS:
        models[str(h)] = {}
        for t in TARGETS:
            band = {}
            for name, level in zip(BAND, levels):
                est = DummyRegressor(strategy="constant", constant=level).fit(gold[cols], gold["now_north"])
                band[name] = f"{t}_h{h}_{name}.joblib"
                joblib.dump(est, path / band[name])
            models[str(h)][t] = dict(model=primary, artifact=band[primary], band=band,
                                     quantile_alpha=quantile_alpha(primary),
                                     training_label_end="2026-01-01T00:00:00+08:00",
                                     test_start="2026-02-01T00:00:00+08:00", final_test=dict(mae=1.0, f2=None))
    (path / g.MANIFEST).write_text(json.dumps(dict(
        schema_version=1, pipeline="hourly", horizons_hours=list(HORIZONS), feature_columns=cols,
        threshold=56.0, quantile_band=list(BAND),
        quantile_selection={str(h): {"chosen": primary} for h in HORIZONS}, models=models)))
    return path


@pytest.fixture
def gold(tmp_path):
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    dates = pd.date_range(end=now, periods=24 * 40, freq="h")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"date": dates, "x": rng.normal(size=len(dates))})
    for t in TARGETS:
        frame[f"now_{t}"] = 30 + rng.random(len(dates))
    frame.to_parquet(tmp_path / "hourly.parquet")
    return frame


def test_forecasts_carry_the_range_into_the_ledger(tmp_path, gold):
    cols = ["x"] + [f"now_{t}" for t in TARGETS]
    models = _range_dir(tmp_path / "models", gold, cols)
    now = datetime.now(timezone.utc) + timedelta(minutes=5)
    records = predict_final(tmp_path / "hourly.parquet", models, now=now)
    assert len(records) == 18
    r = records[0]
    assert (r["pm25_q85"], r["pm25_q90"], r["pm25_q95"]) == (40.0, 45.0, 52.0)
    assert r["forecast_pm25"] == r["pm25_q95"] and r["model"] == "xgboost_q95"
    saved = archive_forecasts(records, tmp_path / "ledger.sqlite", latest_path=tmp_path / "latest.json")
    assert {c for c in BAND_COLUMNS.values()} <= set(saved[0])
    assert json.loads((tmp_path / "latest.json").read_text())[0]["pm25_q85"] == 40.0


def test_old_ledger_gains_nullable_range_columns(tmp_path):
    import sqlite3
    path = tmp_path / "old.sqlite"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE forecasts (forecast_id TEXT PRIMARY KEY, issued_at TEXT NOT NULL, as_of_time TEXT NOT NULL,"
               " forecast_time TEXT NOT NULL, region TEXT NOT NULL, horizon_hours INTEGER NOT NULL, model TEXT NOT NULL,"
               " model_run_id TEXT, manifest_sha256 TEXT NOT NULL, forecast_pm25 REAL NOT NULL, threshold REAL NOT NULL,"
               " elevated_flag INTEGER NOT NULL, feature_age_hours REAL NOT NULL, persistence_pm25 REAL NOT NULL,"
               " UNIQUE(as_of_time, region, horizon_hours))")
    db.close()
    db = connect(path)
    cols = {r["name"]: r for r in db.execute("PRAGMA table_info(forecasts)")}
    db.close()
    assert all(c in cols and cols[c]["notnull"] == 0 for c in BAND_COLUMNS.values())


def test_promotion_copies_and_smoke_tests_the_range_models(tmp_path, gold):
    cols = ["x"] + [f"now_{t}" for t in TARGETS]
    gold_path = tmp_path / "hourly.parquet"
    cand = _range_dir(tmp_path / "cand", gold, cols)
    out = g.promote(cand, tmp_path / "prod", gold_path, ledger_path=tmp_path / "none.sqlite")
    assert out["promoted"]
    manifest = json.loads((tmp_path / "prod" / g.MANIFEST).read_text())
    for entries in manifest["models"].values():
        for e in entries.values():
            assert all((tmp_path / "prod" / a).exists() for a in g.entry_artifacts(e))

    broken = _range_dir(tmp_path / "broken", gold, cols)
    (broken / "north_h6_xgboost_q85.joblib").write_bytes(b"not a model")
    out = g.check_candidate(broken, tmp_path / "prod", gold_path, ledger_path=tmp_path / "none.sqlite")
    assert not out["passed"] and any("xgboost_q85" in p for p in out["checks"]["integrity"]["problems"])


def test_gate_refits_a_range_model_at_its_own_quantile(gold):
    cols = ["x"] + [f"now_{t}" for t in TARGETS]
    train = gold.assign(label_north=gold["now_north"])
    est = g._refit_as_trained("xgboost_q95", {"model": "xgboost_q95"}, train, cols, "north", 6)
    assert est.get_params()["quantile_alpha"] == 0.95
    est = g._refit_as_trained("xgboost_q85", {"model": "xgboost_q85", "quantile_alpha": 0.9}, train, cols, "north", 6)
    assert est.get_params()["quantile_alpha"] == 0.9


def test_every_range_model_can_be_logged_to_mlflow(monkeypatch):
    import mlflow.xgboost
    from hazecast.train import _log_selected_model
    logged = []
    monkeypatch.setattr(mlflow.xgboost, "log_model", lambda model, name=None, **kw: logged.append(model))
    for name in BAND:
        _log_selected_model(_make_model(name), name)
    assert len(logged) == len(BAND)


def test_dual_bar_matches_the_xgb_tuned_definition():
    from hazecast.hourly_model import dual_bar
    y = np.array([10, 60, 70, 80, 20.0])
    pers = np.array([10, 30, 70, 40, 20.0])          # catches 1 of 3 haze hours
    good = dual_bar(y, np.array([12, 62, 72, 78, 25.0]), pers)
    assert good["passed"] and good["recall"] == 1.0 and abs(good["persistence_recall"] - 1 / 3) < 1e-9
    worse = dual_bar(y, np.array([12, 30, 50, 40, 25.0]), pers)     # misses every haze hour
    assert not worse["passed"]
    assert dual_bar(np.array([10, 20.0]), np.array([10, 20.0]), np.array([10, 20.0])) is None
