"""Stability monitoring: season-matched CSI (features) and PSI (forecasts), upstream ingestion
checks, and how they reach the governance decision and the dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from hazecast import config as C
from hazecast import model_monitor as mm
from hazecast.pipeline_store import SGT, connect

COLS = ["x", "now_north"]


def _gold(start="2023-01-01", days=730, shift=0.0, seed=0):
    """x and PM2.5 are higher in the SW monsoon (Jun-Sep), as fire-season inputs are."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=24 * days, freq="h", tz=SGT)
    sw = dates.month.isin(C.MONSOON_SEASONS["SW monsoon"])
    x = rng.normal(size=len(dates)) + 3.0 * sw + shift
    now = 15 + 2 * x + rng.normal(size=len(dates))
    return pd.DataFrame({"date": dates, "x": x, "now_north": now})


@pytest.fixture(scope="module")
def setup():
    ref = _gold()
    est = LinearRegression().fit(ref[COLS], ref["now_north"])
    models = {("north", 6): ("persistence", None), ("south", 6): ("ridge", est)}
    return ref, models, mm.build_baseline(ref, models, COLS)


def _sw_window(shift=0.0, seed=7):
    return _gold("2025-07-01", 30, shift=shift, seed=seed)


def test_seasons_cover_every_month():
    assert sorted(mm.SEASON_OF_MONTH) == list(range(1, 13))


def test_a_normal_monsoon_is_not_drift_but_a_shift_within_the_season_is(setup):
    _, _, base = setup
    normal = mm.data_quality_monitor(_sw_window(), base)
    x = normal["features"]["x"]
    assert normal["season"] == "SW monsoon"
    assert x["psi"] > x["csi"] and x["status"] == "ok"      # all-season PSI sees the monsoon, CSI does not
    shifted = mm.data_quality_monitor(_sw_window(shift=3.0), base)
    assert shifted["features"]["x"]["status"] == "alert" and shifted["features"]["x"]["csi_ratio"] > 1


def test_forecast_psi_flags_unstable_output(setup, monkeypatch):
    _, models, base = setup
    calm = mm.stability_monitor(_sw_window(), models, COLS, base)
    assert calm["season"] == "SW monsoon" and all(m["status"] == "ok" for m in calm["models"].values())
    moved = mm.stability_monitor(_sw_window(shift=3.0), models, COLS, base)
    assert all(m["status"] == "alert" and m["psi_ratio"] > 1 for m in moved["models"].values())
    assert moved["status"] == "warn"                          # 2 models < STABILITY_ALERT_MIN_MODELS
    monkeypatch.setattr(C, "STABILITY_ALERT_MIN_MODELS", 2)
    assert mm.stability_monitor(_sw_window(shift=3.0), models, COLS, base)["status"] == "alert"


def test_old_baselines_still_work(setup):
    _, models, base = setup
    old = {k: v for k, v in base.items() if k != "predictions"}
    old["features"] = {c: {k: v for k, v in f.items() if not k.startswith("season")} for c, f in base["features"].items()}
    assert mm.stability_monitor(_sw_window(), models, COLS, old)["status"] == "no_data"
    dq = mm.data_quality_monitor(_sw_window(), old)
    assert dq["features"]["x"]["csi"] == dq["features"]["x"]["psi"]   # falls back to all seasons


def test_a_stuck_pm25_feed_means_fix_data(setup):
    _, _, base = setup
    w = _sw_window()
    w.loc[100:100 + C.PM25_STUCK_HOURS, "now_north"] = 42.0
    dq = mm.data_quality_monitor(w, base)
    assert dq["status"] == "fail" and dq["stuck_feeds"]["north"] >= C.PM25_STUCK_HOURS
    decision = mm.governance_decision(dict(data_quality=dq, model_quality={}, bias_drift={}, attribution_drift={}))
    assert decision["action"] == "fix_data" and decision["reasons"][0].startswith("stuck_feed")


def test_source_health_names_the_failing_source(tmp_path):
    now = pd.Timestamp("2026-10-30 12:00", tz=SGT)
    local = now.tz_localize(None)
    pd.DataFrame({"ts": [local - pd.Timedelta(hours=1)]}).to_parquet(tmp_path / "pm25.parquet")
    pd.DataFrame({"ts": [local - pd.Timedelta(hours=1)] * len(C.WEATHER_POINTS) + [local - pd.Timedelta(hours=10)],
                  "point": list(C.WEATHER_POINTS) + ["singapore"]}).to_parquet(tmp_path / "weather.parquet")
    # last detection 20 h ago: past the 15 h trust window, inside the 24 h carry-forward
    grid_end = local - pd.Timedelta(hours=20) + pd.Timedelta(hours=C.FIRE_KNOWN_HOURS_AFTER_LAST_DETECTION)
    pd.DataFrame({"ts": [grid_end]}).to_parquet(tmp_path / "firms_hourly.parquet")
    health = mm.source_health(now, tmp_path)
    assert health["pm25"]["status"] == "ok" and health["weather_singapore"]["status"] == "ok"
    assert health["fires"]["status"] == "warn" and health["fires"]["age_hours"] == pytest.approx(20)
    pd.DataFrame({"ts": [local - pd.Timedelta(hours=6)]}).to_parquet(tmp_path / "pm25.parquet")
    assert mm.source_health(now, tmp_path)["pm25"]["status"] == "alert"


def _results(stab="ok", attr="ok"):
    return dict(data_quality=dict(status="ok", n_psi_alert=0, stale=False, missing_columns=[]),
                stability=dict(status=stab, n_alert=4 if stab == "alert" else 0),
                model_quality=dict(status="ok", n_alert=0, n_groups=18),
                bias_drift=dict(status="ok", reasons=[]),
                attribution_drift=dict(status=attr, n_alert=3 if attr == "alert" else 0))


def test_unstable_output_alone_is_investigated_with_attribution_drift_it_retrains():
    assert mm.governance_decision(_results(stab="alert"))["action"] == "investigate"
    d = mm.governance_decision(_results(stab="alert", attr="alert"))
    assert d["action"] == "retrain" and any(r.startswith("output_stability") for r in d["reasons"])


def test_stability_reaches_the_dashboard(setup, tmp_path):
    from hazecast.dashboard import figures, _read
    ref, models, base = setup
    gold = pd.concat([ref, _gold("2025-01-01", 365, seed=3)], ignore_index=True)
    db_path = tmp_path / "ledger.sqlite"
    db = connect(db_path)
    try:
        for end in pd.date_range("2025-03-01", "2025-11-01", freq="30D", tz=SGT):
            results = mm.evaluate_window(gold, db, models, COLS, base, end - pd.Timedelta(days=7), end)
            mm.save_results(db, end, results, mm.governance_decision(results))
    finally:
        db.close()
    titles = [f.layout.title.text for f in figures(*_read(db_path))]
    assert any(t.startswith("Feature stability (CSI)") for t in titles)
    assert any(t.startswith("Forecast stability (PSI)") for t in titles)
    assert any(t.startswith("Stability alerts") for t in titles)
