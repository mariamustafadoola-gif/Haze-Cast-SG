"""Scenario retrains, persistence, cooldown, calibration (coverage), provenance, early-life and
recent-window checks, the shadow period, the threshold review and haze mode."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from hazecast import config as C
from hazecast import governance as gov
from hazecast import model_monitor as mm
from hazecast import provenance
from hazecast.pipeline_store import SGT, TARGETS, connect, iso


def _results(drift=False, attr=False, mq="ok", coverage=()):
    return dict(data_quality=dict(status="alert" if drift else "ok", n_psi_alert=5 if drift else 0, stale=False,
                                  missing_columns=[], n_feature_alert=0),
                stability=dict(status="ok", n_alert=0),
                model_quality=dict(status=mq, n_alert=3 if mq == "alert" else 0, n_groups=18,
                                   coverage_alerts=list(coverage), horizons={}),
                bias_drift=dict(status="ok", reasons=[]),
                attribution_drift=dict(status="alert" if attr else "ok", n_alert=3 if attr else 0))


# ---------------------------------------------------------------- persistence and cooldown
def test_drift_must_repeat_before_anyone_is_asked():
    first = mm.governance_decision(_results(drift=True), previous_reasons=[])
    assert first["action"] == "none" and "first occurrence" in first["reasons"][0]
    again = mm.governance_decision(_results(drift=True), previous_reasons=first["reasons"])
    assert again["action"] == "investigate"
    both = mm.governance_decision(_results(drift=True, attr=True), previous_reasons=again["reasons"] + ["attribution_drift: x"])
    assert both["action"] == "retrain"


def test_performance_alerts_act_at_once():
    assert mm.governance_decision(_results(mq="alert"), previous_reasons=[])["action"] == "retrain"


def test_coverage_outside_its_range_twice_retrains():
    first = mm.governance_decision(_results(coverage=["+6 h 0.60 outside 0.73-0.93"]), previous_reasons=[])
    assert first["action"] == "none"
    assert mm.governance_decision(_results(coverage=["+6 h 0.58"]), previous_reasons=first["reasons"])["action"] == "retrain"


def test_retrains_are_held_for_a_week_except_after_a_code_change():
    now, last = "2026-10-10T00:00:00+00:00", "2026-10-07T00:00:00+00:00"
    held = mm.governance_decision(_results(mq="alert"), now=now, last_retrain_at=last)
    assert held["action"] == "investigate" and held["reasons"][-1].startswith("retrain_cooldown")
    code = mm.governance_decision(_results(), now=now, last_retrain_at=last,
                                  triggers=["feature_code_changed: feature code or settings changed since training"])
    assert code["action"] == "retrain"
    later = mm.governance_decision(_results(mq="alert"), now="2026-10-15T00:00:00+00:00", last_retrain_at=last)
    assert later["action"] == "retrain"


# ---------------------------------------------------------------- scenario triggers and provenance
def _gold(start="2026-06-01", days=60, haze_days=(), value=20.0):
    dates = pd.date_range(start, periods=24 * days, freq="h", tz=SGT)
    g = pd.DataFrame({"date": dates, **{f"now_{t}": value for t in TARGETS}})
    for d in haze_days:
        g.loc[g.date.dt.strftime("%m-%d") == d, [f"now_{t}" for t in TARGETS]] = 90.0
    return g


def _manifest(trained="2026-06-01T00:00:00+00:00", label_end="2026-06-10T00:00:00+08:00", **extra):
    entry = dict(model="persistence", artifact=None, training_label_end=label_end, test_start="2026-06-20T00:00:00+08:00")
    return dict(trained_at=trained, feature_columns=[f"now_{t}" for t in TARGETS], threshold=56.0, horizons_hours=[6, 12, 24],
                models={str(h): {t: dict(entry) for t in TARGETS} for h in (6, 12, 24)}, **extra)


def test_pre_season_and_post_episode_retrains():
    gold = _gold(haze_days=("07-01",))
    early = mm.scenario_triggers("2026-07-10T00:00:00+00:00", gold, _manifest())
    assert early == []                                                           # 9 quiet days: episode may resume
    post = mm.scenario_triggers("2026-07-16T00:00:00+00:00", gold, _manifest())
    assert any(t.startswith("post_episode") for t in post)                       # 24 haze hours, 15 quiet days
    pre = mm.scenario_triggers("2026-07-20T00:00:00+00:00", gold, _manifest())
    assert any(t.startswith("scheduled_pre_season") for t in pre)
    seen = mm.scenario_triggers("2026-07-20T00:00:00+00:00", gold,
                                _manifest(trained="2026-07-18T00:00:00+00:00", label_end="2026-07-18T00:00:00+08:00"))
    assert seen == []                                                            # already retrained on it


def test_a_changed_feature_path_or_library_asks_for_a_retrain(monkeypatch):
    current = provenance.record()
    assert len(current["feature_fingerprint"]) == 16 and current["library_versions"]["pandas"]
    assert provenance.differences({**current}) == []
    stale = {**current, "feature_fingerprint": "0" * 16,
             "library_versions": {**current["library_versions"], "xgboost": "0.1"}}
    diffs = provenance.differences(stale)
    assert "feature code or settings changed since training" in diffs and any(d.startswith("xgboost 0.1") for d in diffs)
    kinds = {t.split(":")[0] for t in mm.scenario_triggers("2026-03-01T00:00:00+00:00", _gold(), {**_manifest(), **stale})}
    assert kinds == {"feature_code_changed", "environment_changed"}


# ---------------------------------------------------------------- coverage
def _holdout(weeks=30, coverage=0.85, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-01-01", periods=24 * 7 * weeks, freq="h", tz=SGT)
    rows = []
    for target in TARGETS:
        actual = 20 + rng.normal(0, 3, len(t))
        above = rng.random(len(t)) < coverage
        rows.append(pd.DataFrame(dict(forecast_time=[iso(x) for x in t], region=target, horizon_hours=6,
                                      actual_pm25=actual, forecast_pm25=np.where(above, actual + 2, actual - 2),
                                      threshold=56.0, persistence_pm25=actual)))
    return pd.concat(rows, ignore_index=True)


def test_coverage_band_and_monitor():
    band = mm.coverage_band(_holdout())
    assert band["6"]["low"] < 0.85 < band["6"]["high"] and band["6"]["n_weeks"] >= C.COVERAGE_MIN_QUIET_WEEKS
    base = {"coverage_band": band, "quality": {}}
    normal = mm.model_quality_monitor(_holdout(weeks=1, seed=5), base)
    assert normal["horizons"]["6"]["coverage_status"] == "ok" and not normal["coverage_alerts"]
    low = mm.model_quality_monitor(_holdout(weeks=1, coverage=0.5, seed=5), base)
    assert low["horizons"]["6"]["coverage_status"] == "alert" and low["coverage_alerts"]
    assert mm.coverage_band(_holdout(weeks=5)) == {}                     # too few quiet weeks to judge


# ---------------------------------------------------------------- early life
def test_early_life_flags_a_new_champion_worse_than_the_one_it_replaced(tmp_path):
    models = tmp_path / "hourly"
    (models / "history").mkdir(parents=True)
    previous = _manifest()                                               # persistence everywhere
    (models / "history" / "old.json").write_text(json.dumps(previous))
    published = pd.Timestamp("2026-06-05 00:00", tz=SGT)
    (models / "model_manifest.json").write_text(json.dumps({**previous, "published_at": iso(published),
                                                           "replaced": str(models / "history" / "old.json")}))
    gold = _gold(days=10)
    db = connect(tmp_path / "ledger.sqlite")
    try:
        for i in range(60):
            origin = published + pd.Timedelta(hours=i)
            for t in TARGETS:
                ft = origin + pd.Timedelta(hours=6)
                db.execute("INSERT INTO forecasts (forecast_id, issued_at, as_of_time, forecast_time, region, horizon_hours,"
                           " model, manifest_sha256, forecast_pm25, threshold, elevated_flag, feature_age_hours, persistence_pm25)"
                           " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (f"{i}{t}", iso(origin), iso(origin), iso(ft), t, 6, "m", "x", 45.0, 56.0, 0, 0.2, 20.0))
                db.execute("INSERT OR REPLACE INTO actuals VALUES (?,?,?,?,?,?)", (iso(ft), t, 20.0, "t", iso(ft), iso(ft)))
        db.commit()
        finding = mm.early_life_check(db, models, gold, published + pd.Timedelta(days=4))
        assert finding and finding.startswith("early_life") and "+6 h MAE 25.0 vs 0.0" in finding
        assert mm.early_life_check(db, models, gold, published + pd.Timedelta(days=10)) is None   # past 7 days
    finally:
        db.close()


# ---------------------------------------------------------------- recent-window gate check
def test_recent_window_check(tmp_path):
    gold = _gold(start="2026-01-01", days=200, haze_days=("06-20", "06-21"))
    # make PM2.5 move so persistence differs from the truth
    gold[[f"now_{t}" for t in TARGETS]] += np.sin(np.arange(len(gold)) / 5.0)[:, None] * 5
    cand = _manifest()
    db = connect(tmp_path / "ledger.sqlite")
    try:
        lab_times = gold.date[(gold.date >= "2026-05-01") & (gold.date < "2026-07-10")]
        for origin in lab_times:
            ft = origin + pd.Timedelta(hours=6)
            actual = gold.loc[gold.date == ft, "now_north"]
            if actual.empty:
                continue
            a = float(actual.iloc[0])
            db.execute("INSERT INTO forecasts (forecast_id, issued_at, as_of_time, forecast_time, region, horizon_hours,"
                       " model, manifest_sha256, forecast_pm25, threshold, elevated_flag, feature_age_hours, persistence_pm25)"
                       " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (iso(origin), iso(origin), iso(origin), iso(ft), "north", 6, "m", "x", a, 56.0, int(a >= 56), 0.2, a))
            db.execute("INSERT OR REPLACE INTO actuals VALUES (?,?,?,?,?,?)", (iso(ft), "north", a, "t", iso(ft), iso(ft)))
        db.commit()
    finally:
        db.close()
    perfect = gov._recent_window_check(gold, cand, tmp_path / "ledger.sqlite")   # served forecasts were perfect
    h6 = perfect["horizons"]["6"]
    assert h6["judged"] and h6["served_f2"] == pytest.approx(1.0)
    assert perfect["passed"] == (h6["challenger_f2"] >= 1.0 - C.PROMOTION_F2_SEVERE_DROP)
    assert gov._recent_window_check(gold, cand, tmp_path / "missing.sqlite")["note"].startswith("no live forecasts")


# ---------------------------------------------------------------- shadow, review, haze mode
def test_shadow_start_report_and_stop(tmp_path):
    cand = tmp_path / "cand"
    cand.mkdir()
    (cand / "model_manifest.json").write_text(json.dumps(_manifest()))
    out = gov.shadow(cand, tmp_path / "shadow", tmp_path / "prod")
    assert (tmp_path / "shadow" / "model_manifest.json").exists() and out["candidate"] == str(cand)
    ledger, shadow_ledger = connect(tmp_path / "p.sqlite"), connect(tmp_path / "s.sqlite")
    try:
        for db, value in ((ledger, 30.0), (shadow_ledger, 22.0)):
            db.execute("INSERT INTO forecasts (forecast_id, issued_at, as_of_time, forecast_time, region, horizon_hours,"
                       " model, manifest_sha256, forecast_pm25, threshold, elevated_flag, feature_age_hours, persistence_pm25)"
                       " VALUES ('a','t0','t0','t6','north',6,'m','x',?,56,0,0.2,20)", (value,))
            db.commit()
        ledger.execute("INSERT INTO actuals VALUES ('t6','north',20,'t','t','t')")
        ledger.commit()
    finally:
        ledger.close()
        shadow_ledger.close()
    report = gov.shadow_report(tmp_path / "s.sqlite", tmp_path / "p.sqlite")
    assert report["horizons"]["6"]["mae"] == dict(champion=10.0, shadow=2.0)
    gov.stop_shadow(tmp_path / "shadow", tmp_path / "prod")
    assert not (tmp_path / "shadow" / "model_manifest.json").exists()


def test_alert_review_counts_alerts_followed_by_performance_loss(tmp_path):
    from hazecast.alert_review import review
    db = connect(tmp_path / "l.sqlite")
    mm._ensure_tables(db)
    runs = [("2026-08-01", "investigate", ["data_drift: x"]), ("2026-08-08", "investigate", ["data_drift: x"]),
            ("2026-08-15", "retrain", ["model_quality: 5 of 18", "data_drift: x"]),
            ("2026-02-01", "investigate", ["data_drift: x"])]
    with db:
        db.executemany("INSERT INTO monitor_runs VALUES (?,?,?,?,?)",
                       [(f"{d}T00:00:00+00:00", "now", a, "{}", json.dumps(r)) for d, a, r in runs])
    db.close()
    r = review(tmp_path / "l.sqlite").set_index("signal")
    assert r.loc["data_drift", "alerts"] == 4 and r.loc["data_drift", "followed_by_performance_loss"] == 3


def test_haze_mode_switches_on_only_during_an_episode(tmp_path, monkeypatch):
    from hazecast import orchestration
    monkeypatch.setattr(orchestration, "GOLD_DIR", tmp_path)
    g = _gold(days=3)
    g["date"] = g.date.dt.tz_localize(None)
    g.to_parquet(tmp_path / "hourly_features.parquet")
    assert orchestration.haze_active() is False
    g.loc[g.index[-5], "now_west"] = 80.0
    g.to_parquet(tmp_path / "hourly_features.parquet")
    assert orchestration.haze_active() is True
