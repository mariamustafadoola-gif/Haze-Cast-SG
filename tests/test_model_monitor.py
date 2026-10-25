"""Model monitor: metric primitives, each monitor, and the governance rule."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from hazecast import model_monitor as mm
from hazecast.pipeline_store import connect, iso, SGT


def test_psi_zero_for_identical_and_large_for_shifted():
    assert mm.psi([0.25] * 4, [0.25] * 4) == pytest.approx(0.0)
    assert mm.psi([0.25] * 4, [0.85, 0.05, 0.05, 0.05]) > 0.25


def test_ndcg_is_one_for_same_ranking_and_drops_when_reordered():
    base = {"a": 0.6, "b": 0.3, "c": 0.1}
    assert mm.ndcg(base, dict(base), k=3) == pytest.approx(1.0)
    assert mm.ndcg(base, {"a": 0.1, "b": 0.3, "c": 0.6}, k=1) < 0.5


def test_attribution_finds_the_feature_the_model_uses():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=500), "noise": rng.normal(size=500), "now_north": rng.normal(size=500)})
    est = LinearRegression().fit(X, 3 * X["x"])
    att = mm.attribution(est, X, "north", list(X))
    assert max(att, key=att.get) == "x" and sum(att.values()) == pytest.approx(1.0)
    assert mm.attribution(None, X, "north", list(X)) == {"x": 0.0, "noise": 0.0, "now_north": 1.0}


def _gold(n_hours=24 * 60, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n_hours, freq="h", tz=SGT)
    now = 20 + 5 * rng.standard_normal(n_hours)
    return pd.DataFrame({"date": dates, "x": rng.normal(size=n_hours) + shift, "now_north": now})


def test_data_quality_flags_shift_and_missing_columns():
    ref = _gold()
    models = {("north", 6): ("persistence", None)}
    base = mm.build_baseline(ref, models, ["x", "now_north"])
    same = mm.data_quality_monitor(_gold(24 * 7, seed=1), base)
    assert same["features"]["x"]["status"] == "ok" and not same["missing_columns"]
    shifted = mm.data_quality_monitor(_gold(24 * 7, shift=3.0, seed=1), base)
    assert shifted["features"]["x"]["status"] == "alert" and shifted["features"]["x"]["psi"] > 0.25
    broken = mm.data_quality_monitor(_gold(24 * 7).drop(columns="x"), base)
    assert broken["status"] == "fail" and broken["missing_columns"] == ["x"]


def _matched(err=0.0, n=200, regions=("north", "south"), elevated_every=10):
    rows = []
    t0 = pd.Timestamp("2026-02-01", tz=SGT)
    for region in regions:
        for i in range(n):
            actual = 80.0 if i % elevated_every == 0 else 20.0
            rows.append(dict(region=region, horizon_hours=6, forecast_time=iso(t0 + pd.Timedelta(hours=i)),
                             forecast_pm25=actual + err, persistence_pm25=actual + 5, threshold=56.0,
                             actual_pm25=actual))
    return pd.DataFrame(rows)


def test_model_quality_alerts_when_error_exceeds_holdout():
    base = {"quality": {"north|6": {"mae": 2.0, "f2": 0.9}, "south|6": {"mae": 2.0, "f2": 0.9}}}
    ok = mm.model_quality_monitor(_matched(err=1.0), base)
    assert ok["groups"]["north|6"]["status"] == "ok"
    bad = mm.model_quality_monitor(_matched(err=-30.0), base)  # misses every elevated hour, MAE 30
    g = bad["groups"]["north|6"]
    assert {"mae_above_holdout", "recall_below_persistence"} <= set(g["reasons"])
    # Over-forecasting (worse MAE than persistence, but every haze hour caught) is the
    # q85 models' design, not an alert -- as long as MAE stays near its holdout level.
    biased = mm.model_quality_monitor(_matched(err=8.0), {"quality": {"north|6": {"mae": 7.0}}})
    q = biased["groups"]["north|6"]
    assert q["skill"] < 0 and q["status"] == "ok"


def test_bias_drift_flags_one_region_falling_behind():
    good = _matched(err=0.0, regions=("north",))
    bad = _matched(err=-30.0, regions=("south",))
    report = mm.bias_drift_monitor(pd.concat([good, bad]), {"bias": {}})
    assert report["horizons"]["6"]["recall_gap"] == pytest.approx(1.0)
    assert report["status"] == "alert"


def _results(dq="ok", n_psi=0, mq="ok", attr="ok", bias="ok"):
    return dict(data_quality=dict(status=dq, n_psi_alert=n_psi, stale=dq == "fail", missing_columns=[]),
                model_quality=dict(status=mq, n_alert=3 if mq == "alert" else 0, n_groups=18),
                bias_drift=dict(status=bias, reasons=["h6_recall_gap"] if bias == "alert" else []),
                attribution_drift=dict(status=attr, n_alert=3 if attr == "alert" else 0))


def test_governance_rule():
    assert mm.governance_decision(_results())["action"] == "none"
    assert mm.governance_decision(_results(dq="fail"))["action"] == "fix_data"
    assert mm.governance_decision(_results(mq="alert"))["action"] == "retrain"
    # input drift alone -> a human looks first; drift that also moved attributions -> retrain
    assert mm.governance_decision(_results(dq="alert", n_psi=5))["action"] == "investigate"
    assert mm.governance_decision(_results(dq="alert", n_psi=5, attr="alert"))["action"] == "retrain"
    assert mm.governance_decision(_results(bias="alert"))["action"] == "investigate"
    old = mm.governance_decision(_results(), model_published_at="2026-01-01T00:00:00+00:00",
                                 now="2026-03-01T00:00:00+00:00")
    assert old["action"] == "retrain" and "model_age" in old["reasons"][0]


def test_evaluate_window_and_save_round_trip(tmp_path):
    gold = _gold()
    models = {("north", 6): ("persistence", None)}
    base = mm.build_baseline(gold.iloc[: 24 * 30], models, ["x", "now_north"])
    db = connect(tmp_path / "ledger.sqlite")
    try:
        end = gold.date.iloc[-1]
        results = mm.evaluate_window(gold, db, models, ["x", "now_north"], base, end - pd.Timedelta(days=7), end)
        assert results["model_quality"]["status"] == "no_data"           # nothing archived yet
        assert results["attribution_drift"]["models"]["north|6"]["ndcg"] == pytest.approx(1.0)
        decision = mm.governance_decision(results)
        mm.save_results(db, end, results, decision)
        n = db.execute("SELECT COUNT(*) FROM monitor_metrics").fetchone()[0]
        assert n > 0 and db.execute("SELECT action FROM monitor_runs").fetchone()[0] == decision["action"]
    finally:
        db.close()
