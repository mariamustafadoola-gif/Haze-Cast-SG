"""Champion/challenger promotion gate and rollback, on tiny fake model directories."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from hazecast import governance as g
from hazecast.pipeline_store import TARGETS, HORIZONS


@pytest.fixture
def gold_path(tmp_path):
    rng = np.random.default_rng(0)
    n = 24 * 40
    dates = pd.date_range("2026-01-01", periods=n, freq="h")
    base = 20 + np.cumsum(rng.normal(size=n)) * 0.3
    gold = pd.DataFrame({"date": dates, "x": rng.normal(size=n)})
    for t in TARGETS:
        gold[f"now_{t}"] = np.clip(base + rng.normal(size=n), 0, None)
    path = tmp_path / "hourly.parquet"
    gold.to_parquet(path)
    return path


def _model_dir(path, gold_path, mae, tag, f2=None, f2_override=None):
    path.mkdir(parents=True)
    gold = pd.read_parquet(gold_path)
    cols = ["x"] + [f"now_{t}" for t in TARGETS]
    est = LinearRegression().fit(gold[cols], gold["now_north"])
    models = {}
    for h in HORIZONS:
        models[str(h)] = {}
        for t in TARGETS:
            name = f"{t}_h{h}_{tag}.joblib"
            joblib.dump(est, path / name)
            models[str(h)][t] = dict(model="ridge", artifact=name, test_start="2026-02-01T00:00:00+08:00",
                                     final_test=dict(mae=mae, f2=(f2_override or {}).get((t, h), f2)),
                                     validation=dict(mae=mae))
    (path / g.MANIFEST).write_text(json.dumps(dict(schema_version=1, pipeline="hourly", horizons_hours=list(HORIZONS),
                                                   feature_columns=cols, threshold=56.0, models=models)))
    return path


def test_first_promotion_then_regression_rejected_then_rollback(tmp_path, gold_path):
    prod = tmp_path / "prod"
    good = _model_dir(tmp_path / "good", gold_path, mae=0.5, tag="a")
    first = g.promote(good, prod, gold_path)
    assert first["promoted"] and (prod / g.MANIFEST).exists()
    assert json.loads((prod / g.MANIFEST).read_text())["published_at"]

    worse = _model_dir(tmp_path / "worse", gold_path, mae=5.0, tag="b")
    second = g.promote(worse, prod, gold_path)
    assert not second["promoted"] and not second["checks"]["no_regression"]["passed"]
    assert json.loads((prod / g.MANIFEST).read_text())["models"]["6"]["north"]["artifact"].endswith("_a.joblib")

    better = _model_dir(tmp_path / "better", gold_path, mae=0.45, tag="c")
    assert g.promote(better, prod, gold_path)["promoted"]
    restored = g.rollback(production_dir=prod)
    assert json.loads((prod / g.MANIFEST).read_text())["models"]["6"]["north"]["artifact"].endswith("_a.joblib")
    assert restored["restored"]
    log = [json.loads(line)["action"] for line in (prod / "promotion_log.jsonl").read_text().splitlines()]
    assert log == ["promoted", "rejected", "promoted", "rollback"]


def test_broken_artifact_fails_integrity(tmp_path, gold_path):
    cand = _model_dir(tmp_path / "cand", gold_path, mae=0.5, tag="a")
    (cand / "north_h6_a.joblib").write_bytes(b"not a model")
    result = g.promote(cand, tmp_path / "prod", gold_path)
    assert not result["promoted"] and not result["checks"]["integrity"]["passed"]


def test_regression_is_judged_on_f2_not_mae(tmp_path, gold_path):
    """A higher-quantile challenger forecasts higher (worse MAE) but catches more haze: promote it.
    One group collapsing on F2 is held for human review instead of promoted automatically."""
    prod = tmp_path / "prod"
    assert g.promote(_model_dir(tmp_path / "champ", gold_path, mae=8.0, tag="a", f2=0.60), prod, gold_path)["promoted"]

    biased = _model_dir(tmp_path / "biased", gold_path, mae=12.0, tag="b", f2=0.66)   # MAE 1.5x, F2 better
    r = g.check_candidate(biased, prod, gold_path)
    assert r["checks"]["no_regression"]["passed"], r["checks"]["no_regression"]

    one_bad = _model_dir(tmp_path / "one_bad", gold_path, mae=12.0, tag="c", f2=0.66, f2_override={("west", 6): 0.45})
    r = g.check_candidate(one_bad, prod, gold_path)
    nr = r["checks"]["no_regression"]
    assert not nr["passed"] and "west|6" in nr["severe_f2_drops"] and "human review" in nr["note"]

    blown = _model_dir(tmp_path / "blown", gold_path, mae=20.0, tag="d", f2=0.70)      # MAE 2.5x: broken
    assert g.check_candidate(blown, prod, gold_path)["checks"]["no_regression"]["mae_blowups"]


def test_forced_promotion_requires_and_records_a_reason(tmp_path, gold_path):
    import pytest
    prod = tmp_path / "prod"
    assert g.promote(_model_dir(tmp_path / "champ", gold_path, mae=8.0, tag="a", f2=0.60), prod, gold_path)["promoted"]
    held = _model_dir(tmp_path / "held", gold_path, mae=8.0, tag="b", f2=0.62, f2_override={("west", 12): 0.40})
    with pytest.raises(ValueError, match="reason"):
        g.promote(held, prod, gold_path, force=True)
    r = g.promote(held, prod, gold_path, force=True, reason="west +12 h accepted: 17/18 groups improve")
    assert r["promoted"] and r["forced"]
    manifest = json.loads((prod / g.MANIFEST).read_text())
    assert manifest["forced_reason"].startswith("west +12 h") and "no_regression" in manifest["forced_over"]
    last = json.loads((prod / "promotion_log.jsonl").read_text().splitlines()[-1])
    assert last["reason"].startswith("west +12 h") and last["forced"]


def test_champion_is_rescored_like_for_like_when_the_holdout_has_haze(tmp_path):
    rng = np.random.default_rng(0)
    n = 24 * 40
    dates = pd.date_range("2026-01-01", periods=n, freq="h")
    level = np.where((np.arange(n) // 24) % 9 == 0, 90.0, 20.0) + rng.normal(0, 2, n)   # a haze day every 9 days
    gold = pd.DataFrame({"date": dates, "x": rng.normal(size=n)})
    for t in TARGETS:
        gold[f"now_{t}"] = np.clip(level + rng.normal(0, 1, n), 0, None)
    path = tmp_path / "hourly.parquet"; gold.to_parquet(path)
    prod = tmp_path / "prod"
    assert g.promote(_model_dir(tmp_path / "champ", path, mae=5.0, tag="a", f2=0.99), prod, path)["promoted"]
    r = g.check_candidate(_model_dir(tmp_path / "cand", path, mae=5.0, tag="b", f2=0.70), prod, path)
    grp = r["groups"]["north|6"]
    assert grp["champion_basis"] == "refitted on this holdout"
    assert grp["champion_f2"] != grp["champion_f2_recorded"]          # recorded 0.99 no longer used
    assert r["checks"]["no_regression"]["champion_compared_like_for_like"] == 18
    assert "groups_beating_persistence" in r["checks"]["skill"]
