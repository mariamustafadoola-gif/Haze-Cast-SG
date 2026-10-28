"""Model files must load outside Airflow: saved with the standard C pickler, and the
promotion gate rejects any artifact that references an undeclared module (dill)."""
from __future__ import annotations

import pickle

import joblib
import numpy as np
from xgboost import XGBRegressor

from hazecast.governance import pickle_modules
from hazecast.train import save_model

# A minimal pickle that references dill._dill._load_type, in both global encodings.
DILL_PROTO2 = b"\x80\x02cdill._dill\n_load_type\nq\x00."
DILL_PROTO4 = (b"\x80\x04\x95\x1f\x00\x00\x00\x00\x00\x00\x00\x8c\ndill._dill\x94\x8c\n_load_type\x94\x93\x94.")


def test_save_model_is_loadable_by_joblib_and_dill_free(tmp_path):
    X, y = np.random.rand(100, 4), np.random.rand(100)
    model = XGBRegressor(n_estimators=5).fit(X, y)
    path = tmp_path / "m.joblib"
    save_model(model, path)
    assert np.array_equal(joblib.load(path).predict(X), model.predict(X))
    assert "dill" not in pickle_modules(path)


def test_scanner_finds_dill_references_in_both_pickle_encodings(tmp_path):
    for i, raw in enumerate((DILL_PROTO2, DILL_PROTO4)):
        p = tmp_path / f"d{i}.pkl"
        p.write_bytes(raw)
        assert "dill" in pickle_modules(p), raw
    clean = tmp_path / "clean.pkl"
    clean.write_bytes(pickle.dumps({"a": np.arange(3)}, protocol=pickle.HIGHEST_PROTOCOL))
    assert "dill" not in pickle_modules(clean)


def test_gate_rejects_a_dill_dependent_artifact(tmp_path):
    import json
    import pandas as pd
    from hazecast import governance as g
    from hazecast.pipeline_store import TARGETS, HORIZONS
    n = 24 * 10
    gold = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=n, freq="h"), "x": np.random.rand(n)})
    for t in TARGETS:
        gold[f"now_{t}"] = 20.0
    gold_path = tmp_path / "gold.parquet"; gold.to_parquet(gold_path)
    cand = tmp_path / "cand"; cand.mkdir()
    cols = ["x"] + [f"now_{t}" for t in TARGETS]
    good = XGBRegressor(n_estimators=3).fit(gold[cols], gold["now_north"])
    models = {}
    for h in HORIZONS:
        models[str(h)] = {}
        for t in TARGETS:
            name = f"{t}_h{h}.joblib"
            if (t, h) == ("west", 6):
                (cand / name).write_bytes(DILL_PROTO2)          # as an Airflow-patched pickle would look
            else:
                save_model(good, cand / name)
            models[str(h)][t] = dict(model="xgboost_q85", artifact=name, test_start="2026-01-08T00:00:00+08:00",
                                     final_test=dict(mae=1.0, f2=None))
    (cand / g.MANIFEST).write_text(json.dumps(dict(schema_version=1, pipeline="hourly", horizons_hours=list(HORIZONS),
                                                   feature_columns=cols, threshold=56.0, models=models)))
    r = g.check_candidate(cand, tmp_path / "no_champion", gold_path)
    problems = r["checks"]["integrity"]["problems"]
    assert not r["passed"] and any("west h=6" in p and "dill" in p for p in problems), problems
