"""LOSO-tuned xgboost_q85 alpha (from the XGB_Tuned branch): the right alpha reaches the model,
and every caller of hourly_model._fit passes the horizon it needs (one did not on XGB_Tuned)."""
from __future__ import annotations
import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from hazecast import config as C
from hazecast.hourly_model import _fit, _xgb_q85_alpha

ROOT = Path(__file__).resolve().parents[1]


def test_tuned_table_covers_all_18_cells_with_sane_values():
    cells = {(t, h) for t in C.SG_REGIONS + ["national"] for h in C.HOURLY_HORIZONS}
    assert set(C.XGB_QUANTILE_ALPHA_TUNED) == cells
    assert all(0.5 < a < 1 for a in C.XGB_QUANTILE_ALPHA_TUNED.values())


def test_fit_uses_the_cell_alpha_and_falls_back_to_the_flat_default():
    rng = np.random.default_rng(0)
    rows = pd.DataFrame({"x": rng.normal(size=200), "now_central": rng.normal(size=200)})
    rows["label_central"] = rows.x * 3
    model = _fit("xgboost_q85", rows, ["x", "now_central"], "central", 24)
    assert model.get_params()["quantile_alpha"] == C.XGB_QUANTILE_ALPHA_TUNED[("central", 24)]
    assert _xgb_q85_alpha("nowhere", 6) == C.XGB_QUANTILE_ALPHA


def test_every_caller_passes_the_horizon():
    assert list(inspect.signature(_fit).parameters) == ["name", "rows", "cols", "target", "horizon"]
    for path in ("src/hazecast/hourly_model.py", "src/hazecast/monitor_backtest.py",
                 "src/hazecast/hourly_diagnostics.py", "experiments/monitoring_evidence.py"):
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "_fit"]
        assert calls, f"{path}: no _fit calls found"
        for c in calls:
            assert len(c.args) + len(c.keywords) == 5, f"{path}:{c.lineno}: _fit call is missing the horizon argument"
