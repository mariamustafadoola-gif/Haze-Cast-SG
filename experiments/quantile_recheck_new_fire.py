"""Recheck of the LOSO-tuned xgboost_q85 quantiles on the corrected fire features (31 Oct 2026).

1. Clean re-tune: the tuned branch's leave-one-haze-season-out sweep (2015, 2019; alphas 0.70-0.95;
   dual bar: recall >= 1.15x persistence AND elevated-hour MAE <= 1.15x persistence on both seasons),
   using only rows before the out-of-time test period.
2. Out-of-time test: each option trained on pre-test rows (purged boundary), scored once on 2024-26.

The previous per-cell table is kept below as SHIPPED (config now uses one quantile, 0.95).

    python experiments/quantile_recheck_new_fire.py
Writes reports/model_comparison/quantile_recheck_{loso_sweep,loso_recommended,holdout}.csv
(about 10 minutes on a laptop).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from hazecast.config import GOLD_DIR, HOURLY_HORIZONS, MODELS_DIR, PM25_ELEVATED_THRESHOLD as THR, REPO_ROOT, XGB_PARAMS
from hazecast.hourly_model import TARGETS, _prepare, _labelled, _feature_columns, _complete, _before
from hazecast.train import _metrics

OUT = REPO_ROOT / "reports" / "model_comparison"
SHIPPED = {
    ("north", 6): 0.90, ("north", 12): 0.90, ("north", 24): 0.95,
    ("south", 6): 0.90, ("south", 12): 0.85, ("south", 24): 0.95,
    ("east", 6): 0.95, ("east", 12): 0.95, ("east", 24): 0.95,
    ("west", 6): 0.95, ("west", 12): 0.85, ("west", 24): 0.90,
    ("central", 6): 0.95, ("central", 12): 0.95, ("central", 24): 0.90,
    ("national", 6): 0.90, ("national", 12): 0.90, ("national", 24): 0.95,
}
GRID, FOLDS, BAR = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95), (2015, 2019), 1.15


def model(alpha):
    return XGBRegressor(**{**XGB_PARAMS, "objective": "reg:quantileerror", "quantile_alpha": alpha})


def mae_el(y, p):
    """MAE on elevated hours (the tuned branch's mae_elevated)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    el = y >= THR
    return float(np.abs(p[el] - y[el]).mean()) if el.any() else float("nan")


def rec(m):
    return m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) else float("nan")


gold = _prepare(pd.read_parquet(GOLD_DIR / "hourly_features.parquet"))
start = pd.Timestamp(json.loads((MODELS_DIR / "hourly" / "model_manifest.json").read_text())["models"]["6"]["north"]["test_start"])
start = start.tz_convert(gold.date.dt.tz)
t0 = time.time()

# 1. clean LOSO sweep on pre-test rows
sweep = []
for h in HOURLY_HORIZONS:
    lab = _labelled(gold, h)
    cols = _feature_columns(lab)
    for t in TARGETS:
        rows = _before(_complete(lab, cols, t, h), start, h)
        tz = rows.date.dt.tz
        for year in FOLDS:
            a, b = pd.Timestamp(f"{year}-01-01", tz=tz), pd.Timestamp(f"{year + 1}-01-01", tz=tz)
            test = rows.loc[(rows.date >= a) & (rows.date < b)]
            train = rows.loc[(rows.date + pd.Timedelta(hours=h) < a) | (rows.date >= b)]
            yt, pers = test[f"label_{t}"].to_numpy(dtype=float), test[f"now_{t}"].to_numpy(dtype=float)
            mp = _metrics(yt, pers)
            for alpha in GRID:
                p = np.maximum(0, model(alpha).fit(train[cols], train[f"label_{t}"]).predict(test[cols]))
                m = _metrics(test[f"label_{t}"], p)
                sweep.append(dict(target=t, horizon=h, year=year, alpha=alpha, f2=m["f2"], recall=rec(m),
                                  persist_recall=rec(mp),
                                  passes=bool(rec(m) >= BAR * rec(mp) and mae_el(yt, p) <= BAR * mae_el(yt, pers))))
        print(f"[{time.time() - t0:6.0f}s] sweep {t} h={h}", flush=True)
sweep = pd.DataFrame(sweep)
sweep.to_csv(OUT / "quantile_recheck_loso_sweep.csv", index=False)
recs = []
for (t, h), g in sweep.groupby(["target", "horizon"]):
    c = g.groupby("alpha").agg(both=("passes", "all"), f2=("f2", "mean"),
                               worst=("recall", "min")).reset_index()
    both = c[c.both]
    best = both.sort_values("f2", ascending=False).iloc[0] if len(both) else c.sort_values("worst", ascending=False).iloc[0]
    ranked = c.sort_values("f2", ascending=False)
    recs.append(dict(target=t, horizon=h, clean_alpha=best.alpha, passes_both=bool(len(both)),
                     shipped_alpha=SHIPPED[(t, h)], f2_gap_to_next=float(ranked.f2.iloc[0] - ranked.f2.iloc[1])))
recs = pd.DataFrame(recs)
recs.to_csv(OUT / "quantile_recheck_loso_recommended.csv", index=False)
clean = {(r.target, r.horizon): r.clean_alpha for r in recs.itertuples()}

# 2. out-of-time test
options = {"flat_0.85": lambda t, h: 0.85, "shipped_table": lambda t, h: SHIPPED[(t, h)],
           "flat_0.90": lambda t, h: 0.90, "flat_0.95": lambda t, h: 0.95,
           "clean_retune": lambda t, h: clean[(t, h)]}
rows_out = []
for h in HOURLY_HORIZONS:
    lab = _labelled(gold, h)
    cols = _feature_columns(lab)
    for t in TARGETS:
        rows = _complete(lab, cols, t, h)
        train, test = _before(rows, start, h), rows.loc[rows.date >= start]
        y = test[f"label_{t}"].to_numpy(dtype=float)
        el = y >= THR
        pers = test[f"now_{t}"].to_numpy(dtype=float)
        mp = _metrics(y, pers)
        fitted = {}
        for name, fn in options.items():
            alpha = fn(t, h)
            if alpha not in fitted:
                fitted[alpha] = np.maximum(0, model(alpha).fit(train[cols], train[f"label_{t}"]).predict(test[cols]))
            p = fitted[alpha]
            m = _metrics(y, p)
            rows_out.append(dict(option=name, target=t, horizon=h, alpha=alpha, recall=rec(m), f2=m["f2"], mae=m["mae"],
                                 false_alarm=m["fp"] / max((~el).sum(), 1), coverage=float((y <= p).mean()),
                                 persist_recall=rec(mp), persist_f2=mp["f2"],
                                 dual_bar=bool(rec(m) >= BAR * rec(mp) and mae_el(y, p) <= BAR * mae_el(y, pers))))
        print(f"[{time.time() - t0:6.0f}s] holdout {t} h={h}", flush=True)
res = pd.DataFrame(rows_out)
res.to_csv(OUT / "quantile_recheck_holdout.csv", index=False)
print("DONE", flush=True)
