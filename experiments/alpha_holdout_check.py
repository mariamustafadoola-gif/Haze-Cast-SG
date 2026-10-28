"""Out-of-time check of the xgboost_q85 quantile alpha options.

Question: on the untouched final holdout (2024-04-15 onward, includes the 2026 haze),
does the per-cell LOSO-tuned alpha table (config.XGB_QUANTILE_ALPHA_TUNED, from the
XGB_Tuned branch) beat simpler alternatives?

Every option is scored the same way, so they are directly comparable:
  - xgboost_q85 only (no candidate selection), one model per (region, horizon);
  - trained on rows whose label time precedes the holdout start (purged boundary,
    as hourly_model does), so nothing from the holdout is seen;
  - scored once on the holdout against persistence.

Options:
  flat_0.85        the previous default (current champion's alpha)
  tuned_table      config.XGB_QUANTILE_ALPHA_TUNED (teammate's LOSO table)
  flat_0.90        one alpha for all cells
  flat_0.95        one alpha for all cells
  tuned_no_holdout LOSO re-run with the holdout period removed from training
                   (only if --no-holdout-table CSV is given)

    python experiments/alpha_holdout_check.py [--no-holdout-table recommended.csv]
Writes reports/model_comparison/alpha_holdout_check.csv (+ _summary.csv).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from hazecast.config import (GOLD_DIR, HOURLY_HORIZONS, MODELS_DIR, PM25_ELEVATED_THRESHOLD as THR,
                             REPO_ROOT, XGB_PARAMS, XGB_QUANTILE_ALPHA_TUNED)
from hazecast.hourly_model import TARGETS, _prepare, _labelled, _feature_columns, _complete, _before
from hazecast.train import _metrics

OUT = REPO_ROOT / "reports" / "model_comparison"
BAR = 1.15  # the project's dual bar: recall >= 1.15x persistence AND mae_elevated <= 1.15x persistence


def holdout_start(gold: pd.DataFrame) -> pd.Timestamp:
    """Same boundary hourly_model uses (first date of the last 20% of rows)."""
    manifest = MODELS_DIR / "hourly" / "model_manifest.json"
    if manifest.exists():
        return pd.Timestamp(json.loads(manifest.read_text())["models"]["6"]["north"]["test_start"])
    return gold.date.iloc[int(len(gold) * 0.8)]


def score(y, p):
    m = _metrics(y, p)
    el = np.asarray(y) >= THR
    recall = m["tp"] / max(m["tp"] + m["fn"], 1)
    return dict(mae=m["mae"], f2=m["f2"], recall=recall, mae_elevated=m["mae_elevated"],
                false_alarm=m["fp"] / max((~el).sum(), 1), n_elevated=m["n_elevated"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-holdout-table", help="recommended-alpha CSV from a LOSO run without holdout rows")
    a = ap.parse_args()
    options = {"flat_0.85": lambda t, h: 0.85, "tuned_table": lambda t, h: XGB_QUANTILE_ALPHA_TUNED[(t, h)],
               "flat_0.90": lambda t, h: 0.90, "flat_0.95": lambda t, h: 0.95}
    if a.no_holdout_table:
        tab = pd.read_csv(a.no_holdout_table).set_index(["target", "horizon"]).recommended_alpha.to_dict()
        options["tuned_no_holdout"] = lambda t, h: tab[(t, h)]

    gold = _prepare(pd.read_parquet(GOLD_DIR / "hourly_features.parquet"))
    start = holdout_start(gold)
    start = start.tz_localize(gold.date.dt.tz) if start.tzinfo is None else start.tz_convert(gold.date.dt.tz)
    rows_out = []
    for h in HOURLY_HORIZONS:
        lab = _labelled(gold, h)
        cols = _feature_columns(lab)
        for t in TARGETS:
            rows = _complete(lab, cols, t, h)
            train, test = _before(rows, start, h), rows.loc[rows.date >= start]
            y = test[f"label_{t}"].to_numpy(dtype=float)
            pers = score(y, test[f"now_{t}"].to_numpy(dtype=float))
            fitted = {}
            for name, alpha_of in options.items():
                alpha = alpha_of(t, h)
                if alpha not in fitted:  # same alpha -> same model; fit once
                    m = XGBRegressor(**{**XGB_PARAMS, "objective": "reg:quantileerror", "quantile_alpha": alpha})
                    m.fit(train[cols], train[f"label_{t}"])
                    fitted[alpha] = np.maximum(0.0, m.predict(test[cols]))
                s = score(y, fitted[alpha])
                s["dual_bar"] = bool(s["recall"] >= BAR * pers["recall"] and s["mae_elevated"] <= BAR * pers["mae_elevated"])
                rows_out.append(dict(target=t, horizon=h, option=name, alpha=alpha, **s,
                                     persistence_recall=pers["recall"], persistence_mae_elevated=pers["mae_elevated"],
                                     persistence_false_alarm=pers["false_alarm"]))
            print(f"{t} h={h} done", flush=True)
    res = pd.DataFrame(rows_out)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "alpha_holdout_check.csv", index=False)
    summary = res.groupby(["option", "horizon"]).agg(
        recall=("recall", "mean"), false_alarm=("false_alarm", "mean"), f2=("f2", "mean"), mae=("mae", "mean"),
        mae_elevated=("mae_elevated", "mean"), dual_bar_cells=("dual_bar", "sum")).round(3)
    summary.to_csv(OUT / "alpha_holdout_check_summary.csv")
    overall = res.groupby("option").agg(recall=("recall", "mean"), false_alarm=("false_alarm", "mean"),
                                        f2=("f2", "mean"), mae=("mae", "mean"), dual_bar_cells=("dual_bar", "sum")).round(3)
    print(summary.to_string()); print(); print(overall.to_string())
    print(f"\nholdout starts {start}; persistence recall by horizon:",
          res.drop_duplicates(["target", "horizon"]).groupby("horizon").persistence_recall.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
