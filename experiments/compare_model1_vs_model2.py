"""Side-by-side comparison: Model 1 (untuned xgboost_q85, flat alpha=0.85)
vs Model 2 (the LOSO-tuned xgboost_q85 on this branch), scored once on the
untouched final_test holdout.

PURPOSE
-------
Clean evidence for the team: did per-cell LOSO alpha tuning (see config.py's
XGB_QUANTILE_ALPHA_TUNED and experiments/xgb_loso_alpha_tuning.py) improve
the model, on the one holdout nothing is ever tuned against?

Model 1 is refit here (single fit per cell, flat alpha=0.85, no candidate
comparison) rather than read from a saved artifact, so this script is
self-contained and reproducible without depending on any untracked scratch
output. Model 2's numbers are read directly from the current, deployed
data/models/hourly/model_manifest.json -- the real production result,
including whichever candidate actually won CV selection per cell (usually
xgboost_q85, occasionally another candidate).

OUTPUT
------
reports/model_comparison/model1_vs_model2_final_test.csv -- one row per
(target, horizon), both models' numbers side by side.
"""
from __future__ import annotations
import json

import pandas as pd
from xgboost import XGBRegressor

from hazecast.config import GOLD_DIR, HOURLY_HORIZONS, REPO_ROOT, XGB_PARAMS, XGB_QUANTILE_ALPHA
from hazecast.hourly_model import TARGETS, _prepare, _labelled, _feature_columns, _complete
from hazecast.train import _metrics

OUT_DIR = REPO_ROOT / "reports" / "model_comparison"
MODEL2_MANIFEST = REPO_ROOT / "data" / "models" / "hourly" / "model_manifest.json"
TUNING_BAR_MULTIPLIER = 1.15


def main():
    gold = _prepare(pd.read_parquet(GOLD_DIR / "hourly_features.parquet"))
    m2 = json.loads(MODEL2_MANIFEST.read_text())

    rows_out = []
    for horizon in HOURLY_HORIZONS:
        labelled = _labelled(gold, horizon)
        cols = _feature_columns(labelled)
        for target in TARGETS:
            rows = _complete(labelled, cols, target, horizon)
            e2 = m2["models"][str(horizon)][target]
            test_start = pd.Timestamp(e2["test_start"])
            train = rows.loc[rows.date < test_start]
            test = rows.loc[rows.date >= test_start]

            m_pers = _metrics(test[f"label_{target}"], test[f"now_{target}"].to_numpy(dtype=float))
            p_recall = m_pers["tp"] / (m_pers["tp"] + m_pers["fn"]) if (m_pers["tp"] + m_pers["fn"]) else float("nan")

            # --- Model 1: flat alpha=0.85, refit here, single fit ---
            model1 = XGBRegressor(**{**XGB_PARAMS, "objective": "reg:quantileerror",
                                      "quantile_alpha": XGB_QUANTILE_ALPHA})
            model1.fit(train[cols], train[f"label_{target}"])
            pred1 = model1.predict(test[cols]).clip(min=0)
            m1 = _metrics(test[f"label_{target}"], pred1)
            recall1 = m1["tp"] / (m1["tp"] + m1["fn"]) if (m1["tp"] + m1["fn"]) else float("nan")
            pass1 = bool(recall1 >= TUNING_BAR_MULTIPLIER * p_recall
                         and m1["mae_elevated"] <= TUNING_BAR_MULTIPLIER * m_pers["mae_elevated"])

            # --- Model 2: as deployed, from the current manifest ---
            ft2 = e2["final_test"]
            recall2 = ft2["tp"] / (ft2["tp"] + ft2["fn"]) if (ft2["tp"] + ft2["fn"]) else float("nan")
            pass2 = bool(recall2 >= TUNING_BAR_MULTIPLIER * p_recall
                         and ft2["mae_elevated"] <= TUNING_BAR_MULTIPLIER * m_pers["mae_elevated"])

            recall_winner = "Model 2" if recall2 > recall1 else ("Model 1" if recall1 > recall2 else "tie")
            mae1, mae2 = m1["mae_elevated"], ft2["mae_elevated"]
            mae_winner = "Model 2" if mae2 < mae1 else ("Model 1" if mae1 < mae2 else "tie")

            rows_out.append(dict(
                target=target, horizon=horizon,
                recall_model1=round(recall1, 3), recall_model2=round(recall2, 3), recall_winner=recall_winner,
                mae_elevated_model1=round(mae1, 2), mae_elevated_model2=round(mae2, 2), mae_elevated_winner=mae_winner,
                dual_bar_pass_model1=pass1, dual_bar_pass_model2=pass2,
                model2_selected_candidate=e2["model"],
                persist_recall=round(p_recall, 3), persist_mae_elevated=round(m_pers["mae_elevated"], 2),
            ))

    out = pd.DataFrame(rows_out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "model1_vs_model2_final_test.csv", index=False)

    n1 = out.dual_bar_pass_model1.sum()
    n2 = out.dual_bar_pass_model2.sum()
    n_recall2 = (out.recall_winner == "Model 2").sum()
    n_mae2 = (out.mae_elevated_winner == "Model 2").sum()
    n_cells = len(HOURLY_HORIZONS) * len(TARGETS)
    print(f"Model 1 (untuned, alpha=0.85): {n1}/{n_cells} cells pass the dual bar on final_test")
    print(f"Model 2 (LOSO-tuned):          {n2}/{n_cells} cells pass the dual bar on final_test")
    print(f"Model 2 has higher recall in {n_recall2}/{n_cells} cells, lower mae_elevated in {n_mae2}/{n_cells} cells")
    print(f"Wrote {OUT_DIR / 'model1_vs_model2_final_test.csv'}")


if __name__ == "__main__":
    main()
