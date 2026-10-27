"""XGBoost quantile_alpha tuning audit: leave-one-haze-season-out (LOSO).

PURPOSE
-------
Traceability artefact, not a pipeline component. It answers one question
for an auditor: "how did we arrive at the per-region/horizon alphas in
hazecast.config.XGB_QUANTILE_ALPHA_TUNED, and was the switch away from
the flat XGB_QUANTILE_ALPHA actually justified by the numbers?"

The xgboost_q85 candidate's alpha was originally tuned (if at all) against
ordinary forward-chronological CV folds -- the same purged expanding
validation hazecast.hourly_model.train_final_hourly_models() uses for
model SELECTION. That protocol is appropriate for picking a winner among
candidates, but most calendar months in this dataset are calm (haze days
are ~2% of rows), so a chronological fold has a real chance of containing
none of them. A hyperparameter tuned against such folds can look great in
validation and still not generalise to an actual haze event it never saw.

HOURLY_TUNED_SETTINGS (the linear quantile model, see config.py) was
already tuned with leave-one-haze-season-out against the 2015 and 2019
seasons for exactly this reason. This script applies the same discipline
to xgboost_q85's alpha, which previously had none.

It is purely additive:
  - Does NOT import, modify, or re-run hazecast.hourly_model's production
    training path (train_final_hourly_models) or hazecast.score.
  - Does NOT touch data/models/hourly/* (the deployed models).
  - Writes its own output to data/audit/ (new, separate path).
  - Reuses the SAME feature columns, label construction, and threshold as
    production (imported directly from hazecast.hourly_model and
    hazecast.train), so the numbers below are guaranteed consistent with
    what the shipped model actually sees -- the only thing this script
    adds is a second, season-aware split used only for alpha selection.

METHOD
------
For each (target, horizon) cell and each candidate alpha: fit once on
every row except calendar year 2015, score only on 2015; fit once on
every row except calendar year 2019, score only on 2019. An alpha is kept
only if it clears the project's own tuning bar (recall_elevated >= 1.15x
persistence AND mae_elevated <= 1.15x persistence) on BOTH held-out
seasons independently -- not pooled, not averaged. Among alphas that
clear both, the one with the highest average F2 across the two seasons
is recommended. If none clears both, the closest worst-case is reported
and flagged, not silently treated as a pass.

The recommended alphas from the run this repo ships with are already
copied into hazecast.config.XGB_QUANTILE_ALPHA_TUNED; re-running this
script should reproduce them (same data, same seed in XGB_PARAMS).

CONFIRMATION, not reflected in this script's output: the recommended
alphas were additionally checked once against the untouched final_test
holdout (2024-04-15 onward) as a one-time confirmation, never as a
selection step -- dual-bar pass rate rose from 5/15 to 9/15 region/horizon
cells versus the flat-alpha baseline. See README "Results" for that table.

HOW TO RUN
----------
    export PYTHONPATH=src
    python experiments/xgb_loso_alpha_tuning.py
Requires data/gold/hourly_features.parquet to already exist.
"""
from __future__ import annotations
import logging
import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from hazecast.config import GOLD_DIR, HOURLY_HORIZONS, REPO_ROOT, XGB_PARAMS
from hazecast.hourly_model import TARGETS, _prepare, _labelled, _feature_columns, _complete
from hazecast.train import _metrics  # same mae_elevated / recall_elevated definitions as production

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

AUDIT_DIR = REPO_ROOT / "data" / "audit"
ALPHA_GRID = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
FOLD_YEARS = (2015, 2019)
TUNING_BAR_MULTIPLIER = 1.15  # the project's own stated bar, both conditions at once


def _prf(m):
    prec = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else float("nan")
    rec = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) else float("nan")
    return prec, rec


def _season_year_folds(rows, horizon):
    """Yield (fold_year, train, test) for each calendar year in FOLD_YEARS.

    A row is excluded from a fold's training set if its own date falls in
    the held-out year, OR its label window (date + horizon) spills into
    the held-out year from just before it -- otherwise the model would be
    trained on a target value measured inside the season it's supposed to
    have never seen.
    """
    tz = rows.date.dt.tz
    for fold_year in FOLD_YEARS:
        fold_start = pd.Timestamp(f"{fold_year}-01-01", tz=tz)
        fold_end = pd.Timestamp(f"{fold_year + 1}-01-01", tz=tz)
        test = rows.loc[(rows.date >= fold_start) & (rows.date < fold_end)]
        train = rows.loc[
            (rows.date + pd.Timedelta(hours=horizon) < fold_start) | (rows.date >= fold_end)
        ]
        if test.empty or train.empty:
            continue
        yield fold_year, train, test


def run_loso_sweep(gold: pd.DataFrame) -> pd.DataFrame:
    results = []
    t0 = time.time()
    for horizon in HOURLY_HORIZONS:
        labelled = _labelled(gold, horizon)
        cols = _feature_columns(labelled)
        for target in TARGETS:
            rows = _complete(labelled, cols, target, horizon)
            for fold_year, train, test in _season_year_folds(rows, horizon):
                m_pers = _metrics(test[f"label_{target}"], test[f"now_{target}"].to_numpy(dtype=float))
                p_prec, p_rec = _prf(m_pers)
                for alpha in ALPHA_GRID:
                    model = XGBRegressor(**{**XGB_PARAMS, "objective": "reg:quantileerror",
                                           "quantile_alpha": alpha})
                    model.fit(train[cols], train[f"label_{target}"])
                    pred = np.maximum(0.0, model.predict(test[cols]))
                    m = _metrics(test[f"label_{target}"], pred)
                    prec, rec = _prf(m)
                    passes = bool((rec >= TUNING_BAR_MULTIPLIER * p_rec)
                                  and (m["mae_elevated"] <= TUNING_BAR_MULTIPLIER * m_pers["mae_elevated"]))
                    results.append(dict(
                        target=target, horizon=horizon, fold_year=fold_year, alpha=alpha,
                        n_test=len(test), n_elevated_test=m_pers["n_elevated"],
                        f2=m["f2"], precision=prec, recall=rec, mae_elevated=m["mae_elevated"],
                        persist_recall=p_rec, persist_mae_elevated=m_pers["mae_elevated"],
                        dual_bar_pass=passes,
                    ))
            log.info("[%7.1fs] target=%s h=%d done", time.time() - t0, target, horizon)
    return pd.DataFrame(results)


def recommend(sweep: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for (target, horizon), grp in sweep.groupby(["target", "horizon"]):
        rows = []
        for alpha, agrp in grp.groupby("alpha"):
            by_fold = agrp.set_index("fold_year")
            if not set(FOLD_YEARS).issubset(by_fold.index):
                continue
            pass_a = bool(by_fold.loc[FOLD_YEARS[0], "dual_bar_pass"])
            pass_b = bool(by_fold.loc[FOLD_YEARS[1], "dual_bar_pass"])
            f2_a, f2_b = by_fold.loc[FOLD_YEARS[0], "f2"], by_fold.loc[FOLD_YEARS[1], "f2"]
            rec_a, rec_b = by_fold.loc[FOLD_YEARS[0], "recall"], by_fold.loc[FOLD_YEARS[1], "recall"]
            pers_a = by_fold.loc[FOLD_YEARS[0], "persist_recall"]
            pers_b = by_fold.loc[FOLD_YEARS[1], "persist_recall"]
            margin_a = rec_a - TUNING_BAR_MULTIPLIER * pers_a
            margin_b = rec_b - TUNING_BAR_MULTIPLIER * pers_b
            rows.append(dict(alpha=alpha, both_pass=pass_a and pass_b,
                              avg_f2=(f2_a + f2_b) / 2, worst_margin=min(margin_a, margin_b)))
        if not rows:
            recs.append(dict(target=target, horizon=horizon, recommended_alpha=float("nan"),
                              status="NO USABLE FOLDS"))
            continue
        cand = pd.DataFrame(rows)
        both = cand[cand["both_pass"]]
        if not both.empty:
            best = both.sort_values("avg_f2", ascending=False).iloc[0]
            status = "pass both folds"
        else:
            best = cand.sort_values("worst_margin", ascending=False).iloc[0]
            status = "NO ALPHA PASSES BOTH FOLDS -- best worst-case shown"
        recs.append(dict(target=target, horizon=horizon, recommended_alpha=best["alpha"],
                          avg_f2=round(best["avg_f2"], 3), status=status))
    return pd.DataFrame(recs).sort_values(["horizon", "target"]).reset_index(drop=True)


if __name__ == "__main__":
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    gold = _prepare(pd.read_parquet(GOLD_DIR / "hourly_features.parquet"))

    sweep = run_loso_sweep(gold)
    sweep.to_csv(AUDIT_DIR / "xgb_loso_alpha_sweep.csv", index=False)

    rec_df = recommend(sweep)
    rec_df.to_csv(AUDIT_DIR / "xgb_loso_alpha_recommended.csv", index=False)

    n_pass = (rec_df["status"] == "pass both folds").sum()
    log.info("Wrote %d sweep rows and %d recommendations to %s", len(sweep), len(rec_df), AUDIT_DIR)
    log.info("%d/%d cells have an alpha that passes the dual bar on BOTH held-out seasons (2015, 2019)",
              n_pass, len(rec_df))
