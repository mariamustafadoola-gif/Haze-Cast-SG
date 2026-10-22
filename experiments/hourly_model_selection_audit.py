"""Hourly model selection audit: plain vanilla LR vs. the quantile-tuned
LR actually shipped (hazecast.hourly_model).

PURPOSE
-------
Traceability artefact, not a pipeline component. It answers one question
for an auditor: "starting from an untuned linear model, how did we arrive
at the per-region/horizon quantile-tuned settings in
hazecast.config.HOURLY_TUNED_SETTINGS, and was the switch actually
justified by the numbers?"

It is purely additive:
  - Does NOT import, modify, or re-run anything in hazecast.hourly_model's
    production path (train_final_hourly_models) or hazecast.score.
  - Does NOT touch data/models/hourly/*.joblib (the deployed models) or
    data/models/hourly_eval_summary.csv (the production eval output).
  - Writes its own output to data/audit/ (new, separate path) so a diff
    against the existing repo shows only new files, nothing changed.
  - Reuses the SAME feature columns, label construction, and season-fold
    splitting as the real evaluation (imported directly from
    hazecast.hourly_model and hazecast.train), so the numbers below are
    guaranteed consistent with production -- the only thing that differs
    between stages is which estimator is fit.

HOW TO RUN
----------
Cell markers ("# %%") make this runnable top-to-bottom as a Jupyter-style
interactive script in VS Code or JupyterLab, or plain:
    export PYTHONPATH=src
    python experiments/hourly_model_selection_audit.py
Requires data/gold/hourly_features.parquet to already exist (run the
ingest -> bronze_to_silver -> silver_to_gold steps first; see README.md).

STAGES
------
  1. Baseline: persistence (this hour's own reading, carried forward).
  2. Plain vanilla: ordinary LinearRegression, no quantile, no
     regularization, same features as production. This is the "what if we
     shipped the untuned default" comparison -- it was not itself a
     production candidate, so there is no committed eval CSV for it; this
     script is what generates it.
  3. Quantile-tuned: the actual shipped model. Calls
     hazecast.hourly_model.fit_predict() directly (not a re-implementation)
     so this stage is byte-for-byte the production logic, just run here
     for side-by-side comparison instead of in isolation.
  4. Verdict: applies the project's own stated tuning bar (recall_elevated
     >= 1.15x persistence AND mae_elevated <= 1.15x persistence, both at
     once) to every region/horizon/season cell, for both the vanilla and
     tuned stages, so the comparison isn't just "which number is bigger"
     but "which one clears the bar we committed to before looking".
"""
# %%
from __future__ import annotations
import logging
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from hazecast.config import (
    GOLD_DIR, HAZE_SEASONS, HOURLY_HORIZONS, PM25_ELEVATED_THRESHOLD, REPO_ROOT,
)
from hazecast.hourly_model import TARGETS, _feature_columns, build_label, fit_predict, get_tuned_setting
from hazecast.train import _metrics  # same mae_elevated / recall_elevated definitions as production

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

AUDIT_DIR = REPO_ROOT / "data" / "audit"
TUNING_BAR_MULTIPLIER = 1.15  # the project's own stated bar, both conditions at once


def _season_folds(gold: pd.DataFrame, target: str, horizon: int, feature_cols: list[str]):
    """Yield (season_name, train, test, label_col) for one (target, horizon).

    Mirrors hazecast.hourly_model.evaluate_all_horizons()'s masking exactly,
    including the leak-buffer fix (train excludes rows starting `horizon`
    hours before the held-out season, since label_h{horizon}_{target} is
    now_{target} shifted back by `horizon`). Kept as a separate, explicit
    helper here -- rather than re-deriving it per stage -- so stages 2 and
    3 below are guaranteed to see identical train/test rows; only the
    fitted estimator differs between them.
    """
    label_col = f"label_h{horizon}_{target}"
    if label_col not in gold.columns:
        gold[label_col] = build_label(gold, target, horizon)
    complete = gold[feature_cols + [label_col]].notna().all(axis=1)

    for season_name, (start, end) in HAZE_SEASONS.items():
        end_exclusive = pd.Timestamp(end) + pd.Timedelta(days=1)
        train_excl_start = pd.Timestamp(start) - pd.Timedelta(hours=horizon)
        test_mask = complete & gold["date"].between(start, end_exclusive, inclusive="left")
        train_mask = complete & ~gold["date"].between(train_excl_start, end_exclusive, inclusive="left")
        train, test = gold.loc[train_mask], gold.loc[test_mask]
        if test.empty or train.empty:
            continue
        yield season_name, train, test, label_col


# %%
# ---------------------------------------------------------------------------
# Load the same gold table the production hourly model trains on. No
# features are recomputed here -- if this file is stale, re-run
# hazecast.silver_to_gold first (see README.md), same as for the real model.
# ---------------------------------------------------------------------------
gold_df = pd.read_parquet(GOLD_DIR / "hourly_features.parquet")
gold_df["date"] = pd.to_datetime(gold_df["date"])
gold_df = gold_df.sort_values("date").reset_index(drop=True)
feature_cols = _feature_columns(gold_df)
log.info("Loaded gold table: %d rows, %d feature columns", len(gold_df), len(feature_cols))


# %%
# ---------------------------------------------------------------------------
# STAGE 1 -- baseline: persistence (this hour's own reading carried
# forward as the forecast). Every candidate below is measured against
# this, per the project's own tuning bar -- not against an absolute
# number, and not against the candidates measuring each other.
# ---------------------------------------------------------------------------
persistence_rows = []
for target in TARGETS:
    now_col = f"now_{target}"
    for horizon in HOURLY_HORIZONS:
        for season_name, train, test, label_col in _season_folds(gold_df, target, horizon, feature_cols):
            pred = test[now_col].to_numpy()
            m = _metrics(test[label_col], pred, PM25_ELEVATED_THRESHOLD)
            m.update(stage="persistence", model="persistence", target=target, horizon=horizon, season=season_name)
            persistence_rows.append(m)

persistence_df = pd.DataFrame(persistence_rows)
log.info("Stage 1 (persistence): %d region/horizon/season cells", len(persistence_df))
persistence_df


# %%
# ---------------------------------------------------------------------------
# STAGE 2 -- plain vanilla: ordinary LinearRegression. No quantile
# objective, no alpha/regularization, same feature set and same
# train/test rows as the tuned model below. This is the "before tuning"
# reference point -- it was never itself shipped, which is exactly why
# there's no committed eval CSV for it; this script produces it fresh.
# ---------------------------------------------------------------------------
vanilla_rows = []
for target in TARGETS:
    for horizon in HOURLY_HORIZONS:
        for season_name, train, test, label_col in _season_folds(gold_df, target, horizon, feature_cols):
            model = LinearRegression().fit(train[feature_cols], train[label_col])
            pred = model.predict(test[feature_cols])
            m = _metrics(test[label_col], pred, PM25_ELEVATED_THRESHOLD)
            m.update(stage="vanilla_linreg", model="linreg_plain", target=target, horizon=horizon, season=season_name)
            vanilla_rows.append(m)

vanilla_df = pd.DataFrame(vanilla_rows)
log.info("Stage 2 (plain vanilla LR): %d region/horizon/season cells", len(vanilla_df))
vanilla_df


# %%
# ---------------------------------------------------------------------------
# STAGE 3 -- quantile-tuned: the actual shipped model. Calls
# hazecast.hourly_model.fit_predict() directly, which internally looks up
# each region/horizon's (quantile, alpha) from
# hazecast.config.HOURLY_TUNED_SETTINGS -- so this is the production
# logic itself, not a re-implementation, run here for side-by-side
# comparison against stages 1 and 2.
# ---------------------------------------------------------------------------
tuned_rows = []
for target in TARGETS:
    for horizon in HOURLY_HORIZONS:
        quantile, alpha = get_tuned_setting(target, horizon)
        for season_name, train, test, label_col in _season_folds(gold_df, target, horizon, feature_cols):
            _, pred = fit_predict(train, test, feature_cols, label_col, target, horizon)
            m = _metrics(test[label_col], pred, PM25_ELEVATED_THRESHOLD)
            m.update(
                stage="quantile_tuned", model="linreg_q", variant=f"q{quantile}_a{alpha}",
                target=target, horizon=horizon, season=season_name,
            )
            tuned_rows.append(m)

tuned_df = pd.DataFrame(tuned_rows)
log.info("Stage 3 (quantile-tuned LR): %d region/horizon/season cells", len(tuned_df))
tuned_df


# %%
# ---------------------------------------------------------------------------
# STAGE 4 -- verdict: apply the project's own stated tuning bar to both
# the vanilla and tuned stages, per region/horizon/season cell:
#   recall_elevated >= 1.15 x persistence's recall_elevated
#   AND
#   mae_elevated    <= 1.15 x persistence's mae_elevated
# Both conditions at once, same as the bar actually used to decide this.
# ---------------------------------------------------------------------------
def _apply_tuning_bar(candidate_df: pd.DataFrame, persistence_df: pd.DataFrame) -> pd.DataFrame:
    merged = candidate_df.merge(
        persistence_df[["target", "horizon", "season", "recall_elevated", "mae_elevated"]],
        on=["target", "horizon", "season"], suffixes=("", "_persistence"),
    )
    merged["recall_bar_pass"] = merged["recall_elevated"] >= TUNING_BAR_MULTIPLIER * merged["recall_elevated_persistence"]
    merged["mae_bar_pass"] = merged["mae_elevated"] <= TUNING_BAR_MULTIPLIER * merged["mae_elevated_persistence"]
    merged["clears_bar"] = merged["recall_bar_pass"] & merged["mae_bar_pass"]
    return merged


vanilla_verdict = _apply_tuning_bar(vanilla_df, persistence_df)
tuned_verdict = _apply_tuning_bar(tuned_df, persistence_df)

comparison = pd.concat([vanilla_verdict, tuned_verdict], ignore_index=True)
summary = (
    comparison.groupby(["stage", "horizon"])["clears_bar"]
    .agg(["sum", "count"])
    .rename(columns={"sum": "cells_passing", "count": "cells_total"})
)
log.info("Pass/fail summary by stage and horizon (out of region x season cells):\n%s", summary)
summary


# %%
# ---------------------------------------------------------------------------
# Save for the team to review/diff. New path, nothing in data/models/
# touched. Everything under data/ stays gitignored as before -- these CSVs
# are for local review, not something to commit.
# ---------------------------------------------------------------------------
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
comparison.to_csv(AUDIT_DIR / "hourly_model_selection_audit.csv", index=False)
summary.to_csv(AUDIT_DIR / "hourly_model_selection_audit_summary.csv")
log.info("Wrote %s and %s", AUDIT_DIR / "hourly_model_selection_audit.csv", AUDIT_DIR / "hourly_model_selection_audit_summary.csv")
