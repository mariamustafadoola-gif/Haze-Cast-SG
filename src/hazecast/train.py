"""Train and evaluate next-day PM2.5 forecasters, one per Singapore region.

Evaluation is leave-one-haze-season-out (see hazecast.config.HAZE_SEASONS),
not a single chronological split. This project's own data profiling found
that "elevated" days (>=56 ug/m3) are about 2% of all days, concentrated in
a handful of seasons (2015, 2019, and the still-unlabelled 2026 episode).
A calendar-year train/validate/test split can land a whole fold with zero
elevated days in it, at which point "recall on elevated days" is undefined
for that fold. Leave-one-season-out guarantees every test fold actually
contains haze to evaluate against.

Four candidates are compared per region, per season:
  - persistence: tomorrow = today (the baseline every model must beat)
  - linreg: plain linear regression on the same feature set
  - xgboost: gradient-boosted trees minimizing MAE (the model in the
    proposal) -- kept as a comparison baseline; see xgboost_q85 below for
    why this is NOT what's shipped as the production model
  - xgboost_q85: gradient-boosted trees targeting the 85th percentile
    (hazecast.config.XGB_QUANTILE_ALPHA) instead of the mean -- THIS is
    the production model (see train_final_models). Plain MAE regression
    under-predicts rare elevated days (~2% of all days); a high-quantile
    objective deliberately biases toward catching them, trading some
    precision (more false alarms) for much better recall on the days
    this tool actually exists to flag. See README "Results" for the
    real evaluated numbers behind this choice.

Results are written to data/models/eval_summary.csv and logged to MLflow.
A final XGBoost model per region, trained on all available labelled rows,
is saved to data/models/ and registered in MLflow for the scoring step.
"""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path

# MLflow tries to stamp each run with the current git commit/branch/remote
# via GitPython. pipeline.Dockerfile doesn't install the `git` binary (the
# api/train/score containers don't need a working repo), so without this,
# every mlflow.start_run() call prints ~10 lines of "git executable not
# found" warnings -- harmless, but noisy enough to look like a crash (see
# README). Must be set before mlflow is imported.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from hazecast.config import (
    USE_SELECTED_FEATURES, SELECTED_FEATURES_DAILY,
    GOLD_DIR, MODELS_DIR, REPO_ROOT, SG_REGIONS,
    HAZE_SEASONS, PM25_ELEVATED_THRESHOLD, XGB_PARAMS, XGB_QUANTILE_ALPHA,
)

log = logging.getLogger(__name__)

TARGETS = SG_REGIONS + ["national"]
NON_FEATURE_COLS = {"date"} | {f"label_{t}" for t in TARGETS} | {"label_low_coverage"}


def _feature_columns(gold: pd.DataFrame) -> list[str]:
    cols = [c for c in gold.columns if c not in NON_FEATURE_COLS]
    if USE_SELECTED_FEATURES:
        # keep only the data team's selected set (see config.SELECTED_FEATURES_DAILY)
        cols = [c for c in cols if c in SELECTED_FEATURES_DAILY]
    return cols


def _usable_seasons(gold: pd.DataFrame, feature_cols: list[str]) -> dict[str, tuple[str, str]]:
    """Only evaluate a season if its rows have complete feature coverage.
    Currently excludes 2015 until the 2014-2018 FIRMS backfill lands, since
    fire features are all-NaN before 2019-01-01 in this project's data.
    """
    usable = {}
    for name, (start, end) in HAZE_SEASONS.items():
        mask = gold["date"].between(start, end)
        if not mask.any():
            log.warning("Season '%s' (%s to %s): no rows in gold table, skipping", name, start, end)
            continue
        coverage = gold.loc[mask, feature_cols].notna().all(axis=1).mean()
        if coverage < 0.95:
            log.warning(
                "Season '%s': only %.0f%% of rows have complete features "
                "(likely missing FIRMS backfill) -- skipping this fold",
                name, coverage * 100,
            )
            continue
        usable[name] = (start, end)
    return usable


def _metrics(y_true: pd.Series, y_pred: np.ndarray, threshold: float) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    elevated = y_true >= threshold
    n_elevated = int(elevated.sum())
    if n_elevated:
        mae_elevated = mean_absolute_error(y_true[elevated], y_pred[elevated])
        recall_elevated = float(((y_pred >= threshold) & elevated).sum() / n_elevated)
    else:
        mae_elevated, recall_elevated = np.nan, np.nan
    flagged = int((y_pred >= threshold).sum())
    precision_elevated = (
        float(((y_pred >= threshold) & elevated).sum() / flagged) if flagged else np.nan
    )
    return {
        "mae": mae,
        "mae_elevated": mae_elevated,
        "recall_elevated": recall_elevated,
        "precision_elevated": precision_elevated,
        "n_test": len(y_true),
        "n_elevated": n_elevated,
    }


def run_season_fold(
    gold: pd.DataFrame, target: str, feature_cols: list[str],
    season_name: str, start: str, end: str,
) -> list[dict]:
    label_col = f"label_{target}"
    complete = gold[feature_cols + [label_col]].notna().all(axis=1)
    test_mask = complete & gold["date"].between(start, end)
    train_mask = complete & ~gold["date"].between(start, end)

    train, test = gold.loc[train_mask], gold.loc[test_mask]
    if len(test) == 0 or len(train) == 0:
        log.warning("target=%s season=%s: empty train or test fold, skipping", target, season_name)
        return []

    X_train, y_train = train[feature_cols], train[label_col]
    X_test, y_test = test[feature_cols], test[label_col]
    today_col = f"today_{target}"

    results = []

    persistence_pred = test[today_col].to_numpy()
    results.append({"model": "persistence", **_metrics(y_test, persistence_pred, PM25_ELEVATED_THRESHOLD)})

    lr = LinearRegression().fit(X_train, y_train)
    results.append({"model": "linreg", **_metrics(y_test, lr.predict(X_test), PM25_ELEVATED_THRESHOLD)})

    xgb = XGBRegressor(**XGB_PARAMS).fit(X_train, y_train)
    results.append({"model": "xgboost", **_metrics(y_test, xgb.predict(X_test), PM25_ELEVATED_THRESHOLD)})

    xgb_q = XGBRegressor(
        **XGB_PARAMS, objective="reg:quantileerror", quantile_alpha=XGB_QUANTILE_ALPHA,
    ).fit(X_train, y_train)
    results.append({"model": "xgboost_q85", **_metrics(y_test, xgb_q.predict(X_test), PM25_ELEVATED_THRESHOLD)})

    for r in results:
        r.update(target=target, season=season_name, n_train=len(train))
    return results


def evaluate_all(gold: pd.DataFrame) -> pd.DataFrame:
    feature_cols = _feature_columns(gold)
    seasons = _usable_seasons(gold, feature_cols)
    if not seasons:
        raise RuntimeError(
            "No haze season has complete feature coverage -- run the "
            "2014-2018 FIRMS backfill, or add a covered season to "
            "hazecast.config.HAZE_SEASONS, before evaluating."
        )
    log.info("Evaluating leave-one-season-out on: %s", list(seasons))

    all_results = []
    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("hazecast-sg-loso")
    for target in TARGETS:
        for season_name, (start, end) in seasons.items():
            with mlflow.start_run(run_name=f"{target}_{season_name}"):
                mlflow.log_params({
                    "target": target, "held_out_season": season_name,
                    "season_start": start, "season_end": end,
                    "n_features": len(feature_cols),
                })
                fold_results = run_season_fold(gold, target, feature_cols, season_name, start, end)
                for r in fold_results:
                    prefix = r["model"]
                    for k, v in r.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            mlflow.log_metric(f"{prefix}_{k}", v)
                all_results.extend(fold_results)
    return pd.DataFrame(all_results)


def train_final_models(gold: pd.DataFrame) -> dict[str, Path]:
    """Fit one XGBoost model per region on ALL available labelled rows
    (not just the LOSO folds) and persist + register each in MLflow. This
    is the model the batch scoring job actually loads.

    Uses the quantile objective (hazecast.config.XGB_QUANTILE_ALPHA), not
    plain MAE regression -- see the module docstring and README "Results"
    for why: evaluated on real data, this roughly triples elevated-day
    recall versus the MAE-minimizing variant, which is what actually
    matters for an early-warning tool. save_model()/load_model() were
    verified to round-trip a quantile model's predictions exactly (the
    objective is stored in the saved file), so hazecast.score doesn't need
    any change to load it correctly.
    """
    feature_cols = _feature_columns(gold)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("hazecast-sg-production")

    paths = {}
    for target in TARGETS:
        label_col = f"label_{target}"
        complete = gold[feature_cols + [label_col]].notna().all(axis=1)
        train = gold.loc[complete]
        if len(train) < 100:
            log.warning("target=%s: only %d fully-featured rows, skipping final fit", target, len(train))
            continue
        model = XGBRegressor(
            **XGB_PARAMS, objective="reg:quantileerror", quantile_alpha=XGB_QUANTILE_ALPHA,
        ).fit(train[feature_cols], train[label_col])

        out_path = MODELS_DIR / f"xgb_{target}.json"
        model.save_model(out_path)
        paths[target] = out_path

        with mlflow.start_run(run_name=f"final_{target}"):
            mlflow.log_params({"target": target, "n_train": len(train), "n_features": len(feature_cols)})
            mlflow.xgboost.log_model(model, name=f"model_{target}")
            mlflow.log_artifact(str(out_path))

    (MODELS_DIR / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2))
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gold_df = pd.read_parquet(GOLD_DIR / "daily_features.parquet")
    gold_df["date"] = pd.to_datetime(gold_df["date"])

    eval_df = evaluate_all(gold_df)
    summary_path = MODELS_DIR / "eval_summary.csv"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(summary_path, index=False)
    log.info("Wrote evaluation summary (%d rows) to %s", len(eval_df), summary_path)

    pivot = eval_df.pivot_table(index=["target", "season"], columns="model", values="mae")
    log.info("MAE by target/season (lower is better), all 4 candidates:\n%s", pivot.round(2))

    model_paths = train_final_models(gold_df)
    log.info("Saved final models: %s", {k: str(v) for k, v in model_paths.items()})
