"""Daily six-model comparison with purged forward validation and final holdout.

Run as ``python -m hazecast.train``. Hourly models/settings remain separate.
Gold features must already be causal: no future/backfilled observations or
unshifted labels may be included in feature engineering.
"""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from hazecast.config import (
    USE_SELECTED_FEATURES, SELECTED_FEATURES_DAILY,
    GOLD_DIR, MODELS_DIR, REPO_ROOT, SG_REGIONS, FORECAST_HORIZON_DAYS,
    PM25_ELEVATED_THRESHOLD, XGB_PARAMS, XGB_QUANTILE_ALPHA,
    MODEL_CANDIDATES, RANDOM_FOREST_PARAMS, RIDGE_ALPHA, LIGHTGBM_PARAMS, CV_SPLITS,
    FINAL_TEST_FRACTION, MIN_TRAIN_ROWS, MLFLOW_EXPERIMENT,
)

log = logging.getLogger(__name__)
TARGETS = SG_REGIONS + ["national"]
NON_FEATURE_COLS = {"date", "label_low_coverage"} | {f"label_{t}" for t in TARGETS}


def _prepare(gold):
    gold = gold.copy()
    gold["date"] = pd.to_datetime(gold["date"], errors="raise")
    if gold["date"].isna().any() or gold["date"].duplicated().any():
        raise ValueError("Daily gold must have one non-null date per row")
    gold = gold.sort_values("date").reset_index(drop=True)
    if gold.empty:
        raise ValueError("Empty daily gold table")
    return gold


def _feature_columns(gold):
    # Exclude every label, including future horizons unknown to this module.
    cols = [c for c in gold if c not in NON_FEATURE_COLS
            and not c.startswith(("label_", "target_"))]
    if USE_SELECTED_FEATURES:
        # keep only the data team's selected set (see config.SELECTED_FEATURES_DAILY)
        cols = [c for c in cols if c in SELECTED_FEATURES_DAILY]
    if not cols or any(not pd.api.types.is_numeric_dtype(gold[c]) for c in cols):
        raise ValueError("Gold features must be nonempty and numeric")
    return cols


def _complete(gold, cols, target):
    mask = np.isfinite(gold[cols + [f"label_{target}"]].to_numpy(dtype=float)).all(axis=1)
    if "label_low_coverage" in gold:
        mask &= gold["label_low_coverage"].eq(False).to_numpy()
    # Labels whose observation time is beyond the table's cutoff are unknown.
    mask &= (gold.date + pd.Timedelta(days=FORECAST_HORIZON_DAYS) <= gold.date.max()).to_numpy()
    return gold.loc[mask]


def _metrics(y_true, y_pred, threshold=PM25_ELEVATED_THRESHOLD):
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if not len(y) or y.shape != p.shape or not np.isfinite(p).all():
        raise ValueError("Metrics require finite, aligned nonempty predictions")
    actual, flagged = y >= threshold, p >= threshold
    tp = int((actual & flagged).sum())
    fp = int((~actual & flagged).sum())
    fn = int((actual & ~flagged).sum())
    # F2 = 5 TP / (5 TP + 4 FN + FP). No positive support is
    # explicitly undefined, so a quiet fold cannot win selection on F2.
    f2 = 5 * tp / (5 * tp + 4 * fn + fp) if actual.any() else float("nan")
    return dict(mae=float(mean_absolute_error(y, p)),
                rmse=float(np.sqrt(mean_squared_error(y, p))), f2=f2,
                tp=tp, fp=fp, fn=fn, n_test=len(y), n_elevated=int(actual.sum()))


def save_model(model, path):
    """Write a fitted model with Python's standard C pickler (loadable by joblib.load).

    Not joblib.dump: joblib pickles with the pure-Python pickler, whose dispatch table
    Airflow's import of `dill` patches, so models trained inside some Airflow task
    processes silently required `dill` to load (found 2026-10-28: the retrain's artifacts
    referenced dill._dill, earlier ones did not). The C pickler is not patched.
    """
    import pickle
    with open(path, "wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _make_model(name):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError(
                "LightGBM is required for this comparison. Install lightgbm "
                "in the same Python environment as the notebook kernel."
            ) from exc
        return LGBMRegressor(**LIGHTGBM_PARAMS)
    if name == "random_forest":
        return RandomForestRegressor(**RANDOM_FOREST_PARAMS)
    if name == "xgboost":
        return XGBRegressor(**{**XGB_PARAMS, "objective": "reg:absoluteerror"})
    if name == "xgboost_q85":
        return XGBRegressor(**{**XGB_PARAMS, "objective": "reg:quantileerror",
                              "quantile_alpha": XGB_QUANTILE_ALPHA})
    if name == "persistence":
        return None
    raise ValueError(f"Unknown model: {name}")


def _predict(model, rows, cols, target):
    pred = (rows[f"today_{target}"].to_numpy(dtype=float) if model is None
            else model.predict(rows[cols]))
    return np.maximum(0.0, np.asarray(pred, dtype=float))


def _fit(name, rows, cols, target):
    model = _make_model(name)
    if model is not None:
        model.fit(rows[cols], rows[f"label_{target}"])
    return model


def _before(rows, boundary):
    # Strict embargo: training label timestamps precede the next origin.
    return rows.loc[rows.date + pd.Timedelta(days=FORECAST_HORIZON_DAYS) < boundary]


def _setup_mlflow():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def _evaluate(gold):
    gold = _prepare(gold)
    cols = _feature_columns(gold)
    if not 0 < FINAL_TEST_FRACTION < 1 or CV_SPLITS < 1:
        raise ValueError("Invalid evaluation split configuration")
    dates = gold.date
    test_start = dates.iloc[int(len(dates) * (1 - FINAL_TEST_FRACTION))]
    development = _before(gold, test_start)
    # First block initializes training; remaining blocks are forward validation.
    blocks = np.array_split(development.date.to_numpy(), CV_SPLITS + 1)
    if any(len(b) == 0 for b in blocks):
        raise ValueError("Insufficient dates for forward validation")
    results, selections = [], {}
    _setup_mlflow()
    for target in TARGETS:
        if f"today_{target}" not in cols:
            raise ValueError(f"Missing persistence feature today_{target}")
        rows = _complete(gold, cols, target)
        pooled = {name: ([], []) for name in MODEL_CANDIDATES}
        for fold, block in enumerate(blocks[1:], 1):
            start, end = pd.Timestamp(block[0]), pd.Timestamp(block[-1])
            train = _before(rows, start)
            valid = rows.loc[rows.date.between(start, end)]
            if len(train) < MIN_TRAIN_ROWS or valid.empty:
                raise ValueError(f"{target}: insufficient rows in validation fold {fold}")
            for name in MODEL_CANDIDATES:
                with mlflow.start_run(run_name=f"{target}_{name}_cv{fold}"):
                    model = _fit(name, train, cols, target)
                    pred = _predict(model, valid, cols, target)
                    metrics = _metrics(valid[f"label_{target}"], pred)
                    params = dict(target=target, model=name, split="validation", fold=fold,
                                  train_end=str(train.date.max()), valid_start=str(start),
                                  n_train=len(train), horizon_days=FORECAST_HORIZON_DAYS,
                                  threshold=PM25_ELEVATED_THRESHOLD)
                    mlflow.log_params(params)
                    if model is not None:
                        mlflow.log_dict(
                            json.loads(json.dumps(model.get_params(), default=str)),
                            "model_parameters.json",
                        )
                    mlflow.log_metrics({k: v for k, v in metrics.items() if np.isfinite(v)})
                    results.append({**params, **metrics})
                pooled[name][0].extend(valid[f"label_{target}"].tolist())
                pooled[name][1].extend(pred.tolist())
        summaries = {n: _metrics(*pair) for n, pair in pooled.items()}
        has_haze = next(iter(summaries.values()))["n_elevated"] > 0
        winner = min(MODEL_CANDIDATES, key=lambda n: (
            (-summaries[n]["f2"], summaries[n]["mae"], summaries[n]["rmse"])
            if has_haze else (summaries[n]["mae"], summaries[n]["rmse"])))
        for name, metrics in summaries.items():
            results.append(dict(target=target, model=name, split="validation_pooled", **metrics))
        # The latest contiguous block is touched only after selection is fixed.
        train = _before(rows, test_start)
        test = rows.loc[rows.date >= test_start]
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            raise ValueError(f"{target}: insufficient training/final holdout rows")
        selected = _fit(winner, train, cols, target)
        holdout = _metrics(test[f"label_{target}"], _predict(selected, test, cols, target))
        results.append(dict(target=target, model=winner, split="final_test", **holdout))
        selections[target] = dict(model=winner, validation=summaries[winner], final_test=holdout,
                                  selection_rule="f2_then_mae_rmse" if has_haze else "mae_then_rmse_no_positive_support",
                                  test_start=test_start.isoformat())
    return pd.DataFrame(results), selections, cols


def evaluate_all(gold):
    """Return validation and selected-model holdout results without publishing models."""
    return _evaluate(gold)[0]


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _log_selected_model(model, candidate):
    """Use native boosting formats and support MLflow 2.x/3.x naming."""
    import inspect
    if candidate in ("xgboost", "xgboost_q85"):
        import mlflow.xgboost
        logger = mlflow.xgboost.log_model
        saved_model = model
        options = {"model_format": "json"}
    elif candidate == "lightgbm":
        import mlflow.lightgbm
        logger = mlflow.lightgbm.log_model
        # A native Booster avoids sklearn/skops serialization of its wrapper.
        saved_model = model.booster_
        options = {}
    elif candidate in ("ridge", "random_forest"):
        import mlflow.sklearn
        logger = mlflow.sklearn.log_model
        saved_model = model
        # These estimators were fitted by this process. Load only trusted outputs.
        options = {"serialization_format": "cloudpickle"}
    else:
        raise ValueError(f"No fitted-model logger for {candidate}")
    location_key = "name" if "name" in inspect.signature(logger).parameters else "artifact_path"
    return logger(saved_model, **{location_key: "model"}, **options)


def train_final_models(gold):
    """Evaluate/select, refit using known labels, and publish manifest last.

    joblib artifacts must only be loaded from a trusted training directory.
    The daily scoring contract stays independent of hourly model artifacts.
    """
    gold = _prepare(gold)
    summary, selections, cols = _evaluate(gold)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(MODELS_DIR / "eval_summary.csv", index=False)
    manifest = dict(schema_version=1, pipeline="daily", feature_columns=cols,
                    horizon_days=FORECAST_HORIZON_DAYS,
                    threshold=PM25_ELEVATED_THRESHOLD, models={})
    paths = {}
    for target, selection in selections.items():
        rows = _complete(gold, cols, target)
        model = _fit(selection["model"], rows, cols, target)
        with mlflow.start_run(run_name=f"final_{target}_{selection['model']}") as run:
            mlflow.log_params(dict(target=target, model=selection["model"], n_train=len(rows),
                                   horizon_days=FORECAST_HORIZON_DAYS, n_features=len(cols)))
            for split in ("validation", "final_test"):
                mlflow.log_metrics({f"{split}_{k}": v for k, v in selection[split].items()
                                    if np.isfinite(v)})
            entry = {**selection, "run_id": run.info.run_id, "artifact": None,
                     "training_end": rows.date.max().isoformat()}
            if model is not None:
                # Unique names keep the previous manifest valid until publication.
                path = MODELS_DIR / f"daily_{target}_{run.info.run_id}.joblib"
                save_model(model, path)
                paths[target] = path
                entry["artifact"] = path.name
                _log_selected_model(model, selection["model"])
                mlflow.log_artifact(str(path))
            mlflow.log_dict(_json_safe(entry), "selection.json")
            manifest["models"][target] = entry
    (MODELS_DIR / "feature_columns.json").write_text(json.dumps(cols, indent=2))
    temporary = MODELS_DIR / "daily_model_manifest.tmp"
    temporary.write_text(json.dumps(_json_safe(manifest), indent=2, allow_nan=False))
    temporary.replace(MODELS_DIR / "daily_model_manifest.json")
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    paths = train_final_models(pd.read_parquet(GOLD_DIR / "daily_features.parquet"))
    log.info("Published daily model manifest and %d fitted artifacts", len(paths))
