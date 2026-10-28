"""Six-candidate hourly forecasts at +6/+12/+24 hours for six targets.

Uses exact future timestamp labels, purged expanding validation, a final
holdout and a separate hourly manifest. Source hourly feature engineering
is unchanged. Timestamps without an offset are interpreted as Singapore time.
"""
from __future__ import annotations
import json
import logging
import os
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
from pathlib import Path
import joblib
import mlflow
import numpy as np
import pandas as pd
from hazecast.config import (
    USE_SELECTED_FEATURES, SELECTED_FEATURES_HOURLY,
    SELECTION_DEFAULT_CANDIDATE, SELECTION_SWITCH_CONFIDENCE,
    SELECTION_BOOTSTRAP_BLOCK_HOURS, SELECTION_BOOTSTRAP_SAMPLES,
    GOLD_DIR, MODELS_DIR, SG_REGIONS, HOURLY_HORIZONS, HOURLY_TIMEZONE,
    PM25_ELEVATED_THRESHOLD, MODEL_CANDIDATES, CV_SPLITS,
    FINAL_TEST_FRACTION, MIN_TRAIN_ROWS, HOURLY_MLFLOW_EXPERIMENT, REPO_ROOT,
)
from hazecast.train import _make_model, _metrics, _json_safe, _log_selected_model, save_model

log = logging.getLogger(__name__)
TARGETS = SG_REGIONS + ["national"]


def _prepare(gold):
    gold = gold.copy()
    dates = pd.to_datetime(gold["date"], errors="raise")
    dates = (dates.dt.tz_localize(HOURLY_TIMEZONE) if dates.dt.tz is None
             else dates.dt.tz_convert(HOURLY_TIMEZONE))
    gold["date"] = dates
    if dates.isna().any() or dates.duplicated().any() or gold.empty:
        raise ValueError("Hourly gold requires unique, non-null timestamps")
    if not dates.eq(dates.dt.floor("h")).all():
        raise ValueError("Hourly gold timestamps must fall on hour boundaries")
    gold = gold.sort_values("date").reset_index(drop=True)
    if len(gold) > 1 and not gold.date.diff().eq(pd.Timedelta(hours=1)).any():
        raise ValueError("Expected hourly data, not daily gold")
    missing = {f"now_{t}" for t in TARGETS} - set(gold.columns)
    if missing:
        raise ValueError(f"Missing hourly PM2.5 columns: {sorted(missing)}")
    return gold


def _feature_columns(gold):
    cols = [c for c in gold if c not in {"date", "label_low_coverage", "fire_carried_hours"}
            and not c.startswith(("label_", "target_"))]
    if USE_SELECTED_FEATURES:
        # keep only the data team's selected set (see config.SELECTED_FEATURES_HOURLY)
        cols = [c for c in cols if c in SELECTED_FEATURES_HOURLY]
    if not cols or any(not pd.api.types.is_numeric_dtype(gold[c]) for c in cols):
        raise ValueError("Hourly features must be nonempty and numeric")
    return cols


def build_label(gold, target, horizon):
    """Match t+h exactly; gaps produce missing labels, never shifted hours."""
    if not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise ValueError("Horizon must be a positive integer number of hours")
    dates = pd.to_datetime(gold["date"], errors="raise")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Labels require unique, non-null timestamps")
    readings = pd.Series(gold[f"now_{target}"].to_numpy(), index=pd.DatetimeIndex(dates))
    future = pd.DatetimeIndex(dates) + pd.Timedelta(hours=int(horizon))
    return pd.Series(readings.reindex(future).to_numpy(), index=gold.index)


def _labelled(gold, horizon):
    gold = _prepare(gold)
    for target in TARGETS:
        # Always rebuild: ignore stale or pre-shifted labels in source gold.
        gold[f"label_{target}"] = build_label(gold, target, horizon)
    return gold


def _complete(gold, cols, target, horizon):
    mask = np.isfinite(gold[cols + [f"label_{target}"]].to_numpy(dtype=float)).all(axis=1)
    if "label_low_coverage" in gold:
        mask &= gold["label_low_coverage"].eq(False).to_numpy()
    mask &= (gold.date + pd.Timedelta(hours=horizon) <= gold.date.max()).to_numpy()
    return gold.loc[mask]


def _before(rows, boundary, horizon):
    return rows.loc[rows.date + pd.Timedelta(hours=horizon) < boundary]


def _fit(name, rows, cols, target):
    model = _make_model(name)
    if model is not None:
        model.fit(rows[cols], rows[f"label_{target}"])
    return model


def _predict(model, rows, cols, target):
    pred = (rows[f"now_{target}"].to_numpy(dtype=float) if model is None
            else model.predict(rows[cols]))
    return np.maximum(0.0, np.asarray(pred, dtype=float))


def switch_confidence(y, dates, p_alt, p_def, *, block_hours=SELECTION_BOOTSTRAP_BLOCK_HOURS,
                      samples=SELECTION_BOOTSTRAP_SAMPLES, seed=42):
    """Share of paired block-bootstrap resamples in which p_alt has the higher F2 than p_def.

    Validation rows are resampled in whole blocks (default one week) so the serial
    dependence of hourly PM2.5, and whole haze episodes, are kept together.
    """
    y, p_alt, p_def = (np.asarray(a, dtype=float) for a in (y, p_alt, p_def))
    t = pd.to_datetime(pd.Series(dates))
    block = ((t - t.min()) // pd.Timedelta(hours=block_hours)).to_numpy()
    groups = [np.flatnonzero(block == b) for b in np.unique(block)]
    rng = np.random.default_rng(seed)
    wins = valid = 0
    for _ in range(samples):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        a, b = _metrics(y[idx], p_alt[idx])["f2"], _metrics(y[idx], p_def[idx])["f2"]
        if np.isfinite(a) and np.isfinite(b):
            valid += 1
            wins += a > b
    return wins / valid if valid else 0.0


def _setup_mlflow():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}"))
    mlflow.set_experiment(HOURLY_MLFLOW_EXPERIMENT)


def _horizons(horizons):
    values = list(HOURLY_HORIZONS if horizons is None else horizons)
    if not values or len(values) != len(set(values)):
        raise ValueError("Horizons must be nonempty and unique")
    if any(not isinstance(h, (int, np.integer)) or h <= 0 for h in values):
        raise ValueError("Horizons must be positive integer hours")
    return [int(h) for h in values]


def _evaluate(gold, horizon):
    gold = _labelled(gold, horizon)
    cols = _feature_columns(gold)
    if not 0 < FINAL_TEST_FRACTION < 1 or CV_SPLITS < 1:
        raise ValueError("Invalid evaluation split configuration")
    dates = gold.date
    test_start = dates.iloc[int(len(dates) * (1 - FINAL_TEST_FRACTION))]
    development = _before(gold, test_start, horizon)
    # First block initializes training; remaining blocks are forward validation.
    blocks = np.array_split(development.date.to_numpy(), CV_SPLITS + 1)
    if any(len(b) == 0 for b in blocks):
        raise ValueError("Insufficient dates for forward validation")
    results, selections = [], {}
    _setup_mlflow()
    for target in TARGETS:
        if f"now_{target}" not in cols:
            raise ValueError(f"Missing persistence feature now_{target}")
        rows = _complete(gold, cols, target, horizon)
        pooled = {name: ([], []) for name in MODEL_CANDIDATES}
        pooled_dates = []
        for fold, block in enumerate(blocks[1:], 1):
            start, end = pd.Timestamp(block[0]), pd.Timestamp(block[-1])
            train = _before(rows, start, horizon)
            valid = rows.loc[rows.date.between(start, end)]
            if len(train) < MIN_TRAIN_ROWS or valid.empty:
                raise ValueError(f"{target}: insufficient rows in validation fold {fold}")
            for name in MODEL_CANDIDATES:
                with mlflow.start_run(run_name=f"{target}_h{horizon}_{name}_cv{fold}"):
                    model = _fit(name, train, cols, target)
                    pred = _predict(model, valid, cols, target)
                    metrics = _metrics(valid[f"label_{target}"], pred)
                    params = dict(target=target, model=name, split="validation", fold=fold,
                                  train_end=str(train.date.max()), valid_start=str(start),
                                  n_train=len(train), horizon_hours=horizon,
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
            pooled_dates.extend(valid.date.tolist())
        summaries = {n: _metrics(*pair) for n, pair in pooled.items()}
        has_haze = next(iter(summaries.values()))["n_elevated"] > 0
        winner = min(MODEL_CANDIDATES, key=lambda n: (
            (-summaries[n]["f2"], summaries[n]["mae"], summaries[n]["rmse"])
            if has_haze else (summaries[n]["mae"], summaries[n]["rmse"])))
        stability = None
        default = SELECTION_DEFAULT_CANDIDATE
        if has_haze and winner != default and default in pooled:
            conf = switch_confidence(pooled[winner][0], pooled_dates, pooled[winner][1], pooled[default][1])
            stability = dict(proposed=winner, default=default, confidence=conf,
                             required=SELECTION_SWITCH_CONFIDENCE,
                             kept=winner if conf >= SELECTION_SWITCH_CONFIDENCE else default)
            log.info("%s h=%d: %s beat %s on pooled F2 in %.0f%% of resamples -> %s", target, horizon,
                     winner, default, 100 * conf, stability["kept"])
            winner = stability["kept"]
        for name, metrics in summaries.items():
            results.append(dict(target=target, model=name, split="validation_pooled", **metrics))
        # The latest contiguous block is touched only after selection is fixed.
        train = _before(rows, test_start, horizon)
        test = rows.loc[rows.date >= test_start]
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            raise ValueError(f"{target}: insufficient training/final holdout rows")
        selected = _fit(winner, train, cols, target)
        holdout = _metrics(test[f"label_{target}"], _predict(selected, test, cols, target))
        results.append(dict(target=target, model=winner, split="final_test", **holdout))
        selections[target] = dict(model=winner, validation=summaries[winner], final_test=holdout,
                                  selection_rule="f2_then_mae_rmse" if has_haze else "mae_then_rmse_no_positive_support",
                                  stability=stability, test_start=test_start.isoformat())
    summary = pd.DataFrame(results)
    summary["horizon_hours"] = horizon
    return summary, selections, cols


def evaluate_all_horizons(gold, haze_seasons=None, *, horizons=None):
    """Compare six candidates per target/horizon using forward splits.

    haze_seasons is accepted for old callers but no longer controls splitting.
    No future seasons are included in any earlier training fold.
    """
    if haze_seasons is not None:
        log.warning("haze_seasons is ignored; hourly evaluation uses forward-only splits")
    return pd.concat([_evaluate(gold, h)[0] for h in _horizons(horizons)], ignore_index=True)


def train_final_hourly_models(gold, *, horizons=None, out_dir=None):
    """Compare/select, refit 18 winners, publish hourly manifest last.

    out_dir defaults to the production directory (MODELS_DIR/hourly); the
    retrain DAG passes a staging directory and promotes only after its gate.

    Returns 18 (target,horizon) entries. Persistence entries are None because
    persistence needs no fitted artifact. Single training publisher required.
    """
    gold = _prepare(gold)
    horizons = _horizons(horizons)
    evaluations = []
    decisions = {}
    # Evaluate all horizons before publishing any new manifest.
    for horizon in horizons:
        summary, selections, cols = _evaluate(gold, horizon)
        evaluations.append(summary)
        decisions[horizon] = selections
    out_dir = Path(out_dir) if out_dir is not None else MODELS_DIR / "hourly"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(evaluations, ignore_index=True)
    summary.to_csv(out_dir / "eval_summary.csv", index=False)
    manifest = dict(schema_version=1, pipeline="hourly", horizons_hours=horizons,
                    timezone=HOURLY_TIMEZONE, feature_columns=cols,
                    threshold=PM25_ELEVATED_THRESHOLD,
                    trained_at=pd.Timestamp.now(tz="UTC").isoformat(), models={})
    paths = {}
    for horizon in horizons:
        labelled = _labelled(gold, horizon)
        manifest["models"][str(horizon)] = {}
        for target, selection in decisions[horizon].items():
            rows = _complete(labelled, cols, target, horizon)
            model = _fit(selection["model"], rows, cols, target)
            with mlflow.start_run(run_name=f"final_{target}_h{horizon}_{selection['model']}") as run:
                mlflow.log_params(dict(target=target, model=selection["model"],
                    horizon_hours=horizon, n_train=len(rows), n_features=len(cols),
                    threshold=PM25_ELEVATED_THRESHOLD))
                for split in ("validation", "final_test"):
                    mlflow.log_metrics({f"{split}_{k}": v for k, v in selection[split].items()
                                        if np.isfinite(v)})
                entry = dict(**selection, artifact=None, run_id=run.info.run_id,
                    training_end=rows.date.max().isoformat(),
                    training_label_end=(rows.date.max() + pd.Timedelta(hours=horizon)).isoformat())
                path = None
                if model is not None:
                    path = out_dir / f"{target}_h{horizon}_{run.info.run_id}.joblib"
                    save_model(model, path)
                    entry["artifact"] = path.name
                    _log_selected_model(model, selection["model"])
                    mlflow.log_artifact(str(path))
                paths[(target, horizon)] = path
                mlflow.log_dict(_json_safe(entry), "selection.json")
                manifest["models"][str(horizon)][target] = entry
    temp = out_dir / "model_manifest.tmp"
    temp.write_text(json.dumps(_json_safe(manifest), indent=2, allow_nan=False))
    temp.replace(out_dir / "model_manifest.json")
    log.info("Published %d hourly selections for horizons %s", len(paths), horizons)
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gold = pd.read_parquet(GOLD_DIR / "hourly_features.parquet")
    train_final_hourly_models(gold)
