"""Batch forecasts: 18 hourly predictions by default; daily mode is explicit.

Run ``python -m hazecast.score`` after hourly training. Existing xgb_REGION.json
artifacts remain supported only when no daily selection manifest exists.
Hourly artifacts and constants are independent of the daily artifacts.
write_forecast writes hourly results to latest_hourly_forecast.json and
daily results to latest_forecast.json.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from hazecast.config import (
    GOLD_DIR, MODELS_DIR, SG_REGIONS, PM25_ELEVATED_THRESHOLD,
    FORECAST_HORIZON_DAYS,
)

log = logging.getLogger(__name__)
TARGETS = SG_REGIONS + ["national"]


def score_latest_daily(gold_path=GOLD_DIR / "daily_features.parquet", models_dir=MODELS_DIR):
    models_dir = Path(models_dir)
    gold = pd.read_parquet(gold_path)
    gold["date"] = pd.to_datetime(gold["date"], errors="raise")
    if gold.empty or gold.date.isna().any() or gold.date.duplicated().any():
        raise ValueError("Daily gold must have unique non-null dates and at least one row")
    gold = gold.sort_values("date")
    manifest_path = models_dir / "daily_model_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != 1 or manifest.get("pipeline") != "daily":
            raise ValueError("Unsupported daily model manifest")
        cols = manifest["feature_columns"]
        horizon, threshold = manifest["horizon_days"], manifest["threshold"]
        entries = manifest["models"]
        if set(entries) != set(TARGETS):
            raise ValueError("Manifest must contain every daily target")
    else:
        log.warning("No selection manifest; loading legacy XGBoost artifacts")
        cols = json.loads((models_dir / "feature_columns.json").read_text())
        horizon, threshold = FORECAST_HORIZON_DAYS, PM25_ELEVATED_THRESHOLD
        entries = {t: dict(model="legacy_xgboost", artifact=f"xgb_{t}.json") for t in TARGETS}
    if horizon < 1 or not np.isfinite(threshold):
        raise ValueError("Invalid horizon or threshold in model metadata")
    missing = set(cols) - set(gold.columns)
    if missing:
        raise ValueError(f"Missing gold features: {sorted(missing)}")
    complete = np.isfinite(gold[cols].to_numpy(dtype=float)).all(axis=1)
    if not complete.any():
        raise RuntimeError("No gold row has complete finite features")
    row = gold.loc[complete].iloc[[-1]]
    if row.date.iloc[0] != gold.date.max():
        log.warning("Scoring stale features as of %s; latest gold date is %s",
                    row.date.iloc[0], gold.date.max())
    forecast_for = row.date.iloc[0] + pd.Timedelta(days=horizon)
    scored_at = datetime.now(timezone.utc).isoformat()
    results = []
    for target in TARGETS:
        entry = entries[target]
        if entry["model"] == "persistence":
            pred = float(row[f"today_{target}"].iloc[0])
        else:
            artifact = entry.get("artifact")
            if not artifact or Path(artifact).name != artifact:
                raise ValueError(f"Invalid model artifact for {target}")
            path = models_dir / artifact
            if entry["model"] == "legacy_xgboost":
                model = XGBRegressor()
                model.load_model(path)
            else:
                # Load only artifacts produced by trusted project training.
                model = joblib.load(path)
            pred = float(model.predict(row[cols])[0])
        if not np.isfinite(pred):
            raise ValueError(f"Nonfinite prediction for {target}")
        pred = max(0.0, pred)
        results.append(dict(scored_at=scored_at,
                            as_of_date=row.date.iloc[0].date().isoformat(),
                            forecast_date=forecast_for.date().isoformat(), region=target,
                            forecast_pm25=round(pred, 1), elevated_flag=bool(pred >= threshold)))
    return pd.DataFrame(results)


def write_forecast(df, gold_dir=GOLD_DIR):
    gold_dir = Path(gold_dir)
    gold_dir.mkdir(parents=True, exist_ok=True)
    filename = "latest_hourly_forecast" if "horizon_hours" in df.columns else "latest_forecast"
    path = gold_dir / f"{filename}.json"
    temporary = gold_dir / f"{filename}.tmp"
    temporary.write_text(df.to_json(orient="records", indent=2))
    temporary.replace(path)
    return str(path)


def score_latest(gold_path=None, models_dir=None, *, pipeline="hourly"):
    """Default: 18 hourly predictions. Use pipeline='daily' for daily scoring."""
    if pipeline == "hourly":
        from hazecast.hourly_score import score_latest_hourly
        return score_latest_hourly(
            gold_path=GOLD_DIR / "hourly_features.parquet" if gold_path is None else gold_path,
            models_dir=MODELS_DIR / "hourly" if models_dir is None else models_dir,
        )
    if pipeline == "daily":
        return score_latest_daily(
            gold_path=GOLD_DIR / "daily_features.parquet" if gold_path is None else gold_path,
            models_dir=MODELS_DIR if models_dir is None else models_dir,
        )
    raise ValueError("pipeline must be 'hourly' or 'daily'")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    forecast = score_latest()
    log.info("Scored %d regions -> %s", len(forecast), write_forecast(forecast))
