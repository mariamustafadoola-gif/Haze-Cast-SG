"""Score exactly 6/12/24h-ahead PM2.5 readings: 18 forecasts per origin."""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from hazecast.config import GOLD_DIR, MODELS_DIR, SG_REGIONS
from hazecast.hourly_model import _prepare

log = logging.getLogger(__name__)
TARGETS = SG_REGIONS + ["national"]
REQUIRED_HORIZONS = [6, 12, 24]


def score_latest_hourly(gold_path=GOLD_DIR / "hourly_features.parquet",
                        models_dir=MODELS_DIR / "hourly"):
    models_dir = Path(models_dir)
    manifest_path = models_dir / "model_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("No hourly selection manifest. Run train_final_hourly_models(hourly_gold) first.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("pipeline") != "hourly":
        raise ValueError("Unsupported hourly model manifest")
    horizons = manifest["horizons_hours"]
    if sorted(horizons) != REQUIRED_HORIZONS:
        raise ValueError("Hourly scoring requires trained 6, 12 and 24 hour horizons")
    cols = manifest["feature_columns"]
    if not cols or len(cols) != len(set(cols)):
        raise ValueError("Invalid hourly feature schema")
    threshold = float(manifest["threshold"])
    if not np.isfinite(threshold):
        raise ValueError("Invalid saved threshold")
    for horizon in horizons:
        if set(manifest["models"].get(str(horizon), {})) != set(TARGETS):
            raise ValueError(f"Missing target entries for horizon {horizon}")
    gold = _prepare(pd.read_parquet(gold_path))
    missing = set(cols) - set(gold.columns)
    if missing:
        raise ValueError(f"Missing hourly features: {sorted(missing)}")
    complete = np.isfinite(gold[cols].to_numpy(dtype=float)).all(axis=1)
    if not complete.any():
        raise RuntimeError("No complete finite hourly feature row; refresh hourly gold")
    row = gold.loc[complete].iloc[[-1]]
    origin = row.date.iloc[0]
    latest = gold.date.max()
    scored_at = datetime.now(timezone.utc)
    age_hours = (pd.Timestamp(scored_at) - origin.tz_convert("UTC")).total_seconds() / 3600
    if origin < latest or age_hours > 1:
        log.warning("Hourly forecasts use features as of %s; latest gold timestamp=%s; age=%.1fh. "
                    "Forecast times remain anchored to this feature origin.", origin, latest, age_hours)
    results = []
    for horizon in sorted(horizons):
        forecast_time = origin + pd.Timedelta(hours=int(horizon))
        for target in TARGETS:
            entry = manifest["models"][str(horizon)][target]
            name = entry["model"]
            if name == "persistence":
                pred = float(row[f"now_{target}"].iloc[0])
            else:
                artifact = entry.get("artifact")
                if not artifact or Path(artifact).name != artifact:
                    raise ValueError(f"Invalid hourly artifact for {target}, h={horizon}")
                model = joblib.load(models_dir / artifact)
                pred = float(model.predict(row[cols])[0])
            if not np.isfinite(pred):
                raise ValueError(f"Nonfinite prediction for {target}, h={horizon}")
            pred = max(0.0, pred)
            results.append(dict(scored_at=scored_at.isoformat(),
                as_of_time=origin.isoformat(), forecast_time=forecast_time.isoformat(),
                horizon_hours=int(horizon), region=target, model=name,
                forecast_pm25=round(pred, 1), elevated_flag=bool(pred >= threshold),
                feature_age_hours=round(age_hours, 2)))
    frame = pd.DataFrame(results)
    if len(frame) != 18 or frame.duplicated(["region", "horizon_hours"]).any():
        raise RuntimeError("Expected exactly 18 distinct target/horizon forecasts")
    return frame
