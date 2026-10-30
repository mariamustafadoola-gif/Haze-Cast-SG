"""Production inference: use final hourly winners; archive 18 forecasts.

Run after hourly gold refresh: python -m hazecast.inference
No training occurs. Forecasts issued after their target time are rejected.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from hazecast.config import GOLD_DIR, MODELS_DIR, DATA_DIR
from hazecast.pipeline_store import (
    TARGETS, HORIZONS, connect, timestamp, iso, finite, atomic_json,
)

log = logging.getLogger(__name__)


def predict_final(gold_path, models_dir, *, max_feature_age_hours=2, now=None):
    """Predict with full precision and freeze model provenance for all 18 rows."""
    import joblib
    import numpy as np
    import pandas as pd

    if not math.isfinite(max_feature_age_hours) or max_feature_age_hours < 0:
        raise ValueError("max_feature_age_hours must be finite and nonnegative")
    now = timestamp(now or datetime.now(timezone.utc))
    models_dir = Path(models_dir)
    raw = (models_dir / "model_manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema_version") != 1 or manifest.get("pipeline") != "hourly":
        raise ValueError("Expected hourly model_manifest.json from final hourly training")
    if sorted(manifest.get("horizons_hours", [])) != list(HORIZONS):
        raise ValueError("Train hourly horizons [6,12,24] before inference")
    from hazecast import provenance
    built_differently = provenance.differences(manifest)
    if built_differently:
        # Not a refusal: the champion keeps serving, and the monitor asks for a gated retrain.
        log.warning("Champion was built with different code or libraries (%s); a retrain is due",
                    "; ".join(built_differently))
    cols = manifest["feature_columns"]
    if not cols or len(set(cols)) != len(cols) or any(c == "date" or c.startswith(("label_", "target_")) for c in cols):
        raise ValueError("Invalid saved feature columns")
    threshold = float(manifest["threshold"])
    if not finite(threshold):
        raise ValueError("Invalid saved threshold")
    gold = pd.read_parquet(gold_path).copy()
    dates = pd.to_datetime(gold["date"], errors="raise")
    gold["date"] = dates.dt.tz_localize("Asia/Singapore") if dates.dt.tz is None else dates.dt.tz_convert("Asia/Singapore")
    if gold.empty or gold.date.isna().any() or gold.date.duplicated().any():
        raise ValueError("Hourly gold requires unique, non-null timestamps")
    if not gold.date.eq(gold.date.dt.floor("h")).all():
        raise ValueError("Hourly timestamps must be on hour boundaries")
    required = list(dict.fromkeys(cols + [f"now_{t}" for t in TARGETS]))
    missing = set(required) - set(gold.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    gold = gold.sort_values("date")
    complete = np.isfinite(gold[required].to_numpy(dtype=float)).all(axis=1)
    complete &= (gold[required].filter(regex=r"^now_").ge(0).all(axis=1)).to_numpy()
    complete &= (gold.date <= pd.Timestamp(now)).to_numpy()
    if not complete.any():
        raise RuntimeError("No complete, available hourly feature row; refresh upstream data")
    row = gold.loc[complete].iloc[[-1]]
    origin = timestamp(row.date.iloc[0].to_pydatetime())
    age = (now - origin).total_seconds() / 3600
    if age > max_feature_age_hours:
        raise RuntimeError(f"Features are {age:.2f}h old (limit {max_feature_age_hours}h). Refresh silver/gold; no forecasts archived.")
    # > 0 when FIRMS was late and the fire features were carried forward (silver_to_gold)
    fire_carried = float(row["fire_carried_hours"].iloc[0]) if "fire_carried_hours" in row else 0.0
    digest = hashlib.sha256(raw).hexdigest()
    records = []
    for horizon in HORIZONS:
        entries = manifest["models"].get(str(horizon), {})
        if set(entries) != set(TARGETS):
            raise ValueError(f"Incomplete model manifest at h={horizon}")
        forecast_time = origin + timedelta(hours=horizon)
        if forecast_time <= now:
            raise RuntimeError("Target time has already passed; predictions cannot be archived as live forecasts")
        for target in TARGETS:
            entry = entries[target]
            # Prevent using a model trained on observations unavailable at issue time.
            if "training_label_end" not in entry:
                raise ValueError("Hourly model provenance missing training_label_end; retrain with the hourly pipeline")
            if timestamp(entry["training_label_end"]) > now:
                raise ValueError("Model training labels extend past inference issue time")
            if entry["model"] == "persistence":
                prediction = float(row[f"now_{target}"].iloc[0])
            else:
                artifact = entry.get("artifact")
                if not artifact or Path(artifact).name != artifact:
                    raise ValueError("Invalid model artifact path")
                model = joblib.load(models_dir / artifact)
                prediction = float(model.predict(row[cols])[0])
            if not math.isfinite(prediction):
                raise ValueError("Nonfinite prediction")
            prediction = max(0.0, prediction)
            key = f"{iso(origin)}|{target}|{horizon}"
            records.append(dict(forecast_id=hashlib.sha256(key.encode()).hexdigest(),
                issued_at=iso(now), as_of_time=iso(origin), forecast_time=iso(forecast_time),
                region=target, horizon_hours=horizon, model=entry["model"],
                model_run_id=entry.get("run_id"), manifest_sha256=digest,
                forecast_pm25=prediction, threshold=threshold,
                elevated_flag=int(prediction >= threshold), feature_age_hours=age,
                persistence_pm25=float(row[f"now_{target}"].iloc[0]),
                fire_carried_hours=fire_carried))
    return records


def archive_forecasts(records, db_path, *, latest_path=None):
    """First-issued forecast per origin/target/horizon wins on retries."""
    keys = {(r["region"], r["horizon_hours"]) for r in records}
    if len(records) != 18 or keys != {(t,h) for t in TARGETS for h in HORIZONS}:
        raise ValueError("Expected 18 complete target/horizon records")
    if len({r['as_of_time'] for r in records}) != 1:
        raise ValueError("All 18 records must share a feature origin")
    for r in records:
        if not finite(r["forecast_pm25"]) or not finite(r["persistence_pm25"]):
            raise ValueError("Invalid forecast concentration")
        if timestamp(r["issued_at"]) >= timestamp(r["forecast_time"]):
            raise ValueError("Forecast must be issued before its target time")
        if timestamp(r["forecast_time"]) - timestamp(r["as_of_time"]) != timedelta(hours=r["horizon_hours"]):
            raise ValueError("Forecast horizon/time mismatch")
    db = connect(db_path)
    try:
        with db:
            for r in records:
                cols = list(r)
                db.execute(f"INSERT INTO forecasts ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)}) "
                           "ON CONFLICT(as_of_time,region,horizon_hours) DO NOTHING", [r[c] for c in cols])
            saved = [dict(row) for row in db.execute(
                "SELECT * FROM forecasts WHERE as_of_time=? ORDER BY horizon_hours,region",
                (records[0]["as_of_time"],))]
        if latest_path is not None:
            # Use archived values on a retry, never silently replace prior predictions.
            atomic_json(latest_path, [{**r, "scored_at": r["issued_at"]} for r in saved])
        return saved
    finally:
        db.close()


def run_inference(gold_path=GOLD_DIR / "hourly_features.parquet", models_dir=MODELS_DIR / "hourly",
                  db_path=DATA_DIR / "monitoring" / "pipeline.sqlite",
                  latest_path=GOLD_DIR / "latest_hourly_forecast.json", max_feature_age_hours=2,
                  refresh_gold=False):
    if refresh_gold:
        import pandas as pd
        from hazecast.config import SILVER_DIR
        from hazecast.silver_to_gold import build_gold_hourly
        gold = build_gold_hourly(
            pd.read_parquet(SILVER_DIR / "pm25.parquet"),
            pd.read_parquet(SILVER_DIR / "weather.parquet"),
            pd.read_parquet(SILVER_DIR / "firms_daily.parquet"),
            pd.read_parquet(SILVER_DIR / "firms_hourly.parquet"),
        )
        gold_path = Path(gold_path)
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep inference downstream of the completed refresh operation.
        gold.to_parquet(gold_path, index=False)
    records = predict_final(gold_path, models_dir, max_feature_age_hours=max_feature_age_hours)
    saved = archive_forecasts(records, db_path, latest_path=latest_path)
    log.info("Archived %d predictions for origin %s", len(saved), saved[0]["as_of_time"])
    _score_shadow(gold_path, Path(db_path).parent / "shadow.sqlite", max_feature_age_hours)
    return saved


def _score_shadow(gold_path, shadow_db, max_feature_age_hours, shadow_dir=MODELS_DIR / "hourly_shadow"):
    """During a shadow period (hazecast.governance shadow), score the challenger into its own ledger.
    Never served, and a failure here never affects the champion's forecasts."""
    if not (Path(shadow_dir) / "model_manifest.json").exists():
        return
    try:
        shadow = archive_forecasts(predict_final(gold_path, shadow_dir, max_feature_age_hours=max_feature_age_hours),
                                   shadow_db)
        log.info("Shadow: archived %d predictions", len(shadow))
    except Exception as exc:  # noqa: BLE001
        log.warning("Shadow scoring failed (champion unaffected): %s", exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-path", type=Path, default=GOLD_DIR / "hourly_features.parquet")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR / "hourly")
    parser.add_argument("--db-path", type=Path, default=DATA_DIR / "monitoring" / "pipeline.sqlite")
    parser.add_argument("--latest-path", type=Path, default=GOLD_DIR / "latest_hourly_forecast.json")
    parser.add_argument("--max-feature-age-hours", type=float, default=2)
    parser.add_argument("--refresh-gold", action="store_true", help="Rebuild hourly gold from existing refreshed silver tables")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_inference(**vars(args))


if __name__ == "__main__":
    main()
