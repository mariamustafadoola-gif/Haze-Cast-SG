"""Replay the final holdout period through the model monitor.

Live monitoring only has history from the day inference starts archiving
forecasts. To see how the selected models perform and how stable they are
across a long period (including a haze season), this module re-creates
honest forecasts for the whole final holdout (~20% of history, about 2.5
years) and runs every monitor on a rolling window across it.

    python -m hazecast.monitor_backtest                 # weekly windows
    python -m hazecast.monitor_backtest --step-days 1   # daily windows (slower)

Honesty rules (same as hourly_diagnostics.backtest_selected_models):
  - each selected model is refitted on rows whose label precedes the holdout
    start; production artifacts (refitted on everything) are never used here;
  - the data-quality and attribution baselines come from that pre-holdout
    training data; the model-quality baseline is each model's pooled
    forward-validation score from the training manifest (also pre-holdout);
  - the bias baseline needs per-region forecasts, which only exist from the
    holdout on, so the first BURN_IN_DAYS of the holdout serve as its
    baseline and are not themselves monitored.

Writes:
  data/monitoring/backtest.sqlite                 forecasts, actuals, monitor metrics per window
  data/models/hourly/holdout_predictions.csv      same format as hourly_diagnostics (feeds the
                                                  production baseline's quality/bias sections)
  reports/monitoring/backtest_dashboard.html      via hazecast.dashboard
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from hazecast import model_monitor as mm
from hazecast.config import GOLD_DIR, MODELS_DIR, MONITORING_DIR, MIN_TRAIN_ROWS, MONITOR_WINDOW_DAYS
from hazecast.pipeline_store import TARGETS, connect, iso

log = logging.getLogger(__name__)
BURN_IN_DAYS = 90


def refit_pre_holdout(gold: pd.DataFrame, manifest: dict):
    """Refit every selected model on pre-holdout rows. Returns (models, holdout predictions, reference rows)."""
    from hazecast.hourly_model import _labelled, _complete, _before, _fit, _predict
    cols = manifest["feature_columns"]
    models, frames, boundaries = {}, [], []
    for horizon in manifest["horizons_hours"]:
        labelled = _labelled(gold, horizon)
        for target in TARGETS:
            entry = manifest["models"][str(horizon)][target]
            boundary = pd.Timestamp(entry["test_start"])
            boundary = boundary.tz_localize(gold.date.dt.tz) if boundary.tzinfo is None else boundary.tz_convert(gold.date.dt.tz)
            rows = _complete(labelled, cols, target, horizon)
            train, test = _before(rows, boundary, horizon), rows.loc[rows.date >= boundary]
            if len(train) < MIN_TRAIN_ROWS or test.empty:
                raise ValueError(f"Insufficient pre-test or final-test rows: {target}, h={horizon}")
            est = _fit(entry["model"], train, cols, target)
            models[(target, int(horizon))] = (entry["model"], est)
            frames.append(pd.DataFrame(dict(
                as_of_time=test.date.to_numpy(), forecast_time=(test.date + pd.Timedelta(hours=horizon)).to_numpy(),
                region=target, horizon_hours=int(horizon), model=entry["model"],
                actual_pm25=test[f"label_{target}"].to_numpy(dtype=float),
                predicted_pm25=_predict(est, test, cols, target),
                persistence_pm25=test[f"now_{target}"].to_numpy(dtype=float),
                threshold=float(manifest["threshold"]), split="final_test", test_start=boundary.isoformat(),
                train_label_end=(train.date.max() + pd.Timedelta(hours=horizon)).isoformat(), n_train=len(train))))
            boundaries.append(boundary)
            log.info("refit %s h=%d %s on %d rows; %d holdout forecasts", target, horizon, entry["model"],
                     len(train), len(test))
    holdout = pd.concat(frames, ignore_index=True)
    reference = gold.loc[gold.date < min(boundaries)]
    return models, holdout, reference


def write_ledger(db_path: Path, holdout: pd.DataFrame, gold: pd.DataFrame, manifest_sha: str):
    db_path = Path(db_path)
    for suffix in ("", "-wal", "-shm"):
        Path(str(db_path) + suffix).unlink(missing_ok=True)
    db = connect(db_path)
    forecasts = [(hashlib.sha256(f"{iso(o)}|{r}|{h}".encode()).hexdigest(), iso(o), iso(o), iso(f), r, int(h), m,
                  None, manifest_sha, float(p), float(thr), int(p >= thr), 0.0, float(pers))
                 for o, f, r, h, m, p, thr, pers in holdout[["as_of_time", "forecast_time", "region", "horizon_hours",
                                                             "model", "predicted_pm25", "threshold",
                                                             "persistence_pm25"]].itertuples(index=False)
                 if np.isfinite(pers)]
    actual_rows = []
    for target in TARGETS:
        vals = gold[["date", f"now_{target}"]].dropna()
        actual_rows += [(iso(d), target, float(v), "gold_now_" + target, iso(d), iso(d)) for d, v in vals.itertuples(index=False)]
    with db:
        # Columns named explicitly: the live table has gained columns since (fire_carried_hours),
        # and a positional insert broke the replay when it did.
        db.executemany("INSERT OR IGNORE INTO forecasts (forecast_id, issued_at, as_of_time, forecast_time, region, "
                       "horizon_hours, model, model_run_id, manifest_sha256, forecast_pm25, threshold, elevated_flag, "
                       "feature_age_hours, persistence_pm25) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", forecasts)
        db.executemany("INSERT OR REPLACE INTO actuals VALUES (?,?,?,?,?,?)", actual_rows)
    log.info("backtest ledger: %d forecasts, %d actuals -> %s", len(forecasts), len(actual_rows), db_path)
    return db


def run_backtest(*, step_days: int = 7, window_days: int = MONITOR_WINDOW_DAYS,
                 gold_path: Path = GOLD_DIR / "hourly_features.parquet",
                 models_dir: Path = MODELS_DIR / "hourly",
                 db_path: Path = MONITORING_DIR / "backtest.sqlite",
                 dashboard_path: Path | None = None) -> dict:
    models_dir = Path(models_dir)
    raw = (models_dir / "model_manifest.json").read_bytes()
    manifest = json.loads(raw)
    cols = manifest["feature_columns"]
    gold = mm.prepare_gold(pd.read_parquet(gold_path))
    models, holdout, reference = refit_pre_holdout(gold, manifest)
    holdout.drop(columns="persistence_pm25").to_csv(models_dir / "holdout_predictions.csv", index=False)

    forecasts = holdout.rename(columns={"predicted_pm25": "forecast_pm25"})
    burn_in_end = pd.to_datetime(forecasts.forecast_time).min() + pd.Timedelta(days=BURN_IN_DAYS)
    baseline = mm.build_baseline(reference, models, cols,
                                 holdout=forecasts.loc[pd.to_datetime(forecasts.forecast_time) < burn_in_end])
    # Model-quality baseline: pre-holdout pooled validation scores, not the holdout itself.
    baseline["quality"] = {f"{t}|{h}": manifest["models"][str(h)][t]["validation"]
                           for (t, h) in models}
    db = write_ledger(db_path, holdout, gold, hashlib.sha256(raw).hexdigest())
    try:
        mm._ensure_tables(db)
        last = pd.to_datetime(forecasts.forecast_time).max()
        ends = pd.date_range(burn_in_end.ceil("D") + pd.Timedelta(days=window_days), last.floor("D"), freq=f"{step_days}D")
        actions = {}
        for end in ends:
            results = mm.evaluate_window(gold, db, models, cols, baseline, end - pd.Timedelta(days=window_days), end)
            decision = mm.governance_decision(results)  # model age is not meaningful in a replay
            mm.save_results(db, end, results, decision)
            actions[str(end.date())] = decision["action"]
            log.info("window ending %s: %s %s", end.date(), decision["action"],
                     {m: results[m]["status"] for m in mm.MONITORS})
    finally:
        db.close()
    from hazecast.dashboard import build_dashboard
    out = build_dashboard(db_path, dashboard_path or (mm.REPORTS_DIR / "backtest_dashboard.html"),
                          title="HazeCast SG: model monitor replay over the final holdout",
                          subtitle=(f"Selected models refitted on data before {holdout.test_start.min()[:10]}; "
                                    f"monitors run on {window_days}-day windows every {step_days} days. "
                                    f"First {BURN_IN_DAYS} days of the holdout are the bias baseline."))
    counts = pd.Series(actions).value_counts().to_dict()
    log.info("Replay done: %d windows %s -> %s", len(actions), counts, out)
    return dict(windows=len(actions), actions=counts, dashboard=str(out))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--step-days", type=int, default=7)
    ap.add_argument("--window-days", type=int, default=MONITOR_WINDOW_DAYS)
    a = ap.parse_args()
    print(json.dumps(run_backtest(step_days=a.step_days, window_days=a.window_days), indent=2))
