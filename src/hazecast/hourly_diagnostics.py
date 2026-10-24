"""Reconstruct honest final-holdout predictions and plot actual vs predicted.

Uses selected model names and split boundaries from the hourly manifest.
Refits fresh estimators on PRE-holdout rows only; production model artifacts
were refit on all labels and must not be used to assess historical accuracy.
Keep the same gold data, model parameters and package versions as training
for a faithful reconstruction of the original final-holdout assessment.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from hazecast.config import MODELS_DIR, REPO_ROOT, MIN_TRAIN_ROWS
from hazecast.hourly_model import (
    TARGETS, _prepare, _feature_columns, _labelled, _complete, _before,
    _fit, _predict, _metrics,
)


def _timestamp(value, timezone):
    stamp = pd.Timestamp(value)
    return stamp.tz_localize(timezone) if stamp.tzinfo is None else stamp.tz_convert(timezone)


def backtest_selected_models(
    hourly_gold, *, targets=None, horizons=None,
    models_dir=MODELS_DIR / "hourly", output_path=None,
):
    """Rebuild selected-model final-test predictions without rerunning CV.

    Each target/horizon is fitted once using exactly its saved pre-test
    boundary and current estimator parameters. No production artifacts are
    loaded or overwritten. Predictions are unrounded and nonnegative.
    """
    manifest = json.loads((Path(models_dir) / "model_manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("pipeline") != "hourly":
        raise ValueError("Expected a version-1 hourly selection manifest")
    gold = _prepare(hourly_gold)
    cols = manifest["feature_columns"]
    if _feature_columns(gold) != cols:
        raise ValueError("Gold feature schema/order differs from the hourly training manifest")
    targets = list(TARGETS if targets is None else targets)
    horizons = list(manifest["horizons_hours"] if horizons is None else horizons)
    if not targets or not horizons or len(set(targets)) != len(targets) or len(set(horizons)) != len(horizons):
        raise ValueError("Targets and horizons must be nonempty and unique")
    if set(targets) - set(TARGETS) or set(horizons) - set(manifest["horizons_hours"]):
        raise ValueError("Unknown target or untrained horizon")
    threshold = float(manifest["threshold"])
    frames = []
    for horizon in horizons:
        labelled = _labelled(gold, horizon)
        for target in targets:
            entry = manifest["models"][str(horizon)][target]
            boundary = _timestamp(entry["test_start"], str(gold.date.dt.tz))
            rows = _complete(labelled, cols, target, horizon)
            train = _before(rows, boundary, horizon)
            test = rows.loc[rows.date >= boundary]
            if len(train) < MIN_TRAIN_ROWS or test.empty:
                raise ValueError(f"Insufficient pre-test or final-test rows: {target}, h={horizon}")
            estimator = _fit(entry["model"], train, cols, target)
            predicted = _predict(estimator, test, cols, target)
            if not np.isfinite(predicted).all():
                raise ValueError("Nonfinite backtest predictions")
            label_end = train.date.max() + pd.Timedelta(hours=horizon)
            if label_end >= boundary:
                raise AssertionError("Training label crosses test boundary")
            frames.append(pd.DataFrame(dict(
                as_of_time=test.date.to_numpy(),
                forecast_time=(test.date + pd.Timedelta(hours=horizon)).to_numpy(),
                region=target, horizon_hours=int(horizon), model=entry["model"],
                actual_pm25=test[f"label_{target}"].to_numpy(dtype=float),
                predicted_pm25=predicted, threshold=threshold,
                split="final_test", test_start=boundary.isoformat(),
                train_label_end=label_end.isoformat(), n_train=len(train),
            )))
    predictions = pd.concat(frames, ignore_index=True).sort_values(
        ["region", "horizon_hours", "forecast_time"]
    ).reset_index(drop=True)
    if output_path is None:
        output_path = Path(models_dir) / "holdout_predictions.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return predictions


def plot_actual_vs_predicted(
    predictions, *, target="central", horizon=6,
    start=None, end=None, output_dir=REPO_ROOT / "reports",
):
    """Plot forecast-time-aligned held-out data; start inclusive, end exclusive.

    Pass unrounded predictions returned by backtest_selected_models. Naive
    start/end values use the timezone of the prediction timestamps.
    Returns (figure, metrics, PNG path); caller may display/close the figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    data = predictions.loc[
        (predictions["region"] == target) & (predictions["horizon_hours"] == horizon)
    ].copy()
    if data.empty:
        raise ValueError(f"No predictions for {target}, h={horizon}")
    data["forecast_time"] = pd.to_datetime(data["forecast_time"], utc=True).dt.tz_convert("Asia/Singapore")
    if not data["split"].eq("final_test").all():
        raise ValueError("Expected final-test predictions only")
    if data.forecast_time.duplicated().any():
        raise ValueError("Duplicate forecast timestamps")
    timezone = str(data.forecast_time.dt.tz)
    if start is not None:
        data = data.loc[data.forecast_time >= _timestamp(start, timezone)]
    if end is not None:
        data = data.loc[data.forecast_time < _timestamp(end, timezone)]
    data = data.sort_values("forecast_time")
    if data.empty:
        raise ValueError("No held-out points in this window. Choose dates within the final holdout; do not use production-fit predictions.")
    threshold_values = data.threshold.unique()
    if len(threshold_values) != 1:
        raise ValueError("Mixed alert thresholds")
    threshold = float(threshold_values[0])
    metrics = _metrics(data.actual_pm25.to_numpy(), data.predicted_pm25.to_numpy(), threshold)
    elevated = data.actual_pm25 >= threshold
    metrics["mae_elevated"] = (float(np.abs(data.loc[elevated, "actual_pm25"] - data.loc[elevated, "predicted_pm25"]).mean())
                               if elevated.any() else float("nan"))
    # Insert missing hours as NaNs so gaps are not drawn as continuous data.
    timeline = pd.date_range(data.forecast_time.min(), data.forecast_time.max(), freq="h")
    display_data = data.set_index("forecast_time").reindex(timeline)
    model_names = ", ".join(sorted(data.model.unique()))
    fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
    ax.plot(display_data.index, display_data.actual_pm25, color="#2679d9", linewidth=1.5,
            label=f"Actual PM2.5 ({target})")
    ax.plot(display_data.index, display_data.predicted_pm25, color="#f47735", linewidth=1.5,
            label=f"Predicted PM2.5 (+{horizon}h, {model_names})")
    ax.axhline(threshold, color="#e25759", linestyle="--", linewidth=1.3,
               label=f"Elevated threshold ({threshold:g} µg/m³)")
    # Shade contiguous elevated intervals, including the final elevated hour.
    mask = display_data.actual_pm25.ge(threshold).fillna(False).to_numpy()
    edges = np.diff(np.r_[False, mask, False].astype(int))
    for first, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        ax.axvspan(timeline[first], timeline[stop - 1] + pd.Timedelta(hours=1),
                   color="#e25759", alpha=.10, linewidth=0)
    f2 = f"{metrics['f2']:.3f}" if np.isfinite(metrics['f2']) else "N/A (no elevated actuals)"
    mae_elevated = f"{metrics['mae_elevated']:.2f}" if np.isfinite(metrics['mae_elevated']) else "N/A"
    text = (f"MAE: {metrics['mae']:.2f} µg/m³   RMSE: {metrics['rmse']:.2f} µg/m³\n"
            f"F2: {f2}   MAE elevated: {mae_elevated}\n"
            f"n={metrics['n_test']:,}; elevated={metrics['n_elevated']:,}")
    ax.text(1, 1.02, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=.4", facecolor="white", edgecolor="#c9c9c9"))
    ax.set_title(f"{target.title()} — actual vs. predicted PM2.5, +{horizon} hours\n"
                 "Final holdout · model refitted using pre-test data only", loc="left", fontsize=12, pad=52)
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_xlabel("Forecast time (Asia/Singapore)")
    locator = mdates.AutoDateLocator(tz=timeline.tz)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=timeline.tz))
    ax.grid(axis="y", alpha=.25)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(bottom=0)
    label_end = pd.to_datetime(data.train_label_end, utc=True).max().tz_convert("Asia/Singapore")
    ax.text(0, -.18, f"Latest training label: {label_end.isoformat()} | Shading: elevated actual hours. "
            "Metrics cover displayed, matched points only.", transform=ax.transAxes, fontsize=8, color="#666666")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{data.forecast_time.min():%Y%m%d}_{data.forecast_time.max():%Y%m%d}"
    path = output_dir / f"{target}_h{horizon}_actual_vs_predicted_{tag}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    return fig, metrics, path
