"""Model monitor for the deployed hourly models: data quality, stability, model
quality, bias drift and feature attribution drift, plus the retraining decision.

Follows the SageMaker Model Monitor pattern: a *baseline* is captured from
the training data when a model is published, and each monitoring run
compares a recent window against it.

    python -m hazecast.model_monitor baseline   # after training (the retrain DAG runs this)
    python -m hazecast.model_monitor run        # daily (the monitoring DAG runs this)

Monitors (thresholds live in hazecast.config, the policy in docs/MODEL_GOVERNANCE.md):
  data_quality         schema, freshness, missing rate, out-of-range share, stuck PM2.5
                       feeds, the age of each upstream source (live runs), and the CSI
                       (characteristic stability index) of every model feature against
                       the same monsoon season in the training years
  stability            PSI (population stability index) of each model's forecasts
                       against its forecasts on the same season in the training years
  model_quality        archived forecasts vs NEA actuals (the forecast ledger
                       that hazecast.inference writes and hazecast.monitoring
                       fills with actuals): MAE vs holdout MAE, skill vs
                       persistence, F2 drop
  bias_drift           per-region recall gap, mean signed error and alert-rate
                       bias vs training; day vs night split reported
  attribution_drift    label-free permutation attribution of each model on the
                       window vs baseline, compared with NDCG@k (as Clarify does)

Every metric is also written to the monitor_metrics table of the ledger,
which is what hazecast.dashboard plots over time.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from hazecast import config as C
from hazecast.config import GOLD_DIR, MODELS_DIR, MONITORING_DIR, REPO_ROOT, SG_REGIONS, SILVER_DIR, WEATHER_POINTS
from hazecast.pipeline_store import SGT, TARGETS, connect, iso, timestamp, atomic_json

log = logging.getLogger(__name__)
BASELINE_FILE = "monitoring_baseline.json"
REPORTS_DIR = REPO_ROOT / "reports" / "monitoring"
MONITORS = ("data_quality", "stability", "model_quality", "bias_drift", "attribution_drift")
SEASON_OF_MONTH = {m: season for season, months in C.MONSOON_SEASONS.items() for m in months}


def season_of(dates) -> np.ndarray:
    """Monsoon season (config.MONSOON_SEASONS) of each timestamp."""
    return pd.to_datetime(pd.Series(dates)).dt.month.map(SEASON_OF_MONTH).to_numpy()


def window_season(frame: pd.DataFrame) -> str | None:
    """A window's season is that of its middle hour (the null windows use the same rule)."""
    return None if frame.empty else season_of(frame.date.iloc[[len(frame) // 2]])[0]


# ---------------------------------------------------------------------------
# Models and data
# ---------------------------------------------------------------------------
def load_production_models(models_dir: Path = MODELS_DIR / "hourly"):
    """(target, horizon) -> (model name, fitted estimator or None for persistence), plus the manifest."""
    import joblib
    models_dir = Path(models_dir)
    manifest = json.loads((models_dir / "model_manifest.json").read_text())
    models = {}
    for horizon, entries in manifest["models"].items():
        for target, entry in entries.items():
            est = None if entry["model"] == "persistence" else joblib.load(models_dir / entry["artifact"])
            models[(target, int(horizon))] = (entry["model"], est)
    return models, manifest["feature_columns"], manifest


def prepare_gold(gold: pd.DataFrame) -> pd.DataFrame:
    from hazecast.hourly_model import _prepare
    return _prepare(gold)


def _window(gold: pd.DataFrame, start, end) -> pd.DataFrame:
    return gold.loc[(gold.date >= start) & (gold.date < end)]


def _complete_rows(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return frame.loc[np.isfinite(frame[cols].to_numpy(dtype=float)).all(axis=1)]


def predict(est, X: pd.DataFrame, target: str) -> np.ndarray:
    pred = X[f"now_{target}"].to_numpy(dtype=float) if est is None else est.predict(X)
    return np.maximum(0.0, np.asarray(pred, dtype=float))


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------
def psi(expected: list[float], actual: list[float], eps: float = 1e-4) -> float:
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _bin_props(values: np.ndarray, inner_edges: list[float]) -> list[float]:
    idx = np.searchsorted(np.asarray(inner_edges, dtype=float), values, side="right")
    counts = np.bincount(idx, minlength=len(inner_edges) + 1)
    return (counts / max(counts.sum(), 1)).tolist()


def attribution(est, X: pd.DataFrame, target: str, cols: list[str], seed: int = 42) -> dict[str, float]:
    """Label-free permutation attribution: mean |change in prediction| when one feature is shuffled,
    normalised to sum to 1. Works for every candidate model type (and persistence)."""
    if est is None:
        return {c: float(c == f"now_{target}") for c in cols}
    rng = np.random.default_rng(seed)
    base = predict(est, X, target)
    scores = {}
    for c in cols:
        shuffled = X.copy()
        shuffled[c] = rng.permutation(shuffled[c].to_numpy())
        scores[c] = float(np.mean(np.abs(predict(est, shuffled, target) - base)))
    total = sum(scores.values())
    return {c: (v / total if total > 0 else 0.0) for c, v in scores.items()}


def ndcg(baseline: dict[str, float], current: dict[str, float], k: int = C.ATTRIBUTION_TOP_K) -> float:
    """NDCG@k of the current feature ranking, using baseline attributions as relevance."""
    top_now = sorted(current, key=current.get, reverse=True)[:k]
    ideal = sorted(baseline.values(), reverse=True)[:k]
    dcg = sum(baseline.get(f, 0.0) / math.log2(i + 2) for i, f in enumerate(top_now))
    idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 1.0


def quality(frame: pd.DataFrame) -> dict:
    """Error and alert metrics for rows with forecast_pm25, actual_pm25, threshold (+ persistence_pm25)."""
    if frame.empty:
        return dict(n=0)
    f, a, thr = (frame[c].to_numpy(dtype=float) for c in ("forecast_pm25", "actual_pm25", "threshold"))
    err = f - a
    actual, flagged = a >= thr, f >= thr
    tp, fp, fn = int((actual & flagged).sum()), int((~actual & flagged).sum()), int((actual & ~flagged).sum())
    out = dict(n=len(frame), mae=float(np.abs(err).mean()), rmse=float(np.sqrt((err ** 2).mean())),
               mean_error=float(err.mean()), n_elevated=int(actual.sum()), tp=tp, fp=fp, fn=fn,
               f2=5 * tp / (5 * tp + 4 * fn + fp) if actual.any() else None,
               recall=tp / actual.sum() if actual.any() else None,
               flag_rate=float(flagged.mean()), actual_rate=float(actual.mean()),
               coverage=float((a <= f).mean()), mean_actual=float(a.mean()))
    if "persistence_pm25" in frame:
        p_mae = float(np.abs(frame["persistence_pm25"].to_numpy(dtype=float) - a).mean())
        p_flag = frame["persistence_pm25"].to_numpy(dtype=float) >= thr
        out.update(persistence_mae=p_mae, skill=1 - out["mae"] / p_mae if p_mae > 0 else None,
                   persistence_recall=float((actual & p_flag).sum() / actual.sum()) if actual.any() else None)
    return out


def _bias_by_region(frame: pd.DataFrame) -> dict:
    """Per horizon: per-region quality + the spread across regions (the bias facets)."""
    out = {}
    for horizon, g in frame.loc[frame.region.isin(SG_REGIONS)].groupby("horizon_hours"):
        regions = {r: quality(gr) for r, gr in g.groupby("region")}
        supported = [q["recall"] for q in regions.values()
                     if q.get("recall") is not None and q["n_elevated"] >= C.MIN_ELEVATED_FOR_RECALL]
        hours = pd.to_datetime(g["forecast_time"], utc=True).dt.tz_convert(SGT).dt.hour
        out[str(int(horizon))] = dict(
            regions=regions,
            recall_gap=(max(supported) - min(supported)) if len(supported) >= 2 else None,
            day=quality(g.loc[hours.between(7, 18)]), night=quality(g.loc[~hours.between(7, 18)]),
        )
    return out


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
def _windows(reference: pd.DataFrame, n: int, seed: int) -> list[pd.DataFrame]:
    """n random DRIFT_WINDOW_DAYS-long slices of the reference period (the "normal weeks" sample)."""
    length = C.DRIFT_WINDOW_DAYS * 24
    if len(reference) <= length:
        return []
    rng = np.random.default_rng(seed)
    starts = rng.choice(len(reference) - length, size=min(n, len(reference) - length), replace=False)
    return [reference.iloc[s:s + length] for s in sorted(starts)]


def drift_null(reference: pd.DataFrame, models: dict, cols: list[str], features: dict,
               attributions: dict, *, seed: int = 42) -> dict:
    """Empirical calibration of the drift thresholds.

    Fixed textbook bands (PSI 0.25, NDCG 0.90) assume independent samples. These
    features are autocorrelated (fire counts repeat for 24 h and move over days), so
    in the holdout replay an ordinary month scored PSI 5-8 on fire features and NDCG
    0.86 at +24 h. Instead, score DRIFT_NULL_WINDOWS random windows *from the training
    period itself* against the baseline and keep their upper percentiles: an alert
    then means "unlike the normal months the model was trained on", whatever the
    feature's natural variability. The textbook values remain as floors / ceilings.
    """
    windows = _windows(reference, C.DRIFT_NULL_WINDOWS, seed)
    psi_null = {c: [] for c in cols}
    ndcg_null = {k: [] for k in attributions}
    for i, w in enumerate(windows):
        for c in cols:
            v = w[c].to_numpy(dtype=float)
            finite = v[np.isfinite(v)]
            if len(finite):
                psi_null[c].append(psi(features[c]["bin_props"], _bin_props(finite, features[c]["bin_edges"])))
        rows = _complete_rows(w, cols)
        if i < C.ATTRIBUTION_NULL_WINDOWS and len(rows) >= 24:
            for (t, h), (_, est) in models.items():
                key = f"{t}|{h}"
                ndcg_null[key].append(ndcg(attributions[key], attribution(est, rows[cols], t, cols, seed + i)))
    q = lambda xs, p: float(np.quantile(xs, p)) if xs else None  # noqa: E731
    return dict(
        psi={c: dict(psi_p95=q(v, 0.95), psi_p99=q(v, 0.99),
                     psi_alert=max(C.PSI_ALERT, q(v, 0.99) or 0.0), psi_warn=max(C.PSI_WARN, q(v, 0.95) or 0.0))
             for c, v in psi_null.items()},
        ndcg={k: dict(p01=q(v, 0.01), p05=q(v, 0.05),
                      alert=min(C.ATTRIBUTION_NDCG_ALERT, q(v, 0.01) if v else C.ATTRIBUTION_NDCG_ALERT),
                      warn=min(C.ATTRIBUTION_NDCG_ALERT, q(v, 0.05) if v else C.ATTRIBUTION_NDCG_ALERT))
              for k, v in ndcg_null.items()},
        n_windows=len(windows))


def coverage_band(holdout: pd.DataFrame) -> dict:
    """Normal range of weekly calibration per horizon: the 1st-99th percentile of the share of hours
    at or below the forecast, pooled over the targets, in holdout weeks without haze (fewer than
    MIN_ELEVATED_FOR_RECALL elevated hours). Returns {} when there are too few quiet weeks."""
    t = pd.to_datetime(holdout["forecast_time"], utc=True).dt.tz_convert(SGT).dt.tz_localize(None)
    frame = holdout.assign(week=t.dt.to_period("W").dt.start_time,
                           ok=holdout["actual_pm25"] <= holdout["forecast_pm25"],
                           el=holdout["actual_pm25"] >= holdout["threshold"])
    elevated = frame.loc[frame.region == "national"].groupby(["week", "horizon_hours"]).el.sum()
    weekly = frame.groupby(["week", "horizon_hours"]).ok.mean()
    quiet = weekly[elevated.reindex(weekly.index, fill_value=0) < C.MIN_ELEVATED_FOR_RECALL]
    out = {}
    for h, v in quiet.groupby(level="horizon_hours"):
        if len(v) >= C.COVERAGE_MIN_QUIET_WEEKS:
            out[str(int(h))] = dict(low=float(v.quantile(0.01)), high=float(v.quantile(0.99)), n_weeks=len(v))
    return out


def _season_windows(reference: pd.DataFrame, n: int, seed: int) -> dict[str, list[pd.DataFrame]]:
    """Up to n random DRIFT_WINDOW_DAYS-long training-period windows per season (by middle hour)."""
    length = C.DRIFT_WINDOW_DAYS * 24
    if len(reference) <= length:
        return {}
    rng = np.random.default_rng(seed)
    starts = np.arange(0, len(reference) - length, 24)
    mid = season_of(reference.date.iloc[starts + length // 2])
    out = {}
    for season in C.MONSOON_SEASONS:
        candidates = starts[mid == season]
        if len(candidates):
            picked = rng.choice(candidates, size=min(n, len(candidates)), replace=False)
            out[season] = [reference.iloc[s:s + length] for s in sorted(picked)]
    return out


def _limits(values: list[float]) -> dict:
    """Alert / warn limits from a null distribution: 99th / 95th percentile, standard bands as floors."""
    if not values:
        return dict(alert=C.PSI_ALERT, warn=C.PSI_WARN)
    return dict(alert=max(C.PSI_ALERT, float(np.quantile(values, 0.99))),
                warn=max(C.PSI_WARN, float(np.quantile(values, 0.95))))


def _deciles(values: np.ndarray) -> list[float]:
    return np.unique(np.quantile(values, np.linspace(0, 1, 11)))[1:-1].tolist() if len(values) else []


def seasonal_stability_baseline(reference: pd.DataFrame, models: dict, cols: list[str], features: dict,
                                *, seed: int = 42) -> dict:
    """Per-season reference distributions and calibrated limits for the stability indices.

    CSI (per feature) and PSI (per model's forecasts) are measured against the same monsoon
    season in the training years, so a normal seasonal change is not read as instability.
    Limits are calibrated like drift_null: score SEASON_NULL_WINDOWS training-period windows
    of each season against that season's reference and keep the 99th / 95th percentiles.
    Adds season_props / season_limits to each entry of `features` and returns the forecast
    ("predictions") baseline.
    """
    seasons = season_of(reference.date)
    for c in cols:
        v = reference[c].to_numpy(dtype=float)
        features[c]["season_props"] = {s: _bin_props(v[(seasons == s) & np.isfinite(v)], features[c]["bin_edges"])
                                       for s in C.MONSOON_SEASONS if ((seasons == s) & np.isfinite(v)).any()}
    complete = _complete_rows(reference, cols)
    c_seasons = season_of(complete.date)
    predictions = {}
    for (t, h), (_, est) in models.items():
        p = predict(est, complete[cols], t)
        edges = _deciles(p)
        predictions[f"{t}|{h}"] = dict(bin_edges=edges, bin_props=_bin_props(p, edges), mean=float(p.mean()),
                                       season_props={s: _bin_props(p[c_seasons == s], edges)
                                                     for s in C.MONSOON_SEASONS if (c_seasons == s).any()})
    csi_null = {c: {} for c in cols}
    psi_null = {k: {} for k in predictions}
    for season, windows in _season_windows(reference, C.SEASON_NULL_WINDOWS, seed).items():
        for w in windows:
            for c in cols:
                v = w[c].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                if len(v) and season in features[c]["season_props"]:
                    csi_null[c].setdefault(season, []).append(
                        psi(features[c]["season_props"][season], _bin_props(v, features[c]["bin_edges"])))
            rows = _complete_rows(w, cols)
            if len(rows) < 24:
                continue
            for (t, h), (_, est) in models.items():
                b = predictions[f"{t}|{h}"]
                if season in b["season_props"]:
                    psi_null[f"{t}|{h}"].setdefault(season, []).append(
                        psi(b["season_props"][season], _bin_props(predict(est, rows[cols], t), b["bin_edges"])))
    for c in cols:
        features[c]["season_limits"] = {s: _limits(v) for s, v in csi_null[c].items()}
    for k, b in predictions.items():
        b["season_limits"] = {s: _limits(v) for s, v in psi_null[k].items()}
    return predictions


def build_baseline(reference: pd.DataFrame, models: dict, cols: list[str], *,
                   holdout: pd.DataFrame | None = None, seed: int = 42) -> dict:
    """Capture what "normal" looked like at training time.

    reference: the hourly gold rows the models were trained on.
    holdout:   honest out-of-sample forecasts (hourly_diagnostics.backtest_selected_models
               output, renamed to forecast_pm25), used for the quality and bias baselines.
    """
    features = {}
    for c in cols:
        v = reference[c].to_numpy(dtype=float)
        finite = v[np.isfinite(v)]
        inner = np.unique(np.quantile(finite, np.linspace(0, 1, 11)))[1:-1].tolist() if len(finite) else []
        features[c] = dict(missing_rate=float(1 - len(finite) / max(len(v), 1)),
                           mean=float(finite.mean()), std=float(finite.std()),
                           min=float(finite.min()), max=float(finite.max()),
                           p01=float(np.quantile(finite, 0.01)), p99=float(np.quantile(finite, 0.99)),
                           bin_edges=inner, bin_props=_bin_props(finite, inner))
    complete = _complete_rows(reference, cols)
    sample = complete.sample(min(C.ATTRIBUTION_SAMPLE_ROWS, len(complete)), random_state=seed)
    attributions = {f"{t}|{h}": attribution(est, sample[cols], t, cols, seed) for (t, h), (_, est) in models.items()}
    null = drift_null(reference, models, cols, features, attributions, seed=seed)
    for c in cols:
        features[c].update(null["psi"][c])
    predictions = seasonal_stability_baseline(reference, models, cols, features, seed=seed)
    baseline = dict(schema_version=1, created_at=iso(datetime.now(timezone.utc)), ndcg_null=null["ndcg"],
                    reference_start=str(reference.date.min()), reference_end=str(reference.date.max()),
                    n_reference=len(reference), feature_columns=cols, features=features,
                    predictions=predictions, seasons=C.MONSOON_SEASONS,
                    attributions=attributions, models={f"{t}|{h}": name for (t, h), (name, _) in models.items()},
                    quality={}, bias={})
    if holdout is not None and not holdout.empty:
        baseline["quality"] = {f"{t}|{int(h)}": quality(g) for (t, h), g in holdout.groupby(["region", "horizon_hours"])}
        baseline["bias"] = _bias_by_region(holdout)
        baseline["coverage_band"] = coverage_band(holdout)
        baseline["holdout_start"] = str(pd.to_datetime(holdout["forecast_time"], utc=True).min())
    return baseline


def _holdout_from_diagnostics(path: Path, gold: pd.DataFrame) -> pd.DataFrame | None:
    """hourly_diagnostics writes predicted_pm25; add persistence from gold so skill is available."""
    if not path.exists():
        return None
    h = pd.read_csv(path).rename(columns={"predicted_pm25": "forecast_pm25"})
    origin = pd.to_datetime(h["as_of_time"], utc=True).dt.tz_convert(SGT)
    now_vals = gold.set_index("date")
    h["persistence_pm25"] = [now_vals.at[o, f"now_{r}"] if o in now_vals.index else np.nan
                             for o, r in zip(origin, h["region"])]
    return h.dropna(subset=["actual_pm25", "forecast_pm25"])


def write_production_baseline(gold_path: Path = GOLD_DIR / "hourly_features.parquet",
                              models_dir: Path = MODELS_DIR / "hourly") -> Path:
    models_dir = Path(models_dir)
    models, cols, manifest = load_production_models(models_dir)
    gold = prepare_gold(pd.read_parquet(gold_path))
    trained_to = max(pd.Timestamp(e["training_end"]) for entries in manifest["models"].values() for e in entries.values())
    reference = gold.loc[gold.date <= trained_to]
    baseline = build_baseline(reference, models, cols,
                              holdout=_holdout_from_diagnostics(models_dir / "holdout_predictions.csv", gold))
    path = models_dir / BASELINE_FILE
    atomic_json(path, _json_safe(baseline))
    log.info("Baseline from %d training rows (%s to %s) -> %s", len(reference),
             baseline["reference_start"], baseline["reference_end"], path)
    return path


# ---------------------------------------------------------------------------
# Monitors
# ---------------------------------------------------------------------------
def _longest_run(values: np.ndarray) -> int:
    """Longest run of identical consecutive finite values."""
    v = values[np.isfinite(values)]
    if not len(v):
        return 0
    breaks = np.flatnonzero(np.diff(v) != 0)
    edges = np.concatenate([[-1], breaks, [len(v) - 1]])
    return int(np.diff(edges).max())


def stuck_feeds(window: pd.DataFrame) -> dict[str, int]:
    """Regions whose PM2.5 repeated the same value for at least PM25_STUCK_HOURS hours: {region: hours}."""
    out = {}
    for region in SG_REGIONS:
        col = f"now_{region}"
        if col in window:
            run = _longest_run(window.sort_values("date")[col].to_numpy(dtype=float))
            if run >= C.PM25_STUCK_HOURS:
                out[region] = run
    return out


def source_health(now, silver_dir: Path = SILVER_DIR) -> dict:
    """Age of the newest data from each upstream source (live runs only), so a failure points
    at the source that caused it. PM2.5 and weather older than FEATURE_MAX_AGE_HOURS is an
    alert. For fires the age is measured from the hour fire counts are known to (the end of
    the silver grid: 15 h after the last detection, or later if FIRMS reports those hours as
    processed), not from the last detection, so a quiet spell with no fires is not a stale
    feed. Beyond FIRE_LATENCY_HOURS + 1 the carry-forward is in use (warn); beyond that plus
    FIRE_CARRY_FORWARD_HOURS forecasts have stopped (alert)."""
    now_sgt = pd.Timestamp(timestamp(now)).tz_convert(SGT).tz_localize(None)
    out = {}

    def entry(newest, alert_after, warn_after=None, **extra):
        if newest is None or pd.isna(newest):
            return dict(newest=None, age_hours=None, status="alert", **extra)
        # never negative: fire counts can be known a few hours ahead of now (the 15 h trust window)
        age = max(0.0, (now_sgt - pd.Timestamp(newest)).total_seconds() / 3600)
        status = "alert" if age > alert_after else "warn" if warn_after is not None and age > warn_after else "ok"
        return dict(newest=str(newest), age_hours=age, status=status, **extra)

    def newest(name, column="ts", **filters):
        try:
            frame = pd.read_parquet(Path(silver_dir) / name)
        except (FileNotFoundError, OSError):
            return None
        for k, v in filters.items():
            frame = frame.loc[frame[k] == v]
        return frame[column].max() if len(frame) else None

    out["pm25"] = entry(newest("pm25.parquet"), C.FEATURE_MAX_AGE_HOURS)
    for point in WEATHER_POINTS:
        out[f"weather_{point}"] = entry(newest("weather.parquet", point=point), C.FEATURE_MAX_AGE_HOURS)
    known_to = newest("firms_hourly.parquet")
    lag = C.FIRE_LATENCY_HOURS + 1          # the features at hour t use counts up to t - lag
    try:
        grid = pd.read_parquet(Path(silver_dir) / "firms_hourly.parquet", columns=["ts", "total"])
        last_detection = grid.loc[grid.total > 0, "ts"].max()
    except (FileNotFoundError, OSError, KeyError, ValueError):   # pyarrow's ArrowInvalid is a ValueError
        last_detection = None
    out["fires"] = entry(known_to, lag + C.FIRE_CARRY_FORWARD_HOURS, warn_after=lag,
                         last_detection=None if last_detection is None or pd.isna(last_detection) else str(last_detection))
    return out


def data_quality_monitor(window: pd.DataFrame, baseline: dict, *, now=None, latest_complete=None,
                         sources: dict | None = None) -> dict:
    cols = baseline["feature_columns"]
    missing_cols = sorted(set(cols) - set(window.columns))
    season = window_season(window)
    features, n_csi_alert = {}, 0
    for c in [c for c in cols if c in window.columns]:
        b = baseline["features"][c]
        v = window[c].to_numpy(dtype=float)
        finite = v[np.isfinite(v)]
        missing = float(1 - len(finite) / max(len(v), 1))
        oor = float(((finite < b["min"]) | (finite > b["max"])).mean()) if len(finite) else 0.0
        props = _bin_props(finite, b["bin_edges"]) if len(finite) else None
        value = psi(b["bin_props"], props) if props else None
        # CSI: the same index against the same season in the training years. Baselines from
        # before seasonal stability was added have no season_props: fall back to all seasons.
        season_ref, season_lim = b.get("season_props", {}).get(season), b.get("season_limits", {}).get(season)
        psi_alert, psi_warn = b.get("psi_alert", C.PSI_ALERT), b.get("psi_warn", C.PSI_WARN)
        seasonal = bool(props and season_ref and season_lim)
        csi = psi(season_ref, props) if seasonal else value
        alert_at, warn_at = (season_lim["alert"], season_lim["warn"]) if seasonal else (psi_alert, psi_warn)
        # A feature counts as drifted only when it is unusual both for its season (CSI) and
        # against all training months (PSI). In the holdout replay, PSI alone gave 8 drift-only
        # investigate weeks and CSI alone 16 (a stricter reference also flags years that were
        # merely unusual for their season); requiring both gave 5, with the same 10 Aug 2026
        # warning ahead of the haze.
        drifted = csi is not None and csi > alert_at and value > psi_alert
        status = ("alert" if drifted or missing - b["missing_rate"] > C.MISSING_RATE_MAX_INCREASE
                  or oor > C.OUT_OF_RANGE_MAX_FRACTION
                  else "warn" if csi is not None and csi > warn_at and value > psi_warn else "ok")
        n_csi_alert += drifted
        features[c] = dict(csi=csi, csi_alert_at=alert_at, csi_ratio=csi / alert_at if csi is not None else None,
                           psi=value, psi_alert_at=psi_alert,
                           missing_rate=missing, missing_rate_baseline=b["missing_rate"],
                           out_of_range=oor, mean=float(finite.mean()) if len(finite) else None,
                           mean_baseline=b["mean"], status=status)
    age = None
    if now is not None and latest_complete is not None:
        age = (timestamp(now) - timestamp(latest_complete.to_pydatetime())).total_seconds() / 3600
    stale = age is not None and age > C.FEATURE_MAX_AGE_HOURS
    stuck = stuck_feeds(window)
    failed_sources = sorted(k for k, s in (sources or {}).items() if s["status"] == "alert")
    return dict(n_rows=len(window), season=season, missing_columns=missing_cols, feature_age_hours=age, stale=stale,
                stuck_feeds=stuck, sources=sources or {}, failed_sources=failed_sources,
                # n_psi_alert is the drift count the governance rule reads; it is now season-matched (CSI)
                n_csi_alert=int(n_csi_alert), n_psi_alert=int(n_csi_alert),
                n_feature_alert=sum(f["status"] == "alert" for f in features.values()),
                features=features,
                status="fail" if missing_cols or stale or stuck or failed_sources
                else "alert" if n_csi_alert >= C.PSI_ALERT_MIN_FEATURES
                or any(f["missing_rate"] - f["missing_rate_baseline"] > C.MISSING_RATE_MAX_INCREASE
                       or f["out_of_range"] > C.OUT_OF_RANGE_MAX_FRACTION for f in features.values())
                else "warn" if any(f["status"] != "ok" for f in features.values())
                or any(s["status"] == "warn" for s in (sources or {}).values()) else "ok")


def stability_monitor(window: pd.DataFrame, models: dict, cols: list[str], baseline: dict,
                      min_rows: int = 24) -> dict:
    """PSI of each model's forecast distribution on the window against its forecasts on the same
    season in the training years. Shows whether the model's output is stable even when no
    actuals are available to judge accuracy."""
    rows = _complete_rows(window, cols)
    ref = baseline.get("predictions") or {}
    if len(rows) < min_rows or not ref:
        return dict(n_rows=len(rows), season=window_season(window), models={}, n_alert=0, status="no_data")
    season = window_season(rows)
    out, n_alert = {}, 0
    for (target, horizon), (name, est) in models.items():
        key = f"{target}|{horizon}"
        if key not in ref:
            continue
        b = ref[key]
        p = predict(est, rows[cols], target)
        props = _bin_props(p, b["bin_edges"])
        season_ref = b.get("season_props", {}).get(season)
        value = psi(season_ref or b["bin_props"], props)
        lim = b.get("season_limits", {}).get(season) or dict(alert=C.PSI_ALERT, warn=C.PSI_WARN)
        status = "alert" if value > lim["alert"] else "warn" if value > lim["warn"] else "ok"
        n_alert += status == "alert"
        out[key] = dict(model=name, psi=value, psi_alert_at=lim["alert"], psi_ratio=value / lim["alert"],
                        mean_forecast=float(p.mean()), mean_baseline=b["mean"], status=status)
    return dict(n_rows=len(rows), season=season, models=out, n_alert=n_alert,
                status="alert" if n_alert >= C.STABILITY_ALERT_MIN_MODELS
                else "warn" if any(m["status"] != "ok" for m in out.values()) else "ok")


def matched_forecasts(db, start, end) -> pd.DataFrame:
    """Forecasts whose target time falls in [start, end) joined to their NEA actual."""
    rows = db.execute("""
        SELECT f.region, f.horizon_hours, f.forecast_time, f.as_of_time, f.model,
               f.forecast_pm25, f.persistence_pm25, f.threshold, a.actual_pm25
        FROM forecasts f JOIN actuals a ON f.forecast_time = a.observation_time AND f.region = a.region
        WHERE f.forecast_time >= ? AND f.forecast_time < ? AND a.actual_pm25 IS NOT NULL""",
        (iso(start), iso(end))).fetchall()
    return pd.DataFrame([dict(r) for r in rows],
                        columns=["region", "horizon_hours", "forecast_time", "as_of_time", "model",
                                 "forecast_pm25", "persistence_pm25", "threshold", "actual_pm25"])


def model_quality_monitor(matched: pd.DataFrame, baseline: dict) -> dict:
    groups, n_alert = {}, 0
    for (target, horizon), g in matched.groupby(["region", "horizon_hours"]):
        q = quality(g)
        base = baseline.get("quality", {}).get(f"{target}|{int(horizon)}", {})
        q["mae_baseline"], q["f2_baseline"] = base.get("mae"), base.get("f2")
        q["mae_ratio"] = q["mae"] / base["mae"] if base.get("mae") else None
        reasons = []
        if q["n"] >= C.MIN_MATCHED_FOR_QUALITY:
            if q["mae_ratio"] is not None and q["mae_ratio"] > C.MODEL_MAE_RATIO_ALERT:
                reasons.append("mae_above_holdout")
            # The product is an early warning: the failure that matters is catching fewer
            # haze hours than the naive "same as now" forecast would have. (MAE skill vs
            # persistence is reported but not alerted on: the q85 models are upward-biased
            # by design and lose on MAE most weeks -- median skill -0.42 in the replay.)
            if (q["n_elevated"] >= C.MIN_ELEVATED_FOR_RECALL and q.get("persistence_recall") is not None
                    and q["recall"] < q["persistence_recall"] - C.RECALL_VS_PERSISTENCE_MARGIN):
                reasons.append("recall_below_persistence")
        q["status"], q["reasons"] = ("alert" if reasons else "ok"), reasons
        n_alert += bool(reasons)
        groups[f"{target}|{int(horizon)}"] = q
    # Calibration and level, pooled over the targets per horizon. Coverage can be judged every
    # week, haze or not; the level ratio tells a person whether a high MAE ratio came with a
    # higher pollution level (a harder week) rather than a worse model.
    horizons, coverage_out = {}, []
    for horizon, g in matched.groupby("horizon_hours"):
        q = quality(g)
        band = baseline.get("coverage_band", {}).get(str(int(horizon)))
        base_level = np.mean([b["mean_actual"] for k, b in baseline.get("quality", {}).items()
                              if k.endswith(f"|{int(horizon)}") and b.get("mean_actual")] or [np.nan])
        out_of_band = (band is not None and q["n"] >= C.MIN_MATCHED_FOR_QUALITY * len(TARGETS)
                       and not band["low"] <= q["coverage"] <= band["high"])
        horizons[str(int(horizon))] = dict(n=q["n"], coverage=q["coverage"], coverage_band=band,
                                           level_ratio=q["mean_actual"] / base_level if np.isfinite(base_level) else None,
                                           coverage_status="alert" if out_of_band else "ok" if band else "not_judged")
        if out_of_band:
            coverage_out.append(f"+{int(horizon)} h {q['coverage']:.2f} outside {band['low']:.2f}-{band['high']:.2f}")
    return dict(n_matched=len(matched), n_groups=len(groups), n_alert=n_alert, groups=groups,
                horizons=horizons, coverage_alerts=coverage_out,
                status="no_data" if matched.empty else
                "alert" if n_alert >= C.MODEL_QUALITY_ALERT_MIN_GROUPS
                else "warn" if n_alert or coverage_out else "ok")


def bias_drift_monitor(matched: pd.DataFrame, baseline: dict) -> dict:
    current, reasons = _bias_by_region(matched), []
    for horizon, cur in current.items():
        base = baseline.get("bias", {}).get(horizon, {})
        gap, base_gap = cur["recall_gap"], base.get("recall_gap")
        cur["recall_gap_baseline"] = base_gap
        if gap is not None and (gap > C.BIAS_RECALL_GAP_ALERT
                                or (base_gap is not None and gap - base_gap > C.BIAS_RECALL_GAP_DRIFT)):
            reasons.append(f"h{horizon}_recall_gap")
        for region, q in cur["regions"].items():
            b = base.get("regions", {}).get(region, {})
            q["mean_error_baseline"] = b.get("mean_error")
            q["mean_error_shift"] = (q["mean_error"] - b["mean_error"]
                                     if q.get("n", 0) >= C.MIN_MATCHED_FOR_QUALITY and b.get("mean_error") is not None
                                     else None)
            if q["mean_error_shift"] is not None and abs(q["mean_error_shift"]) > C.BIAS_MEAN_ERROR_DRIFT:
                reasons.append(f"h{horizon}_{region}_mean_error_shift")
    return dict(horizons=current, reasons=reasons,
                status="no_data" if matched.empty else "alert" if reasons else "ok")


def attribution_drift_monitor(window: pd.DataFrame, models: dict, cols: list[str], baseline: dict,
                              min_rows: int = 24) -> dict:
    rows = _complete_rows(window, cols)
    if len(rows) < min_rows:
        return dict(n_rows=len(rows), models={}, n_alert=0, status="no_data")
    out, n_alert = {}, 0
    for (target, horizon), (name, est) in models.items():
        key = f"{target}|{horizon}"
        if key not in baseline["attributions"]:
            continue
        cur = attribution(est, rows[cols], target, cols)
        score = ndcg(baseline["attributions"][key], cur)
        top = sorted(cur, key=cur.get, reverse=True)[:5]
        limits = baseline.get("ndcg_null", {}).get(key, {})
        alert_at, warn_at = limits.get("alert", C.ATTRIBUTION_NDCG_ALERT), limits.get("warn", C.ATTRIBUTION_NDCG_ALERT)
        status = "alert" if score < alert_at else "warn" if score < warn_at else "ok"
        n_alert += status == "alert"
        out[key] = dict(model=name, ndcg=score, ndcg_alert_at=alert_at, top_features=top, attributions=cur,
                        status=status)
    return dict(n_rows=len(rows), models=out, n_alert=n_alert,
                status="alert" if n_alert >= C.MODEL_QUALITY_ALERT_MIN_GROUPS
                else "warn" if n_alert or any(m["status"] == "warn" for m in out.values()) else "ok")


# ---------------------------------------------------------------------------
# Governance decision (docs/MODEL_GOVERNANCE.md, section "Retraining triggers")
# ---------------------------------------------------------------------------
PERSISTENT_SIGNALS = ("data_drift", "output_stability", "attribution_drift", "coverage")
COOLDOWN_EXEMPT = ("feature_code_changed", "environment_changed")


def governance_decision(results: dict, *, model_published_at=None, now=None, previous_reasons=None,
                        triggers=(), last_retrain_at=None, early_life=None) -> dict:
    """One action from the monitor results (docs/MODEL_GOVERNANCE.md §5).

    previous_reasons: the previous run's reasons. When given, a drift, output-stability,
    attribution or coverage signal only counts if it also appeared in that run (persistence);
    None keeps the single-run behaviour. triggers: scenario retrains (scenario_triggers).
    last_retrain_at: when the last retrain was attempted; automatic retrains are held for
    RETRAIN_COOLDOWN_DAYS after it, except after a change in the feature code or libraries.
    early_life: a finding of early_life_check, which asks a person to review.
    """
    dq, mq = results["data_quality"], results["model_quality"]
    bias, attr = results["bias_drift"], results["attribution_drift"]
    reasons = []
    age = None
    if model_published_at is not None and now is not None:
        age = (timestamp(now) - timestamp(model_published_at)).total_seconds() / 86400
    if dq["status"] == "fail":
        if dq.get("failed_sources"):
            reasons.append(f"upstream_source_failure: {', '.join(dq['failed_sources'])}")
        if dq.get("stuck_feeds"):
            reasons.append("stuck_feed: PM2.5 unchanged for " + ", ".join(
                f"{r} {h} h" for r, h in dq["stuck_feeds"].items()))
        if dq["stale"]:
            reasons.append("data_pipeline_failure: stale features")
        if dq["missing_columns"]:
            reasons.append(f"data_pipeline_failure: missing columns {dq['missing_columns']}")
        return dict(action="fix_data", reasons=reasons, model_age_days=age)
    stab = results.get("stability") or dict(status="no_data", n_alert=0)
    seen_before = None if previous_reasons is None else {r.split(":")[0] for r in previous_reasons}

    def signal(name: str, active: bool, text: str) -> bool:
        """Record a persistent signal; it counts only if the previous run had it too."""
        if not active:
            return False
        confirmed = seen_before is None or name in seen_before
        reasons.append(f"{name}: {text}" + ("" if confirmed else " (first occurrence; acted on if it repeats)"))
        return confirmed

    season = f" ({dq['season']})" if dq.get("season") else ""
    other_alert = False
    performance = mq["status"] == "alert"
    if performance:
        levels = [h["level_ratio"] for h in mq.get("horizons", {}).values() if h.get("level_ratio")]
        level = f"; pollution level {max(levels):.1f}x the holdout average" if levels and max(levels) >= 1.5 else ""
        reasons.append(f"model_quality: {mq['n_alert']} of {mq['n_groups']} target/horizon groups in alert{level}")
    drift = signal("data_drift", dq["n_psi_alert"] >= C.PSI_ALERT_MIN_FEATURES,
                   f"{dq['n_psi_alert']} features beyond their calibrated CSI and PSI limits{season}")
    output_shift = signal("output_stability", stab["status"] == "alert",
                          f"forecasts of {stab.get('n_alert', 0)} models beyond their calibrated PSI limit{season}")
    attribution_shift = signal("attribution_drift", attr.get("status") == "alert",
                               f"{attr.get('n_alert', 0)} models below their calibrated NDCG limit")
    coverage = signal("coverage", bool(mq.get("coverage_alerts")),
                      "calibration outside its normal weekly range: " + "; ".join(mq.get("coverage_alerts", [])))
    if dq["status"] == "alert" and dq["n_psi_alert"] < C.PSI_ALERT_MIN_FEATURES:
        reasons.append(f"data_quality: {dq['n_feature_alert']} features out of range or missing more often")
        other_alert = True
    if bias.get("status") == "alert":
        reasons.append("bias_drift: " + ", ".join(bias["reasons"][:5]))
        other_alert = True
    aged = age is not None and age > C.MAX_MODEL_AGE_DAYS
    if aged:
        reasons.append(f"model_age: {age:.0f} days > {C.MAX_MODEL_AGE_DAYS}")
    reasons.extend(triggers)
    if early_life:
        reasons.append(early_life)
        other_alert = True
    # Retrain on evidence the model is wrong (performance or calibration), on input drift or
    # unstable output that has also changed what the model relies on, on age, or on a scenario
    # (before the fire season, after an episode, after a change in the feature path).
    # Drift, output instability, attribution or bias alone -> a person looks first.
    retrain = (performance or coverage or ((drift or output_shift) and attribution_shift)
               or aged or bool(triggers))
    exempt = any(t.startswith(COOLDOWN_EXEMPT) for t in triggers)
    if retrain and not exempt and last_retrain_at is not None and now is not None:
        since = (timestamp(now) - timestamp(last_retrain_at)).total_seconds() / 86400
        if since < C.RETRAIN_COOLDOWN_DAYS:
            reasons.append(f"retrain_cooldown: last retrain {since:.1f} days ago; the next is allowed after "
                           f"{C.RETRAIN_COOLDOWN_DAYS} days")
            retrain, other_alert = False, True
    # Warnings and unconfirmed signals stay on the dashboard but do not page anyone (alarm fatigue).
    attention = other_alert or drift or output_shift or attribution_shift or coverage or performance
    return dict(action="retrain" if retrain else "investigate" if attention else "none",
                reasons=reasons, model_age_days=age)


def scenario_triggers(now, gold: pd.DataFrame, manifest: dict) -> list[str]:
    """Retrains no threshold would ask for (docs/MODEL_GOVERNANCE.md §5): before the fire season,
    after a haze episode the model has not trained on, and after a change in the feature code,
    feature settings or model libraries."""
    from hazecast import provenance
    out = []
    now_sgt = pd.Timestamp(timestamp(now)).tz_convert(SGT)
    trained = pd.Timestamp(manifest.get("trained_at") or manifest.get("published_at") or now_sgt)
    trained = trained.tz_localize("UTC") if trained.tzinfo is None else trained
    month, day = C.PRE_SEASON_RETRAIN
    season_start = pd.Timestamp(now_sgt.year, month, day, tz=SGT)
    if season_start <= now_sgt < season_start + pd.Timedelta(days=31) and trained < season_start:
        out.append("scheduled_pre_season: refresh before the August-October fire season")
    label_ends = [pd.Timestamp(e["training_label_end"]) for es in manifest.get("models", {}).values()
                  for e in es.values() if e.get("training_label_end")]
    if label_ends and "date" in gold:
        seen_to = max(label_ends)
        recent = gold.loc[gold.date > seen_to.tz_convert(gold.date.dt.tz)]
        hot = recent.loc[(recent[[f"now_{r}" for r in SG_REGIONS if f"now_{r}" in recent]]
                          >= C.PM25_ELEVATED_THRESHOLD).any(axis=1)]
        if len(hot) >= C.MIN_ELEVATED_FOR_RECALL:
            last = hot.date.max()
            if (now_sgt - last).total_seconds() / 86400 >= C.POST_EPISODE_QUIET_DAYS:
                out.append(f"post_episode: {len(hot)} elevated hours since the model's training data ended "
                           f"(last {last:%d %b}); retrain so the episode enters training and validation")
    for diff in provenance.differences(manifest):
        kind = "feature_code_changed" if diff.startswith("feature") else "environment_changed"
        out.append(f"{kind}: {diff}")
    return out


def last_retrain_attempt(models_dir: Path = MODELS_DIR / "hourly") -> str | None:
    """When the last retrain was attempted: the newest staging directory or promotion-log entry."""
    times = [datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
             for p in (Path(models_dir).parent / "hourly_staging").glob("*") if p.is_dir()]
    log_path = Path(models_dir) / "promotion_log.jsonl"
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("action") in ("promoted", "rejected") and entry.get("decided_at"):
                times.append(timestamp(entry["decided_at"]))
    return iso(max(times)) if times else None


def early_life_check(db, models_dir: Path, gold: pd.DataFrame, now) -> str | None:
    """In a new champion's first EARLY_LIFE_DAYS, compare its live forecasts with what the champion
    it replaced would have forecast for the same hours. Returns a reason when it is clearly worse."""
    import joblib
    models_dir = Path(models_dir)
    manifest = json.loads((models_dir / "model_manifest.json").read_text())
    published, replaced = manifest.get("published_at"), manifest.get("replaced")
    if not published or not replaced or not Path(replaced).exists():
        return None
    if (timestamp(now) - timestamp(published)).total_seconds() / 86400 > C.EARLY_LIFE_DAYS:
        return None
    previous = json.loads(Path(replaced).read_text())
    matched = matched_forecasts(db, timestamp(published), timestamp(now))
    # only forecasts the new champion issued (not the old one's, whose target time fell after the switch)
    matched = matched.loc[pd.to_datetime(matched["as_of_time"], utc=True) >= pd.Timestamp(timestamp(published))]
    if matched.empty:
        return None
    rows = gold.set_index("date")
    findings = []
    for horizon, g in matched.groupby("horizon_hours"):
        if len(g) < C.MIN_MATCHED_FOR_QUALITY * len(TARGETS):
            continue
        origins = pd.to_datetime(g["as_of_time"], utc=True).dt.tz_convert(SGT)
        old = np.full(len(g), np.nan)
        for target in g.region.unique():
            entry = previous["models"][str(int(horizon))][target]
            sel = (g.region == target).to_numpy() & origins.isin(rows.index).to_numpy()
            if not sel.any():
                continue
            X = rows.loc[origins[sel], previous["feature_columns"]]
            est = None if entry["model"] == "persistence" else joblib.load(models_dir / entry["artifact"])
            old[sel] = predict(est, X, target)
        ok = np.isfinite(old)
        if ok.sum() < C.MIN_MATCHED_FOR_QUALITY * len(TARGETS):
            continue
        new_q = quality(g.loc[ok])
        old_q = quality(g.loc[ok].assign(forecast_pm25=old[ok]))
        if old_q.get("mae") is not None and new_q["mae"] > C.EARLY_LIFE_MAE_RATIO * old_q["mae"]:
            findings.append(f"+{int(horizon)} h MAE {new_q['mae']:.1f} vs {old_q['mae']:.1f}")
        if (new_q["n_elevated"] >= C.MIN_ELEVATED_FOR_RECALL and old_q["recall"] is not None
                and new_q["recall"] < old_q["recall"] - C.RECALL_VS_PERSISTENCE_MARGIN):
            findings.append(f"+{int(horizon)} h recall {new_q['recall']:.2f} vs {old_q['recall']:.2f}")
    if not findings:
        return None
    return ("early_life: the new champion is doing worse than the one it replaced on the same hours ("
            + "; ".join(findings) + "); review and consider `python -m hazecast.governance rollback`")


# ---------------------------------------------------------------------------
# Persistence of results
# ---------------------------------------------------------------------------
def _ensure_tables(db) -> None:
    db.executescript("""
        CREATE TABLE IF NOT EXISTS monitor_metrics (
            window_end TEXT NOT NULL, monitor TEXT NOT NULL, region TEXT NOT NULL,
            horizon_hours INTEGER NOT NULL, feature TEXT NOT NULL, metric TEXT NOT NULL,
            value REAL, status TEXT,
            PRIMARY KEY (window_end, monitor, region, horizon_hours, feature, metric));
        CREATE TABLE IF NOT EXISTS monitor_runs (
            window_end TEXT PRIMARY KEY, generated_at TEXT NOT NULL, action TEXT NOT NULL,
            statuses TEXT NOT NULL, reasons TEXT NOT NULL);
    """)


def _metric_rows(window_end: str, results: dict) -> list[tuple]:
    rows = []

    def add(monitor, metric, value, status=None, region="", horizon=0, feature=""):
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return
        rows.append((window_end, monitor, region, int(horizon), feature, metric, float(value), status))

    dq = results["data_quality"]
    add("data_quality", "n_psi_alert", dq["n_psi_alert"], dq["status"])
    add("data_quality", "n_csi_alert", dq.get("n_csi_alert"), dq["status"])
    add("data_quality", "feature_age_hours", dq["feature_age_hours"], dq["status"])
    for f, m in dq["features"].items():
        add("data_quality", "psi", m["psi"], m["status"], feature=f)
        add("data_quality", "csi", m.get("csi"), m["status"], feature=f)
        add("data_quality", "csi_ratio", m.get("csi_ratio"), m["status"], feature=f)
        add("data_quality", "missing_rate", m["missing_rate"], m["status"], feature=f)
        add("data_quality", "out_of_range", m["out_of_range"], m["status"], feature=f)
    for region, hours in dq.get("stuck_feeds", {}).items():
        add("data_quality", "stuck_hours", hours, "alert", region=region)
    for source, s in dq.get("sources", {}).items():
        add("data_quality", "source_age_hours", s.get("age_hours"), s["status"], feature=source)
    stab = results.get("stability") or {}
    add("stability", "n_psi_alert", stab.get("n_alert"), stab.get("status"))
    for key, s in stab.get("models", {}).items():
        region, horizon = key.split("|")
        add("stability", "psi", s["psi"], s["status"], region, horizon)
        add("stability", "psi_ratio", s["psi_ratio"], s["status"], region, horizon)
        add("stability", "mean_forecast", s["mean_forecast"], s["status"], region, horizon)
    for key, q in results["model_quality"]["groups"].items():
        region, horizon = key.split("|")
        for metric in ("mae", "rmse", "f2", "recall", "skill", "mae_ratio", "persistence_mae", "n", "n_elevated",
                       "coverage"):
            add("model_quality", metric, q.get(metric), q["status"], region, horizon)
    for horizon, h in results["model_quality"].get("horizons", {}).items():
        add("model_quality", "coverage_pooled", h["coverage"], h["coverage_status"], horizon=horizon)
        add("model_quality", "level_ratio", h.get("level_ratio"), None, horizon=horizon)
    for horizon, cur in results["bias_drift"]["horizons"].items():
        add("bias_drift", "recall_gap", cur["recall_gap"], results["bias_drift"]["status"], horizon=horizon)
        for part in ("day", "night"):
            add("bias_drift", f"mae_{part}", cur[part].get("mae"), None, horizon=horizon)
        for region, q in cur["regions"].items():
            add("bias_drift", "mean_error", q.get("mean_error"), None, region, horizon)
            add("bias_drift", "recall", q.get("recall"), None, region, horizon)
            add("bias_drift", "alert_rate_bias", (q["flag_rate"] - q["actual_rate"]) if q.get("n") else None,
                None, region, horizon)
    for key, a in results["attribution_drift"]["models"].items():
        region, horizon = key.split("|")
        add("attribution_drift", "ndcg", a["ndcg"], a["status"], region, horizon)
    return rows


def save_results(db, window_end, results: dict, decision: dict) -> None:
    _ensure_tables(db)
    end = iso(window_end)
    with db:
        db.executemany("INSERT OR REPLACE INTO monitor_metrics VALUES (?,?,?,?,?,?,?,?)", _metric_rows(end, results))
        db.execute("INSERT OR REPLACE INTO monitor_runs VALUES (?,?,?,?,?)",
                   (end, iso(datetime.now(timezone.utc)),
                    decision["action"], json.dumps({m: results[m]["status"] for m in MONITORS if m in results}),
                    json.dumps(decision["reasons"])))


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


# ---------------------------------------------------------------------------
def evaluate_window(gold: pd.DataFrame, db, models: dict, cols: list[str], baseline: dict,
                    window_start, window_end, *, now=None, sources: dict | None = None) -> dict:
    """Run all five monitors. [window_start, window_end) is the performance window (forecasts
    vs actuals); the drift and stability monitors look at the DRIFT_WINDOW_DAYS ending at
    window_end, because a week of slow-moving features is too small a sample to compare
    distributions on. `now` enables the freshness check and `sources` the upstream source
    ages (live runs only)."""
    window = _window(gold, window_end - pd.Timedelta(days=C.DRIFT_WINDOW_DAYS), window_end)
    latest = None
    if now is not None:
        done = _complete_rows(gold.loc[gold.date <= window_end], cols)
        latest = done.date.max() if len(done) else None
    matched = matched_forecasts(db, window_start, window_end)
    return dict(
        data_quality=data_quality_monitor(window, baseline, now=now, latest_complete=latest, sources=sources),
        stability=stability_monitor(window, models, cols, baseline),
        model_quality=model_quality_monitor(matched, baseline),
        bias_drift=bias_drift_monitor(matched, baseline),
        attribution_drift=attribution_drift_monitor(window, models, cols, baseline),
    )


def run_model_monitor(*, now=None, window_days: int = C.MONITOR_WINDOW_DAYS,
                      gold_path: Path = GOLD_DIR / "hourly_features.parquet",
                      models_dir: Path = MODELS_DIR / "hourly",
                      db_path: Path = MONITORING_DIR / "pipeline.sqlite",
                      output_dir: Path = MONITORING_DIR) -> dict:
    now = timestamp(now or datetime.now(timezone.utc))
    models_dir = Path(models_dir)
    baseline_path = models_dir / BASELINE_FILE
    if not baseline_path.exists():
        raise FileNotFoundError(f"No {baseline_path}; run `python -m hazecast.model_monitor baseline` after training")
    baseline = json.loads(baseline_path.read_text())
    models, cols, manifest = load_production_models(models_dir)
    gold = prepare_gold(pd.read_parquet(gold_path))
    end = pd.Timestamp(now).tz_convert(SGT).floor("h") + pd.Timedelta(hours=1)
    start = end - pd.Timedelta(days=window_days)
    db = connect(db_path)
    try:
        results = evaluate_window(gold, db, models, cols, baseline, start, end, now=now, sources=source_health(now))
        published = manifest.get("published_at") or datetime.fromtimestamp(
            (models_dir / "model_manifest.json").stat().st_mtime, timezone.utc).isoformat()
        _ensure_tables(db)
        last = db.execute("SELECT reasons FROM monitor_runs WHERE window_end < ? ORDER BY window_end DESC LIMIT 1",
                          (iso(end),)).fetchone()
        try:
            early = early_life_check(db, models_dir, gold, now)
        except Exception as exc:  # noqa: BLE001 - a failed comparison must not stop the monitor
            log.warning("Early-life check skipped: %s", exc)
            early = None
        decision = governance_decision(results, model_published_at=published, now=now,
                                       previous_reasons=json.loads(last[0]) if last else [],
                                       triggers=scenario_triggers(now, gold, manifest),
                                       last_retrain_at=last_retrain_attempt(models_dir), early_life=early)
        save_results(db, end, results, decision)
    finally:
        db.close()
    report = _json_safe(dict(generated_at=iso(now), window_start=str(start), window_end=str(end),
                             decision=decision, **results))
    output_dir = Path(output_dir)
    atomic_json(output_dir / "model_monitor" / f"{end:%Y-%m-%dT%H}.json", report)
    atomic_json(output_dir / "latest_model_monitor.json", report)
    log.info("Model monitor %s -> %s: action=%s %s", start, end, decision["action"],
             {m: results[m]["status"] for m in MONITORS})
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("baseline", help="capture the training baseline for the published hourly models")
    run = sub.add_parser("run", help="run all five monitors on the latest window")
    run.add_argument("--window-days", type=int, default=C.MONITOR_WINDOW_DAYS)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.cmd == "baseline":
        write_production_baseline()
    else:
        report = run_model_monitor(window_days=args.window_days)
        print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
