"""Model monitor for the deployed hourly models: data quality, model quality,
bias drift and feature attribution drift, plus the retraining decision.

Follows the SageMaker Model Monitor pattern: a *baseline* is captured from
the training data when a model is published, and each monitoring run
compares a recent window against it.

    python -m hazecast.model_monitor baseline   # after training (the retrain DAG runs this)
    python -m hazecast.model_monitor run        # daily (the monitoring DAG runs this)

Monitors (thresholds live in hazecast.config, the policy in docs/MODEL_GOVERNANCE.md):
  data_quality         schema, freshness, missing rate, out-of-range share and
                       PSI of every model feature in the window vs training
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
from hazecast.config import GOLD_DIR, MODELS_DIR, MONITORING_DIR, REPO_ROOT, SG_REGIONS
from hazecast.pipeline_store import SGT, TARGETS, connect, iso, timestamp, atomic_json

log = logging.getLogger(__name__)
BASELINE_FILE = "monitoring_baseline.json"
REPORTS_DIR = REPO_ROOT / "reports" / "monitoring"
MONITORS = ("data_quality", "model_quality", "bias_drift", "attribution_drift")


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
               flag_rate=float(flagged.mean()), actual_rate=float(actual.mean()))
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
    baseline = dict(schema_version=1, created_at=iso(datetime.now(timezone.utc)), ndcg_null=null["ndcg"],
                    reference_start=str(reference.date.min()), reference_end=str(reference.date.max()),
                    n_reference=len(reference), feature_columns=cols, features=features,
                    attributions=attributions, models={f"{t}|{h}": name for (t, h), (name, _) in models.items()},
                    quality={}, bias={})
    if holdout is not None and not holdout.empty:
        baseline["quality"] = {f"{t}|{int(h)}": quality(g) for (t, h), g in holdout.groupby(["region", "horizon_hours"])}
        baseline["bias"] = _bias_by_region(holdout)
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
def data_quality_monitor(window: pd.DataFrame, baseline: dict, *, now=None, latest_complete=None) -> dict:
    cols = baseline["feature_columns"]
    missing_cols = sorted(set(cols) - set(window.columns))
    features, n_psi_alert = {}, 0
    for c in [c for c in cols if c in window.columns]:
        b = baseline["features"][c]
        v = window[c].to_numpy(dtype=float)
        finite = v[np.isfinite(v)]
        missing = float(1 - len(finite) / max(len(v), 1))
        oor = float(((finite < b["min"]) | (finite > b["max"])).mean()) if len(finite) else 0.0
        value = psi(b["bin_props"], _bin_props(finite, b["bin_edges"])) if len(finite) else None
        alert_at, warn_at = b.get("psi_alert", C.PSI_ALERT), b.get("psi_warn", C.PSI_WARN)
        drifted = value is not None and value > alert_at
        status = ("alert" if drifted or missing - b["missing_rate"] > C.MISSING_RATE_MAX_INCREASE
                  or oor > C.OUT_OF_RANGE_MAX_FRACTION
                  else "warn" if value is not None and value > warn_at else "ok")
        n_psi_alert += drifted
        features[c] = dict(psi=value, psi_alert_at=alert_at, missing_rate=missing, missing_rate_baseline=b["missing_rate"],
                           out_of_range=oor, mean=float(finite.mean()) if len(finite) else None,
                           mean_baseline=b["mean"], status=status)
    age = None
    if now is not None and latest_complete is not None:
        age = (timestamp(now) - timestamp(latest_complete.to_pydatetime())).total_seconds() / 3600
    stale = age is not None and age > C.FEATURE_MAX_AGE_HOURS
    return dict(n_rows=len(window), missing_columns=missing_cols, feature_age_hours=age, stale=stale,
                n_psi_alert=int(n_psi_alert), n_feature_alert=sum(f["status"] == "alert" for f in features.values()),
                features=features,
                status="fail" if missing_cols or stale
                else "alert" if n_psi_alert >= C.PSI_ALERT_MIN_FEATURES
                or any(f["missing_rate"] - f["missing_rate_baseline"] > C.MISSING_RATE_MAX_INCREASE
                       or f["out_of_range"] > C.OUT_OF_RANGE_MAX_FRACTION for f in features.values())
                else "warn" if any(f["status"] != "ok" for f in features.values()) else "ok")


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
    return dict(n_matched=len(matched), n_groups=len(groups), n_alert=n_alert, groups=groups,
                status="no_data" if matched.empty else
                "alert" if n_alert >= C.MODEL_QUALITY_ALERT_MIN_GROUPS else "warn" if n_alert else "ok")


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
def governance_decision(results: dict, *, model_published_at=None, now=None) -> dict:
    dq, mq = results["data_quality"], results["model_quality"]
    bias, attr = results["bias_drift"], results["attribution_drift"]
    reasons = []
    age = None
    if model_published_at is not None and now is not None:
        age = (timestamp(now) - timestamp(model_published_at)).total_seconds() / 86400
    if dq["status"] == "fail":
        reasons.append("data_pipeline_failure: " + ("stale features" if dq["stale"] else
                                                    f"missing columns {dq['missing_columns']}"))
        return dict(action="fix_data", reasons=reasons, model_age_days=age)
    performance = mq["status"] == "alert"
    drift = dq["n_psi_alert"] >= C.PSI_ALERT_MIN_FEATURES
    attribution_shift = attr["status"] == "alert"
    if performance:
        reasons.append(f"model_quality: {mq['n_alert']} of {mq['n_groups']} target/horizon groups in alert")
    if drift:
        reasons.append(f"data_drift: {dq['n_psi_alert']} features beyond their calibrated PSI limit")
    if attribution_shift:
        reasons.append(f"attribution_drift: {attr['n_alert']} models below their calibrated NDCG limit")
    if dq["status"] == "alert" and not drift:
        reasons.append(f"data_quality: {dq['n_feature_alert']} features out of range or missing more often")
    if bias["status"] == "alert":
        reasons.append("bias_drift: " + ", ".join(bias["reasons"][:5]))
    if age is not None and age > C.MAX_MODEL_AGE_DAYS:
        reasons.append(f"model_age: {age:.0f} days > {C.MAX_MODEL_AGE_DAYS}")
    # Retrain on evidence the model is wrong (performance), on input drift that has
    # also changed what the model relies on, or on age. Drift / bias alone -> a human looks first.
    retrain = performance or (drift and attribution_shift) or (age is not None and age > C.MAX_MODEL_AGE_DAYS)
    # Warnings are shown on the dashboard but do not page anyone (alarm fatigue): only
    # an alert in some monitor asks a human to look.
    attention = reasons or any(results[m]["status"] == "alert" for m in MONITORS)
    return dict(action="retrain" if retrain else "investigate" if attention else "none",
                reasons=reasons, model_age_days=age)


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
    add("data_quality", "feature_age_hours", dq["feature_age_hours"], dq["status"])
    for f, m in dq["features"].items():
        add("data_quality", "psi", m["psi"], m["status"], feature=f)
        add("data_quality", "missing_rate", m["missing_rate"], m["status"], feature=f)
        add("data_quality", "out_of_range", m["out_of_range"], m["status"], feature=f)
    for key, q in results["model_quality"]["groups"].items():
        region, horizon = key.split("|")
        for metric in ("mae", "rmse", "f2", "recall", "skill", "mae_ratio", "persistence_mae", "n", "n_elevated"):
            add("model_quality", metric, q.get(metric), q["status"], region, horizon)
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
                    decision["action"], json.dumps({m: results[m]["status"] for m in MONITORS}),
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
                    window_start, window_end, *, now=None) -> dict:
    """Run all four monitors. [window_start, window_end) is the performance window (forecasts
    vs actuals); the drift monitors look at the DRIFT_WINDOW_DAYS ending at window_end, because
    a week of slow-moving features is too small a sample to compare distributions on.
    `now` enables the freshness check (live runs only)."""
    window = _window(gold, window_end - pd.Timedelta(days=C.DRIFT_WINDOW_DAYS), window_end)
    latest = None
    if now is not None:
        done = _complete_rows(gold.loc[gold.date <= window_end], cols)
        latest = done.date.max() if len(done) else None
    matched = matched_forecasts(db, window_start, window_end)
    return dict(
        data_quality=data_quality_monitor(window, baseline, now=now, latest_complete=latest),
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
        results = evaluate_window(gold, db, models, cols, baseline, start, end, now=now)
        published = manifest.get("published_at") or datetime.fromtimestamp(
            (models_dir / "model_manifest.json").stat().st_mtime, timezone.utc).isoformat()
        decision = governance_decision(results, model_published_at=published, now=now)
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
    run = sub.add_parser("run", help="run all four monitors on the latest window")
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
