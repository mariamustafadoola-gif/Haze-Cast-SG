"""Context for the "Haze or Hangout?" page: what the page shows around the forecast.

The hourly forecast (data/gold/latest_hourly_forecast.json) says what PM2.5 will
be in 6, 12 and 24 hours. This module adds what the page needs around it:

  now        the latest measured PM2.5 per region (the "Now" tab)
  accuracy   how often each forecast horizon caught unhealthy hours on the
             held-out test period (reports/hourly_actual_vs_predicted_metrics.csv)
  monitoring the latest model-monitor decision, when the monitoring DAG has run

Everything is read from files the pipeline already writes; nothing is scored here.
Served by the API at /context.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from hazecast.config import GOLD_DIR, MONITORING_DIR, REPO_ROOT, SG_REGIONS

GOLD_HOURLY_PATH = GOLD_DIR / "hourly_features.parquet"
METRICS_PATH = REPO_ROOT / "reports" / "hourly_actual_vs_predicted_metrics.csv"
MONITOR_PATH = MONITORING_DIR / "latest_model_monitor.json"


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, 1) if math.isfinite(f) else None


def accuracy_by_horizon(metrics_path: Path = METRICS_PATH) -> dict:
    """Share of unhealthy hours each horizon caught on the test period, all targets pooled."""
    if not Path(metrics_path).exists():
        return {}
    m = pd.read_csv(metrics_path)
    out = {}
    for h, g in m.groupby("horizon_hours"):
        tp, fp, fn = g["tp"].sum(), g["fp"].sum(), g["fn"].sum()
        out[str(int(h))] = dict(
            caught_share=round(tp / (tp + fn), 3) if tp + fn else None,
            right_when_warned=round(tp / (tp + fp), 3) if tp + fp else None,
            mae=_num(g["mae"].mean()),
        )
    chart = str(m["chart"].iloc[0]) if "chart" in m and len(m) else ""
    period = None
    parts = [p for p in Path(chart).stem.split("_") if p.isdigit() and len(p) == 8]
    if len(parts) == 2:
        period = [f"{p[:4]}-{p[4:6]}-{p[6:]}" for p in parts]
    return dict(horizons=out, test_period=period)


def monitoring_summary(monitor_path: Path = MONITOR_PATH) -> dict | None:
    if not Path(monitor_path).exists():
        return None
    r = json.loads(Path(monitor_path).read_text())
    statuses = {k: r[k]["status"] for k in ("data_quality", "model_quality", "bias_drift", "attribution_drift")
                if isinstance(r.get(k), dict) and "status" in r[k]}
    d = r.get("decision") or {}
    # action is "none" (all good), "investigate" (a person should look) or "retrain" / "fix_data"
    return dict(generated_at=r.get("generated_at"), action=d.get("action") if isinstance(d, dict) else d,
                statuses=statuses)


def build_context(gold_path: Path = GOLD_HOURLY_PATH, metrics_path: Path = METRICS_PATH,
                  monitor_path: Path = MONITOR_PATH) -> dict:
    gold = pd.read_parquet(gold_path, columns=["date", *(f"now_{r}" for r in (*SG_REGIONS, "national"))])
    gold = gold[gold["now_national"].notna()]
    if gold.empty:
        raise ValueError("gold table has no measured PM2.5 rows")
    row = gold.sort_values("date").iloc[-1]
    return dict(
        as_of_time=pd.Timestamp(row["date"]).tz_localize("Asia/Singapore").isoformat(),
        now={r: _num(row.get(f"now_{r}")) for r in (*SG_REGIONS, "national")},
        accuracy=accuracy_by_horizon(metrics_path),
        monitoring=monitoring_summary(monitor_path),
    )


if __name__ == "__main__":
    print(json.dumps(build_context(), indent=2))
