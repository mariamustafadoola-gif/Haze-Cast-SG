"""Task functions for the Airflow DAGs in dags/.

Each function is one Airflow task: plain Python, importable and runnable
without Airflow (handy for debugging a failed task locally), returning small
JSON-serialisable summaries that Airflow stores as XCom. The DAG files only
wire these together and decide schedules.

    hazecast_backfill    manual       fetch_history -> build_layers -> (trigger hazecast_retrain)
    hazecast_hourly      hourly       fetch_live -> build_layers -> features_fresh? -> run_inference
    hazecast_monitoring  daily        topup_history -> collect_actuals -> model_monitor -> dashboards
                                      -> branch(retrain | investigate | fix_data | none)
    hazecast_retrain     monthly or   train_challenger -> replay_holdout -> promote? -> write_baseline
                         triggered    (+ train_daily in parallel)
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from hazecast.config import GOLD_DIR, MODELS_DIR, MONITORING_DIR

log = logging.getLogger(__name__)
HOURLY_DIR = MODELS_DIR / "hourly"
STAGING_ROOT = MODELS_DIR / "hourly_staging"
LEDGER = MONITORING_DIR / "pipeline.sqlite"


def _firms_key_present() -> bool:
    from hazecast.ingest.live import _env
    return bool(_env("FIRMS_MAP_KEY"))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def fetch_history(start: str | None = None, end: str | None = None, lookback_days: int | None = None,
                  sources: list[str] | None = None) -> dict:
    """Backfill / top up data/raw. FIRMS is skipped with a warning when no MAP_KEY is configured
    (unless explicitly requested), so PM2.5 and weather still update."""
    from hazecast.ingest import historical
    sources = list(sources or historical.SOURCES)
    if "firms" in sources and not _firms_key_present():
        log.warning("FIRMS_MAP_KEY not set: skipping fire history. Add it to .env to keep fire features current.")
        sources.remove("firms")
    end_d = date.fromisoformat(end) if end else None
    start_d = date.fromisoformat(start) if start else None
    if lookback_days is not None:
        start_d = (end_d or historical._yesterday_sgt()) - timedelta(days=lookback_days)
    return historical.fetch_history(start_d, end_d, tuple(sources))


def fetch_live(days: int = 3) -> dict:
    """Latest hours from the live APIs into the *live* raw files (overwritten each run)."""
    from hazecast.ingest import live
    now = live.SGT_NOW()
    start = (now - timedelta(days=days)).date()
    out = {"pm25": str(live.write_pm25_raw(live.fetch_pm25(start, now.date()))),
           "weather": [str(p) for p in live.write_weather_raw(days + 1, now=now)]}
    if _firms_key_present():
        out["firms"] = str(live.write_firms_raw(live.fetch_firms(start, now.date())))
    else:
        log.warning("FIRMS_MAP_KEY not set: fire features will go stale (fire_data_gap_recent will flag it)")
    return out


def build_layers() -> dict:
    """raw -> bronze -> silver -> gold (daily + hourly) + data-quality report."""
    from hazecast import refresh
    gold = refresh.run(fetch=False)
    newest = refresh.latest_complete_hour(gold)
    return {"hourly_rows": len(gold), "newest_complete_hour": str(newest)}


def features_fresh(max_age_hours: float = 2.0) -> bool:
    """Gate for inference: is the newest complete hourly gold row recent enough to forecast from?"""
    from hazecast.hourly_model import _feature_columns
    gold = pd.read_parquet(GOLD_DIR / "hourly_features.parquet")
    cols = _feature_columns(gold)
    ok = gold[cols].notna().all(axis=1)
    if not ok.any():
        log.error("No complete hourly feature row")
        return False
    newest = pd.Timestamp(gold.loc[ok, "date"].max()).tz_localize("Asia/Singapore")
    age = (pd.Timestamp.now(tz="UTC") - newest).total_seconds() / 3600
    log.info("Newest complete hour %s is %.2f h old (limit %.1f h)", newest, age, max_age_hours)
    if "fire_carried_hours" in gold:
        carried = float(gold.loc[ok, "fire_carried_hours"].iloc[-1])
        if carried > 0:
            log.warning("FIRMS is late: that hour's fire features were carried forward %.0f h", carried)
    return age <= max_age_hours


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(max_feature_age_hours: float = 2.0) -> dict:
    from hazecast.inference import run_inference as _run
    saved = _run(max_feature_age_hours=max_feature_age_hours)
    return {"archived": len(saved), "as_of_time": saved[0]["as_of_time"],
            "elevated": sorted({f"{r['region']}+{r['horizon_hours']}h" for r in saved if r["elevated_flag"]})}


def score_daily() -> dict:
    """Next-day daily-mean forecast (the original project objective), if a daily model is published."""
    from hazecast.score import score_latest, write_forecast
    if not (MODELS_DIR / "daily_model_manifest.json").exists():
        log.warning("No daily model published yet; skipping daily forecast")
        return {"skipped": True}
    return {"path": write_forecast(score_latest(pipeline="daily"))}


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------
def collect_actuals(lookback_days: int = 3) -> dict:
    """Fetch NEA actuals for the last few days into the ledger and score archived forecasts."""
    from hazecast.monitoring import run_monitoring
    if not LEDGER.exists():
        log.warning("No forecast ledger yet (inference has not run); nothing to verify")
        return {"status": "no_ledger"}
    report = run_monitoring(lookback_days=lookback_days)
    return {"status": report["status"], "alerts": len(report["alerts"]), "api_errors": len(report["api_errors"])}


def model_monitor() -> dict:
    from hazecast.model_monitor import run_model_monitor, MONITORS
    report = run_model_monitor()
    return {"action": report["decision"]["action"], "reasons": report["decision"]["reasons"],
            "statuses": {m: report[m]["status"] for m in MONITORS}}


def dashboards() -> dict:
    from hazecast.dashboard import build_dashboard
    out = {}
    if LEDGER.exists():
        out["live"] = str(build_dashboard(LEDGER))
    return out


def haze_active(hours: int | None = None) -> bool:
    """Is a haze episode under way: did any region read >= the elevated threshold in the last
    HAZE_ACTIVE_HOURS hours of gold? Gates the 6-hourly haze-mode monitoring."""
    from hazecast import config as C
    from hazecast.config import SG_REGIONS
    hours = hours or C.HAZE_ACTIVE_HOURS
    cols = [f"now_{r}" for r in SG_REGIONS]
    gold = pd.read_parquet(GOLD_DIR / "hourly_features.parquet", columns=["date", *cols])
    recent = gold.loc[pd.to_datetime(gold.date) >= pd.to_datetime(gold.date).max() - pd.Timedelta(hours=hours)]
    active = bool((recent[cols] >= C.PM25_ELEVATED_THRESHOLD).to_numpy().any())
    log.info("Haze %s in the last %d h", "active" if active else "not active", hours)
    return active


def choose_branch(action: str) -> str:
    """Map the governance action to the next task id in the monitoring DAG."""
    return {"retrain": "trigger_retrain", "fix_data": "alert_data_failure",
            "investigate": "notify_investigate"}.get(action, "no_action")


# ---------------------------------------------------------------------------
# Training / governance
# ---------------------------------------------------------------------------
def train_challenger(run_id: str | None = None) -> str:
    """Train the six-candidate hourly comparison into a fresh staging directory."""
    from hazecast.hourly_model import train_final_hourly_models
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")))
    out_dir = STAGING_ROOT / safe
    train_final_hourly_models(pd.read_parquet(GOLD_DIR / "hourly_features.parquet"), out_dir=out_dir)
    return str(out_dir)


def replay_holdout(models_dir: str) -> dict:
    """Honest holdout forecasts for the challenger (feeds the gate, baseline and replay dashboard)."""
    from hazecast.monitor_backtest import run_backtest
    return run_backtest(models_dir=Path(models_dir))


def promote(models_dir: str) -> bool:
    from hazecast.governance import promote as _promote
    result = _promote(Path(models_dir))
    log.info("Promotion gate: %s", json.dumps({k: v["passed"] for k, v in result["checks"].items()}))
    return bool(result["promoted"])


def write_baseline() -> str:
    from hazecast.model_monitor import write_production_baseline
    return str(write_production_baseline())


def train_daily() -> dict:
    from hazecast.train import train_final_models
    paths = train_final_models(pd.read_parquet(GOLD_DIR / "daily_features.parquet"))
    return {"artifacts": len(paths)}
