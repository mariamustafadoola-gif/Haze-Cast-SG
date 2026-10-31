"""FastAPI serving layer for HazeCast SG.

This does NOT score on request -- batch scoring happens in Airflow
(dags/hazecast_hourly_dag.py -> hazecast.inference every hour, and
dags/hazecast_monitoring_dag.py -> the daily next-day model). The API serves
the most recently written files cheaply: the deployment pattern is batch
inference behind a read-only endpoint (see docs/MODEL_GOVERNANCE.md,
"Deployment options").

The forecast dashboard (serving/static/index.html, designed on the XGB_Tuned
branch) is served at /ui and calls /config, /forecast/hourly and /health on the
same origin. If the API or the forecast file is unavailable it falls back to an
embedded demo snapshot and says so on screen. docs/index.html is the same page
for static hosting (GitHub Pages), where it always runs on the demo snapshot.

Run locally with: uvicorn serving.app:app --reload
Run in Docker via the `api` service in docker-compose.yml.
"""
from __future__ import annotations
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from hazecast.config import GOLD_DIR, MODELS_DIR, MONITORING_DIR, PM25_ELEVATED_THRESHOLD, REPO_ROOT
from hazecast import context as ctx

app = FastAPI(title="HazeCast SG", version="0.2.0")

FORECAST_PATH = GOLD_DIR / "latest_forecast.json"
HOURLY_FORECAST_PATH = GOLD_DIR / "latest_hourly_forecast.json"
MONITOR_PATH = MONITORING_DIR / "latest_model_monitor.json"
DASHBOARD_DIR = REPO_ROOT / "reports" / "monitoring"
GOLD_HOURLY_PATH = ctx.GOLD_HOURLY_PATH
METRICS_PATH = ctx.METRICS_PATH


def _read_json(path: Path, missing: str):
    if not path.exists():
        raise HTTPException(status_code=503, detail=missing)
    return json.loads(path.read_text())


@app.get("/", include_in_schema=False)
def root():
    """Open the dashboard by default."""
    return RedirectResponse(url="/ui/")


@app.get("/health")
def health():
    return {"status": "ok"}


_SOURCES_CACHE: dict = {}


@app.get("/health/sources")
def source_status():
    """How fresh each upstream source is (PM2.5, weather per point, fire counts), so the dashboard
    can say which source is holding a forecast back. Cached for a minute; it reads silver."""
    from datetime import datetime, timezone
    from hazecast.model_monitor import source_health
    now = datetime.now(timezone.utc)
    cached = _SOURCES_CACHE.get("value")
    if cached is None or (now - _SOURCES_CACHE["at"]).total_seconds() > 60:
        cached = {"checked_at": now.isoformat(), "sources": source_health(now)}
        _SOURCES_CACHE.update(value=cached, at=now)
    return cached


@app.get("/forecast")
def get_forecast():
    """Return the latest next-day (daily-mean) forecast for all regions."""
    return _read_json(FORECAST_PATH, "No daily forecast available yet -- has the monitoring DAG run?")


@app.get("/forecast/hourly")
def get_hourly_forecast(region: str | None = None, horizon: int | None = None):
    """Latest 6/12/24-hour-ahead forecasts (18 rows), optionally filtered."""
    rows = _read_json(HOURLY_FORECAST_PATH, "No hourly forecast available yet -- has the hourly DAG run?")
    rows = [r for r in rows if (region is None or r["region"] == region)
            and (horizon is None or r["horizon_hours"] == horizon)]
    if not rows:
        raise HTTPException(status_code=404, detail="No hourly forecast for that region/horizon")
    return rows


@app.get("/forecast/{region}")
def get_region_forecast(region: str):
    data = get_forecast()
    for row in data:
        if row["region"] == region:
            return row
    raise HTTPException(status_code=404, detail=f"No forecast for region '{region}'")


@app.get("/model")
def get_model():
    """Which model serves each region x horizon, and when it was published."""
    m = _read_json(MODELS_DIR / "hourly" / "model_manifest.json", "No hourly model published yet")
    return {"published_at": m.get("published_at"), "trained_at": m.get("trained_at"),
            "models": {h: {t: {"model": e["model"], "training_end": e.get("training_end"),
                               "holdout_mae": (e.get("final_test") or {}).get("mae"),
                               "holdout_f2": (e.get("final_test") or {}).get("f2")}
                           for t, e in entries.items()} for h, entries in m["models"].items()}}


@app.get("/monitoring")
def get_monitoring():
    """Latest model-monitor report: four monitor statuses and the governance decision."""
    report = _read_json(MONITOR_PATH, "No monitoring report yet -- has the monitoring DAG run?")
    keep = ("generated_at", "window_start", "window_end", "decision")
    return {**{k: report[k] for k in keep if k in report},
            "statuses": {m: report[m]["status"] for m in ("data_quality", "model_quality", "bias_drift", "attribution_drift")
                         if m in report}}


@app.get("/monitoring/dashboard")
def get_dashboard(kind: str = "live"):
    """The monitoring dashboard (kind=live) or the holdout replay (kind=backtest)."""
    path = DASHBOARD_DIR / ("backtest_dashboard.html" if kind == "backtest" else "dashboard.html")
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"No {kind} dashboard yet")
    return FileResponse(path, media_type="text/html")


@app.get("/config")
def get_config():
    """Expose the decision threshold so the dashboard/alerting stays in
    sync with what the model was evaluated against."""
    return {"elevated_threshold_ug_m3": PM25_ELEVATED_THRESHOLD}


@app.get("/context")
def get_context():
    """What the page shows around the forecast: latest readings, test-period
    accuracy and the monitoring decision."""
    if not GOLD_HOURLY_PATH.exists():
        raise HTTPException(status_code=503, detail="gold hourly table not built yet; run the pipeline")
    try:
        return ctx.build_context(GOLD_HOURLY_PATH, METRICS_PATH, MONITOR_PATH)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))


# Dashboard: static page calling the endpoints above on the same origin (no CORS
# needed). Mounted last so it can never shadow an API route.
app.mount("/ui", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")
