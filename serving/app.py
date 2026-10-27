"""Minimal FastAPI serving layer for HazeCast SG.

This does NOT score on request -- scoring happens once a day in the
Airflow DAG (see dags/hazecast_daily_dag.py -> hazecast.score). The API
just serves the most recently written forecast file cheaply, matching the
proposal's "one scoring run a day, light read traffic" usage pattern.

Run locally with: uvicorn serving.app:app --reload
Run in Docker via the `api` service in docker-compose.yml.
"""
from __future__ import annotations
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from hazecast.config import GOLD_DIR, PM25_ELEVATED_THRESHOLD

app = FastAPI(title="HazeCast SG", version="0.1.0")

FORECAST_PATH = GOLD_DIR / "latest_forecast.json"
HOURLY_FORECAST_PATH = GOLD_DIR / "latest_hourly_forecast.json"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecast")
def get_forecast():
    """Return the latest hourly forecast (6/12/24h ahead) for all regions."""
    if not HOURLY_FORECAST_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="No forecast available yet -- has hazecast.score been run?",
        )
    return json.loads(HOURLY_FORECAST_PATH.read_text())


@app.get("/forecast/daily")
def get_daily_forecast():
    """Return the latest daily forecast for all regions (original endpoint)."""
    if not FORECAST_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="No forecast available yet -- has the daily pipeline run?",
        )
    return json.loads(FORECAST_PATH.read_text())


@app.get("/forecast/{region}")
def get_region_forecast(region: str):
    """Return every horizon's forecast for one region (the hourly forecast
    has up to 3 rows per region -- 6/12/24h ahead -- not one)."""
    data = get_forecast()
    rows = [row for row in data if row["region"] == region]
    if not rows:
        raise HTTPException(status_code=404, detail=f"No forecast for region '{region}'")
    return rows


@app.get("/config")
def get_config():
    """Expose the decision threshold so the dashboard/alerting stays in
    sync with what the model was evaluated against."""
    return {"elevated_threshold_ug_m3": PM25_ELEVATED_THRESHOLD}


# Dashboard: static HTML/JS that calls the endpoints above (same origin,
# no CORS needed). Served last so it never shadows an API route.
app.mount("/ui", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")
