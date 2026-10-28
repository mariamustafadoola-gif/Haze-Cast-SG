"""Shared SQLite forecast ledger. All stored timestamps are UTC with offsets."""
from __future__ import annotations
import json
import math
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

SGT = ZoneInfo("Asia/Singapore")
REGIONS = ("north", "south", "east", "west", "central")
TARGETS = REGIONS + ("national",)
HORIZONS = (6, 12, 24)


def timestamp(value):
    result = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        result = result.replace(tzinfo=SGT)
    return result.astimezone(timezone.utc)


def iso(value):
    return timestamp(value).isoformat(timespec="microseconds")


def finite(value):
    try:
        return not isinstance(value, bool) and math.isfinite(float(value)) and float(value) >= 0
    except (ValueError, TypeError):
        return False


def connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=60)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript('''
        CREATE TABLE IF NOT EXISTS forecasts (
            forecast_id TEXT PRIMARY KEY,
            issued_at TEXT NOT NULL, as_of_time TEXT NOT NULL,
            forecast_time TEXT NOT NULL, region TEXT NOT NULL,
            horizon_hours INTEGER NOT NULL, model TEXT NOT NULL,
            model_run_id TEXT, manifest_sha256 TEXT NOT NULL,
            forecast_pm25 REAL NOT NULL, threshold REAL NOT NULL,
            elevated_flag INTEGER NOT NULL, feature_age_hours REAL NOT NULL,
            persistence_pm25 REAL NOT NULL,
            UNIQUE(as_of_time, region, horizon_hours)
        );
        CREATE INDEX IF NOT EXISTS idx_forecast_time ON forecasts(forecast_time);
        CREATE TABLE IF NOT EXISTS actuals (
            observation_time TEXT NOT NULL, region TEXT NOT NULL,
            actual_pm25 REAL, source TEXT NOT NULL,
            source_updated_at TEXT NOT NULL, fetched_at TEXT NOT NULL,
            PRIMARY KEY(observation_time, region)
        );
    ''')
    # Added after the first archives were written: how long the fire features had been
    # carried forward because FIRMS was late (0 = current).
    if "fire_carried_hours" not in {r["name"] for r in db.execute("PRAGMA table_info(forecasts)")}:
        db.execute("ALTER TABLE forecasts ADD COLUMN fire_carried_hours REAL NOT NULL DEFAULT 0")
    return db


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
