"""What a model was built with: the feature code, the libraries and the commit.

Recorded in every hourly manifest at training time and compared at inference and
monitoring time. A model fed features built by different code gives wrong forecasts
without any error (the column names still match); a different xgboost version gave a
different F2 for the same model (0.759 vs 0.790, 28 Oct 2026). A mismatch therefore
asks for a retrain through the normal gate (model_monitor.scenario_triggers).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from importlib.metadata import PackageNotFoundError, version

from hazecast import config as C
from hazecast.config import REPO_ROOT

# Everything that decides what a gold feature means.
FEATURE_FILES = ("ingest/pm25.py", "ingest/weather.py", "ingest/firms.py",
                 "bronze_to_silver.py", "silver_to_gold.py")
FEATURE_SETTINGS = ("USE_SELECTED_FEATURES", "SELECTED_FEATURES_HOURLY", "FIRE_REGIONS", "WEATHER_POINTS",
                    "MAX_FILL_HOURS", "HOTSPOT_LAGS", "ROLLING_WINDOWS", "HOURLY_RAIN_ROLLING_WINDOWS",
                    "FIRE_LATENCY_HOURS", "FIRE_KNOWN_HOURS_AFTER_LAST_DETECTION", "FIRE_CARRY_FORWARD_HOURS",
                    "FIRE_DAY_AVAILABLE_HOUR_SGT")
# Libraries whose version can change a model's predictions.
MODEL_LIBRARIES = ("xgboost", "lightgbm", "scikit-learn", "numpy", "pandas")


def feature_fingerprint() -> str:
    """Short hash of the feature-building code and settings. Line endings are normalised, so a
    Windows and a Linux checkout of the same commit agree. Any edit to these files, comments
    included, changes it: a deliberately cautious trigger (the cost is one gated retrain)."""
    h = hashlib.sha256()
    src = REPO_ROOT / "src" / "hazecast"
    for name in FEATURE_FILES:
        h.update(name.encode())
        h.update((src / name).read_bytes().replace(b"\r\n", b"\n"))
    h.update(json.dumps({k: getattr(C, k) for k in FEATURE_SETTINGS}, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def library_versions() -> dict[str, str | None]:
    out = {}
    for name in MODEL_LIBRARIES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=5, check=True).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None   # no git in the container, or not a checkout


def record() -> dict:
    """The provenance block written into a manifest."""
    return dict(feature_fingerprint=feature_fingerprint(), library_versions=library_versions(), git_commit=git_commit())


def differences(manifest: dict) -> list[str]:
    """What differs between how the manifest's models were built and the code running now.
    Manifests from before provenance was recorded report nothing."""
    out = []
    recorded = manifest.get("feature_fingerprint")
    if recorded and recorded != feature_fingerprint():
        out.append("feature code or settings changed since training")
    for lib, was in (manifest.get("library_versions") or {}).items():
        now = library_versions().get(lib)
        if was and now and was != now:
            out.append(f"{lib} {was} -> {now}")
    return out
