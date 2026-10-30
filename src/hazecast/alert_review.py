"""Threshold review: how many of the monitor's alerts were followed by a real loss in performance?

Run after every haze season and each quarter (docs/MODEL_GOVERNANCE.md §5). For every monitoring
run that asked for a person or a retrain, it checks whether a model-quality alert (errors or
recall against NEA actuals) followed within --follow-days. A signal type with many alerts and
few followed by degradation is a candidate for a stricter limit; one that only fires after
model quality has already dropped adds nothing.

    python -m hazecast.alert_review                                    # live ledger
    python -m hazecast.alert_review --db data/monitoring/backtest.sqlite
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

from hazecast.config import MONITORING_DIR


def review(db_path: Path = MONITORING_DIR / "pipeline.sqlite", follow_days: int = 28) -> pd.DataFrame:
    """One row per signal type: alerts, how many came before or with a model-quality alert within
    follow_days, and how many weeks of lead they gave on the first model-quality alert."""
    with sqlite3.connect(db_path) as db:
        runs = pd.read_sql("SELECT window_end, action, reasons FROM monitor_runs ORDER BY window_end", db)
    if runs.empty:
        return pd.DataFrame(columns=["signal", "alerts", "followed_by_performance_loss", "share_followed"])
    runs["window_end"] = pd.to_datetime(runs.window_end, utc=True, format="ISO8601")
    runs["kinds"] = runs.reasons.map(lambda r: [x.split(":")[0] for x in json.loads(r)])
    perf = runs.loc[runs.kinds.map(lambda k: "model_quality" in k), "window_end"]
    rows = []
    alerting = runs.loc[runs.action.isin(["investigate", "retrain"])]
    for kind in sorted({k for ks in alerting.kinds for k in ks}):
        hits = alerting.loc[alerting.kinds.map(lambda ks: kind in ks), "window_end"]
        window = pd.Timedelta(follow_days, unit="D")   # keyword form warns under NumPy 2.4
        followed = [bool(perf.between(t, t + window).any()) for t in hits]
        rows.append(dict(signal=kind, alerts=len(hits), followed_by_performance_loss=sum(followed),
                         share_followed=round(sum(followed) / len(hits), 2) if len(hits) else None))
    return pd.DataFrame(rows).sort_values("alerts", ascending=False, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=MONITORING_DIR / "pipeline.sqlite")
    ap.add_argument("--follow-days", type=int, default=28)
    a = ap.parse_args()
    print(review(a.db, a.follow_days).to_string(index=False))


if __name__ == "__main__":
    main()
