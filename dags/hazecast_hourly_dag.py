"""hazecast_hourly: hourly inference (the production forecast).

fetch_live -> build_layers -> features_fresh? -> run_inference

At 10 past every hour (SGT): pull the last few days from the live APIs,
rebuild bronze/silver/gold, and if the newest complete feature row is fresh
enough, archive 18 forecasts (5 regions + national x +6/12/24 h) in the
forecast ledger and data/gold/latest_hourly_forecast.json, which the API
serves. If features are stale the run stops (skipped, not failed) and no
forecast is issued -- a stale forecast labelled as current is worse than none.
"""
from __future__ import annotations

from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

from _common import DEFAULT_ARGS, START, TAGS

MAX_FEATURE_AGE_HOURS = 2.0


def _fetch():
    from hazecast.orchestration import fetch_live
    return fetch_live(days=3)


def _build():
    from hazecast.orchestration import build_layers
    return build_layers()


def _fresh():
    from hazecast.orchestration import features_fresh
    return features_fresh(MAX_FEATURE_AGE_HOURS)


def _infer():
    from hazecast.orchestration import run_inference
    return run_inference(MAX_FEATURE_AGE_HOURS)


with DAG(
    dag_id="hazecast_hourly",
    description="Live fetch -> bronze/silver/gold -> hourly 6/12/24h forecasts",
    default_args={**DEFAULT_ARGS, "retries": 1, "retry_delay": timedelta(minutes=3)},
    schedule="10 * * * *",
    start_date=START,
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=50),
    tags=TAGS,
) as dag:
    fetch = PythonOperator(task_id="fetch_live", python_callable=_fetch)
    build = PythonOperator(task_id="build_layers", python_callable=_build)
    fresh = ShortCircuitOperator(task_id="features_fresh", python_callable=_fresh)
    infer = PythonOperator(task_id="run_inference", python_callable=_infer)

    fetch >> build >> fresh >> infer
