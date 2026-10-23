"""Airflow DAG: the nightly HazeCast SG pipeline.

fetch_live -> ingest (pm25, firms, weather) -> bronze -> silver -> gold -> score

fetch_live (hazecast.ingest.live) downloads the latest PM2.5, weather and
fire data into data/raw/; the ingest tasks then read data/raw/ as before.
It needs FIRMS_MAP_KEY in the environment (passed in from .env by
docker-compose).

Retraining is a SEPARATE dag (hazecast_retrain_dag.py, monthly / drift-
triggered per the proposal) -- deliberately not run every night, since
same-day retraining on a single new label would be noisy and slow.

The task functions are covered by the tests in tests/. Check the DAG
structure with `airflow dags test hazecast_daily` inside Docker.
"""
from __future__ import annotations
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def _fetch_live():
    from hazecast.ingest.live import fetch_all
    fetch_all(days=10)


def _ingest_pm25():
    from hazecast.ingest.pm25 import load_pm25_bronze, write_bronze
    write_bronze(load_pm25_bronze())


def _ingest_firms():
    from hazecast.ingest.firms import load_firms_bronze, write_bronze
    write_bronze(load_firms_bronze())


def _ingest_weather():
    from hazecast.ingest.weather import load_weather_bronze, write_bronze
    write_bronze(load_weather_bronze())


def _bronze_to_silver():
    from hazecast.bronze_to_silver import pm25_silver, weather_silver, firms_daily_silver, write_silver
    write_silver(pm25_silver(), "pm25")
    write_silver(weather_silver(), "weather")
    write_silver(firms_daily_silver(), "firms_daily")


def _silver_to_gold():
    import pandas as pd
    from hazecast.config import SILVER_DIR
    from hazecast.silver_to_gold import build_gold, write_gold, build_gold_hourly, write_gold_hourly
    pm25 = pd.read_parquet(SILVER_DIR / "pm25.parquet")
    weather = pd.read_parquet(SILVER_DIR / "weather.parquet")
    firms = pd.read_parquet(SILVER_DIR / "firms_daily.parquet")
    write_gold(build_gold(pm25, weather, firms))
    write_gold_hourly(build_gold_hourly(pm25, weather, firms))


def _score():
    from hazecast.score import score_latest, write_forecast
    write_forecast(score_latest())


default_args = {
    "owner": "hazecast-sg",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="hazecast_daily",
    description="Nightly ingest -> bronze -> silver -> gold -> score for HazeCast SG",
    default_args=default_args,
    schedule="30 23 * * *",  # 23:30 SGT, per the proposal's framing choice
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["hazecast-sg"],
) as dag:

    fetch_live = PythonOperator(task_id="fetch_live", python_callable=_fetch_live)
    ingest_pm25 = PythonOperator(task_id="ingest_pm25", python_callable=_ingest_pm25)
    ingest_firms = PythonOperator(task_id="ingest_firms", python_callable=_ingest_firms)
    ingest_weather = PythonOperator(task_id="ingest_weather", python_callable=_ingest_weather)

    bronze_to_silver = PythonOperator(task_id="bronze_to_silver", python_callable=_bronze_to_silver)
    silver_to_gold = PythonOperator(task_id="silver_to_gold", python_callable=_silver_to_gold)
    score = PythonOperator(task_id="score", python_callable=_score)

    fetch_live >> [ingest_pm25, ingest_firms, ingest_weather]
    [ingest_pm25, ingest_firms, ingest_weather] >> bronze_to_silver >> silver_to_gold >> score
