"""hazecast_backfill: one-off (manual) historical download for model training.

Trigger from the Airflow UI ("Trigger DAG w/ config") or
    airflow dags trigger hazecast_backfill --conf '{"start": "2014-04-01"}'

fetch_history -> build_layers -> retrain_requested? -> trigger hazecast_retrain

fetch_history only downloads what data/raw/ is missing (see
hazecast.ingest.historical), so re-triggering is safe, and a fresh clone
without the LFS files gets the full 2014-onwards history.
"""
from __future__ import annotations

from datetime import timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from _common import DEFAULT_ARGS, START, TAGS


def _fetch(params):
    from hazecast.orchestration import fetch_history
    return fetch_history(start=params.get("start"), end=params.get("end"))


def _build():
    from hazecast.orchestration import build_layers
    return build_layers()


def _retrain_requested(params):
    return bool(params.get("retrain", True))


with DAG(
    dag_id="hazecast_backfill",
    description="Download historical PM2.5, weather and fire data, rebuild layers, then retrain",
    default_args=DEFAULT_ARGS,
    schedule=None,
    start_date=START,
    catchup=False,
    max_active_runs=1,
    tags=TAGS,
    params={
        "start": Param(None, type=["null", "string"], description="first day YYYY-MM-DD (default HISTORY_START)"),
        "end": Param(None, type=["null", "string"], description="last day YYYY-MM-DD (default yesterday SGT)"),
        "retrain": Param(True, type="boolean", description="trigger hazecast_retrain afterwards"),
    },
) as dag:
    fetch = PythonOperator(task_id="fetch_history", python_callable=_fetch, execution_timeout=timedelta(hours=3))
    build = PythonOperator(task_id="build_layers", python_callable=_build)
    gate = ShortCircuitOperator(task_id="retrain_requested", python_callable=_retrain_requested)
    retrain = TriggerDagRunOperator(task_id="trigger_retrain", trigger_dag_id="hazecast_retrain",
                                    conf={"reason": "backfill"})

    fetch >> build >> gate >> retrain
