"""hazecast_retrain: champion/challenger retraining (monthly, or triggered).

Triggered by: the 1st of each month (scheduled refresh), hazecast_monitoring
when the governance rule says "retrain", hazecast_backfill, or a person.

build_layers -> train_challenger -> train_daily -> replay_holdout -> promote? -> write_baseline -> dashboards

train_challenger  six-candidate forward-validated comparison per region x horizon,
                  published to data/models/hourly_staging/<run_id>/ (not production)
replay_holdout    honest holdout forecasts + monitor replay for the challenger
promote           governance gate (hazecast.governance): integrity, skill vs
                  persistence, no regression vs the champion. If it fails the run
                  stops here (skipped) and the champion keeps serving.
write_baseline    monitoring baseline for the newly promoted champion

The two trainings run one after the other on purpose: each uses every core
(XGBoost/LightGBM OpenMP threads, scikit-learn joblib workers), and running
them side by side oversubscribed the CPUs so badly that the hourly comparison
slowed down ~20x in testing. replay_holdout runs even if train_daily fails
(trigger_rule="all_done"), so the secondary daily model never blocks the
hourly promotion.
"""
from __future__ import annotations

from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

from _common import DEFAULT_ARGS, START, TAGS


def _build():
    from hazecast.orchestration import build_layers
    return build_layers()


def _train(run_id):
    from hazecast.orchestration import train_challenger
    return train_challenger(run_id)


def _replay(ti):
    from hazecast.orchestration import replay_holdout
    return replay_holdout(ti.xcom_pull(task_ids="train_challenger"))


def _promote(ti):
    from hazecast.orchestration import promote
    return promote(ti.xcom_pull(task_ids="train_challenger"))


def _baseline():
    from hazecast.orchestration import write_baseline
    return write_baseline()


def _dashboards():
    from hazecast.orchestration import dashboards
    return dashboards()


def _daily():
    from hazecast.orchestration import train_daily
    return train_daily()


with DAG(
    dag_id="hazecast_retrain",
    description="Train challenger, gate it against the champion, promote, refresh monitoring baseline",
    default_args={**DEFAULT_ARGS, "retries": 1, "retry_delay": timedelta(minutes=15)},
    schedule="0 3 1 * *",  # 03:00 SGT on the 1st of each month
    start_date=START,
    catchup=False,
    max_active_runs=1,  # single training publisher
    tags=TAGS,
) as dag:
    build = PythonOperator(task_id="build_layers", python_callable=_build)
    train = PythonOperator(task_id="train_challenger", python_callable=_train, execution_timeout=timedelta(hours=6))
    replay = PythonOperator(task_id="replay_holdout", python_callable=_replay, execution_timeout=timedelta(hours=2),
                            trigger_rule="all_done")
    gate = ShortCircuitOperator(task_id="promote", python_callable=_promote)
    baseline = PythonOperator(task_id="write_baseline", python_callable=_baseline)
    dash = PythonOperator(task_id="dashboards", python_callable=_dashboards)
    daily = PythonOperator(task_id="train_daily", python_callable=_daily, execution_timeout=timedelta(hours=3))

    build >> train >> daily >> replay >> gate >> baseline >> dash
