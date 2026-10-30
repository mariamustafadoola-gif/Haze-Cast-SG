"""hazecast_haze_monitoring: the monitor every 6 hours while a haze episode is under way.

haze_active? -> collect_actuals -> model_monitor -> dashboards -> decide
                                                                 |-> trigger_retrain
                                                                 |-> notify_investigate
                                                                 |-> alert_data_failure
                                                                 '-> no_action

haze_active      any region read >= 56 ug/m3 in the last 24 h (config.HAZE_ACTIVE_HOURS); otherwise
                 the run stops here and the daily 01:30 run in hazecast_monitoring is enough
model_monitor    the same monitors and governance rule as the daily run; persistence counts
                 consecutive runs (so 6 h apart here), and retrains stay limited to one per
                 RETRAIN_COOLDOWN_DAYS, so a long episode does not retrain every few hours
"""
from __future__ import annotations

import logging

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator, ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from _common import DEFAULT_ARGS, START, TAGS


def _active():
    from hazecast.orchestration import haze_active
    return haze_active()


def _actuals():
    from hazecast.orchestration import collect_actuals
    return collect_actuals(lookback_days=1)


def _monitor():
    from hazecast.orchestration import model_monitor
    return model_monitor()


def _dashboards():
    from hazecast.orchestration import dashboards
    return dashboards()


def _decide(ti):
    from hazecast.orchestration import choose_branch
    return choose_branch(ti.xcom_pull(task_ids="model_monitor")["action"])


def _notify(ti):
    report = ti.xcom_pull(task_ids="model_monitor")
    logging.getLogger("hazecast.governance").warning(
        "Haze-mode monitor needs a human (%s): %s", report["action"],
        "; ".join(report["reasons"]) or report["statuses"])


with DAG(
    dag_id="hazecast_haze_monitoring",
    description="During a haze episode: run the model monitor every 6 hours",
    default_args=DEFAULT_ARGS,
    schedule="30 7,13,19 * * *",  # with the daily 01:30 run, every 6 h (SGT)
    start_date=START,
    catchup=False,
    max_active_runs=1,
    tags=TAGS,
) as dag:
    active = ShortCircuitOperator(task_id="haze_active", python_callable=_active)
    actuals = PythonOperator(task_id="collect_actuals", python_callable=_actuals)
    monitor = PythonOperator(task_id="model_monitor", python_callable=_monitor)
    dash = PythonOperator(task_id="dashboards", python_callable=_dashboards)
    decide = BranchPythonOperator(task_id="decide", python_callable=_decide)
    retrain = TriggerDagRunOperator(task_id="trigger_retrain", trigger_dag_id="hazecast_retrain",
                                    conf={"reason": "haze_mode_monitor"})
    investigate = PythonOperator(task_id="notify_investigate", python_callable=_notify)
    data_failure = PythonOperator(task_id="alert_data_failure", python_callable=_notify)
    nothing = EmptyOperator(task_id="no_action")

    active >> actuals >> monitor >> dash >> decide >> [retrain, investigate, data_failure, nothing]
