"""hazecast_monitoring: daily model monitoring and governance decision.

topup_history -> collect_actuals -> model_monitor -> dashboards -> decide
             \                                                     |-> trigger_retrain
              '-> daily_forecast                                   |-> notify_investigate
                                                                   |-> alert_data_failure
                                                                   '-> no_action

topup_history    makes the last 10 days permanent in data/raw/ (live files are overwritten)
collect_actuals  NEA actuals for the last 3 days -> forecast ledger (model-quality input)
model_monitor    data quality, model quality, bias drift, feature attribution drift vs the
                 training baseline; writes data/monitoring/latest_model_monitor.json
decide           the retraining rule in docs/MODEL_GOVERNANCE.md
daily_forecast   the next-day daily-mean model (original project objective)
"""
from __future__ import annotations

import logging
from datetime import timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from _common import DEFAULT_ARGS, START, TAGS


def _topup():
    from hazecast.orchestration import fetch_history
    return fetch_history(lookback_days=10)


def _actuals():
    from hazecast.orchestration import collect_actuals
    return collect_actuals(lookback_days=3)


def _monitor():
    from hazecast.orchestration import model_monitor
    return model_monitor()


def _dashboards():
    from hazecast.orchestration import dashboards
    return dashboards()


def _daily():
    from hazecast.orchestration import score_daily
    return score_daily()


def _decide(ti):
    from hazecast.orchestration import choose_branch
    return choose_branch(ti.xcom_pull(task_ids="model_monitor")["action"])


def _notify(ti):
    report = ti.xcom_pull(task_ids="model_monitor")
    logging.getLogger("hazecast.governance").warning(
        "Model monitor needs a human (%s): %s", report["action"],
        "; ".join(report["reasons"]) or report["statuses"])


with DAG(
    dag_id="hazecast_monitoring",
    description="Daily: top up history, verify forecasts, run model monitors, decide on retraining",
    default_args=DEFAULT_ARGS,
    schedule="30 1 * * *",  # 01:30 SGT, once the previous day's NEA readings are complete
    start_date=START,
    catchup=False,
    max_active_runs=1,
    tags=TAGS,
) as dag:
    topup = PythonOperator(task_id="topup_history", python_callable=_topup, execution_timeout=timedelta(hours=1))
    actuals = PythonOperator(task_id="collect_actuals", python_callable=_actuals)
    monitor = PythonOperator(task_id="model_monitor", python_callable=_monitor)
    dash = PythonOperator(task_id="dashboards", python_callable=_dashboards)
    daily = PythonOperator(task_id="daily_forecast", python_callable=_daily)
    decide = BranchPythonOperator(task_id="decide", python_callable=_decide)
    retrain = TriggerDagRunOperator(task_id="trigger_retrain", trigger_dag_id="hazecast_retrain",
                                    conf={"reason": "model_monitor"})
    investigate = PythonOperator(task_id="notify_investigate", python_callable=_notify)
    data_failure = PythonOperator(task_id="alert_data_failure", python_callable=_notify)
    nothing = EmptyOperator(task_id="no_action")

    topup >> actuals >> monitor >> dash >> decide >> [retrain, investigate, data_failure, nothing]
    topup >> daily
