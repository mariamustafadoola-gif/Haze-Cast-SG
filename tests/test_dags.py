"""DAG structure checks. Needs Airflow, so it is skipped outside the Airflow image:
    docker compose run --rm airflow-scheduler python -m pytest /opt/airflow/tests/test_dags.py
"""
from __future__ import annotations
from pathlib import Path

import pytest

pytest.importorskip("airflow")
from airflow.models import DagBag  # noqa: E402

DAGS = Path(__file__).resolve().parents[1] / "dags"


@pytest.fixture(scope="module")
def bag():
    return DagBag(dag_folder=str(DAGS), include_examples=False)


def test_no_import_errors(bag):
    assert bag.import_errors == {}


def test_expected_dags_and_edges(bag):
    assert set(bag.dag_ids) == {"hazecast_backfill", "hazecast_hourly", "hazecast_monitoring", "hazecast_retrain"}
    hourly = bag.get_dag("hazecast_hourly")
    assert hourly.get_task("run_inference").upstream_task_ids == {"features_fresh"}
    monitoring = bag.get_dag("hazecast_monitoring")
    assert monitoring.get_task("decide").downstream_task_ids == {
        "trigger_retrain", "notify_investigate", "alert_data_failure", "no_action"}
    retrain = bag.get_dag("hazecast_retrain")
    assert retrain.get_task("write_baseline").upstream_task_ids == {"promote"}
    # trainings are sequential (CPU oversubscription), and daily can't block the hourly promotion
    assert retrain.get_task("train_daily").upstream_task_ids == {"train_challenger"}
    assert retrain.get_task("replay_holdout").trigger_rule == "all_done"
    assert all(d.max_active_runs == 1 for d in bag.dags.values())
