"""Shared settings for the HazeCast SG DAGs (not a DAG itself)."""
from __future__ import annotations

import logging
from datetime import timedelta

import pendulum

SGT = pendulum.timezone("Asia/Singapore")
START = pendulum.datetime(2026, 10, 1, tz=SGT)  # cron schedules in the DAGs are Singapore time
TAGS = ["hazecast-sg"]

log = logging.getLogger("hazecast.dags")


def on_failure(context):
    """Single place to hook up paging / Slack / email; logs loudly for now."""
    ti = context.get("task_instance")
    log.error("HazeCast task failed: %s.%s (run %s, try %s)", ti.dag_id, ti.task_id,
              context.get("run_id"), ti.try_number)


DEFAULT_ARGS = {
    "owner": "hazecast-sg",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": on_failure,
}
