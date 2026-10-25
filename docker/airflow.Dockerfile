# Airflow image with the pipeline's dependencies baked in (instead of
# _PIP_ADDITIONAL_REQUIREMENTS, which reinstalls on every container start).
ARG AIRFLOW_VERSION=2.10.5
FROM apache/airflow:${AIRFLOW_VERSION}-python3.11
ARG AIRFLOW_VERSION

USER root
# libgomp1: OpenMP runtime needed by LightGBM and XGBoost wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow

COPY requirements-pipeline.txt /tmp/requirements-pipeline.txt
# Pinning apache-airflow in the same install stops pip from upgrading any of
# Airflow's own dependencies to versions it doesn't support.
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements-pipeline.txt
