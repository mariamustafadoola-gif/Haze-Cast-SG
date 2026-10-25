# Shared image for the scoring API and JupyterLab -- both just need the
# hazecast package and its dependencies on PYTHONPATH. Which one runs is
# chosen by `command` in docker-compose.yml. (Airflow uses airflow.Dockerfile.)
FROM python:3.11-slim

# libgomp1: OpenMP runtime needed by LightGBM and XGBoost wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-pipeline.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY serving/ serving/
ENV PYTHONPATH=/app/src:/app

# No CMD here on purpose -- docker-compose.yml sets one per service.
