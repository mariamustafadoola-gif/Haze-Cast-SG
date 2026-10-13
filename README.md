# HazeCast SG

Regional PM2.5 haze forecasts for NEA: every morning at 06:00, forecast the average PM2.5
for the next 6, 12 and 24 hours in each of Singapore's five regions.

CS611 Machine Learning Engineering, group project.

## Start the environment

You need Docker Desktop running.

```
docker-compose build
docker-compose up
```

Open http://localhost:8888 in your browser. JupyterLab opens with this folder.
To run a script, open a Terminal inside JupyterLab.

## Branches

| Branch | Who | What |
|---|---|---|
| main | everyone | Working, shared code only. Do not push to it directly; merge through a pull request. |
| data | data team | Data pipeline: ingest, bronze, silver, gold |
| ai | AI team | Training, evaluation, registry, forecasting |

## Data for the AI team

Start with the dummy gold tables. They have the real columns but made-up values.

```python
from utils.load_gold import load_training_table
df = load_training_table(source="dummy")   # switch to "real" later
```

Column meanings, join keys and the agreed time-based split are in `docs/data_contract.md`.
To rebuild the dummy tables: `python scripts/make_dummy_data.py`.

## Folders

| Folder | Contents |
|---|---|
| utils/ | Python modules for each pipeline step |
| scripts/ | One-off helpers, such as the dummy data generator |
| dummy_data/ | Dummy gold tables for the AI team |
| docs/ | Data contract and design notes |
| dags/ | Airflow DAGs (to come) |
| notebooks/ | Exploration notebooks, named with your initials |
| data/raw/ | Raw downloads (not committed to git) |
| datamart/ | Bronze, silver and gold output (not committed to git) |

## Secrets

The NASA FIRMS key goes in a file called `.env` (copy `.env.example`). `.env` is never committed.
