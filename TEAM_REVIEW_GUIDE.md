# HazeCast SG — Team Review Guide

Repo: `https://github.com/limes0606/hazecast-sg` (confirm with Brandon
whether it's private with you all as collaborators, or public -- check
before assuming either way).

This guide maps the workload split to specific files/sections, and shows
how to get the data and results needed to review your own and each
other's portions on GitHub.

## 1. Clone and set up

```bash
git clone https://github.com/limes0606/hazecast-sg.git
cd hazecast-sg
pip install -r requirements.txt
export PYTHONPATH=src
```

## 2. Getting the data (only if you want to re-run anything)

`data/raw/`, `data/bronze/`, `data/silver/`, `data/gold/*.parquet`, and
`data/models/*` are **not** in the repo (see `.gitignore`) -- licences/size,
not secrecy. Full download steps for the 3 real sources (NEA PM2.5, Open-Meteo
weather, NASA FIRMS) are in [`data/README.md`](data/README.md). After
downloading, reproduce everything with:

```bash
python -m hazecast.ingest.pm25 && python -m hazecast.ingest.firms && python -m hazecast.ingest.weather
python -m hazecast.bronze_to_silver
python -m hazecast.silver_to_gold
python -m hazecast.train      # writes data/models/eval_summary.csv + models
python -m hazecast.score
```

**You don't have to do this just to review.** `docs/TECHNICAL_GUIDE.md`'s "Results"
section and the committed `notebooks/pipeline_walkthrough.ipynb` already
contain real, already-executed output (row counts, eval tables, the
forecast) from a real run -- read those first.

## 3. EDA + medallion architecture (2 pax)

Files to review:
- `src/hazecast/config.py` -- regions, fire-region boxes, weather points, thresholds, haze seasons (every assumption is commented)
- `src/hazecast/ingest/pm25.py`, `ingest/firms.py`, `ingest/weather.py` -- the 3 source ingestions
- `src/hazecast/bronze_to_silver.py` -- cleaning, gap-flagging, hourly alignment
- `src/hazecast/silver_to_gold.py` -- daily feature + label table, built leak-free
- `data/README.md` -- data sourcing
- Notebook sections 1-3 (`notebooks/pipeline_walkthrough.ipynb`)

**Gap to flag:** there is currently no standalone EDA notebook/writeup
(label distributions, missingness, correlation plots, skewness) anywhere
in the repo -- only informal profiling referenced in code comments. If
your submission needs a visible EDA artifact, this pair should add one;
it does not exist yet.

## 4. Linear regression (1 pax)

- `src/hazecast/train.py` -- the `linreg` candidate (`LinearRegression().fit(...)`) inside `run_season_fold`
- `docs/TECHNICAL_GUIDE.md` -- "Results" section (MAE / recall / precision tables and the written analysis)
- `data/models/eval_summary.csv` (after running training) -- full per-region numbers, not just national

## 5. XGBoost (1 pax)

- `src/hazecast/train.py` -- `xgboost` (plain MAE) and `xgboost_q85` (quantile, the shipped model) candidates, plus `train_final_models`
- `docs/TECHNICAL_GUIDE.md` -- "Why a quantile model, and how `XGB_QUANTILE_ALPHA=0.85` was chosen" section
- `data/models/eval_summary.csv` and `data/models/xgb_*.json` (saved per-region models, after running training)

## 6. Reviewing each other's work on GitHub

- Browse any file at `github.com/limes0606/hazecast-sg` -- click into it to read.
- **Blame:** open a file, click "Blame" (top-right) to see who changed each line and when.
- **Line comments:** push your changes to a branch and open a Pull Request against `main` instead of committing straight to `main` -- that lets anyone comment on specific lines before it merges.
- **Broader discussion** not tied to one line: open a GitHub Issue.
- **History:** `git log --oneline` (or the "History" view on a file's page) shows every commit.

## Notes

- Nothing under `data/` (except the `.gitkeep` placeholders) is in GitHub -- it regenerates locally when each of you runs the pipeline after downloading the 3 sources yourselves.
- `mlflow.db` and `mlruns/` (experiment tracking) are also local-only and gitignored -- MLflow's UI (`localhost:5000` via `docker compose up`) only shows runs from whoever generated them locally.
- The numbers in `docs/TECHNICAL_GUIDE.md` and the committed notebook are real and already reproducible without anyone re-running anything -- start there before deciding whether you need to regenerate data yourself.
