# HazeCast SG

Forecasting tomorrow's PM2.5 in each region of Singapore from regional
fire hotspots, wind and rain. CS611 Machine Learning Engineering group
project.

This is a **proof of concept**, built and validated end-to-end against
real downloaded data (not synthetic placeholders) before being handed to
the team. What's real vs. what still needs work is called out explicitly
in [What's tested vs. not](#whats-tested-vs-not) below -- please read that
section before assuming a piece is production-ready.

## Business problem

Transboundary haze from Indonesian fires periodically pushes Singapore's
PM2.5 into unhealthy territory (2015, 2019, and the ongoing Sep 2026
episode being the clearest examples), and outdoor operations teams
currently only find out once readings are already high. NEA already
publishes current PSI/PM2.5 readings and a regional outlook -- this
project doesn't try to out-forecast NEA. It turns those public upwind
signals into an automated, per-region next-day forecast that a specific
operator (an events/construction/schools operations team) can act on,
which today requires someone manually checking multiple dashboards and
using judgement.

**Objective:** forecast next-day daily-mean PM2.5 for each of Singapore's
5 NEA regions (north, south, east, west, central), every day, and flag
any region forecast to cross the NEA "Elevated" band (56 ug/m3).

## Architecture

This branch (`e2e-pipeline`) joins the data pipeline (`data` branch) and
the model pipeline (`Model_build_train_inference_monitor` branch) into one
pipeline orchestrated by Airflow:

```
 Sources: data.gov.sg PM2.5 · Open-Meteo weather · NASA FIRMS fires
   |  hazecast.ingest.historical (backfill / daily top-up)   hazecast.ingest.live (hourly)
   v
 data/raw -> bronze (fixed raw) -> silver (hourly grid, short gaps filled) -> gold (hourly + daily features)
   |                                                                            |
   |  hazecast_retrain (monthly / triggered)                                   |  hazecast_hourly (every hour)
   |   six candidates x 6 targets x 3 horizons, forward validation, MLflow     |   freshness gate -> 18 forecasts
   |   -> challenger in staging -> holdout replay -> promotion gate             |   -> forecast ledger (SQLite)
   |   -> champion (data/models/hourly) -> monitoring baseline                  |   -> latest_hourly_forecast.json -> FastAPI
   v                                                                            v
 hazecast_monitoring (daily): NEA actuals -> data quality · model quality · bias drift ·
   feature attribution drift -> dashboard -> governance decision -> (trigger retrain | investigate | fix data)
```

Everything runs locally via `docker compose up` (Airflow + Postgres +
MLflow + the API); see [End-to-end pipeline](#end-to-end-pipeline). The
governance and model-refresh SOP is in
[`docs/MODEL_GOVERNANCE.md`](docs/MODEL_GOVERNANCE.md).

## Repo layout

```
src/hazecast/
  config.py              regions, fire boxes, weather points, thresholds, selected features, monitoring thresholds
  ingest/historical.py   automated history download for training (coverage-driven, resumable)
  ingest/live.py         latest hours from the live APIs for inference
  ingest/pm25.py, firms.py, weather.py   raw files -> bronze
  bronze_to_silver.py    cleaning, short-gap filling, gap flags
  silver_to_gold.py      hourly + daily feature/label tables (leak-free by construction)
  data_quality.py        data-quality report on the medallion layers
  refresh.py             one call: (fetch) -> bronze -> silver -> gold -> data-quality report
  hourly_model.py        six-candidate hourly comparison (+6/12/24 h), forward validation, MLflow
  train.py               same comparison for the next-day daily model
  inference.py           hourly production inference -> forecast ledger
  monitoring.py          verifies archived forecasts against NEA actuals
  model_monitor.py       data quality, model quality, bias drift, attribution drift + retraining rule
  monitor_backtest.py    replays the holdout through the monitors (performance over ~2.5 years)
  dashboard.py           performance & stability dashboard (HTML)
  governance.py          champion/challenger promotion gate, rollback
  orchestration.py       the task functions the DAGs call
serving/app.py            FastAPI: /ui (dashboard), /forecast/hourly, /forecast, /model, /monitoring, /monitoring/dashboard
serving/static/index.html forecast dashboard (map of the 5 regions, +6/12/24 h), with a built-in demo fallback
docs/index.html           the same dashboard for static hosting (GitHub Pages; always demo data)
dags/                     hazecast_backfill, hazecast_hourly, hazecast_monitoring, hazecast_retrain
docker/                   pipeline.Dockerfile (API, Jupyter), airflow.Dockerfile
docs/MODEL_GOVERNANCE.md  deployment choice, monitors, retraining triggers, promotion/rollback SOP
notebooks/                EDA, cleaning/feature selection, pipeline walkthrough
tests/                    fast tests on synthetic fixtures and mocked APIs (no real data needed)
data/README.md            how the 3 datasets are downloaded (automated) and the manual reference steps
```

## Data

Three public sources, documented in full in
[`data/README.md`](data/README.md):

| Source | Grain | History used | Access |
|---|---|---|---|
| [NEA Historical 1-hr PM2.5](https://data.gov.sg/datasets/d_6c0d5fc34838b12472475fdb73c0af29/view) | hourly, 5 regions | 2014-04-01 to 2026-08-01 (validated) | Open Data Licence |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) (VIIRS Suomi-NPP) | per detection | 2019-01-01 to 2026-09-21 (validated); **2014-2018 not yet obtained** | free MAP_KEY |
| [Open-Meteo Historical Weather](https://open-meteo.com/en/docs/historical-weather-api) (ERA5) | hourly, 5 points | 2014-04-01 to 2026-07-31 (validated) | free, non-commercial |

None of the raw files are committed to git -- run the download steps in
`data/README.md` once, locally, before running the pipeline.

## Results (proof of concept, real data)

Leave-one-season-out evaluation, now with the 2014-2018 FIRMS backfill in
place, so **both** haze seasons in `hazecast.config.HAZE_SEASONS` are
evaluated (2015 and 2019 -- 2026 still has no labels past 1 Aug). Four
candidates are compared per region, per season (see `hazecast.train`):
`persistence` (tomorrow = today), `linreg`, `xgboost` (plain
MAE-minimizing regression -- the model in the original proposal), and
`xgboost_q85` (gradient-boosted trees targeting the **85th percentile** of
next-day PM2.5 instead of the mean -- see "Why a quantile model" below).
Full per-fold numbers are in `data/models/eval_summary.csv` after running
training.

**MAE by held-out season** (ug/m3, lower is better; national target shown,
full per-region table in the CSV):

| season | persistence | linreg | xgboost | xgboost_q85 |
|---|---:|---:|---:|---:|
| 2015 | 17.92 | **15.74** | 18.58 | 17.35 |
| 2019 | 6.44 | 5.84 | **5.05** | 7.01 |

**Elevated-day recall (national), the metric that actually matters for
this use case -- of the days that were genuinely elevated, what fraction
did each model flag?**

| season | persistence | linreg | xgboost | xgboost_q85 |
|---|---:|---:|---:|---:|
| 2015 (31 elevated days in the test fold) | 0.68 | 0.61 | 0.13 | **0.45** |
| 2019 (4 elevated days in the test fold) | 0.25 | 0.75 | 0.50 | **1.00** |
| mean across both folds | 0.46 | 0.68 | 0.31 | **0.73** |

**Elevated-day precision (national) -- of the days each model flagged,
what fraction were genuinely elevated:**

| season | persistence | linreg | xgboost | xgboost_q85 |
|---|---:|---:|---:|---:|
| 2015 | 0.68 | 0.68 | 0.67 | 0.70 |
| 2019 | 0.25 | 0.38 | 0.67 | 0.40 |

**Honest read of this, not a sales pitch:** plain XGBoost (mean
regression) systematically under-predicts the rare elevated days --
with haze at ~2% of all days, a mean-minimizer gets pulled toward the
calm majority. `xgboost_q85` was built specifically to test whether
targeting a high quantile instead of the mean fixes this, and on real
data it roughly **triples** XGBoost's 2015 recall (0.13 -> 0.45) and
reaches **perfect** recall on the 2019 fold (0.50 -> 1.00), at a real
but modest precision cost (more false alarms) and a season-dependent
MAE change (better on 2015, worse on 2019). Averaged across both folds,
`xgboost_q85` (0.73) edges out `linreg` (0.68) on recall.

That said, this is **not a clean win worth overselling**: the two folds
disagree on which model is better, and 2015 -- the fold with by far the
larger, more statistically reliable sample of elevated days (31, vs.
2019's 4) -- is actually won on recall by `linreg`, not `xgboost_q85`.
2019's "perfect recall" is 4-for-4, which one region flipping the wrong
way would erase; treat it as encouraging, not proven. Two haze seasons is
a small sample for a rare event either way, and no model's ranking should
be treated as final.

A Gamma-distributed GLM with a log link was also tried as a fifth
candidate, on the reasoning that PM2.5 labels are strictly positive and
heavily right-skewed. It produced the best elevated-day recall of any
candidate on the 2015 fold, but its predictions blew up to physically
implausible magnitudes (into the thousands of ug/m3) on that same fold --
2015's fire-hotspot values fall far outside the training range, and a
log-link model's `exp(...)` compounds that extrapolation error rather
than degrading gracefully like a tree model would. Not usable as a
forecast number, so it was dropped rather than shipped or kept as a
documented baseline.

**What's shipped:** `hazecast.train.train_final_models` fits and saves
`xgboost_q85` as the production model (loaded by `hazecast.score`) --
chosen because it's now competitive with `linreg` on average and
meaningfully fixes plain XGBoost's worst failure mode, and is the model
family named in the original proposal, tuned to address its specific
documented weakness rather than replaced outright. This is a defensible
choice, not the *only* defensible choice -- swapping the shipped model to
`linreg` is one config change (see `train_final_models`) and would also
be justifiable given the 2015 numbers above. Document whichever the team
picks, and why, in the final report -- that choice, backed by these
numbers, is exactly what the brief rewards.

**Why a quantile model, and how `XGB_QUANTILE_ALPHA=0.85` was chosen:**
a quantile regressor targeting the 85th percentile predicts "the value
this day's PM2.5 has an 85% chance of being at or below" rather than the
expected value -- for a right-skewed quantity like PM2.5 (mostly low,
occasionally very high), that systematically biases predictions upward,
which is exactly the direction that helps catch elevated days at the
cost of more false alarms. 0.85 wasn't guessed: quantiles from 0.70 to
0.95 were swept against real data (see git history / ask Brandon for the
sweep script), and 0.70-0.80 caught meaningfully fewer elevated days
while 0.90-0.95 added false alarms without a matching recall gain over
0.85-0.90. This is a tunable trade-off, not a fixed constant -- lower
`XGB_QUANTILE_ALPHA` for fewer false alarms (more conservative alerts),
raise it to catch more true events at the cost of more false ones,
depending on what the eventual end-user (the events/construction/schools
operations team from the business problem above) actually needs.

Per the module brief: a simple, honestly-evaluated model with rigorous
MLOps is the goal here, not a leaderboard score. Both `linreg` and
`xgboost_q85` are defensible choices for the decision-support objective;
document whichever is shipped, and the recall numbers behind it, in the
final report.

## End-to-end pipeline

**With Docker (recommended):**
```bash
cp .env.example .env              # add FIRMS_MAP_KEY; change *_PORT if 8080/5000/8000 are taken
docker compose up -d --build      # postgres, airflow (init/webserver/scheduler), mlflow, api
```
1. Airflow at `localhost:8080` (admin / admin). Trigger **`hazecast_backfill`** once.
   It downloads whatever history `data/raw/` is missing, rebuilds the layers and
   triggers **`hazecast_retrain`** (about an hour on a laptop). That run trains,
   replays the holdout, promotes the first champion and writes the monitoring baseline.
2. After that, **`hazecast_hourly`** issues 18 forecasts every hour, and
   **`hazecast_monitoring`** runs the four monitors each night and triggers
   retraining when the governance rule says so.
3. Results: the **forecast dashboard at `localhost:8000`** (redirects to `/ui/`), `localhost:8000/forecast/hourly`, `/monitoring`,
   `/monitoring/dashboard` (live) and `/monitoring/dashboard?kind=backtest`
   (holdout replay); MLflow at `localhost:5000`.

**Without Docker, the same steps by hand:**
```bash
pip install -r requirements.txt && export PYTHONPATH=src
python -m hazecast.ingest.historical          # training history (only what's missing)
python -m hazecast.refresh --days 3           # live hours + bronze/silver/gold
python -m hazecast.hourly_model               # train + select 18 models (straight to production)
python -m hazecast.monitor_backtest           # holdout replay -> reports/monitoring/backtest_dashboard.html
python -m hazecast.model_monitor baseline     # monitoring baseline for the champion
python -m hazecast.inference                  # 18 forecasts -> ledger + latest_hourly_forecast.json
python -m hazecast.monitoring                 # NEA actuals for archived forecasts (next day)
python -m hazecast.model_monitor run          # four monitors + governance decision
python -m hazecast.dashboard                  # reports/monitoring/dashboard.html
```

Model refresh, promotion and rollback are in
[`docs/MODEL_GOVERNANCE.md`](docs/MODEL_GOVERNANCE.md)
(`python -m hazecast.governance status | promote | rollback`).

## Running it

**1. Get the data** -- see [`data/README.md`](data/README.md).

**2. Run the pipeline locally (fastest way to iterate):**
```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m hazecast.ingest.pm25 && python -m hazecast.ingest.firms && python -m hazecast.ingest.weather
python -m hazecast.bronze_to_silver
python -m hazecast.silver_to_gold
python -m hazecast.train      # writes data/models/eval_summary.csv + the final models
python -m hazecast.score      # writes data/gold/latest_forecast.json
uvicorn serving.app:app --reload   # then GET localhost:8000/forecast
```

**3. Run the fast smoke tests** (no real data needed):
```bash
pytest tests/ -v
```

**4. Or run the same 5 steps interactively in JupyterLab** (same pattern as
Assignment 1 -- Docker + docker-compose + browser, no local Python install
needed):
```bash
docker compose up --build jupyter
```
Then open `localhost:8888/lab` (no token -- local dev only, don't expose
this port beyond localhost) and open `notebooks/pipeline_walkthrough.ipynb`.
It runs ingest -> bronze -> silver -> gold -> train -> score one cell at a
time, printing row counts/date ranges/eval tables as it goes, plus an
optional last cell for the smoke tests. Already executed once end-to-end
against the real data in this proof of concept, so the committed notebook's
outputs are real, not placeholders -- rerun it yourself to confirm on your
own data. (Prefer a local JupyterLab instead of Docker? `pip install -r
requirements.txt && jupyter lab`, then open the same notebook -- it detects
either environment automatically.)

**5. Run the full stack in Docker (all services):**
```bash
docker compose up --build
```
MLflow UI: `localhost:5000`. Airflow UI: `localhost:8080` (see the note
at the top of `docker-compose.yml` about first-run DB init). API:
`localhost:8000/forecast`. JupyterLab: `localhost:8888/lab`.

## Data Pipeline

Owned by the data team (Maria, Anu). **Status: complete** on the `data` branch.
Data currently runs from April 2014 to the latest live refresh.

| Layer | What it does | Code |
|---|---|---|
| Bronze | Reads raw files and fixes known errors: broken timestamps, repeated hours, repeated fire records, fires outside Sumatra and Kalimantan, gas flares and volcanoes | `src/hazecast/ingest/` |
| Silver | One complete hourly clock; gaps of up to 3 hours filled; satellite gaps in the fire data flagged, never filled | `src/hazecast/bronze_to_silver.py` |
| Gold | Hourly and daily model tables, 56 candidate features each | `src/hazecast/silver_to_gold.py` |
| Feature selection | 56 features cut to 36 (hourly) and 37 (daily) with the same accuracy; training uses them automatically | `config.py` (`SELECTED_FEATURES_HOURLY`, `USE_SELECTED_FEATURES`) |
| Data quality | Counts every fix and gap, runs 7 checks | `src/hazecast/data_quality.py` |
| Live refresh | Downloads the latest PM2.5 (data.gov.sg), weather (Open-Meteo) and fires (NASA FIRMS), then rebuilds bronze, silver and gold | `src/hazecast/ingest/live.py`, `src/hazecast/refresh.py` |

Two notebooks explain every step with charts: `notebooks/01_eda.ipynb` (what the data shows)
and `notebooks/02_cleaning_and_feature_selection.ipynb` (how it was cleaned and why 36 features).

### Refreshing to Live Data

No local Python needed; this uses the project's Docker image.

1. Copy `.env.example` to `.env` and add a free NASA FIRMS key
   ([get one here](https://firms.modaps.eosdis.nasa.gov/api/map_key/)). `.env` is never committed.
2. Start Docker Desktop, then from the project folder: `docker compose up --build jupyter`
3. Open `localhost:8888/lab`, then **File > New > Terminal**, and run:
   ```bash
   python -m hazecast.refresh --days 10
   ```
4. It ends with the newest hour that is ready to forecast from. Verified on 23 October 2026:
   all three sources downloaded and the gold table was current to 16:00 SGT that day.

The nightly Airflow DAG (`dags/hazecast_daily_dag.py`) runs the same download as its first task.
Use `--no-fetch` to rebuild from the files already in `data/raw/`.

### Next Steps for the Model Team

1. Add an hourly forecast step that predicts 1 and 6 hours ahead from the newest row of
   `data/gold/hourly_features.parquet` (`score.py` currently scores the daily model only).
2. Re-check `HOURLY_TUNED_SETTINGS` on the 36 selected features (they were tuned on all 56).
3. Demo days (3 and 4 November): run the live refresh, then the forecast. Rehearse on 2 November.
4. Decide whether the 2026 haze season becomes a third test season once it ends. It was kept
   out of feature selection so that test stays fair.

## What's tested vs. not

### End-to-end pipeline (`e2e-pipeline` branch, verified 2026-10-25)

- **Automated history download** (`hazecast.ingest.historical`): run against the real data.gov.sg and Open-Meteo APIs in both modes. Fresh clone: bulk PM2.5 + per-day API + 3 years of weather, loaded through bronze. Top-up: only the missing days, and a second run requests nothing. FIRMS: mocked tests only (needs a MAP_KEY).
- **Training on the merged code:** `hazecast.hourly_model` on 110,015 real hours (~35 min). It selected `xgboost_q85` in all 18 groups, and the holdout matches the model team's numbers (central +6 h MAE 7.78, F2 0.69).
- **Monitoring:** `hazecast.monitor_backtest` replayed 117 weekly windows over the 2024-26 holdout; thresholds calibrated from it (see `docs/PIPELINE_WALKTHROUGH.md` §4.3).
- **Docker:** both images build. The full stack (Postgres, Airflow 2.10.5 LocalExecutor, MLflow, API) runs, and all four DAGs parse. `hazecast_hourly` and `hazecast_monitoring` ran end to end on schedule, and the freshness gate and `fix_data` branch behaved as designed without a FIRMS key. A manual `hazecast_retrain` trained, replayed, passed the promotion gate, promoted, archived the old champion and rebuilt the baseline (details in `docs/PIPELINE_WALKTHROUGH.md` §4.1).
- **Tests:** 30 pass inside the Airflow image (`docker compose exec airflow-scheduler python -m pytest /opt/airflow/tests`), including DAG structure, promotion/rollback, the monitors, and the history fetcher.
- **Not verified:** live hourly inference end to end (needs `FIRMS_MAP_KEY` for fresh fire features); alert delivery (the failure callback and `notify_investigate` only log, ready to be wired to Slack/email).

### Original proof of concept (historical notes)

This section exists so nobody re-discovers the same gap the hard way.

**Actually run against real downloaded data in this proof of concept:**
- All three ingestion scripts (`hazecast.ingest.*`) -- row counts match
  independent manual profiling of the same files.
- `bronze_to_silver` -- correctly found the same known data quirks
  identified during manual profiling (3 missing PM2.5 hours, ~73 likely
  FIRMS data-gap days including the May 2026 gap).
- `silver_to_gold` -- verified leak-free (label on day D matches
  `today_*` on day D+1 for >99.9% of rows; the tiny remainder is
  low-coverage days) and produces sane feature correlations.
- `train` -- full leave-one-season-out evaluation ran (both the 2015 and
  2019 haze seasons) across all 4 candidates and produced the results
  tables above; the `xgboost_q85` quantile variant's save/load round-trip
  was verified to reproduce identical predictions (XGBoost stores the
  objective in the saved file); models saved and logged to a local
  MLflow (sqlite backend). A Gamma GLM candidate was also tried and
  dropped -- see "Results" above.
- `score` -- produced a real forecast from the latest available day,
  using the `xgboost_q85` production model.
- `serving/app.py` -- smoke-tested with FastAPI's TestClient (health,
  forecast, per-region, 404 path all correct).
- `docker-compose.yml` and `docker/pipeline.Dockerfile` -- YAML validated
  with `docker compose config`; `docker` itself is present.
- The 3 pytest smoke tests in `tests/`.
- `notebooks/pipeline_walkthrough.ipynb` -- executed end-to-end (via
  `jupyter nbconvert --execute`) against the real data; every cell's
  committed output (row counts, eval tables, forecast) is a real result,
  not a placeholder.
- The `jupyter` service in `docker-compose.yml` -- unlike the other
  services (no Docker daemon in the sandbox this was built in), this one
  was actually run with `docker compose up --build jupyter` on a real
  machine. The image built cleanly, but the container exited immediately
  on first run (`Running as root is not recommended. Use --allow-root to
  bypass.`) -- `pipeline.Dockerfile` never creates a non-root user, so
  JupyterLab refused to start. Fixed by adding `--allow-root` to the
  service's command; if you're reading this before re-testing that fix,
  treat it as unconfirmed and re-run `docker compose up --build jupyter`
  yourself.

**Live data:** the live download of all three sources was run end to end on
23 October 2026 (see [Refreshing to Live Data](#refreshing-to-live-data)).

**Not yet verified:**
- Actually building/running the Docker images (`docker build` failed
  here with no daemon available) and the Airflow container specifically
  (its image is large; wasn't pulled here). The DAG files were syntax-
  checked and every function they call was confirmed to exist with a
  matching signature, but the DAG has not executed inside a real
  scheduler.
- ~~Any monitoring/drift job~~ Built on `e2e-pipeline`: `hazecast.model_monitor`
  + `hazecast_monitoring` DAG, which triggers `hazecast_retrain`.
- GitHub Actions / CI -- none set up yet.

## Next steps for the team

1. Decide the shipped model given the recall numbers above -- linreg is
   the current default recommendation for the elevated-day-recall
   objective; if the team wants to keep exploring XGBoost, look at class
   weighting or a quantile/focal-style loss rather than plain MAE
   regression, since the 2015 recall collapse looks like a
   class-imbalance problem, not a "needs more data" problem alone.
2. Build the monitoring job (data quality, PSI/feature drift, 30-day
   rolling error vs. persistence) the proposal describes -- currently
   the biggest structural gap vs. the proposal deck.
3. ~~Verify the live-API ingestion paths~~ Done: see
   [Data Pipeline](#data-pipeline). Still open: run the Airflow container
   itself and confirm the nightly DAG pulls fresh data on schedule.
4. Once the 2026 haze season's PM2.5 labels are complete, add a "2026"
   entry to `hazecast.config.HAZE_SEASONS` and re-run `hazecast.train`
   for a third fold -- no other code changes needed.
