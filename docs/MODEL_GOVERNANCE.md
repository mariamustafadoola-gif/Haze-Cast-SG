# HazeCast SG: model governance and refresh SOP

This document covers how the production PM2.5 forecaster is deployed,
monitored, refreshed and rolled back, and who decides what. Every
threshold named here is a constant in `src/hazecast/config.py` (section
"Model monitoring + governance"), so the code and this document can't
drift apart. Change a threshold in config, then update this page in the
same pull request.

## 1. What is in production

| Item | Value |
|---|---|
| Product | PM2.5 forecasts **6, 12 and 24 hours ahead** for the 5 NEA regions + national (18 forecasts per hour), with an *elevated* flag at 56 µg/m³ |
| Models | One model per region × horizon, chosen from six candidates (persistence, ridge, random forest, XGBoost, LightGBM, XGBoost q85) by forward validation: pooled F2 on elevated hours first, then MAE and RMSE (`hazecast.hourly_model`) |
| Champion | `data/models/hourly/model_manifest.json` plus the `.joblib` artifacts it names. This is the only directory inference reads |
| Secondary product | Next-day daily-mean forecast (`hazecast.train` / `hazecast.score --pipeline daily`), retrained alongside but not gated |
| Tracking | MLflow experiments `hazecast-sg-hourly-forward` (hourly) and `hazecast-sg-forward` (daily): every CV fold, holdout score and final fit |
| Forecast ledger | `data/monitoring/pipeline.sqlite`: every issued forecast (first-issued wins, never overwritten), NEA actuals, and every monitor metric over time |

## 2. Deployment options and the choice made

| Option | How it would work | Verdict |
|---|---|---|
| **Batch inference + read-only API** *(chosen)* | Airflow scores once an hour after new data lands, writes the ledger and `latest_hourly_forecast.json`; FastAPI serves the file | The inputs (NEA hourly PM2.5, Open-Meteo hourly weather, daily-ish FIRMS) only change once an hour, so scoring on request adds nothing. It is cheap, every forecast is reproducible from the ledger, and the API never touches model code |
| Real-time endpoint (model behind `/predict`) | Load the 18 models in the API process and score each request | No benefit while features update hourly. It adds latency, scaling and model-loading concerns, plus a second code path that could diverge from what is monitored |
| Managed endpoint (SageMaker / Vertex / Azure ML) | Register the model, deploy an endpoint, use the platform's Model Monitor | The natural next step if the project moves to the cloud. The four monitors here deliberately mirror SageMaker Model Monitor's four types (data quality, model quality, bias drift, feature attribution drift), so the move is mostly plumbing |
| Shadow / canary | Score the challenger alongside the champion for N days before switching | Supported cheaply because forecasts are batch: see §6, "Optional shadow period" |

## 3. Pipelines (Airflow DAGs)

| DAG | Schedule (SGT) | What it does |
|---|---|---|
| `hazecast_backfill` | manual | Downloads any missing history (2014-04 onwards) from data.gov.sg, Open-Meteo and NASA FIRMS, rebuilds bronze/silver/gold, triggers `hazecast_retrain` |
| `hazecast_hourly` | :10 every hour | Live fetch, then bronze/silver/gold, then a **freshness gate** (newest complete feature row ≤ 2 h old), then archives 18 forecasts. Stale features mean **no forecast** (the run is skipped, not failed) |
| `hazecast_monitoring` | 01:30 daily | Makes the last 10 days of raw data permanent, fetches NEA actuals, runs the four monitors, rebuilds the dashboard, applies the **retraining rule** (§5) and branches |
| `hazecast_retrain` | 03:00 on the 1st, or triggered | Trains a challenger in staging, replays its holdout, runs the **promotion gate** (§6), promotes it or keeps the champion, then refreshes the monitoring baseline |

## 4. What is monitored (`hazecast.model_monitor`)

A **baseline** is captured when a model is promoted: feature distributions
and decile bins from the training data, each model's feature attributions,
its holdout error and per-region bias, and **calibrated alert limits**.
Every day the monitors compare recent data against it. All metrics are
stored in the ledger and plotted on the dashboard (`/monitoring/dashboard`).
How every threshold was derived, with the replay evidence, is in
[`PIPELINE_WALKTHROUGH.md` §4.3](PIPELINE_WALKTHROUGH.md#43-thresholds-how-each-was-set-and-the-evidence).

| Monitor | Question | Metrics | Alert when |
|---|---|---|---|
| **Data quality** (30-day window) | Are the model's inputs still healthy and like the training data? | schema, freshness, missing rate, share outside training range, **PSI** vs training deciles | missing column or features > 3 h old (*fail*); ≥ 3 features above their **calibrated PSI limit** (99th percentile of PSI over normal 30-day windows of the training period); missing rate +5 pp or > 5 % out of range |
| **Model quality** (7-day window) | Are forecasts still doing their job, verified against NEA? | MAE, RMSE, F2, recall, MAE skill and **recall of persistence on the same hours** | with ≥ 48 matched hours: MAE > 1.5 × holdout MAE, *or* (with ≥ 12 haze hours) recall more than 0.25 below persistence's. Alert at ≥ 3 of 18 groups. MAE skill vs persistence is reported, not alerted: the q85 models are upward-biased by design |
| **Bias drift** (7-day window) | Is any region (or time of day) served systematically worse? | per-region recall, mean signed error, alert rate vs actual rate; day vs night MAE | recall gap between regions (each ≥ 12 haze hours) > 0.30, or > 0.20 wider than at training; a region's mean signed error shifts > 8 µg/m³ |
| **Feature attribution drift** (30-day window) | Has what the model relies on changed? | label-free permutation attribution; **NDCG@10** vs baseline ranking | NDCG below the model's **calibrated limit** (1st percentile over training-period windows, at most 0.90). Alert at ≥ 3 of 18 models |

Two kinds of monitoring are kept separate on purpose. `hazecast.monitoring`
verifies archived forecasts against NEA actuals; `hazecast.model_monitor`
adds the drift and bias views and makes the retraining decision.

## 5. Retraining triggers: when to refresh the model

The monitoring DAG turns the four monitor results into one action
(`model_monitor.governance_decision`):

| Action | Rule | What happens |
|---|---|---|
| **fix_data** | data-quality *fail*: missing feature columns or stale features | No retraining: a model retrained on broken data would be worse. Data owner investigates the failing source (§7). Inference is already refusing stale features |
| **retrain** | model-quality *alert* (≥ 3 groups), **or** data drift **together with** attribution drift, **or** the champion is older than 35 days | `hazecast_retrain` is triggered automatically; the promotion gate still decides whether anything changes |
| **investigate** | any other *alert*: drift alone, attribution drift alone, bias drift | ML owner reviews the dashboard within 2 working days and decides retrain / threshold change / no-op, recording why |
| **none** | no alerts (warnings are shown on the dashboard only) | nothing |

Evidence from replaying the 2024–26 holdout (117 weekly windows):
- **Overall:** 103 none, 6 investigate, 8 retrain.
- **Investigate:** 4 drift-only alerts in normal periods (about one every 6 months, none followed by degradation). Drift on 10 Aug and bias drift on 17 Aug 2026 led the haze episode by 1–2 weeks.
- **Retrain:** all 8 fell in the 2026 haze episode, when the model really did fall behind.

That is why drift alone asks a human rather than retraining, and why
confirmed performance loss retrains automatically.

**What retraining fixes and what it doesn't** (tested: retrain at the
8 Sep 2026 signal, score the rest of the episode):
- **It fixes calibration:** false alarms 36 % → 12 % at +24 h, and MAE falls.
- **It doesn't fix +24 h recall:** 0.53 against 0.69 for persistence.

During a long episode persistence is hard to beat a day ahead. Hence:

**Decision for the team (recommended, not yet switched on):** for +12 h
and +24 h, set `elevated_flag` when **either** the model or persistence
is ≥ 56 µg/m³. On the 2024–26 holdout, +24 h recall goes 0.56 → 0.74 and
false alarms 1.7 % → 2.0 % of non-haze hours. This changes the meaning of
the public flag, so it needs the approver's sign-off (§7) before the
inference code changes.

Scheduled refresh: monthly (1st, 03:00), so new labels are absorbed even
when nothing alerts. The 2026 haze season becomes a validation period
automatically once its labels are in the development window.

## 6. How a model is refreshed: promotion gate and rollback

1. `train_challenger`: the full six-candidate comparison on all data, written to `data/models/hourly_staging/<run_id>/`. Production is untouched.
2. `replay_holdout`: refits the selected models on pre-holdout data only and replays the holdout through every monitor (`hazecast.monitor_backtest`). This produces honest holdout predictions and the replay dashboard reviewers use.
3. `promote` (`hazecast.governance`); **all** must pass:
   - **Integrity:** 18 entries, every artifact loads and gives a finite prediction on the latest gold row.
   - **Skill:** holdout **F2** within 0.10 of persistence's in ≥ 50 % of the groups with haze hours. F2, not MAE: the q85 models never beat persistence on MAE (0 of 18 groups), so an MAE gate would reject every model. Persistence is a floor, not a target, because it cannot forecast the onset of an episode. Current champion: 15 of 18 within tolerance.
   - **No regression, judged on F2:**
     - F2 within 0.05 of the champion's in at least half the groups.
     - **No single group's F2 falling by more than 0.10.** If one does, the challenger is held for a person to decide (`--force` with a recorded reason), not promoted automatically.
     - MAE at most 2× the champion's in every group, as a blow-up guard.
     The first version of this check used MAE (within 10%). Evaluating the LOSO-tuned model from the `XGB_Tuned` branch showed why that was wrong: it improved F2 in 13 of 18 groups, but a higher quantile forecasts higher by design (MAE up to 1.8×), so the MAE rule rejected it. It also did not single out the one genuine regression (west +6 h, F2 0.779 → 0.628). The F2 rule passes the broad improvement and flags exactly that group.
4. **Pass:** artifacts are copied, the old manifest is archived to `history/`, and the new manifest is written **last**, so inference never sees a half-published model. The monitoring baseline (including the calibrated limits) is rebuilt for the new champion. **Fail:** the champion keeps serving, and the rejection and its reasons go to `promotion_log.jsonl`.
5. **Rollback** (any time, one command, no retraining): `python -m hazecast.governance rollback` restores the previous manifest. Artifacts are never deleted, so any archived version can be restored with `--to`.
6. **Manual override:** `python -m hazecast.governance promote --candidate <dir> --force` promotes despite a failed gate. Only the ML owner may use it, and the reason must be written in the log entry and the team channel.

**Optional shadow period** (recommended before the first haze season in
production): rather than promoting at step 4, run inference for the
staging directory into a second ledger for 7 days, then compare both
ledgers' model-quality metrics on identical hours before switching.

## 7. Roles and response times

| Role | Owns | Responds to |
|---|---|---|
| Data owner (data team) | ingest, bronze/silver/gold, API keys | `fix_data`, ingest task failures, within 1 working day (same day during a haze episode) |
| ML owner (model team) | training code, thresholds, promotion | `investigate` within 2 working days, rejected challengers, forced promotions |
| Approver (project lead) | this document | threshold changes, forced promotions, deployment-option changes |

During an active haze episode (any region's actual PM2.5 ≥ 56 µg/m³ in the
last 24 h), a failed hourly run is handled the same day, and promotions are
frozen unless they fix a model-quality alert.

## 8. Audit trail

- every issued forecast, with the manifest hash and model run id that produced it: forecast ledger
- every monitoring window's metrics and decision: `monitor_metrics` / `monitor_runs` tables, `data/monitoring/model_monitor/*.json`
- every promotion, rejection, rollback and forced promotion: `data/models/hourly/promotion_log.jsonl`, archived manifests in `history/`
- every training run's folds, scores and fitted model: MLflow

## 9. Known limitations

- Bias here means *regional and time-of-day* disparity in forecast quality. No protected attributes exist in this data. Regions are the closest analogue to a "facet": a model that under-warns one part of the island is the harm to watch for.
- Model-quality monitoring is only as complete as NEA's published actuals. Hours NEA has not published stay "pending" and are not scored.
- Without `FIRMS_MAP_KEY`, fire features stop updating, so the freshness gate stops issuing forecasts and the monitor reports `fix_data` (verified in Airflow). The key is required for live operation.
- Fire features use only detections FIRMS has delivered (rolling windows ending 3 h before each hour), and are left empty more than 15 h after the latest detection, so a FIRMS outage pauses forecasts rather than being read as "no fires" (walkthrough §2.7).
- The daily (next-day) product still uses calendar-day fire counts, including the current day's incomplete count at 23:30. This look-ahead issue is open for that product only.
- Thresholds are calibrated on one replay (one out-of-time haze season). Revisit them after the first live haze season; the drift limits are recomputed from the training data with every new baseline.
