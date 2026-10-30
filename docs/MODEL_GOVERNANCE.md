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
| `hazecast_monitoring` | 01:30 daily | Makes the last 10 days of raw data permanent, fetches NEA actuals, runs the five monitors, rebuilds the dashboard, applies the **retraining rule** (§5) and branches |
| `hazecast_retrain` | 03:00 on the 1st, or triggered | Trains a challenger in staging, replays its holdout, runs the **promotion gate** (§6), promotes it or keeps the champion, then refreshes the monitoring baseline |
| `hazecast_haze_monitoring` | 07:30, 13:30, 19:30 | **Haze mode**: stops at once unless a region read ≥ 56 µg/m³ in the last 24 h; during an episode it runs the monitor and the retraining rule, so with the 01:30 run the model is checked every 6 h |

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
| **Data quality** (30-day window) | Are the model's inputs still healthy and like the training data? | schema, freshness, missing rate, share outside training range, stuck PM2.5 feeds, age of each upstream source (live), and **CSI** (characteristic stability index: PSI of each feature) against the **same monsoon season** in the training years | *fail*: missing column, features > 3 h old, a source past its limit (PM2.5 or weather > 3 h; fire detections > 39 h, i.e. past the 15 h trust window and the 24 h carry-forward), or a region's PM2.5 unchanged for ≥ 12 h. *Alert*: ≥ 3 features beyond **both** their calibrated CSI limit (99th percentile over normal 30-day windows of that season in the training period) and their all-season PSI limit; missing rate +5 pp or > 5 % out of range |
| **Stability** (30-day window) | Is the model's output still distributed as it was in this season? | **PSI** (population stability index) of each model's forecasts against its forecasts on the same season in the training years | ≥ 3 of 18 models above their calibrated PSI limit (99th percentile over normal windows of that season). Works without actuals, so it reacts before accuracy can be judged |
| **Model quality** (7-day window) | Are forecasts still doing their job, verified against NEA? | MAE, RMSE, F2, recall, MAE skill, **recall of persistence on the same hours**, **coverage** (share of hours at or below the q85 forecast) and the pollution level relative to the holdout | with ≥ 48 matched hours: MAE > 1.5 × holdout MAE, *or* (with ≥ 12 haze hours) recall more than 0.25 below persistence's. Alert at ≥ 3 of 18 groups. **Coverage** pooled per horizon outside its normal weekly range (1st–99th percentile of quiet holdout weeks) is a calibration alert, judged every week, haze or not. MAE skill vs persistence is reported, not alerted: the q85 models are upward-biased by design. When an MAE alert comes with a level 1.5× the holdout average or more, the reason says so (a harder week, not necessarily a worse model) |
| **Bias drift** (7-day window) | Is any region (or time of day) served systematically worse? | per-region recall, mean signed error, alert rate vs actual rate; day vs night MAE | recall gap between regions (each ≥ 12 haze hours) > 0.30, or > 0.20 wider than at training; a region's mean signed error shifts > 8 µg/m³ |
| **Feature attribution drift** (30-day window) | Has what the model relies on changed? | label-free permutation attribution; **NDCG@10** vs baseline ranking | NDCG below the model's **calibrated limit** (1st percentile over training-period windows, at most 0.90). Alert at ≥ 3 of 18 models |

Two kinds of monitoring are kept separate on purpose. `hazecast.monitoring`
verifies archived forecasts against NEA actuals; `hazecast.model_monitor`
adds the drift, stability and bias views and makes the retraining decision.

**Seasonality.** Singapore's year has four monsoon seasons (`MONSOON_SEASONS`:
NE monsoon Dec–Mar, inter-monsoon Apr–May, SW monsoon Jun–Sep, inter-monsoon
Oct–Nov), and the inputs move with them.
- **In the model:** season enters through the drivers (fire counts, wind, rain, upwind alignment). Adding explicit seasonal inputs did not help on the 2024–26 holdout (`experiments/seasonality_features.py`): day of year lowered F2 from 0.655 to 0.630, monsoon phase moved it by 0.002 (noise), wind direction by −0.003. Outside the 2026 haze the model is calibrated in every season (q85 coverage 0.83–0.86 against a target of 0.85, no false alarms).
- **In monitoring:** CSI and PSI compare each 30-day window with the same season in the training years, with limits calibrated per season. A feature counts as drifted only when it is unusual both for its season and against all training months. In the holdout replay this cut drift-only investigate weeks from 8 (all-season only) to 5, while season-matched CSI alone gave 16; all three gave a warning by 10 Aug 2026, ahead of the haze (walkthrough §4.2).

## 5. Retraining triggers: when to refresh the model

The monitoring DAG turns the five monitor results into one action
(`model_monitor.governance_decision`):

| Action | Rule | What happens |
|---|---|---|
| **fix_data** | data-quality *fail*: missing feature columns, stale features, a source past its age limit, or a stuck PM2.5 feed | No retraining: a model retrained on broken data would be worse. Data owner investigates the failing source (§7) following the data-incident SOP (§10). Inference is already refusing stale features |
| **retrain** | model-quality *alert* (≥ 3 groups); **or** calibration (coverage) outside its normal weekly range; **or** data drift (CSI) or unstable forecasts (PSI) **together with** attribution drift; **or** the champion is older than 35 days; **or** a scenario (below) | `hazecast_retrain` is triggered automatically; the promotion gate still decides whether anything changes. At most one automatic retrain per 7 days (`RETRAIN_COOLDOWN_DAYS`), except after a change in the feature path or libraries; a retrain held by the cooldown becomes *investigate* |
| **investigate** | any other *alert*: drift alone, unstable forecasts alone, attribution drift alone, bias drift, a new champion doing worse than the one it replaced (early life), a retrain held by the cooldown | ML owner reviews the dashboard within 2 working days and decides retrain / threshold change / rollback / no-op, recording why |
| **none** | no alerts, or a drift / stability / attribution / coverage signal seen for the first time (warnings are shown on the dashboard only) | nothing |

**Persistence.** Drift (CSI), unstable forecasts (PSI), attribution drift and
calibration (coverage) count only when the previous monitoring run had the
same signal: a single noisy window never pages anyone (live: two days in a row;
during haze mode, 6 h apart). Performance alerts against actuals and data
failures act at once.

**Scenario retrains** (no threshold would ask for them;
`model_monitor.scenario_triggers`):
- **Before the fire season:** from 15 July, if the champion was trained before then (`PRE_SEASON_RETRAIN`).
- **After an episode:** 12 or more elevated hours since the champion's training data ended, then 14 quiet days (`POST_EPISODE_QUIET_DAYS`), so the episode enters training and validation.
- **After a change in the feature path or libraries:** the manifest records a fingerprint of the feature code and settings and the model libraries' versions (`hazecast.provenance`). A mismatch with the running code retrains even inside the cooldown, because the champion would otherwise get features it was not trained on; inference logs a warning meanwhile.

**Haze mode.** While any region read ≥ 56 µg/m³ in the last 24 h,
`hazecast_haze_monitoring` adds three runs a day. The cooldown keeps a long
episode from retraining more than weekly (the replay recommended a retrain
in 9 consecutive weeks).

**Threshold review.** After every haze season and each quarter, run
`python -m hazecast.alert_review` (and `--db data/monitoring/backtest.sqlite`
for the replay). It lists each signal type with how many of its alerts were
followed by a real loss in performance within 4 weeks. A signal with many
alerts and few followed is a candidate for a stricter limit; record any change
in this document.

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
   - **Integrity:** 18 entries, every artifact loads and gives a finite prediction on the latest gold row, and no artifact needs a module outside the declared requirements (added 28 Oct, after Airflow wrote `dill`-dependent model files).
   - **Skill:** holdout **F2** within 0.10 of persistence's in ≥ 50 % of the groups with haze hours. F2, not MAE: the q85 models never beat persistence on MAE (0 of 18 groups), so an MAE gate would reject every model. Persistence is a floor, not a target, because it cannot forecast the onset of an episode. Current champion: 15 of 18 within tolerance.
   - **No regression, judged on F2, like-for-like:** the champion's model choices are refitted on the challenger's own pre-holdout rows with the current features and scored on the same holdout. The champion's recorded scores came from another window and, after the fire-feature fix, from leaky features, which made an unchanged model look 15/18 better; like-for-like it matched the champion exactly wherever both chose `xgboost_q85`.
     - F2 within 0.05 of the champion's in at least half the groups.
     - **No single group's F2 falling by more than 0.10.** If one does, the challenger is held for a person to decide (`--force` with a recorded reason), not promoted automatically.
     - MAE at most 2× the champion's in every group, as a blow-up guard.
   - **Recent window** (`RECENT_WINDOW_DAYS` = 90): the challenger's model choices, refitted without the last 90 days, must not fall more than 0.10 F2 behind the forecasts actually served in those days (pooled per horizon). Judged only with 12 or more haze hours in the window; otherwise noted as not judged. The holdout is mostly older data; this checks the challenger on the conditions it meets first. A failure holds the challenger for a person.

     The first version of the no-regression check used MAE (within 10%). Evaluating the LOSO-tuned model from the `XGB_Tuned` branch showed why that was wrong: it improved F2 in 13 of 18 groups, but a higher quantile forecasts higher by design (MAE up to 1.8×), so the MAE rule rejected it. It also did not single out the one genuine regression (west +6 h, F2 0.779 → 0.628). The F2 rule passes the broad improvement and flags exactly that group.
4. **Pass:** artifacts are copied, the old manifest is archived to `history/`, and the new manifest is written **last**, so inference never sees a half-published model. The monitoring baseline (including the calibrated limits) is rebuilt for the new champion. **Fail:** the champion keeps serving, and the rejection and its reasons go to `promotion_log.jsonl`.
5. **Rollback** (any time, one command, no retraining): `python -m hazecast.governance rollback` restores the previous manifest. Artifacts are never deleted, so any archived version can be restored with `--to`.
6. **Manual override:** `python -m hazecast.governance promote --candidate <dir> --force --reason "..."` promotes despite a failed gate. The command refuses without a reason, and stores it in `promotion_log.jsonl` and in the published manifest (`forced_reason`, `forced_over`). Only the ML owner may use it, and the team channel should be told.
7. **Model selection stability:** a candidate other than `xgboost_q85` must beat it in ≥ 90 % of paired weekly bootstrap resamples of the validation predictions (walkthrough §3.5). This prevents promotion holds caused by switches on thin, noisy margins.
8. **Reproducibility:** train, gate and replay in the same environment (the Airflow image). On 28 Oct the same model scored F2 0.759 with xgboost 3.2.0 (Airflow) and 0.790 with 3.4.1 (a local environment), because the requirements were lower bounds only. Since 30 Oct `requirements-pipeline.txt` pins exact versions (those of the running images); upgrade deliberately, rebuild, and let the provenance check trigger the retrain.
9. **Early life:** for 7 days after a promotion, the monitor compares the new champion's live forecasts with what the champion it replaced would have forecast for the same hours (its archived artifacts). MAE above 1.25× the previous one's at any horizon, or recall 0.25 lower with 12+ haze hours, asks for a review, with rollback as the likely action.

**Shadow period** (recommended before the first haze season in
production, and for any challenger the team is unsure about):
```bash
python -m hazecast.governance shadow --candidate data/models/hourly_staging/<run>
python -m hazecast.governance shadow-report      # champion vs shadow on identical hours
python -m hazecast.governance shadow-stop
```
While a shadow is set, hourly inference also scores it into
`data/monitoring/shadow.sqlite`. It is never served, and a failure there never
affects the champion. After 7 days, compare the report's MAE, coverage and
recall per horizon, then promote (or not) through the normal gate.

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
- Fire features use only detections FIRMS has delivered (rolling windows ending 3 h before each hour), and are trusted up to 15 h after the latest detection. If FIRMS is late, the last known fire features are carried forward for up to 24 h, and each forecast records this in `fire_carried_hours`. After that, forecasts pause rather than reading an outage as "no fires" (walkthrough §2.7).
- The daily (next-day) product still uses calendar-day fire counts, including the current day's incomplete count at 23:30. This look-ahead issue is open for that product only.
- Thresholds are calibrated on one replay (one out-of-time haze season). Revisit them after the first live haze season; the drift limits are recomputed from the training data with every new baseline.

## 10. Data incidents: upstream outages and ingestion changes

A data incident is any period in which an input is missing, late, frozen,
malformed or means something different from what the model was trained on.
Three rules apply to every incident:
1. **Never serve a forecast from bad inputs.** The freshness gate skips the hourly run, and the dashboard says the forecast is old. A gap is better than a wrong number.
2. **Never retrain on suspect data.** The monitor answers `fix_data`, not `retrain`. Promotions are frozen until the incident is closed.
3. **Close every incident with a check.** Backfill, confirm the layers, replay the monitors, write a short note.

### 10.1 Scenarios

| Scenario | Detected by | What the pipeline does by itself | Owner action | Retrain? |
|---|---|---|---|---|
| **Source API down** (NEA, Open-Meteo or FIRMS returns errors or nothing) | `fetch_live` task failure or retry; source age (`source_age_hours`) past its limit; freshness gate | Skips forecasts once the newest complete row is > 2 h old. Fire data: carried forward up to 24 h, labelled | Data owner: check the provider's status page and our key; wait or switch to the provider's alternative endpoint; the nightly top-up backfills automatically once it is back | No |
| **Late delivery** (FIRMS satellite pass hours late; 28 Oct 2026: > 10 h) | `fires` source in *warn* (past the 15 h trust window); `fire_carried_hours` > 0 on forecasts and the dashboard | Carries fire features forward up to 24 h, labels each forecast, then pauses | None under 24 h; note it if over 12 h. Over 24 h: as "API down" | No |
| **Partial delivery** (a region, a weather point or a satellite missing) | Missing rate up > 5 pp for that feature; a weather point's source age past its limit | Feature row incomplete → no forecast once the gap reaches the newest row | Data owner: re-fetch that source; check whether the provider dropped a station or point | No |
| **Frozen values** (a feed repeating the last reading) | `stuck_feed`: a region's PM2.5 unchanged for ≥ 12 h (never in 12 years of real data; longest genuine run 11 h) | Monitor answers `fix_data` | Data owner: confirm against NEA's website; record the affected hours so they are excluded from training labels | Only after the hours are excluded |
| **Schema or format change** (renamed field, new file layout, API version) | Ingestion parse error; `missing_columns` in the monitor | Task fails; no forecast; `fix_data` | Data owner: adapt the parser, add a test with a sample of the new format, backfill | No |
| **Meaning change** (station moved, units or scaling changed, weather model upgraded, new satellite or product version) | Step change in the CSI of one source's features, values out of training range, while other sources are stable; model quality may follow | Drift alert → `investigate` (or `retrain` if attribution drift agrees) | ML owner and data owner: confirm the change with the provider; align the new data to the old (or rebuild history consistently); then retrain through the normal gate (change process) | Yes, after history is consistent |
| **Revisions** (provider corrects recent values) | Values in the ledger's actuals table updated (`source_updated_at`) | Re-fetches the last 2 days of weather on every top-up; newest values win | None; model quality for affected hours is recomputed automatically | No |
| **Our own ingestion bug** (a code change breaks a feature) | Tests in CI; `missing_columns`, out-of-range share or a CSI step on the day of a deploy | `fix_data` or drift alert | Author: revert or fix the change; replay the monitors; models are unaffected unless they were retrained on the bad data (then roll back) | No (roll back if a retrain used the bad data) |
| **Expired key or rate limit** | `fetch_live` 401/429 errors; source age rising | As "API down" | Data owner: rotate the key in `.env`, restart the stack, confirm the next hourly run | No |
| **Time-zone or timestamp shift** | Model quality drops sharply while CSI is calm; actual and forecast series offset on the dashboard | Model-quality alert → a retrain is triggered, which must not be promoted on shifted data | ML owner: compare raw timestamps with the source's website before the gate result is accepted; fix the parser, rebuild the layers | No (reject the triggered challenger) |

### 10.2 Severity and response

| Severity | Meaning | Examples | Response |
|---|---|---|---|
| **S1** | No forecasts, or wrong data being served | API down > 2 h, stuck feed, timestamp shift | Data owner the same day (within 2 h during an active haze episode); team channel told |
| **S2** | Forecasts continue in a degraded way | Fire data carried forward, one weather point late, partial delivery | Data owner within 1 working day |
| **S3** | No effect on forecasts | Revision of past values, a warning on the dashboard | Weekly review |

### 10.3 Closing an incident

1. The source is back: the newest complete gold row is fresh and every source shows *ok*.
2. Backfill: the nightly top-up repairs the last 10 days; longer gaps need `hazecast_backfill`.
3. Verify: bronze and silver have no missing or duplicate hours for the period; the dashboard's actual series matches the provider's website for a sample of hours.
4. Replay: run the monitor for the affected windows; no remaining `fix_data`.
5. Record: what happened, how long, which hours are excluded from training (if any), and what was changed. Lift the promotion freeze.
