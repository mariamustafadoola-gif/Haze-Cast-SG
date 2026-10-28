"""Shared constants for the HazeCast SG pipeline.

Everything that is an assumption rather than a measured fact is called out
in a comment. Change these in one place; every stage imports from here.
"""
from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (medallion layers). Override with env vars in production / Airflow.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
MODELS_DIR = DATA_DIR / "models"
MONITORING_DIR = DATA_DIR / "monitoring"

# First day of training history (matches the manual downloads in
# data/README.md). hazecast.ingest.historical backfills from here.
HISTORY_START = "2014-04-01"
# Weather hours re-requested on every history top-up, even if already stored: Open-Meteo
# revises the newest hours for about a day after first serving them.
WEATHER_REFETCH_DAYS = 2

# ---------------------------------------------------------------------------
# Regions: Singapore's 5 NEA reporting regions (must match the PM2.5 file's
# column names exactly).
# ---------------------------------------------------------------------------
SG_REGIONS = ["north", "south", "east", "west", "central"]

# Approximate bounding boxes for the two dominant fire source regions,
# used to tag FIRMS detections. These are coarse rectangles, not
# administrative boundaries -- documented assumption, not a measured fact.
FIRE_REGIONS = {
    "sumatra": {"lon_min": 95.0, "lon_max": 106.5, "lat_min": -6.2, "lat_max": 6.0},
    "kalimantan": {"lon_min": 108.5, "lon_max": 119.5, "lat_min": -4.6, "lat_max": 4.6},
}

# Weather reference points used for upwind features (lat, lon), one per
# fire province plus Singapore itself. Matches the Open-Meteo files
# requested for this project.
WEATHER_POINTS = {
    "singapore": (1.37, 103.80),
    "riau": (0.53, 101.42),          # Sumatra (north)
    "palembang": (-2.99, 104.78),    # Sumatra (south)
    "pontianak": (0.04, 109.28),     # Kalimantan (west)
    "palangka": (-2.21, 113.92),     # Kalimantan (central)
}

# ---------------------------------------------------------------------------
# Labels / thresholds
# ---------------------------------------------------------------------------
# NEA 1-hour PM2.5 bands (ug/m3), verified against NEA/haze.gov.sg guidance
# reported by Malay Mail (27 Jun 2016) and cross-checked in this project's
# own data profiling:
#   Normal: 0-55, Elevated: 56-150, High: 151-250, Very High: >250
PM25_ELEVATED_THRESHOLD = 56.0

# Silver gap filling (data team, 2026-10). A gap of up to this many hours
# in an hourly series is filled by straight-line interpolation between the
# readings either side (wind direction: carried forward, since averaging
# 359 and 1 degrees would give 180). Longer gaps are left empty on purpose:
# guessing across a long gap could invent or erase a haze episode.
# Every filled hour is marked in a `filled` column in silver.
MAX_FILL_HOURS = 3

# Forecast horizon: predict the next calendar day's daily-mean PM2.5,
# per region, using only data observed up to "today". No same-day or
# future weather is used (tomorrow's wind is unknown at forecast time).
FORECAST_HORIZON_DAYS = 1

# Hotspot lag features to build (days). Same-day fires (lag 0) matter most,
# per this project's correlation analysis; we keep out to 3 days.
HOTSPOT_LAGS = [0, 1, 2, 3]

# Rolling windows (days) for rain/hotspot accumulation features.
ROLLING_WINDOWS = [3, 7]

# "Haze seasons" used for leave-one-season-out evaluation. These are the
# periods this project's own data profiling found to contain the vast
# majority of elevated (>=56 ug/m3) days. Both 2015 and 2019 have full
# feature coverage as of the 2014-2018 FIRMS backfill; add 2026 once that
# season's PM2.5 labels are complete (currently only through 1 Aug 2026).
HAZE_SEASONS = {
    "2015": ("2015-08-01", "2015-11-15"),
    "2019": ("2019-08-15", "2019-11-30"),
    # "2026": ("2026-08-01", "2026-10-31"),  # add once PM2.5 labels cover it
}

# ---------------------------------------------------------------------------
# Hourly pipeline (revised core target: hourly PM2.5 per region, refreshed
# as new weather data lands -- see build_gold_hourly() in silver_to_gold.py).
# Additive to the constants above: the original daily pipeline/labels are
# unchanged and still used by the shipped daily model.
# ---------------------------------------------------------------------------
# Predict PM2.5 this many hours ahead, using only data observed up to the
# current hour.
FORECAST_HORIZON_HOURS = 1

# Rolling windows (HOURS, not days) for precipitation accumulation at
# hourly resolution. Fire-hotspot accumulation keeps using the daily
# ROLLING_WINDOWS above -- FIRMS is a satellite-pass source (~1-2
# overpasses/day/location), not something that gains meaning at a 3h/24h
# grain; see the data-staleness note on _fire_features_hourly.
HOURLY_RAIN_ROLLING_WINDOWS = [3, 24]

# Hourly fire features are rolling windows on detection timestamps: at hour t,
# fire_<region>_lag0 = detections in the 24 h ending FIRE_LATENCY_HOURS full hours before t,
# lag1..lag3 the 24 h blocks before that, roll3 / roll7 the 72 h / 168 h ending at the same
# point. Every row only uses detections that had been delivered by its hour (no look-ahead),
# the meaning is the same at every hour, and a forecast never waits for "today's" first
# detection. Chosen over day-based alternatives on the 2024-26 holdout (see
# docs/PIPELINE_WALKTHROUGH.md section 2.6): same haze recall as the old (leaky) rule,
# fewer false alarms, better F2 and MAE.
FIRE_LATENCY_HOURS = 3                       # FIRMS near-real-time delivery: within ~3 h of the pass
# Hourly counts are trusted up to this long after the latest detection. Passes are ~11 h apart
# (afternoon 13:00-15:59 SGT, night 01:00-04:59 SGT); a longer silence means a missing pass or a
# stalled feed, so the features are left empty (the forecast gate refuses) rather than reading an
# outage as "no fires". 15 h = one pass interval + ~3 h delivery margin; in 12 years of data,
# silences > 15 h occurred 295 times vs 298 for > 13 h, so the margin costs no outage detection.
FIRE_KNOWN_HOURS_AFTER_LAST_DETECTION = 15
# When FIRMS is late (28 Oct 2026: the 13:00 SGT pass was still undelivered 6 h later, so the
# fire features went empty and two hourly forecasts were skipped), the last known fire features
# are carried forward for up to this many hours instead. Only delivered detections are used (no
# look-ahead); the windows are 24-168 h long, so a few hours' extra age changes them little.
# Each forecast records how long its fire features were carried (fire_carried_hours). The
# carried values exist only at the live edge: gold is rebuilt every hour, and once FIRMS
# delivers, those hours get their true counts, so training never sees carried values. Past this
# limit the features stay empty and the forecast gate refuses, as before.
FIRE_CARRY_FORWARD_HOURS = 24
# FIRMS dates detections by UTC day (ends 08:00 SGT); a day's count is complete from 11:00 SGT
# the next day. Used for the day-based satellite-gap flag in the hourly table.
FIRE_DAY_AVAILABLE_HOUR_SGT = 11

# ---------------------------------------------------------------------------
# Hourly multi-horizon deliverable (6/12/24h ahead, refreshed hourly).
# Active hourly models are now selected by forward validation. Historical
# quantile linear regression tuning below is retained for reference only;
# it is not used by the six-candidate hourly comparison.
# ---------------------------------------------------------------------------
HOURLY_HORIZONS = [6, 12, 24]

# Per (region, horizon) QuantileRegressor settings, tuned against the dual
# bar: recall_elevated >= 1.15x persistence AND mae_elevated <= 1.15x
# persistence, per region x haze season (both at once). Deliberately
# upward-biased by design -- see the deployment-decision note in the
# project log (over-reporting during real haze is judged the safer
# failure mode than an unbiased number).
#
# h=1 is the primary tuned setting per region. h=3 only gets its own entry
# where the h=1 setting didn't carry over cleanly (north/south/east needed
# a higher quantile; central/west/national did not improve over the h=1
# setting at h=3 and reuse it, see caveats below). h=6/h=12 reuse each
# region's h=1 setting everywhere -- an alpha=0.5 sweep at those horizons
# (2026-10) found no systematic gain over alpha=0, so this is not a gap,
# it's a confirmed "nothing left on the table" result.
#
# Known open gaps, carried forward rather than silently fixed:
#   - west, h=3: regularization does not help here (2019 MAE gets WORSE
#     as alpha increases, opposite of every other region). Reuses the h=1
#     setting as the least-bad fallback; still short of the dual bar.
#   - national, h=1 and h=3: an information ceiling, not an instability
#     problem -- alpha is flat/unresponsive. Best found for h=3
#     (q=0.91, alpha=0.3) clears 2019 fully and 2015's MAE comfortably but
#     misses 2015 recall by ~1.6pp; logged as accepted near-ceiling, not
#     silently treated as a pass.
#   - national/2019 at h=1 is a boundary case under seed robustness
#     checking (passes under seed=42, fails under seed=123 by a hair on
#     mae_elevated) -- not a stable pass, flagged rather than asserted.
HOURLY_TUNED_SETTINGS: dict[tuple[str, int], tuple[float, float]] = {
    # region: (quantile, alpha) at h=1 -- the primary setting, reused
    # as-is at h=6 and h=12 for every region.
    ("north", 1): (0.85, 0.3),
    ("south", 1): (0.80, 0.0),
    ("east", 1): (0.80, 0.0),
    ("west", 1): (0.85, 0.0),
    ("central", 1): (0.90, 0.5),
    ("national", 1): (0.88, 0.0),
    # h=3 overrides where a different setting was actually needed/better.
    ("north", 3): (0.90, 0.0),
    ("south", 3): (0.90, 0.0),
    ("east", 3): (0.90, 0.3),
    ("national", 3): (0.91, 0.3),  # best found; see "known open gaps" above
    # west, h=3 intentionally absent -> falls back to the h=1 setting
    # (0.85, 0.0) in hourly_model.get_tuned_setting(); see "known open
    # gaps" above. central, h=3 also absent -> falls back to h=1's
    # (0.90, 0.5), which already clears the dual bar at h=3.
}

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
# Shared XGBoost tree/boosting params for every XGBoost variant compared in
# hazecast.train -- only the objective differs between variants.
XGB_PARAMS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)

# The production model forecasts a HIGH quantile of next-day PM2.5, not the
# mean. Plain MAE-minimizing regression (see the "xgboost" candidate in
# hazecast.train, kept only as a comparison baseline) systematically
# under-predicts the rare elevated days: with haze days at ~2% of the
# dataset, a mean-minimizer is pulled toward the calm majority. Evaluated
# against real data (see README "Results"), targeting the 85th percentile
# instead roughly triples elevated-day recall on the worse of the two haze
# seasons, at a real but acceptable cost in precision (more false alarms)
# and a season-dependent change in MAE. This is a deliberate choice for an
# early-warning use case, where a missed elevated day (false negative) is
# more costly than an extra false alarm -- tune lower for a more
# conservative/precise alert, higher to catch more true events at the cost
# of more false alarms; 0.70-0.95 were swept, 0.85 was the best balance.
XGB_QUANTILE_ALPHA = 0.85


# ---------------------------------------------------------------------------
# Selected features (data team, 2026-10). The gold tables still contain all
# 56 candidate features; training only uses the lists below.
#
# How they were chosen, step by step with charts:
#     notebooks/01_eda.ipynb (why) and
#     notebooks/02_cleaning_and_feature_selection.ipynb (how)
# Re-run the selection (rewrites reports/feature_reduction/):
#     python experiments/feature_reduction.py --grain hourly --quick
#
# Validated with leave-one-haze-season-out (2015 and 2019): the reduced
# hourly set (36) matched all 56 at h=1 and h=6: average error within
# 0.3 ug/m3 and unhealthy-hours-caught within 1 percentage point. NOTE for the
# model team: HOURLY_TUNED_SETTINGS above were tuned on all 56 features;
# re-check them on this set before reporting final numbers.
#
# Set USE_SELECTED_FEATURES = False to go back to all 56 (e.g. to compare).
# ---------------------------------------------------------------------------
USE_SELECTED_FEATURES = True

SELECTED_FEATURES_HOURLY = [
    "fire_sumatra_lag0",
    "fire_sumatra_lag2",
    "fire_sumatra_lag3",
    "fire_sumatra_roll3",
    "fire_kalimantan_lag0",
    "fire_kalimantan_lag2",
    "fire_kalimantan_lag3",
    "fire_kalimantan_roll3",
    "fire_data_gap_recent",
    "wind_speed_singapore",
    "rain_singapore_mm",
    "rain_singapore_roll3h",
    "rain_singapore_roll24h",
    "wind_speed_riau",
    "rain_riau_mm",
    "rain_riau_roll24h",
    "wind_speed_palembang",
    "rain_palembang_mm",
    "rain_palembang_roll3h",
    "rain_palembang_roll24h",
    "wind_speed_pontianak",
    "rain_pontianak_mm",
    "rain_pontianak_roll3h",
    "rain_pontianak_roll24h",
    "wind_speed_palangka",
    "rain_palangka_mm",
    "rain_palangka_roll3h",
    "rain_palangka_roll24h",
    "upwind_sumatra",
    "upwind_kalimantan",
    "now_north",
    "now_south",
    "now_east",
    "now_west",
    "now_central",
    "now_national",
]

SELECTED_FEATURES_DAILY = [
    "fire_sumatra_lag0",
    "fire_sumatra_lag1",
    "fire_sumatra_lag2",
    "fire_sumatra_lag3",
    "fire_sumatra_roll7",
    "fire_kalimantan_lag0",
    "fire_kalimantan_lag2",
    "fire_kalimantan_lag3",
    "fire_kalimantan_roll3",
    "fire_data_gap_recent",
    "wind_speed_singapore",
    "rain_singapore_mm",
    "rain_singapore_roll3",
    "rain_singapore_roll7",
    "wind_speed_riau",
    "rain_riau_mm",
    "rain_riau_roll3",
    "rain_riau_roll7",
    "wind_speed_palembang",
    "rain_palembang_mm",
    "rain_palembang_roll3",
    "rain_palembang_roll7",
    "wind_speed_pontianak",
    "rain_pontianak_mm",
    "rain_pontianak_roll3",
    "wind_speed_palangka",
    "rain_palangka_mm",
    "rain_palangka_roll3",
    "rain_palangka_roll7",
    "upwind_sumatra",
    "upwind_kalimantan",
    "today_north",
    "today_south",
    "today_east",
    "today_west",
    "today_central",
    "today_national",
]


# Daily model comparison. Existing hourly constants above are preserved.
# All six candidates are enabled. Set this flag False for the five-model
# comparison without quantile XGBoost. Restart notebook kernels after edits.
ENABLE_XGB_QUANTILE = True
MODEL_CANDIDATES = (
    "persistence", "ridge", "random_forest", "xgboost", "lightgbm",
) + (("xgboost_q85",) if ENABLE_XGB_QUANTILE else ())
# Scaling is fitted inside each training fold, never on validation/test rows.
RIDGE_ALPHA = 10.0
LIGHTGBM_PARAMS = dict(
    objective="regression_l1", n_estimators=300, learning_rate=0.05,
    num_leaves=15, max_depth=-1, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=-1,
)
RANDOM_FOREST_PARAMS = dict(
    n_estimators=300, max_depth=12, min_samples_leaf=3,
    max_features=0.8, random_state=42, n_jobs=-1,
)
CV_SPLITS = 3
FINAL_TEST_FRACTION = 0.20
MIN_TRAIN_ROWS = 100
# Rank pooled validation F2 descending, then MAE/RMSE ascending. If no
# elevated validation observations exist, rank by MAE then RMSE instead.
MLFLOW_EXPERIMENT = "hazecast-sg-forward"

# Selection stability (hourly). The pooled validation has only ~130-150 elevated hours per
# group, so small F2 differences between candidates are noise. Twice, a thin validation win
# switched the west region away from xgboost_q85 (ridge at +6 h by 0.002, random forest at
# +12 h by 0.038) and the switch then lost out of time (F2 -0.151, -0.104). A non-default
# winner is now kept only if it beats the default in at least SELECTION_SWITCH_CONFIDENCE of
# paired block-bootstrap resamples of the validation predictions (whole weeks resampled, so
# haze episodes stay intact). Measured: west +12 h random forest won 73% of resamples,
# central +6 h persistence 43% -- neither reliable.
SELECTION_DEFAULT_CANDIDATE = "xgboost_q85"
SELECTION_SWITCH_CONFIDENCE = 0.90
SELECTION_BOOTSTRAP_BLOCK_HOURS = 168
SELECTION_BOOTSTRAP_SAMPLES = 500

# Active hourly comparison: independent selected model per region/horizon.
HOURLY_MLFLOW_EXPERIMENT = "hazecast-sg-hourly-forward"
HOURLY_TIMEZONE = "Asia/Singapore"


# ---------------------------------------------------------------------------
# Model monitoring + governance (hazecast.model_monitor, dags/, and the SOP in
# docs/MODEL_GOVERNANCE.md). Starting values, not tuned facts: review them
# after the first haze season of live monitoring.
# ---------------------------------------------------------------------------
MONITOR_WINDOW_DAYS = 7            # performance window: forecasts vs actuals (model quality, bias)
DRIFT_WINDOW_DAYS = 30             # drift window: a week of slow, autocorrelated features is too few samples
DRIFT_NULL_WINDOWS = 120           # training-period windows used to calibrate the PSI limits
ATTRIBUTION_NULL_WINDOWS = 30      # ... and the NDCG limits (attribution is costlier to compute)
# Data quality: the gold features the deployed model reads vs its training data.
# Per-feature PSI limits are calibrated (model_monitor.drift_null): alert above the 99th
# percentile of PSI over normal training-period windows, warn above the 95th. The standard
# bands below are only floors, never the limit by themselves.
PSI_WARN, PSI_ALERT = 0.10, 0.25
PSI_ALERT_MIN_FEATURES = 3         # drift counts toward retraining only if this many features are in alert
MISSING_RATE_MAX_INCREASE = 0.05   # absolute increase in a feature's missing rate vs training
OUT_OF_RANGE_MAX_FRACTION = 0.05   # share of window values outside the training min/max
FEATURE_MAX_AGE_HOURS = 3          # newest complete gold row older than this = stale pipeline
# Model quality: archived forecasts vs NEA actuals (forecast ledger).
MODEL_MAE_RATIO_ALERT = 1.5        # rolling MAE / holdout MAE of the deployed model
RECALL_VS_PERSISTENCE_MARGIN = 0.25  # alert if the model catches 25 pp fewer haze hours than persistence
MIN_ELEVATED_FOR_RECALL = 12       # haze hours needed (per region, per window) before recall is judged
MIN_MATCHED_FOR_QUALITY = 48       # matched forecast hours needed before quality alerts count
MODEL_QUALITY_ALERT_MIN_GROUPS = 3 # of the 18 target x horizon groups
# Bias drift: are some regions / times of day served systematically worse?
BIAS_RECALL_GAP_ALERT = 0.30       # max - min elevated-hour recall across the 5 regions (each >= 12 haze hours)
BIAS_RECALL_GAP_DRIFT = 0.20       # ... or this much wider than at training time
BIAS_MEAN_ERROR_DRIFT = 8.0        # ug/m3 shift in a region's mean signed error vs training
# Feature attribution drift: has what the model relies on changed?
ATTRIBUTION_TOP_K = 10
ATTRIBUTION_NDCG_ALERT = 0.90      # SageMaker Clarify's default; a ceiling -- per-model limits are the 1st
                                   # (alert) / 5th (warn) percentile of NDCG over training-period windows
ATTRIBUTION_SAMPLE_ROWS = 3000
# Governance
MAX_MODEL_AGE_DAYS = 35            # scheduled refresh even without drift (monthly retrain + slack)
# Regression vs the champion is judged on F2 (the selection objective), not MAE: a higher
# quantile forecasts higher by design (the LOSO-tuned challenger had 13/18 groups better on
# F2 but MAE up to 1.8x the champion's).
PROMOTION_F2_REGRESSION = 0.05        # F2 may trail the champion's by at most this in >= half the groups
PROMOTION_F2_SEVERE_DROP = 0.10       # any group dropping more than this -> human review, no auto-promotion
PROMOTION_MAX_MAE_RATIO = 2.0         # MAE blow-up guard per group (designed bias reached 1.8x)
PROMOTION_F2_TOLERANCE = 0.10         # challenger holdout F2 may trail persistence's by at most this ...
PROMOTION_MIN_SKILL_TARGETS = 0.5     # ... in at least this share of groups with haze hours in the holdout
                                      # (2024-26 holdout: 14 of 18 groups within 0.10; only 6 of 18 ahead)
