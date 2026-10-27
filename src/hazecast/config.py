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

# Per-cell tuned alpha for the xgboost_q85 candidate in the hourly
# comparison, selected via leave-one-haze-season-out (2015, 2019 -- the
# same methodology HOURLY_TUNED_SETTINGS above uses for the linear
# quantile model). For each candidate alpha, two models are fit: one
# trained on everything except 2015 and scored only on 2015, one trained
# on everything except 2019 and scored only on 2019. An alpha is only
# kept if it clears the dual bar (recall >=1.15x persistence AND
# mae_elevated <=1.15x persistence) on BOTH held-out seasons
# independently -- not pooled, not averaged. Among alphas that clear
# both, highest average F2 wins. The ordinary forward-CV folds used
# elsewhere in this file under-sample real haze events (most calendar
# months are calm), so this is a stricter, season-aware check run only
# for this one hyperparameter.
#
# Checked once against the untouched final_test holdout (2024-04-15
# onward) as confirmation, not selection: dual-bar pass rate rose from
# 5/15 to 9/15 cells vs. the prior flat/forward-CV-tuned alpha. Falls
# back to XGB_QUANTILE_ALPHA above for any (target, horizon) not listed.
#
# Known open gap: west/h=6 and central/h=24 -- no alpha cleared both
# LOSO folds; the value below is the best worst-case recall margin, not
# a confirmed pass. Same "accepted near-ceiling, not silently treated as
# a pass" status as the linreg gaps noted above.
XGB_QUANTILE_ALPHA_TUNED: dict[tuple[str, int], float] = {
    ("north", 6): 0.90, ("north", 12): 0.90, ("north", 24): 0.95,
    ("south", 6): 0.90, ("south", 12): 0.85, ("south", 24): 0.95,
    ("east", 6): 0.95, ("east", 12): 0.95, ("east", 24): 0.95,
    ("west", 6): 0.95, ("west", 12): 0.85, ("west", 24): 0.90,  # h=6, h=24: best worst-case, not a confirmed pass
    ("central", 6): 0.95, ("central", 12): 0.95, ("central", 24): 0.90,  # h=24: best worst-case, not a confirmed pass
    ("national", 6): 0.90, ("national", 12): 0.90, ("national", 24): 0.95,
}

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

# Active hourly comparison: independent selected model per region/horizon.
HOURLY_MLFLOW_EXPERIMENT = "hazecast-sg-hourly-forward"
HOURLY_TIMEZONE = "Asia/Singapore"


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
# re-check them on this set before reporting final numbers. The six-candidate
# hourly comparison (MODEL_CANDIDATES) also trains on this set when
# USE_SELECTED_FEATURES is True.
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
