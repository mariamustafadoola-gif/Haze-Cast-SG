# HazeCast SG revised daily pipeline

Replace `src/hazecast/train.py`, `src/hazecast/score.py` and
`src/hazecast/config.py` with the files supplied here (or use your existing
`hazecast/` package directory). Keep the current project layout: config
still derives REPO_ROOT with `parents[2]`. No hourly module is replaced.
All original hourly settings, horizons and tuned values are unchanged.

## Behavior

Five candidates: persistence, linear regression, random forest, XGBoost
with absolute-error objective, and XGBoost quantile regression at the
existing configured quantile (0.85). The referenced chat did not expose
its earlier five-model recommendation; these extend the four candidates
in the actual uploaded train.py with random forest.

The most recent 20% of daily origins form a final holdout. The preceding
period is divided into four chronological blocks: initial training plus
three expanding forward validation folds. Before every boundary, training
rows are purged unless their label observation date is strictly earlier
than the next validation/test origin. This replaces the original
leave-one-season-out training that included future years. HAZE_SEASONS
remains in config for other modules; it no longer controls this evaluation.

MAE and RMSE measure concentrations. F2 measures elevated-event detection
using actual/predicted PM2.5 >= 56, with twice the recall weight of precision.
All candidates use the same rows and nonnegative prediction clipping.
F2 is null/NaN when there are no observed elevated events. Selection ranks
pooled out-of-fold F2 descending, breaking ties by MAE then RMSE ascending;
when validation contains no elevated observations, MAE/RMSE determine the
winner. Pooled F2 is computed from all validation predictions together,
not averaged over folds. Low-support validation still needs human review.

Only the selected candidate is evaluated on the untouched final holdout.
The final deployed model is then refitted on all complete, known labels,
including the holdout; holdout metrics describe the pre-refit evaluation.
No inference about real-world performance has been made here.

Training writes eval_summary.csv, feature_columns.json, uniquely named
model artifacts, and daily_model_manifest.json under data/models. The
manifest records each region's winner, metrics, threshold and horizon.
Persistence winners require no estimator file. The manifest is published
last; previous artifacts are retained. Run only one training publisher at
a time. Failed candidate fitting aborts publication rather than silently
changing the candidate set.

MLflow records per-fold parameters, metrics and estimator parameters, plus
final fitted models and selection metadata. MLFLOW_TRACKING_URI is honored;
the default remains the project's local SQLite store. Automatic registry
promotion is not implemented; local scoring uses the selection manifest.

Scoring keeps the existing output columns and latest_forecast.json path.
It sorts input dates, selects the latest complete finite row, warns if it
is stale, and uses the saved training feature order. Missing/corrupt selected
artifacts fail explicitly. If no manifest exists, it supports the original
xgb_REGION.json models. Load joblib files only from trusted training output.

## Dependencies and running

Use the project's Python environment with NumPy, pandas, scikit-learn,
joblib, pyarrow, MLflow and XGBoost >= 2.0 (required for quantile objective).
For example, install the accompanying requirements file, then run from the
project root with the existing hazecast package import path:

    python -m hazecast.train
    python -m hazecast.score

Daily gold is still data/gold/daily_features.parquet, one unique date per
row, numeric causal feature columns, today_REGION persistence columns,
and label_REGION next-day targets for all five regions plus national.
The daily forecast horizon still comes from FORECAST_HORIZON_DAYS.
Hourly labels/data must not be passed into this daily entry point. No
hourly forecasting or tuned-setting logic has been changed.

At least MIN_TRAIN_ROWS (100) complete rows are required in each target's
first training fold. With the defaults, roughly 500+ daily observations
are needed before accounting for missingness and horizon purging. Missing
labels, nonfinite features and low-coverage labels are excluded. Labels
whose timestamp exceeds the table's latest origin are excluded as unknown.

Upstream features must use only information available at the forecast
origin, including fire/weather publication delays. These three files
cannot verify upstream feature engineering or whether existing manually
tuned settings used future evaluation data. The hourly threshold comment
is preserved from config; its suitability for daily-mean alerts is a
project assumption, not a newly validated public-health classification.

## Validation performed

Passed on the revised source:
- Python syntax for all three files.
- Exact equality of the four original hourly setting constants.
- Actual helper execution: date sorting, duplicate-date rejection,
  exclusion of label/target columns, finite/known-label filtering,
  low-coverage filtering and strict forecast-horizon embargo.
- MAE, RMSE and F2 hand-calculated cases, including no-positive support,
  missed events and perfect detection; JSON null handling.

Not run: estimator fitting, real-data model comparison, MLflow integration,
model persistence round-trip or parquet scoring. The available runtimes
lack scikit-learn, XGBoost and MLflow, and network access to obtain them
was not granted. Consequently the files are source/helper validated,
not certified end to end. Run both commands above in the project environment
and inspect eval_summary.csv and MLflow before deployment.
