"""Hourly multi-horizon PM2.5 forecaster: 1/3/6/12h ahead, refreshed hourly.

This is the consolidated, "shippable" version of what was explored across
a long run of ad hoc scripts (kept under experiments/ for the record -- see
experiments/README.md). Nothing here is a new model; it packages the SAME
per-region quantile linear regression approach already validated there, so
the tuned settings live in one place (hazecast.config.HOURLY_TUNED_SETTINGS)
instead of being copy-pasted across a dozen one-off scripts.

Design, in brief (see the project log for the full reasoning trail):
  - One QuantileRegressor per (region, horizon), reusing the SAME feature
    set as the 1h model -- h=3/6/12 labels are built by shifting the
    target region's own reading further into the future; no new features
    and no future weather are needed.
  - Deliberately upward-biased (high quantile, e.g. 0.85-0.91) rather than
    MAE-minimizing: this project's own metric is recall_elevated (catch
    real haze) and mae_elevated (how close the number is) together, with
    persistence (this hour's reading carried forward) as the baseline to
    beat. Over-reporting when PM2.5 is already trending high was judged
    the safer failure mode for a public-health tool than an unbiased
    number that risks under-warning -- see the deployment-decision note
    in the project log.
  - Calibration caveat (also in the project log): during actually-elevated
    hours specifically, empirical coverage runs well below the nominal
    quantile everywhere, and in two cells (north/2015, central/2015) it's
    at or below a coin flip. The upward bias is real on average but
    weaker, and in a couple of cases absent, exactly when it matters most
    -- worth restating in the write-up, not just in this docstring.
"""
from __future__ import annotations
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from hazecast.config import (
    GOLD_DIR, MODELS_DIR, HOURLY_HORIZONS, HOURLY_TUNED_SETTINGS, PM25_ELEVATED_THRESHOLD,
)
from hazecast.train import _metrics  # same elevated-recall/MAE metrics as the daily model

log = logging.getLogger(__name__)

TARGETS = ["north", "south", "east", "west", "central", "national"]
NON_FEATURE_COLS = {"date"} | {f"now_{t}" for t in TARGETS}
TRAIN_SUBSAMPLE_ROWS = 20_000  # QuantileRegressor's LP solver scales poorly past this
TRAIN_SEED = 42  # checked against seed=123 for robustness; see project log


def _feature_columns(gold: pd.DataFrame) -> list[str]:
    cols = [c for c in gold.columns if c not in NON_FEATURE_COLS and not c.startswith("label_h")]
    return cols


def get_tuned_setting(target: str, horizon: int) -> tuple[float, float]:
    """(quantile, alpha) for this region/horizon, falling back to the
    region's h=1 (primary) setting when this horizon has no override --
    see the "known open gaps" note on HOURLY_TUNED_SETTINGS for which
    fallbacks are a confirmed-fine reuse (h=6/h=12 everywhere, central/west
    at h=3) versus an accepted shortfall (west at h=3; national at h=1/h=3
    use their own best-found entry, not a silent pass).
    """
    if (target, horizon) in HOURLY_TUNED_SETTINGS:
        return HOURLY_TUNED_SETTINGS[(target, horizon)]
    return HOURLY_TUNED_SETTINGS[(target, 1)]


def build_label(gold: pd.DataFrame, target: str, horizon: int) -> pd.Series:
    """label_h{horizon}_{target} = that region's own reading, shifted
    `horizon` hours earlier in the row so each row's features (as of "now")
    line up with the reading `horizon` hours ahead.
    """
    return gold[f"now_{target}"].shift(-horizon)


def fit_predict(
    train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str],
    label_col: str, target: str, horizon: int,
) -> "tuple[QuantileRegressor, pd.Series]":
    quantile, alpha = get_tuned_setting(target, horizon)
    train_sub = train.sample(n=min(TRAIN_SUBSAMPLE_ROWS, len(train)), random_state=TRAIN_SEED)
    model = QuantileRegressor(quantile=quantile, alpha=alpha, solver="highs").fit(
        train_sub[feature_cols], train_sub[label_col]
    )
    pred = model.predict(test[feature_cols])
    return model, pred


def evaluate_all_horizons(gold: pd.DataFrame, haze_seasons: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """LOSO evaluation across every (target, horizon, season), persistence
    included, for a before-you-ship sanity check. Mirrors the ad hoc sweep
    scripts this consolidates; see experiments/ for the original runs this
    was validated against.
    """
    feature_cols = _feature_columns(gold)
    rows = []
    for target in TARGETS:
        now_col = f"now_{target}"
        for horizon in HOURLY_HORIZONS:
            label_col = f"label_h{horizon}_{target}"
            gold[label_col] = build_label(gold, target, horizon)
            complete = gold[feature_cols + [label_col]].notna().all(axis=1)
            for season_name, (start, end) in haze_seasons.items():
                end_exclusive = pd.Timestamp(end) + pd.Timedelta(days=1)
                test_mask = complete & gold["date"].between(start, end_exclusive, inclusive="left")
                train_mask = complete & ~gold["date"].between(start, end_exclusive, inclusive="left")
                train, test = gold.loc[train_mask], gold.loc[test_mask]
                if test.empty or train.empty:
                    continue

                persistence_pred = test[now_col].to_numpy()
                m = _metrics(test[label_col], persistence_pred, PM25_ELEVATED_THRESHOLD)
                m.update(model="persistence", target=target, horizon=horizon, season=season_name)
                rows.append(m)

                _, pred = fit_predict(train, test, feature_cols, label_col, target, horizon)
                m = _metrics(test[label_col], pred, PM25_ELEVATED_THRESHOLD)
                q, a = get_tuned_setting(target, horizon)
                m.update(model="linreg_q", variant=f"q{q}_a{a}", target=target, horizon=horizon, season=season_name)
                rows.append(m)
        log.info("evaluated: target=%s", target)
    return pd.DataFrame(rows)


def train_final_hourly_models(gold: pd.DataFrame) -> dict[tuple[str, int], Path]:
    """Fit one QuantileRegressor per (region, horizon) on ALL available
    labelled rows and persist to data/models/hourly/ -- this is what the
    live hourly scoring loop should load (not yet built; see project log
    "pending tasks"). Mirrors train.train_final_models()'s structure for
    the daily model.
    """
    feature_cols = _feature_columns(gold)
    out_dir = MODELS_DIR / "hourly"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for target in TARGETS:
        for horizon in HOURLY_HORIZONS:
            label_col = f"label_h{horizon}_{target}"
            if label_col not in gold.columns:
                gold[label_col] = build_label(gold, target, horizon)
            complete = gold[feature_cols + [label_col]].notna().all(axis=1)
            train = gold.loc[complete]
            if len(train) < 100:
                log.warning("target=%s horizon=%d: only %d fully-featured rows, skipping", target, horizon, len(train))
                continue
            quantile, alpha = get_tuned_setting(target, horizon)
            train_sub = train.sample(n=min(TRAIN_SUBSAMPLE_ROWS, len(train)), random_state=TRAIN_SEED)
            model = QuantileRegressor(quantile=quantile, alpha=alpha, solver="highs").fit(
                train_sub[feature_cols], train_sub[label_col]
            )
            out_path = out_dir / f"linreg_q_{target}_h{horizon}.joblib"
            joblib.dump({"model": model, "feature_cols": feature_cols, "quantile": quantile, "alpha": alpha}, out_path)
            paths[(target, horizon)] = out_path
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from hazecast.config import HAZE_SEASONS

    gold_df = pd.read_parquet(GOLD_DIR / "hourly_features.parquet")
    gold_df["date"] = pd.to_datetime(gold_df["date"])
    gold_df = gold_df.sort_values("date").reset_index(drop=True)

    eval_df = evaluate_all_horizons(gold_df, HAZE_SEASONS)
    summary_path = MODELS_DIR / "hourly_eval_summary.csv"
    eval_df.to_csv(summary_path, index=False)
    log.info("Wrote hourly multi-horizon evaluation summary (%d rows) to %s", len(eval_df), summary_path)

    model_paths = train_final_hourly_models(gold_df)
    log.info("Trained and saved %d final hourly models to data/models/hourly/", len(model_paths))
