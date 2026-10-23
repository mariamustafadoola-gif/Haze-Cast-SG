"""Feature reduction for the HazeCast gold table (56 features -> a smaller set).

Why: many of the 56 gold features say the same thing in different ways
(for example fire_total_* is just fire_sumatra_* + fire_kalimantan_*).
Duplicates make linear and quantile-linear models unstable, slow down
training, and make the model harder to explain. This script finds the
duplicates and the features that do not help, proposes a smaller set,
and then proves the smaller set forecasts at least as well as the full one.

It works in four stages. Every decision, and the reason for it, is written
to reports/feature_reduction/feature_audit.csv.

  1. Rules: drop features that are an exact copy or sum of other features,
     or that a linear model cannot read correctly (raw compass degrees).
  2. Data quality: drop features that are mostly missing or never change.
  3. Duplicates: if two features move together almost perfectly
     (|Spearman r| above the threshold), keep the one more related to the label.
  4. Usefulness: drop features that do not improve an XGBoost model
     (permutation importance at or below zero).
  Then a validation step re-runs the leave-one-haze-season-out evaluation
  with the full and the reduced set side by side.

No leakage: stages 2 to 4 only look at rows OUTSIDE the haze seasons in
hazecast.config.HAZE_SEASONS, the same seasons used as test folds, so the
test seasons never influence which features are chosen.

The persistence columns (today_* in the daily table, now_* in the hourly
table) are always kept: each region's model needs its own current reading.

Run from the repo root, after the pipeline has built the gold table:
    export PYTHONPATH=src
    python experiments/feature_reduction.py --grain hourly --quick   # national, h=1 and h=6 only (minutes)
    python experiments/feature_reduction.py --grain hourly           # every region and horizon (slow)
    python experiments/feature_reduction.py --grain daily
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

from hazecast.config import GOLD_DIR, HAZE_SEASONS, PM25_ELEVATED_THRESHOLD, REPO_ROOT, SG_REGIONS, XGB_PARAMS

log = logging.getLogger("feature_reduction")
OUT_DIR = REPO_ROOT / "reports" / "feature_reduction"
TARGETS = SG_REGIONS + ["national"]

RULES = [
    # (test on the column name, reason)
    (lambda c: c.startswith("fire_total_"),
     "exact sum of the matching fire_sumatra_* and fire_kalimantan_* columns"),
    (lambda c: c == "upwind_total",
     "exact sum of upwind_sumatra and upwind_kalimantan"),
    (lambda c: c.startswith("wind_dir_"),
     "raw compass degrees: 359 and 1 are neighbours but look far apart to a linear model; "
     "Singapore's wind direction already enters through the upwind_* features"),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_gold(grain: str) -> pd.DataFrame:
    path = GOLD_DIR / f"{grain}_features.parquet"
    gold = pd.read_parquet(path)
    gold["date"] = pd.to_datetime(gold["date"])
    log.info("loaded %s: %d rows, %d columns", path, len(gold), gold.shape[1])
    return gold.sort_values("date").reset_index(drop=True)


def feature_columns(gold: pd.DataFrame) -> list[str]:
    return [c for c in gold.columns
            if c != "date" and not c.startswith("label_")]


def persistence_columns(features: list[str]) -> list[str]:
    return [c for c in features if c.startswith(("today_", "now_"))]


# Haze periods kept out of feature selection on top of config.HAZE_SEASONS:
# the 2026 season is still running (labels incomplete), so it is not a test
# season yet, but it will likely become one -- choosing features on it now
# would make that future test unfair.
EXTRA_HOLDOUT = {"2026": ("2026-08-01", "2026-12-31")}


def outside_haze_seasons(gold: pd.DataFrame) -> pd.Series:
    """Rows we may use to CHOOSE features (never a test season, current or likely future)."""
    in_any = pd.Series(False, index=gold.index)
    for start, end in [*HAZE_SEASONS.values(), *EXTRA_HOLDOUT.values()]:
        in_any |= gold["date"].between(pd.Timestamp(start) - pd.Timedelta(days=1),
                                       pd.Timestamp(end) + pd.Timedelta(days=1))
    return ~in_any


# ---------------------------------------------------------------------------
# selection stages
# ---------------------------------------------------------------------------
def select_features(gold: pd.DataFrame, corr_threshold: float, max_missing: float, seed: int) -> pd.DataFrame:
    features = feature_columns(gold)
    protected = set(persistence_columns(features))
    label = "label_national"
    sel = gold.loc[outside_haze_seasons(gold)]
    log.info("choosing features on %d rows outside the haze seasons", len(sel))

    audit = pd.DataFrame({"feature": features}).set_index("feature")
    audit["group"] = [f.split("_")[0] for f in features]
    audit["missing_rate"] = sel[features].isna().mean()
    audit["spearman_with_label"] = sel[features].corrwith(sel[label], method="spearman")
    audit["decision"] = "keep"
    audit["reason"] = ""

    def drop(f, reason):
        if f in protected:
            return
        if audit.at[f, "decision"] == "keep":
            audit.at[f, "decision"] = "drop"
            audit.at[f, "reason"] = reason

    # stage 1: rules
    for f in features:
        for test, reason in RULES:
            if test(f):
                drop(f, f"stage 1, rule: {reason}")
                break

    # stage 2: data quality
    for f in features:
        if audit.at[f, "missing_rate"] > max_missing:
            drop(f, f"stage 2: {audit.at[f, 'missing_rate']:.0%} missing")
        elif sel[f].nunique(dropna=True) <= 1:
            drop(f, "stage 2: never changes")

    # stage 3: near-duplicates, keep the one more related to the label
    alive = [f for f in features if audit.at[f, "decision"] == "keep"]
    corr = sel[alive].corr(method="spearman").abs()
    order = sorted(alive, key=lambda f: (f not in protected, -abs(audit.at[f, "spearman_with_label"] or 0)))
    kept: list[str] = []
    for f in order:
        twins = [k for k in kept if corr.at[f, k] > corr_threshold] if f not in protected else []
        if twins:
            k = max(twins, key=lambda t: corr.at[f, t])
            drop(f, f"stage 3: near-duplicate of {k} (|r| = {corr.at[f, k]:.2f})")
        else:
            kept.append(f)

    # stage 4: usefulness. The selection rows are split into 4 time blocks; each block is held
    # out once while XGBoost trains on the other three, and permutation importance is measured
    # on the held-out block. A feature is dropped only if it helped in NONE of the 4 blocks,
    # so a feature that matters only in some periods (fires, during smoky spells) is kept.
    alive = [f for f in features if audit.at[f, "decision"] == "keep"]
    data = sel.dropna(subset=[label]).reset_index(drop=True)
    blocks = np.array_split(np.arange(len(data)), 4)
    fold_imp = []
    for i, hold_idx in enumerate(blocks):
        fit_idx = np.concatenate([b for j, b in enumerate(blocks) if j != i])
        model = XGBRegressor(**XGB_PARAMS).fit(data.loc[fit_idx, alive], data.loc[fit_idx, label])
        imp = permutation_importance(model, data.loc[hold_idx, alive], data.loc[hold_idx, label],
                                     scoring="neg_mean_absolute_error", n_repeats=10, random_state=seed)
        fold_imp.append(imp.importances_mean)
    fold_imp = np.vstack(fold_imp)
    audit.loc[alive, "perm_importance_mean"] = fold_imp.mean(axis=0)
    audit.loc[alive, "perm_importance_best_block"] = fold_imp.max(axis=0)
    for f, best in zip(alive, fold_imp.max(axis=0)):
        if best <= 0:
            drop(f, "stage 4: did not improve forecasts in any of 4 held-out time blocks")

    audit["protected"] = audit.index.isin(protected)
    return audit.reset_index()


# ---------------------------------------------------------------------------
# validation: leave-one-haze-season-out, full set vs reduced set
# ---------------------------------------------------------------------------
def validate_daily(gold, full, reduced, targets):
    from hazecast.train import run_season_fold, _usable_seasons
    rows = []
    seasons = _usable_seasons(gold, full)
    for target in targets:
        for season, (start, end) in seasons.items():
            for name, cols in (("full", full), ("reduced", reduced)):
                for r in run_season_fold(gold, target, cols, season, start, end):
                    rows.append({**r, "feature_set": name, "n_features": len(cols), "horizon": "next day"})
    return pd.DataFrame(rows)


def validate_hourly(gold, full, reduced, targets, horizons):
    from hazecast.hourly_model import build_label, fit_predict
    from hazecast.train import _metrics
    rows = []
    for target in targets:
        for horizon in horizons:
            label_col = f"label_h{horizon}_{target}"
            gold[label_col] = build_label(gold, target, horizon)
            for season, (start, end) in HAZE_SEASONS.items():
                end_x = pd.Timestamp(end) + pd.Timedelta(days=1)
                train_start = pd.Timestamp(start) - pd.Timedelta(hours=horizon)
                for name, cols in (("full", full), ("reduced", reduced)):
                    # same rows for both sets, so the comparison is fair
                    complete = gold[full + [label_col]].notna().all(axis=1)
                    test = gold.loc[complete & gold["date"].between(start, end_x, inclusive="left")]
                    train = gold.loc[complete & ~gold["date"].between(train_start, end_x, inclusive="left")]
                    if test.empty or train.empty:
                        continue
                    if name == "full":
                        m = _metrics(test[label_col], test[f"now_{target}"].to_numpy(), PM25_ELEVATED_THRESHOLD)
                        rows.append({**m, "model": "persistence", "feature_set": "-", "n_features": 0,
                                     "target": target, "horizon": horizon, "season": season})
                    _, pred = fit_predict(train, test, cols, label_col, target, horizon)
                    m = _metrics(test[label_col], pred, PM25_ELEVATED_THRESHOLD)
                    rows.append({**m, "model": "linreg_q", "feature_set": name, "n_features": len(cols),
                                 "target": target, "horizon": horizon, "season": season})
                log.info("validated target=%s horizon=%d season=%s", target, horizon, season)
    return pd.DataFrame(rows)


def _table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown()
    except ImportError:  # tabulate not installed
        return "```\n" + df.to_string() + "\n```"


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--grain", choices=["hourly", "daily"], default="hourly")
    ap.add_argument("--corr-threshold", type=float, default=0.90)
    ap.add_argument("--max-missing", type=float, default=0.20)
    ap.add_argument("--quick", action="store_true", help="validate national only (hourly: h=1 and h=6)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    gold = load_gold(args.grain)
    audit = select_features(gold, args.corr_threshold, args.max_missing, args.seed)
    full = audit["feature"].tolist()
    reduced = audit.loc[audit["decision"] == "keep", "feature"].tolist()
    log.info("features: %d before, %d after", len(full), len(reduced))

    targets = ["national"] if args.quick else TARGETS
    if args.grain == "daily":
        val = validate_daily(gold, full, reduced, targets)
    else:
        horizons = [1, 6] if args.quick else [1, 3, 6, 12]
        val = validate_hourly(gold, full, reduced, targets, horizons)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT_DIR / f"feature_audit_{args.grain}.csv", index=False)
    val.to_csv(OUT_DIR / f"validation_{args.grain}.csv", index=False)
    (OUT_DIR / f"selected_features_{args.grain}.txt").write_text("\n".join(reduced) + "\n")

    keys = ["model", "feature_set"] + (["horizon"] if "horizon" in val else [])
    summary = (val.groupby(keys)[["mae", "mae_elevated", "recall_elevated", "precision_elevated"]]
               .mean().round(3))
    counts = audit.groupby("decision").size().to_dict()
    by_stage = audit.loc[audit.decision == "drop", "reason"].str.extract(r"^(stage \d)")[0].value_counts().sort_index()
    md = [f"# Feature reduction ({args.grain})", "",
          f"- Features before: {len(full)}", f"- Features after: {len(reduced)}",
          f"- Dropped by stage: " + ", ".join(f"{k}: {v}" for k, v in by_stage.items()), "",
          "## Average over regions and haze seasons", "", _table(summary), "",
          "## Kept features", "", ", ".join(reduced), ""]
    (OUT_DIR / f"summary_{args.grain}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
