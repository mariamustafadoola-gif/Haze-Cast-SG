"""Why the hourly fire features are rolling windows on detection timestamps.

Compares ways of attaching FIRMS fire counts to each hour, scored the same way:
xgboost_q85 (flat 0.85) per region x horizon, trained on pre-holdout rows (purged
boundary), scored once on the 2024-26 holdout, all hours and 00:00-10:59 separately.

  A  old rule: hour of Singapore day D gets FIRMS (UTC) day D-1. Reference only: live it
     leaves no features until day D's first detection (~midday), and training rows at
     00:00-10:59 include the night pass (01:00-04:59 SGT) made after them.
  B  as-of FIRMS day: D-1 from 11:00 SGT, D-2 before (no leak, but a column's age depends on the hour)
  C  B + a fire_age_hours feature
  D  always D-2 (consistent age, staler)
  E  rolling windows on detection timestamps, ending FIRE_LATENCY_HOURS full hours before
     the hour (what the pipeline now uses: silver_to_gold._fire_features_hourly)

    python experiments/fire_feature_options.py   -> reports/fire_feature_options.csv
Needs data/silver/{pm25,weather,firms_daily,firms_hourly}.parquet and a trained hourly manifest
(for the holdout boundary).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import hazecast.silver_to_gold as s2g
from hazecast.config import MODELS_DIR, REPO_ROOT, SILVER_DIR, XGB_PARAMS, HOURLY_HORIZONS
from hazecast.hourly_model import TARGETS, _prepare, _labelled, _feature_columns, _complete, _before
from hazecast.train import _metrics


def day_based(pm, we, fi, fh, fire_day):
    """Hourly gold whose fire columns come from daily features of FIRMS day fire_day(ts)."""
    gold = s2g.build_gold_hourly(pm, we, fi, fh)
    daily = s2g._fire_features(fi)
    daily["date"] = pd.to_datetime(daily["date"])
    fire_cols = [c for c in daily.columns if c != "date"]
    keyed = gold[["date"]].assign(fire_date=fire_day(gold["date"]))
    vals = keyed.merge(daily.rename(columns={"date": "fire_date"}), on="fire_date", how="left")
    gold[fire_cols] = vals[fire_cols].to_numpy()
    gold = gold.drop(columns=[c for c in gold.columns if c.startswith("upwind_")])
    up = s2g._upwind_features_hourly(gold)
    return pd.concat([gold, up], axis=1)


def main():
    pm, we = pd.read_parquet(SILVER_DIR / "pm25.parquet"), pd.read_parquet(SILVER_DIR / "weather.parquet")
    fi, fh = pd.read_parquet(SILVER_DIR / "firms_daily.parquet"), pd.read_parquet(SILVER_DIR / "firms_hourly.parquet")
    options = {
        "A old (leaky)": day_based(pm, we, fi, fh, lambda ts: ts.dt.floor("D") - pd.Timedelta(days=1)),
        "B as-of day": day_based(pm, we, fi, fh, s2g.fire_day_for_hour),
        "D always D-2": day_based(pm, we, fi, fh, lambda ts: ts.dt.floor("D") - pd.Timedelta(days=2)),
        "E rolling windows (pipeline)": s2g.build_gold_hourly(pm, we, fi, fh),
    }
    c = options["B as-of day"].copy()
    c["fire_age_hours"] = (c["date"] - (s2g.fire_day_for_hour(c["date"]) + pd.Timedelta(days=1, hours=8))).dt.total_seconds() / 3600
    options["C as-of day + age"] = c

    manifest = json.loads((MODELS_DIR / "hourly" / "model_manifest.json").read_text())
    start = pd.Timestamp(manifest["models"]["6"]["north"]["test_start"])
    out = []
    for name, gold in options.items():
        gold = _prepare(gold)
        st = start.tz_convert(gold.date.dt.tz)
        for h in HOURLY_HORIZONS:
            lab = _labelled(gold, h)
            cols = _feature_columns(lab) + (["fire_age_hours"] if "fire_age_hours" in lab.columns else [])
            for t in TARGETS:
                rows = _complete(lab, cols, t, h)
                tr, te = _before(rows, st, h), rows.loc[rows.date >= st]
                m = XGBRegressor(**{**XGB_PARAMS, "objective": "reg:quantileerror", "quantile_alpha": 0.85})
                p = np.maximum(0, m.fit(tr[cols], tr[f"label_{t}"]).predict(te[cols]))
                y = te[f"label_{t}"].to_numpy()
                for part, sel in (("all", np.ones(len(te), bool)), ("00-10h", (te.date.dt.hour < 11).to_numpy())):
                    mm = _metrics(y[sel], p[sel]); el = y[sel] >= 56
                    out.append(dict(option=name, h=h, target=t, part=part, recall=mm["tp"] / max(mm["tp"] + mm["fn"], 1),
                                    false_alarm=mm["fp"] / max((~el).sum(), 1), f2=mm["f2"], mae=mm["mae"]))
        print(name, "done", flush=True)
    r = pd.DataFrame(out)
    r.to_csv(REPO_ROOT / "reports" / "fire_feature_options.csv", index=False)
    print(r.groupby(["part", "option"])[["recall", "false_alarm", "f2", "mae"]].mean().round(3).to_string())
    print(r[r.part == "all"].groupby(["option", "h"]).recall.mean().round(3).unstack().to_string())


if __name__ == "__main__":
    main()
