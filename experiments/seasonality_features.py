"""Does the hourly model need explicit seasonal inputs?

The selected features carry season only through their drivers (fire counts, wind speed,
rain, upwind alignment). This compares adding seasonal inputs, scored the same way as
experiments/fire_feature_options.py: xgboost_q85 per region x horizon, trained on
pre-holdout rows (purged boundary), scored once on the 2024-26 holdout.

  base         the selected features (what the pipeline uses)
  +doy         day of year as sin/cos (a calendar seasonal cycle)
  +monsoon     the four Singapore monsoon phases as indicators (config.MONSOON_SEASONS)
  +wind_dir    wind direction as sin/cos at Singapore and the four fire-region points
               (the physical monsoon signal: the SW monsoon carries Sumatran smoke)

Reported per variant: all holdout hours; calm fire seasons (Sep-Oct 2024 and 2025, where a
calendar feature could raise false alarms); the 2026 haze (Aug-Oct 2026).

    python experiments/seasonality_features.py   -> reports/seasonality_features.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from hazecast.config import GOLD_DIR, HOURLY_HORIZONS, MODELS_DIR, MONSOON_SEASONS, REPO_ROOT, WEATHER_POINTS, XGB_PARAMS
from hazecast.hourly_model import TARGETS, _before, _complete, _feature_columns, _labelled, _prepare
from hazecast.train import _metrics


def add_seasonal(gold: pd.DataFrame) -> dict[str, list[str]]:
    doy = gold.date.dt.dayofyear.to_numpy()
    gold["doy_sin"], gold["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    month = gold.date.dt.month
    monsoon = []
    for name, months in MONSOON_SEASONS.items():
        col = "season_" + name.lower().replace(" ", "_").replace("-", "_")
        gold[col] = month.isin(months).astype(float)
        monsoon.append(col)
    wind = []
    for p in WEATHER_POINTS:
        rad = np.radians(gold[f"wind_dir_{p}"].to_numpy(dtype=float))
        gold[f"wind_dir_{p}_sin"], gold[f"wind_dir_{p}_cos"] = np.sin(rad), np.cos(rad)
        wind += [f"wind_dir_{p}_sin", f"wind_dir_{p}_cos"]
    return {"base": [], "+doy": ["doy_sin", "doy_cos"], "+monsoon": monsoon, "+wind_dir": wind}


def main():
    gold = _prepare(pd.read_parquet(GOLD_DIR / "hourly_features.parquet"))
    variants = add_seasonal(gold)
    manifest = json.loads((MODELS_DIR / "hourly" / "model_manifest.json").read_text())
    start = pd.Timestamp(manifest["models"]["6"]["north"]["test_start"]).tz_convert(gold.date.dt.tz)
    parts = {
        "all": lambda d: np.ones(len(d), bool),
        "calm Sep-Oct 2024-25": lambda d: (d.dt.month.isin([9, 10]) & d.dt.year.isin([2024, 2025])).to_numpy(),
        "2026 haze (Aug-Oct)": lambda d: ((d.dt.year == 2026) & d.dt.month.isin([8, 9, 10])).to_numpy(),
    }
    out = []
    for h in HOURLY_HORIZONS:
        lab = _labelled(gold, h)
        base = _feature_columns(lab)
        for name, extra in variants.items():
            cols = base + extra
            for t in TARGETS:
                rows = _complete(lab, cols, t, h)
                tr, te = _before(rows, start, h), rows.loc[rows.date >= start]
                m = XGBRegressor(**{**XGB_PARAMS, "objective": "reg:quantileerror", "quantile_alpha": 0.85})
                p = np.maximum(0, m.fit(tr[cols], tr[f"label_{t}"]).predict(te[cols]))
                y = te[f"label_{t}"].to_numpy()
                for part, sel_fn in parts.items():
                    sel = sel_fn(te.date)
                    mm = _metrics(y[sel], p[sel])
                    el = y[sel] >= 56
                    out.append(dict(variant=name, h=h, target=t, part=part, n_elevated=int(el.sum()),
                                    recall=mm["tp"] / max(mm["tp"] + mm["fn"], 1) if el.any() else np.nan,
                                    false_alarm=mm["fp"] / max((~el).sum(), 1), f2=mm["f2"], mae=mm["mae"],
                                    coverage=float((y[sel] <= p[sel]).mean())))
            print(f"h={h} {name} done", flush=True)
    r = pd.DataFrame(out)
    r.to_csv(REPO_ROOT / "reports" / "seasonality_features.csv", index=False)
    summary = r.groupby(["part", "variant"])[["recall", "false_alarm", "f2", "mae", "coverage"]].mean().round(3)
    print(summary.to_string())


if __name__ == "__main__":
    main()
