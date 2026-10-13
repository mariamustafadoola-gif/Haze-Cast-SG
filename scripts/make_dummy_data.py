"""
Make DUMMY gold tables so the AI team can start building models before the
real data pipeline is ready.

The dummy tables have exactly the same columns, types and keys as the real
gold tables will have (see docs/data_contract.md). Only the values are made up.
They follow simple, realistic patterns (fires plus south-west winds raise
PM2.5, rain lowers it, haze seasons around Sep to Oct), so a model trained on
them will learn something, but the scores mean nothing.

Run from the project folder:
    python scripts/make_dummy_data.py
Output:
    dummy_data/gold_feature_store.parquet  (+ .csv copy)
    dummy_data/gold_label_store.parquet    (+ .csv copy)
"""

import os

import numpy as np
import pandas as pd

SEED = 611
START, END = "2016-02-01", "2024-12-31"
REGIONS = ["north", "south", "east", "west", "central"]
REGION_OFFSET = {"north": 1.5, "south": -0.5, "east": 0.0, "west": 2.5, "central": 0.5}  # small fixed differences
HAZE_EPISODES = [("2019-09-05", "2019-10-05", 1.0), ("2023-09-25", "2023-10-12", 0.6),
                 ("2016-08-20", "2016-09-10", 0.4), ("2021-09-10", "2021-09-20", 0.25)]
OUT_DIR = "dummy_data"


def build_daily_drivers(rng, days):
    """Island-wide daily drivers: fires, wind and rain."""
    doy = days.dayofyear.values
    # dry season fires peak around August to October
    season = np.clip(np.sin((doy - 150) / 365 * 2 * np.pi), 0, None) ** 2
    episode = np.zeros(len(days))
    for s, e, strength in HAZE_EPISODES:
        mask = (days >= s) & (days <= e)
        episode[mask] = strength * np.sin(np.linspace(0, np.pi, mask.sum()))
    fire_level = 30 + 400 * season + 4000 * episode
    fire_count_24h = rng.poisson(fire_level * rng.lognormal(0, 0.35, len(days)))
    frp_per_fire = rng.gamma(4, 6, len(days))
    # south-west monsoon (June to September) blows smoke from Sumatra towards Singapore
    sw_monsoon = ((days.month >= 6) & (days.month <= 9)).astype(int)
    wind_from_sw_share = np.clip(0.15 + 0.55 * sw_monsoon + rng.normal(0, 0.12, len(days)), 0, 1)
    wind_dir_deg = np.where(rng.random(len(days)) < wind_from_sw_share, rng.normal(225, 20, len(days)), rng.normal(45, 40, len(days))) % 360
    wind_speed = np.clip(rng.gamma(4, 0.9, len(days)), 0.3, None)
    rain = np.where(rng.random(len(days)) < 0.45, rng.gamma(1.2, 9, len(days)), 0.0)
    return pd.DataFrame({
        "snapshot_date": days,
        "fire_count_24h": fire_count_24h,
        "fire_frp_sum_24h": fire_count_24h * frp_per_fire,
        "wind_speed_mean_24h": wind_speed,
        "wind_dir_deg": wind_dir_deg,
        "wind_from_sw_share_24h": wind_from_sw_share,
        "rain_sum_24h": rain,
        "is_sw_monsoon": sw_monsoon,
    })


def main():
    rng = np.random.default_rng(SEED)
    days = pd.date_range(START, END, freq="D")
    d = build_daily_drivers(rng, days)

    # rolling 3-day fire totals, known by 06:00 (uses today and the two days before)
    d["fire_count_72h"] = d["fire_count_24h"].rolling(3, min_periods=1).sum().astype(int)
    d["fire_frp_sum_72h"] = d["fire_frp_sum_24h"].rolling(3, min_periods=1).sum()
    d["wind_dir_sin"] = np.sin(np.radians(d["wind_dir_deg"]))
    d["wind_dir_cos"] = np.cos(np.radians(d["wind_dir_deg"]))

    # island-wide smoke signal that drives PM2.5, lagging the fires by about a day
    smoke = (d["fire_frp_sum_72h"] / 6000) * (0.3 + d["wind_from_sw_share_24h"]) / (1 + d["rain_sum_24h"] / 15)

    rows_f, rows_l = [], []
    for region in REGIONS:
        noise = rng.normal(0, 3, (len(days), 4))
        base = 14 + REGION_OFFSET[region]
        # hourly-style series summarised per day: past window (features) and future windows (labels)
        past24 = base + 0.9 * smoke.shift(1).fillna(0).values + noise[:, 0]
        now = base + 1.0 * smoke.values + noise[:, 1]
        next6 = base + 1.05 * smoke.values + noise[:, 2]
        next12 = base + 1.1 * (0.6 * smoke.values + 0.4 * smoke.shift(-1).fillna(0).values) + noise[:, 3]
        next24 = base + 1.1 * (0.3 * smoke.values + 0.7 * smoke.shift(-1).fillna(0).values) + rng.normal(0, 3.5, len(days))
        clip = lambda a: np.clip(a, 3, None).round(1)
        f = d[["snapshot_date", "fire_count_24h", "fire_frp_sum_24h", "fire_count_72h", "fire_frp_sum_72h",
               "wind_speed_mean_24h", "wind_dir_sin", "wind_dir_cos", "wind_from_sw_share_24h", "rain_sum_24h", "is_sw_monsoon"]].copy()
        f.insert(0, "region", region)
        f["pm25_now"] = clip(now)
        f["pm25_mean_6h"] = clip(0.6 * now + 0.4 * past24)
        f["pm25_mean_24h"] = clip(past24)
        f["pm25_max_24h"] = clip(np.maximum(now, past24) + np.abs(rng.normal(4, 2, len(days))))
        f["pm25_change_6h"] = (f["pm25_now"] - f["pm25_mean_6h"]).round(1)
        f["day_of_year"] = days.dayofyear.astype("int64")
        f["month"] = days.month.astype("int64")
        rows_f.append(f)
        l = pd.DataFrame({"region": region, "snapshot_date": days,
                          "label_pm25_mean_next_6h": clip(next6),
                          "label_pm25_mean_next_12h": clip(next12),
                          "label_pm25_mean_next_24h": clip(next24)})
        rows_l.append(l)

    fs = pd.concat(rows_f, ignore_index=True)
    ls = pd.concat(rows_l, ignore_index=True)

    # light missing values, as the real data will have some gaps
    for col in ["pm25_now", "pm25_mean_6h", "rain_sum_24h", "wind_speed_mean_24h"]:
        fs.loc[rng.random(len(fs)) < 0.01, col] = np.nan

    # tidy types and order to match the data contract
    fs["snapshot_date"] = fs["snapshot_date"].dt.date
    ls["snapshot_date"] = ls["snapshot_date"].dt.date
    for c in ["fire_frp_sum_24h", "fire_frp_sum_72h", "wind_speed_mean_24h", "wind_dir_sin", "wind_dir_cos",
              "wind_from_sw_share_24h", "rain_sum_24h"]:
        fs[c] = fs[c].astype(float).round(3)
    fs["snapshot_ts"] = pd.to_datetime(fs["snapshot_date"].astype(str) + " 06:00:00")
    fs = fs[["region", "snapshot_date", "snapshot_ts",
             "pm25_now", "pm25_mean_6h", "pm25_mean_24h", "pm25_max_24h", "pm25_change_6h",
             "fire_count_24h", "fire_frp_sum_24h", "fire_count_72h", "fire_frp_sum_72h",
             "wind_speed_mean_24h", "wind_dir_sin", "wind_dir_cos", "wind_from_sw_share_24h", "rain_sum_24h",
             "day_of_year", "month", "is_sw_monsoon"]]
    ls["label_def"] = "mean_pm25_next_window_ugm3"
    ls["is_haze_next_24h"] = (ls["label_pm25_mean_next_24h"] > 55).astype(int)  # NEA "Elevated" band starts at 56

    os.makedirs(OUT_DIR, exist_ok=True)
    fs.to_parquet(f"{OUT_DIR}/gold_feature_store.parquet", index=False)
    ls.to_parquet(f"{OUT_DIR}/gold_label_store.parquet", index=False)
    fs.to_csv(f"{OUT_DIR}/gold_feature_store.csv", index=False)
    ls.to_csv(f"{OUT_DIR}/gold_label_store.csv", index=False)
    print(f"feature store: {len(fs):,} rows, {fs.shape[1]} columns")
    print(f"label store:   {len(ls):,} rows, haze days (next 24 h above 55): {ls['is_haze_next_24h'].sum():,}")


if __name__ == "__main__":
    main()
