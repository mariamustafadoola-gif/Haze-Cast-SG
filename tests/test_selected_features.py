"""Checks on the data team's selected feature lists (config.SELECTED_FEATURES_*).

These make sure the lists stay in step with what silver_to_gold actually
builds, so a renamed column fails here instead of silently disappearing
from training.
"""
from __future__ import annotations
import pandas as pd

from hazecast.config import SELECTED_FEATURES_HOURLY, SELECTED_FEATURES_DAILY, SG_REGIONS
from hazecast.silver_to_gold import build_gold, build_gold_hourly
from hazecast.bronze_to_silver import pm25_silver, weather_silver, firms_daily_silver
from tests.test_pipeline_smoke import pm25_bronze, weather_bronze, firms_bronze  # noqa: F401 (fixtures)


def _silver(pm25_bronze, weather_bronze, firms_bronze, tmp_path):
    paths = [tmp_path / f for f in ("pm25.parquet", "weather.parquet", "firms.parquet")]
    for df, p in zip((pm25_bronze, weather_bronze, firms_bronze), paths):
        df.to_parquet(p, index=False)
    return pm25_silver(bronze_path=paths[0]), weather_silver(bronze_path=paths[1]), firms_daily_silver(bronze_path=paths[2])


def test_lists_have_no_duplicates_or_labels():
    for lst in (SELECTED_FEATURES_HOURLY, SELECTED_FEATURES_DAILY):
        assert len(lst) == len(set(lst))
        assert not any(c.startswith("label_") or c == "date" for c in lst)


def test_persistence_features_are_always_kept():
    for r in SG_REGIONS + ["national"]:
        assert f"now_{r}" in SELECTED_FEATURES_HOURLY
        assert f"today_{r}" in SELECTED_FEATURES_DAILY


def test_every_selected_feature_exists_in_gold(pm25_bronze, weather_bronze, firms_bronze, tmp_path):
    pm25_s, weather_s, firms_s = _silver(pm25_bronze, weather_bronze, firms_bronze, tmp_path)
    hourly_cols = set(build_gold_hourly(pm25_s, weather_s, firms_s).columns)
    daily_cols = set(build_gold(pm25_s, weather_s, firms_s).columns)
    assert set(SELECTED_FEATURES_HOURLY) <= hourly_cols, set(SELECTED_FEATURES_HOURLY) - hourly_cols
    assert set(SELECTED_FEATURES_DAILY) <= daily_cols, set(SELECTED_FEATURES_DAILY) - daily_cols
