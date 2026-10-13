"""
One place to load the gold tables, for the AI team.

Use source="dummy" now. When the data team's pipeline is ready, switch to
source="real" and nothing else in your code has to change.

    from utils.load_gold import load_training_table
    df = load_training_table(source="dummy")
"""

import pandas as pd

PATHS = {
    "dummy": ("dummy_data/gold_feature_store.parquet", "dummy_data/gold_label_store.parquet"),
    "real": ("datamart/gold/feature_store", "datamart/gold/label_store"),
}
KEYS = ["region", "snapshot_date"]


def load_feature_store(source="dummy"):
    df = pd.read_parquet(PATHS[source][0])
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
    return df


def load_label_store(source="dummy"):
    df = pd.read_parquet(PATHS[source][1])
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
    return df


def load_training_table(source="dummy"):
    """Features and labels joined on region and snapshot_date."""
    return load_feature_store(source).merge(load_label_store(source), on=KEYS, how="inner")
