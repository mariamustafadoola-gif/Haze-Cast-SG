"""Silver gap filling: short gaps are filled, long gaps are left empty."""
from __future__ import annotations
import numpy as np
import pandas as pd

from hazecast.bronze_to_silver import fill_short_gaps
from hazecast.config import MAX_FILL_HOURS


def _series(missing_hours):
    idx = pd.date_range("2024-01-01", periods=30, freq="h")
    df = pd.DataFrame({"v": np.arange(30, dtype=float), "dir": 350.0}, index=idx)
    df.iloc[missing_hours, :] = np.nan
    return df


def test_short_gap_is_filled_by_straight_line():
    df, filled = fill_short_gaps(_series([10, 11]), ["v", "dir"], carry_forward=("dir",))
    assert filled.sum() == 2
    assert df["v"].iloc[10] == 10.0 and df["v"].iloc[11] == 11.0
    assert df["dir"].iloc[11] == 350.0  # direction carried forward, not averaged


def test_long_gap_is_left_empty():
    hours = list(range(10, 10 + MAX_FILL_HOURS + 1))
    df, filled = fill_short_gaps(_series(hours), ["v"])
    assert filled.sum() == 0
    assert df["v"].iloc[hours].isna().all()


def test_gap_at_the_edge_is_left_empty():
    df, filled = fill_short_gaps(_series([0, 29]), ["v"])
    assert filled.sum() == 0
    assert df["v"].iloc[[0, 29]].isna().all()
