"""Selection stability: a non-default winner must beat xgboost_q85 reliably, not by noise."""
from __future__ import annotations

import numpy as np
import pandas as pd

from hazecast.hourly_model import switch_confidence

DATES = pd.date_range("2026-01-01", periods=24 * 70, freq="h")


def _truth(seed=0):
    rng = np.random.default_rng(seed)
    y = np.full(len(DATES), 20.0)
    for start in rng.choice(len(DATES) - 48, 12, replace=False):   # 12 haze episodes of 1-2 days
        y[start:start + rng.integers(24, 48)] = 80.0
    return y


def test_clear_advantage_is_reliable():
    y = _truth()
    catches_all, misses_half = y.copy(), np.where(np.arange(len(y)) % 2 == 0, y, 20.0)
    assert switch_confidence(y, DATES, catches_all, misses_half) >= 0.95


def test_identical_models_never_switch():
    y = _truth()
    assert switch_confidence(y, DATES, y.copy(), y.copy()) == 0.0


def test_a_thin_noisy_edge_is_not_reliable():
    y = _truth()
    rng = np.random.default_rng(1)
    base = np.where(rng.random(len(y)) < 0.80, y, 20.0)                  # catches ~80% of haze hours
    slightly_more = np.where(rng.random(len(y)) < 0.82, y, 20.0)          # ~82%: a coin flip by weeks
    assert switch_confidence(y, DATES, slightly_more, base) < 0.90
