"""Hourly fire features: rolling windows on detection timestamps, using only detections
delivered by each hour.

Guards the two bugs of the earlier day-based rule (no fire features, so no forecast, from
midnight until the day's first detection; training rows using detections made after
them) and the live-edge rule (an outage is never read as "no fires").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hazecast.config import FIRE_KNOWN_HOURS_AFTER_LAST_DETECTION, FIRE_LATENCY_HOURS, SG_REGIONS, WEATHER_POINTS
from hazecast.silver_to_gold import build_gold_hourly, fire_day_for_hour

GAP = FIRE_LATENCY_HOURS + 1


def _silver(hours, detections, fire_end=None):
    """`detections`: {hour offset: count} in Sumatra. Fire grid ends at `fire_end` (hour offset)."""
    t0 = pd.Timestamp("2026-10-01 00:00")
    ts = pd.date_range(t0, periods=hours, freq="h")
    pm = pd.DataFrame({"ts": ts, **{r: 20.0 for r in SG_REGIONS}})
    pm["national_mean"] = 20.0
    we = pd.concat([pd.DataFrame({"ts": ts, "point": p, "wind_speed_kmh": 5.0, "wind_dir_deg": 200.0, "rain_mm": 0.0})
                    for p in WEATHER_POINTS], ignore_index=True)
    grid = pd.date_range(t0, periods=fire_end if fire_end is not None else hours, freq="h")
    fh = pd.DataFrame({"ts": grid, "sumatra": 0, "kalimantan": 0})
    for h, n in detections.items():
        fh.loc[h, "sumatra"] = n
    fh["total"] = fh.sumatra + fh.kalimantan
    days = pd.date_range(t0, periods=hours // 24 + 1, freq="D")
    fd = pd.DataFrame({"date": days.date, "sumatra": 1, "kalimantan": 1, "total": 2, "data_gap": False})
    return pm, we, fd, fh


def test_fire_day_for_gap_flag_switches_at_11():
    ts = pd.Series(pd.to_datetime(["2026-10-28 10:59", "2026-10-28 11:00"]))
    assert list(fire_day_for_hour(ts).dt.strftime("%m-%d")) == ["10-26", "10-27"]


def test_a_detection_counts_only_after_delivery_and_for_24_hours():
    h_det = 200                                  # one detection in hour 200
    gold = build_gold_hourly(*_silver(hours=24 * 14, detections={h_det: 7})).set_index("date")
    lag0 = gold["fire_sumatra_lag0"].reset_index(drop=True)
    on = lag0[lag0 == 7].index
    assert on.min() == h_det + GAP              # not before it was delivered: no look-ahead
    assert on.max() == h_det + GAP + 23          # stays in the 24 h window, then moves to lag1
    assert gold["fire_sumatra_lag1"].reset_index(drop=True)[h_det + GAP + 24] == 7


def test_no_row_uses_a_detection_made_after_it():
    rng = np.random.default_rng(0)
    det = {int(h): int(n) for h, n in zip(rng.choice(24 * 13, 60, replace=False), rng.integers(1, 9, 60))}
    pm, we, fd, fh = _silver(hours=24 * 14, detections=det)
    gold = build_gold_hourly(pm, we, fd, fh).reset_index(drop=True)
    counts = fh.set_index(fh.index)["sumatra"]
    for i in range(24 * 2, len(gold)):          # recompute from detections strictly delivered before hour i
        usable = counts.loc[: i - GAP]
        expected = usable.iloc[-24:].sum() if len(usable) >= 24 else np.nan
        got = gold.loc[i, "fire_sumatra_lag0"]
        assert (np.isnan(expected) and np.isnan(got)) or got == expected, i


def test_forecastable_all_morning_but_not_through_an_outage():
    # Last detection at 02:00 on day 12 (the night pass); the feed then goes quiet.
    last = 24 * 11 + 2
    pm, we, fd, fh = _silver(hours=24 * 13, detections={last: 5}, fire_end=last + 1 + FIRE_KNOWN_HOURS_AFTER_LAST_DETECTION)
    gold = build_gold_hourly(pm, we, fd, fh).set_index("date")
    day12 = gold.loc["2026-10-12"]
    known_until = gold.index[last] + pd.Timedelta(hours=FIRE_KNOWN_HOURS_AFTER_LAST_DETECTION)
    assert day12.loc[:known_until, "fire_sumatra_lag0"].notna().all()   # the whole morning is forecastable
    assert gold.loc["2026-10-13", "fire_sumatra_lag0"].isna().all()     # a silent feed is not "no fires"
