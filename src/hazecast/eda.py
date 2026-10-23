"""Exploratory data analysis (EDA) on the silver layer.

Each function answers one plain question about the data and returns a small
table. The notebook notebooks/01_eda.ipynb draws these as charts; running
this module writes the same tables to reports/eda/ so the findings can be
checked without opening a notebook.

    python -m hazecast.eda

The findings here are what the cleaning and feature choices rest on:
  - haze is rare and seasonal          -> test on the haze seasons, report recall
  - time of day barely matters         -> no hour-of-day feature needed
  - regions move together              -> one shared feature set for all regions
  - fires matter for days, fading      -> fire lags and multi-day totals
  - wind from the south and east       -> upwind features, not raw direction
  - rain lowers PM2.5 a little         -> rain totals kept as features
  - PM2.5 now predicts PM2.5 soon      -> current readings always kept
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from hazecast.config import SILVER_DIR, REPO_ROOT, SG_REGIONS, PM25_ELEVATED_THRESHOLD

log = logging.getLogger(__name__)
OUT_DIR = REPO_ROOT / "reports" / "eda"
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def load_silver(silver_dir: Path = SILVER_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pm25 = pd.read_parquet(silver_dir / "pm25.parquet")
    weather = pd.read_parquet(silver_dir / "weather.parquet")
    firms = pd.read_parquet(silver_dir / "firms_daily.parquet")
    return pm25, weather, firms


def overview(pm25: pd.DataFrame, weather: pd.DataFrame, firms: pd.DataFrame) -> pd.DataFrame:
    """What we have: size and date range of each source."""
    return pd.DataFrame([
        ("PM2.5", f"{len(pm25):,} hours", pm25["ts"].min(), pm25["ts"].max(), "5 regions + national average"),
        ("Weather", f"{len(weather):,} place-hours", weather["ts"].min(), weather["ts"].max(),
         f"{weather['point'].nunique()} places"),
        ("Fires", f"{len(firms):,} days", pd.Timestamp(firms["date"].min()), pd.Timestamp(firms["date"].max()),
         "daily counts for Sumatra and Kalimantan"),
    ], columns=["source", "size", "from", "to", "detail"])


def pm25_spread(pm25: pd.DataFrame) -> pd.DataFrame:
    """How high does PM2.5 usually go? Percentiles of the national hourly reading."""
    n = pm25["national_mean"].dropna()
    q = n.quantile([0.5, 0.9, 0.99, 0.999])
    return pd.DataFrame({
        "measure": ["Typical hour (median)", "1 hour in 10 is above", "1 hour in 100 is above",
                    "1 hour in 1,000 is above", "Share of hours unhealthy (56 or more)"],
        "value": [f"{q[0.5]:.0f}", f"{q[0.9]:.0f}", f"{q[0.99]:.0f}", f"{q[0.999]:.0f}",
                  f"{(n >= PM25_ELEVATED_THRESHOLD).mean():.1%}"],
    })


def by_month(pm25: pd.DataFrame, firms: pd.DataFrame) -> pd.DataFrame:
    """When in the year does haze happen, and when do fires burn?"""
    n = pm25.set_index("ts")["national_mean"]
    f = firms.set_index(pd.to_datetime(firms["date"]))
    out = pd.DataFrame({
        "unhealthy_hours": (n >= PM25_ELEVATED_THRESHOLD).groupby(n.index.month).sum(),
        "avg_pm25": n.groupby(n.index.month).mean().round(1),
        "fires_sumatra_k": (f["sumatra"].groupby(f.index.month).sum() / 1000).round(1),
        "fires_kalimantan_k": (f["kalimantan"].groupby(f.index.month).sum() / 1000).round(1),
    })
    out.index = pd.Index(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], name="month")
    return out


def by_hour(pm25: pd.DataFrame) -> pd.DataFrame:
    """Does the time of day matter?"""
    n = pm25.set_index("ts")["national_mean"]
    return n.groupby(n.index.hour).mean().round(1).rename("avg_pm25").rename_axis("hour").to_frame()


def region_similarity(pm25: pd.DataFrame) -> pd.DataFrame:
    """Do the 5 regions move together? Correlation of hourly readings (1 = identical)."""
    return pm25[SG_REGIONS].corr().round(2)


def fire_lag_link(pm25: pd.DataFrame, firms: pd.DataFrame, max_lag: int = 7) -> pd.DataFrame:
    """How strongly do fires N days ago relate to today's PM2.5? (rank correlation, daily)"""
    daily = pm25.set_index("ts")["national_mean"].resample("D").mean()
    f = firms.set_index(pd.to_datetime(firms["date"])).reindex(daily.index)
    rows = []
    for lag in range(max_lag + 1):
        rows.append({
            "days_ago": lag,
            "sumatra": round(daily.corr(f["sumatra"].shift(lag), method="spearman"), 2),
            "kalimantan": round(daily.corr(f["kalimantan"].shift(lag), method="spearman"), 2),
        })
    return pd.DataFrame(rows).set_index("days_ago")


def _daily_sg_weather(weather: pd.DataFrame) -> pd.DataFrame:
    sg = weather[weather["point"] == "singapore"].set_index("ts")
    rad = np.radians(sg["wind_dir_deg"])
    sin_d, cos_d = np.sin(rad).resample("D").mean(), np.cos(rad).resample("D").mean()
    return pd.DataFrame({
        "wind_dir": (np.degrees(np.arctan2(sin_d, cos_d)) + 360) % 360,
        "rain_mm": sg["rain_mm"].resample("D").sum(),
    })


def wind_on_haze_days(pm25: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Where does Singapore's wind come from on haze days vs normal days? (share of days)"""
    daily = pm25.set_index("ts")["national_mean"].resample("D").mean().rename("pm25")
    d = _daily_sg_weather(weather).join(daily, how="inner").dropna()
    d["from"] = [COMPASS[int(((a + 22.5) % 360) // 45)] for a in d["wind_dir"]]
    d["day"] = np.where(d["pm25"] >= PM25_ELEVATED_THRESHOLD, "haze days", "normal days")
    t = pd.crosstab(d["from"], d["day"], normalize="columns").reindex(COMPASS).fillna(0)
    return (t * 100).round(0)


def rain_effect(pm25: pd.DataFrame, weather: pd.DataFrame, wet_mm: float = 5.0) -> pd.DataFrame:
    """Is PM2.5 lower on rainy days in Singapore?"""
    daily = pm25.set_index("ts")["national_mean"].resample("D").mean().rename("pm25")
    d = _daily_sg_weather(weather).join(daily, how="inner").dropna()
    d["day"] = np.where(d["rain_mm"] >= wet_mm, f"rainy (≥{wet_mm:g} mm)", "dry or light rain")
    return d.groupby("day")["pm25"].agg(days="size", avg_pm25="mean").round(1)


def persistence(pm25: pd.DataFrame, hours=(1, 3, 6, 12, 24, 48)) -> pd.DataFrame:
    """How much does PM2.5 now tell us about PM2.5 later? (correlation with itself N hours ahead)"""
    n = pm25.set_index("ts")["national_mean"]
    return pd.DataFrame({"hours_ahead": list(hours),
                         "correlation": [round(n.autocorr(h), 2) for h in hours]}).set_index("hours_ahead")


ALL = {
    "overview": overview, "pm25_spread": pm25_spread, "by_month": by_month, "by_hour": by_hour,
    "region_similarity": region_similarity, "fire_lag_link": fire_lag_link,
    "wind_on_haze_days": wind_on_haze_days, "rain_effect": rain_effect, "persistence": persistence,
}


def write_reports(out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pm25, weather, firms = load_silver()
    args = {"pm25": pm25, "weather": weather, "firms": firms}
    for name, fn in ALL.items():
        params = fn.__code__.co_varnames[: fn.__code__.co_argcount]
        table = fn(**{k: v for k, v in args.items() if k in params})
        table.to_csv(out_dir / f"{name}.csv")
        log.info("wrote %s", out_dir / f"{name}.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    write_reports()
