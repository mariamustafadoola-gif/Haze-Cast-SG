"""Charts and small helpers for the data team notebooks
(notebooks/01_eda.ipynb and notebooks/02_cleaning_and_feature_selection.ipynb).

Kept out of the notebook so its cells stay short and easy to read.
Nothing here changes the data; it only counts, checks and draws.
"""
from __future__ import annotations

import glob
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from hazecast.config import (
    RAW_DIR, HAZE_SEASONS, PM25_ELEVATED_THRESHOLD, FIRE_REGIONS, SG_REGIONS,
)

# ---------------------------------------------------------------------------
# Look and feel
# ---------------------------------------------------------------------------
# Sky palette: blues for the data, an orange haze accent for "unhealthy".
BLUE = "#1F5FA8"        # main series / kept
SKY = "#2EA3C9"         # second series (Kalimantan, normal days)
SKY_LIGHT = "#D6ECF7"   # highlight backgrounds
CLOUD = "#C9D4DE"       # dropped / "all 56" / not important
MIST = "#EEF3F7"        # light card backgrounds
HAZE = "#E07B24"        # unhealthy threshold, haze seasons, warnings
NAVY = "#0B2545"        # titles and dark text
SLATE = "#5C6B7A"       # secondary text

plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#C9D4DE",
    "axes.labelcolor": SLATE,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "axes.grid": True,
    "grid.color": "#E8EEF3",
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "axes.axisbelow": True,
})


def load_feature_reduction(repo_root: Path):
    """Import experiments/feature_reduction.py as a module (it is a script, not a package)."""
    path = repo_root / "experiments" / "feature_reduction.py"
    spec = importlib.util.spec_from_file_location("feature_reduction", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Step 1: what came in, and what the raw -> bronze step removed
# ---------------------------------------------------------------------------
def raw_file_table() -> pd.DataFrame:
    rows = []
    for source in ["pm25", "weather", "firms"]:
        for f in sorted(glob.glob(str(RAW_DIR / source / "*.csv"))):
            p = Path(f)
            with open(p, "rb") as fh:
                n_lines = sum(1 for _ in fh)
            header_lines = 4 if source == "weather" else 1
            rows.append({
                "source": source,
                "file": p.name,
                "size (MB)": round(p.stat().st_size / 1e6, 1),
                "rows": n_lines - header_lines,
            })
    return pd.DataFrame(rows)


def style_checks(df: pd.DataFrame):
    def colour(v):
        return (f"background-color:{SKY_LIGHT}; color:{BLUE}; font-weight:bold" if v == "Pass"
                else f"background-color:#FBE6D3; color:{HAZE}; font-weight:bold")
    return df.style.map(colour, subset=["result"]).hide(axis="index")


def plot_pm25_history(pm25_s: pd.DataFrame):
    daily = pm25_s.set_index("ts")["national_mean"].resample("D").mean()
    fig, ax = plt.subplots(figsize=(11, 3.8))
    seasons = dict(HAZE_SEASONS)
    if daily.index.max() >= pd.Timestamp("2026-09-01"):
        seasons["2026 (now)"] = ("2026-08-01", str(daily.index.max().date()))
    for name, (start, end) in seasons.items():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=HAZE, alpha=0.12, lw=0)
        ax.text(pd.Timestamp(start), daily.max() * 0.97, f" {name}", color=HAZE,
                fontsize=10, fontweight="bold", va="top", ha="right" if "now" in name else "left")
    ax.plot(daily.index, daily.values, color=BLUE, lw=0.9)
    ax.axhline(PM25_ELEVATED_THRESHOLD, color=HAZE, ls="--", lw=1.2)
    ax.text(pd.Timestamp("2021-06-01"), PM25_ELEVATED_THRESHOLD + 3, "unhealthy from 56", color=HAZE,
            ha="center", fontsize=10)
    ax.set_title("Singapore's daily average PM2.5, 2014 to 2026")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_ylim(0, None)
    plt.show()
    elevated = (daily >= PM25_ELEVATED_THRESHOLD).groupby(daily.index.year).sum()
    table = elevated[elevated > 0].rename("days at or above 56").to_frame().T
    table.columns.name = "year"
    return table


def plot_fires_per_year(firms_s: pd.DataFrame):
    df = firms_s.assign(year=pd.to_datetime(firms_s["date"]).dt.year)
    yearly = df.groupby("year")[["sumatra", "kalimantan"]].sum() / 1000
    x = np.arange(len(yearly))
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.bar(x - 0.2, yearly["sumatra"], 0.38, color=BLUE, label="Sumatra")
    ax.bar(x + 0.2, yearly["kalimantan"], 0.38, color=SKY, label="Kalimantan")
    ax.set_xticks(x, yearly.index)
    ax.set_ylabel("fire detections (thousands)")
    ax.set_title("Fire detections per year: the big haze years stand out")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    plt.show()


# ---------------------------------------------------------------------------
# Step 3 and 4: features
# ---------------------------------------------------------------------------
def feature_group_table(features: list[str]) -> pd.DataFrame:
    def group(f):
        if f.startswith("fire_"):
            return "Fire hotspots", "how many fires, today and over recent days"
        if f.startswith(("wind_", "rain_")):
            return "Weather", "wind and rain at Singapore and 4 places near the fires"
        if f.startswith("upwind_"):
            return "Wind blowing smoke our way", "fires multiplied by how much the wind points at Singapore"
        return "PM2.5 right now", "the current reading in each region"
    g = pd.DataFrame([(*group(f), f) for f in features], columns=["group", "what it means", "feature"])
    return (g.groupby(["group", "what it means"], sort=False).size()
             .rename("number of features").reset_index())


def show_compass_problem():
    fig = plt.figure(figsize=(10, 3.4))
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    for deg, c in [(359, BLUE), (1, SKY)]:
        ax.annotate("", xy=(np.radians(deg), 1), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2.5))
    ax.set_yticks([]); ax.set_title("359° and 1° on a compass:\nalmost the same wind", fontsize=11)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.barh(["1°", "359°"], [1, 359], color=[SKY, BLUE], height=0.5)
    ax2.set_xlim(0, 380); ax2.set_title("The same two winds as plain numbers:\nthey look 358 apart", fontsize=11)
    ax2.grid(axis="y", visible=False)
    plt.tight_layout(); plt.show()


def plot_twin_heatmap(gold: pd.DataFrame, region: str = "sumatra"):
    cols = [f"fire_{region}_{c}" for c in ["lag0", "lag1", "lag2", "lag3", "roll3", "roll7", "log_lag0"]]
    corr = gold[cols].corr(method="spearman")
    short = [c.replace(f"fire_{region}_", "") for c in cols]
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    im = ax.imshow(corr.values, cmap="Blues", vmin=0.5, vmax=1)
    ax.set_xticks(range(len(cols)), short, rotation=0)
    ax.set_yticks(range(len(cols)), short)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.85 else NAVY, fontweight="bold" if (v > 0.9 and i != j) else None)
    ax.set_title(f"How closely the {region.capitalize()} fire features move together\n(1.00 = identical, above 0.90 = near-copy)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.show()


def plot_usefulness(audit: pd.DataFrame):
    # The PM2.5-right-now features are always kept and are so strong they would squash
    # every other bar, so they are left out of this chart.
    a = audit.dropna(subset=["perm_importance_best_block"])
    a = a[~a["protected"]].sort_values("perm_importance_best_block")
    colours = [BLUE if v > 0 else HAZE for v in a["perm_importance_best_block"]]
    fig, ax = plt.subplots(figsize=(9, 0.26 * len(a) + 1.2))
    ax.barh(a["feature"], a["perm_importance_best_block"], color=colours, height=0.65)
    ax.axvline(0, color=SLATE, lw=1)
    ax.set_xlabel("how much worse the forecast gets when this feature is scrambled\n(best of 4 time periods, in µg/m³)")
    ax.set_title("Does each remaining feature actually help the forecast?\n"
                 "(PM2.5-right-now features not shown: they are always kept)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=9)
    for lab in ax.get_yticklabels():
        if lab.get_text() in set(a.loc[a["perm_importance_best_block"] <= 0, "feature"]):
            lab.set_color(HAZE); lab.set_fontweight("bold")
    plt.show()


def short_reason(reason: str) -> str:
    if not isinstance(reason, str) or not reason:
        return ""
    if "exact sum" in reason:
        return "just a total"
    if "compass" in reason:
        return "wind direction"
    if "near-duplicate" in reason:
        twin = reason.split("of ")[1].split(" (")[0]
        return "copy of " + twin.replace("fire_sumatra_", "").replace("fire_kalimantan_", "")
    if "stage 4" in reason:
        return "did not help"
    if "missing" in reason:
        return "too many gaps"
    return "never changes"


def plot_feature_map(audit: pd.DataFrame, grain: str = "hourly"):
    """All 56 features as boxes: green = kept, grey = dropped with the reason inside."""
    dec = dict(zip(audit["feature"], audit["decision"]))
    rea = dict(zip(audit["feature"], audit["reason"]))
    now = "now" if grain == "hourly" else "today"
    r3, r24 = ("roll3h", "roll24h") if grain == "hourly" else ("roll3", "roll7")
    pts = ["singapore", "riau", "palembang", "pontianak", "palangka"]
    wc = {"wind_speed": "wind_speed_{}", "wind_dir": "wind_dir_{}", "rain_mm": "rain_{}_mm",
          f"rain_{r3}": "rain_{}_" + r3, f"rain_{r24}": "rain_{}_" + r24}
    sections = [
        ("Fire hotspots", "full name = fire_<row>_<column>", ["sumatra", "kalimantan", "total"],
         ["lag0", "lag1", "lag2", "lag3", "roll3", "roll7", "log_lag0"], lambda r, c: f"fire_{r}_{c}", True),
        ("Weather", "full name follows the column header, e.g. rain_riau_mm", pts, list(wc),
         lambda r, c: wc[c].format(r), True),
        ("Wind blowing smoke towards Singapore", "", [None], ["sumatra", "kalimantan", "total"],
         lambda r, c: f"upwind_{c}", False),
        ("PM2.5 right now" if grain == "hourly" else "PM2.5 today", "", [None],
         ["north", "south", "east", "west", "central", "national"], lambda r, c: f"{now}_{c}", False),
        ("Fire data gap flag", "", [None], ["fire_data_gap_recent"], lambda r, c: c, False),
    ]
    W, H = 13.5, 16.6
    fig = plt.figure(figsize=(W * 0.72, H * 0.72))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    kept = sum(v == "keep" for v in dec.values())
    ax.text(0.5, H - 0.5, f"{kept} of {len(dec)} features kept", fontsize=20, weight="bold", color=NAVY, va="top")
    ax.text(0.5, H - 1.15, "Green = kept.   Grey = dropped, with the reason written in the box.",
            fontsize=12, color=SLATE, va="top")
    y, x0, tw, th, gap = H - 2.0, 2.6, 1.45, 0.62, 0.1
    rend = fig.canvas.get_renderer()
    for title, note, rows, cols, fn, grid in sections:
        feats = [fn(r, c) for r in rows for c in cols]
        k = sum(dec.get(f) == "keep" for f in feats)
        t = ax.text(0.5, y, title, fontsize=14, weight="bold", color=NAVY, va="top")
        bb = t.get_window_extent(renderer=rend).transformed(ax.transData.inverted())
        ax.text(bb.x1 + 0.25, y - 0.02, f"{k} of {len(feats)} kept", fontsize=12, color=BLUE, weight="bold", va="top")
        y -= 0.55
        if note:
            ax.text(0.5, y, note, fontsize=10, color=SLATE, va="top", style="italic"); y -= 0.42
        cw = tw * 3 if len(cols) == 1 else tw
        if grid:
            for j, c in enumerate(cols):
                ax.text(x0 + j * (cw + gap) + cw / 2, y, c, fontsize=10, color=SLATE, ha="center", va="top", weight="bold")
            y -= 0.4
        for r in rows:
            if r:
                ax.text(x0 - 0.2, y - th / 2, r.capitalize(), fontsize=11.5, color=NAVY, ha="right", va="center")
            for j, c in enumerate(cols):
                f = fn(r, c); keep = dec.get(f) == "keep"; xx = x0 + j * (cw + gap)
                ax.add_patch(FancyBboxPatch((xx, y - th), cw, th, boxstyle="round,pad=0,rounding_size=0.08",
                                            fc=BLUE if keep else MIST, ec="none"))
                label = None if grid else f
                if keep:
                    ax.text(xx + cw / 2, y - th / 2, label or "kept", fontsize=8.4 if label else 10.5,
                            color="white", ha="center", va="center", weight="bold")
                elif label:
                    ax.text(xx + cw / 2, y - th * 0.32, label, fontsize=8.4, color=SLATE, ha="center", va="center")
                    ax.text(xx + cw / 2, y - th * 0.72, short_reason(rea.get(f)), fontsize=8.6, color=NAVY,
                            ha="center", va="center", weight="bold")
                else:
                    ax.text(xx + cw / 2, y - th / 2, short_reason(rea.get(f)), fontsize=8.8, color=NAVY,
                            ha="center", va="center", weight="bold")
            y -= th + gap
        y -= 0.4
    plt.show()


def plot_validation(val: pd.DataFrame):
    """Full 56 vs selected set, per horizon: average error and share of haze hours caught."""
    v = val[val["model"] != "persistence"].groupby(["feature_set", "horizon"])[
        ["mae", "recall_elevated"]].mean().reset_index()
    horizons = sorted(v["horizon"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, metric, title, fmt in [
        (axes[0], "mae", "Average error (lower is better)", "{:.1f}"),
        (axes[1], "recall_elevated", "Unhealthy hours caught (higher is better)", "{:.0%}"),
    ]:
        x = np.arange(len(horizons))
        for i, (fs, colour, name) in enumerate([("full", CLOUD, "all 56"), ("reduced", BLUE, "selected set")]):
            vals = [v[(v.feature_set == fs) & (v.horizon == h)][metric].iloc[0] for h in horizons]
            bars = ax.bar(x + (i - 0.5) * 0.36, vals, 0.34, color=colour, label=name)
            for b, val_ in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(val_),
                        ha="center", va="bottom", fontsize=9.5, color=NAVY)
        ax.set_xticks(x, [f"{h} hour{'s' if h > 1 else ''} ahead" for h in horizons])
        ax.set_title(title, fontsize=12)
        ax.grid(axis="x", visible=False)
        if metric == "recall_elevated":
            ax.set_ylim(0, 1.1)
    axes[0].legend(loc="upper left")
    plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------------
# Interactive lookup
# ---------------------------------------------------------------------------
_PLAIN = {
    "fire": "Number of fire detections",
    "wind_speed": "Wind speed",
    "wind_dir": "Wind direction in compass degrees",
    "rain": "Rainfall",
    "upwind": "Fires multiplied by how much the wind blows from them towards Singapore",
    "now": "PM2.5 reading this hour",
    "today": "PM2.5 average today",
}


def explain_feature(audit: pd.DataFrame, name: str) -> str:
    row = audit[audit["feature"] == name]
    if row.empty:
        return f"'{name}' is not one of the {len(audit)} features. Check the spelling."
    r = row.iloc[0]
    kind = next((v for k, v in _PLAIN.items() if name.startswith(k)), "Feature")
    if r["decision"] == "keep":
        verdict = "KEPT"
        why = ("Always kept on purpose: the current reading is the strongest clue for the next few hours."
               if r.get("protected") else "Passed all four checks.")
    else:
        verdict = "DROPPED"
        why = r["reason"]
    return (f"{name}\n  What it is : {kind}\n  Decision   : {verdict}\n  Why        : {why}\n"
            f"  Link to PM2.5 (outside haze seasons, -1 to 1): {r['spearman_with_label']:.2f}")


def feature_lookup(audit: pd.DataFrame):
    """A dropdown in Jupyter. Falls back to plain text if ipywidgets is not installed."""
    try:
        import ipywidgets as w
        from IPython.display import display
    except ImportError:
        print("ipywidgets is not installed, so here is the text version.\n"
              "Use explain_feature(audit, 'fire_total_lag0') for any feature.")
        return
    options = sorted(audit["feature"])
    box = w.Dropdown(options=options, value="fire_total_lag0", description="Feature:",
                     layout=w.Layout(width="420px"))
    out = w.Output(layout=w.Layout(border="1px solid #C9D4DE", padding="6px 10px"))

    def show(change=None):
        out.clear_output()
        with out:
            print(explain_feature(audit, box.value))

    box.observe(show, names="value")
    show()
    display(w.VBox([box, out]))


# ---------------------------------------------------------------------------
# Data quality tables (numbers come from hazecast.data_quality)
# ---------------------------------------------------------------------------
def style_fixes(df: pd.DataFrame):
    """Bronze fixes table: starting rows and kept rows in bold, removals in orange."""
    def row_style(r):
        if r["problem"].startswith(("Rows in", "Detections in", "Rows across")):
            return [f"font-weight:bold; background-color:{MIST}"] * len(r)
        if r["problem"].endswith("kept"):
            return [f"font-weight:bold; color:{BLUE}; background-color:{SKY_LIGHT}"] * len(r)
        if r["count"] > 0:
            return ["", f"color:{HAZE}; font-weight:bold", "", ""]
        return [""] * len(r)
    return (df.style.apply(row_style, axis=1).format({"count": "{:,}"}).hide(axis="index")
              .set_properties(subset=["problem", "action"], **{"text-align": "left"}))


def style_gaps(df: pd.DataFrame):
    def hi(v):
        return f"color:{HAZE}; font-weight:bold" if isinstance(v, (int, np.integer)) and v > 0 else ""
    return (df.style.map(hi, subset=["missing", "filled", "left_empty", "flagged"])
              .format({"expected": "{:,}"}).hide(axis="index")
              .set_properties(subset=["note"], **{"text-align": "left"}))


# ---------------------------------------------------------------------------
# EDA charts (numbers come from hazecast.eda)
# ---------------------------------------------------------------------------
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def plot_unhealthy_by_month(by_month: pd.DataFrame):
    v = by_month["unhealthy_hours"]
    colours = [HAZE if x > 0.05 * v.max() else CLOUD for x in v]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    bars = ax.bar(MONTHS, v, color=colours, width=0.7)
    for b, x in zip(bars, v):
        if x > 0:
            ax.text(b.get_x() + b.get_width() / 2, x, f"{int(x)}", ha="center", va="bottom", fontsize=10, color=NAVY)
    ax.set_title("Unhealthy hours by month, 2014 to 2026")
    ax.set_ylabel("hours at 56 or more")
    ax.grid(axis="x", visible=False)
    plt.show()


def plot_fires_by_month(by_month: pd.DataFrame):
    x = np.arange(12)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.bar(x - 0.2, by_month["fires_sumatra_k"], 0.38, color=BLUE, label="Sumatra")
    ax.bar(x + 0.2, by_month["fires_kalimantan_k"], 0.38, color=SKY, label="Kalimantan")
    ax.set_xticks(x, MONTHS)
    ax.set_title("Fire detections by month, all years added up")
    ax.set_ylabel("detections (thousands)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    plt.show()


def plot_by_hour(by_hour: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.plot(by_hour.index, by_hour["avg_pm25"], color=BLUE, lw=2.2, marker="o", ms=4)
    ax.set_ylim(0, 30)
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.set_title("Average PM2.5 by hour of the day: almost flat")
    ax.set_ylabel("PM2.5 (µg/m³)")
    plt.show()


def plot_region_similarity(corr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(corr.values, cmap="Blues", vmin=0.6, vmax=1)
    labels = [c.capitalize() for c in corr.columns]
    ax.set_xticks(range(5), labels); ax.set_yticks(range(5), labels); ax.grid(False)
    for i in range(5):
        for j in range(5):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10, color="white" if v > 0.85 else NAVY)
    ax.set_title("How closely the 5 regions move together")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.show()


def plot_fire_lag(lag: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for col, colour in [("sumatra", BLUE), ("kalimantan", SKY)]:
        ax.plot(lag.index, lag[col], color=colour, lw=2.4, marker="o", ms=6, label=col.capitalize())
        ax.text(lag.index[-1] + 0.1, lag[col].iloc[-1], col.capitalize(), color=colour, va="center", fontsize=11, fontweight="bold")
    ax.set_xticks(lag.index, ["today"] + [f"{d} day{'s' if d > 1 else ''} ago" for d in lag.index[1:]])
    ax.set_ylim(0, 0.55); ax.set_xlim(-0.3, lag.index[-1] + 1.2)
    ax.set_title("Link between fires and today's PM2.5 (0 = none, 1 = perfect)")
    ax.legend(loc="lower left")
    plt.show()


def plot_wind_roses(wind: pd.DataFrame):
    dirs = list(wind.index)
    theta = np.radians(np.arange(0, 360, 45))
    fig = plt.figure(figsize=(10, 4.4))
    for k, (col, colour, title) in enumerate([("normal days", SKY, "Normal days"), ("haze days", HAZE, "Haze days")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="polar")
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.bar(theta, wind[col].values, width=np.radians(40), color=colour, alpha=0.9, edgecolor="white")
        ax.set_xticks(theta, dirs); ax.set_yticklabels([])
        ax.set_title(f"{title}: wind comes from...", pad=14, fontsize=12)
    plt.tight_layout(); plt.show()


def plot_rain_effect(rain: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    order = list(rain.index)
    bars = ax.barh(order, rain["avg_pm25"], color=[CLOUD if "dry" in o else SKY for o in order], height=0.55)
    for b, v in zip(bars, rain["avg_pm25"]):
        ax.text(v + 0.3, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=11, color=NAVY)
    ax.set_xlim(0, rain["avg_pm25"].max() * 1.25)
    ax.set_title("Average PM2.5 on rainy vs dry days")
    ax.set_xlabel("PM2.5 (µg/m³)"); ax.grid(axis="y", visible=False)
    plt.show()


def plot_persistence(p: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    bars = ax.bar([f"{h} h" for h in p.index], p["correlation"], color=BLUE, width=0.6)
    for b, v in zip(bars, p["correlation"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=11, color=NAVY)
    ax.set_ylim(0, 1.1)
    ax.set_title("How well PM2.5 now matches PM2.5 later (1 = perfectly)")
    ax.set_xlabel("hours ahead"); ax.grid(axis="x", visible=False)
    plt.show()


def plot_season_2026(pm25_s: pd.DataFrame):
    """The 2026 haze season so far: daily national average from 1 August."""
    d = pm25_s.set_index("ts")["national_mean"].resample("D").mean().loc["2026-08-01":]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.fill_between(d.index, d.values, color=SKY, alpha=0.25, lw=0)
    ax.plot(d.index, d.values, color=BLUE, lw=2.2)
    over = d[d >= PM25_ELEVATED_THRESHOLD]
    ax.scatter(over.index, over.values, color=HAZE, s=28, zorder=3, label="unhealthy day")
    ax.axhline(PM25_ELEVATED_THRESHOLD, color=HAZE, ls="--", lw=1.2)
    peak_day, peak = d.idxmax(), d.max()
    ax.annotate(f"peak {peak:.0f} on {peak_day:%d %b}", (peak_day, peak), xytext=(-230, 2),
                textcoords="offset points", color=NAVY, fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=SLATE))
    ax.set_ylim(0, peak * 1.15)
    ax.set_title("2026 so far: daily average PM2.5 since 1 August")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend(loc="upper left")
    plt.show()
    return pd.DataFrame({"unhealthy days so far": [int(len(over))], "highest daily average": [round(float(peak))],
                         "on": [f"{peak_day:%d %b %Y}"]})
