"""Evidence behind the monitoring thresholds and intervention rules in
docs/PIPELINE_WALKTHROUGH.md and docs/MODEL_GOVERNANCE.md, plus its charts.

Needs: data/gold/hourly_features.parquet, data/models/hourly (trained) and
data/monitoring/backtest.sqlite (python -m hazecast.monitor_backtest).

    python experiments/monitoring_evidence.py

Writes docs/img/*.png and reports/monitoring_evidence/*.csv:
  1. seasonality      share of elevated hours by month (why haze-aware metrics)
  2. candidates       pooled forward-validation MAE and F2 per candidate model
  3. replay_actions   governance decision per weekly window over the holdout
  4. recall_weekly    model vs persistence recall in haze weeks (the quality alarm)
  5. retrain_test     would retraining at the first strong signal have helped?
  6. alert_rule       "model" vs "model OR persistence" alert rules
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hazecast.config import GOLD_DIR, MODELS_DIR, MONITORING_DIR, PM25_ELEVATED_THRESHOLD as THR  # noqa: E402

IMG, OUT = ROOT / "docs" / "img", ROOT / "reports" / "monitoring_evidence"
# Reference palette (dataviz skill): categorical slots 1-3, status colours, ink, grid.
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
STATUS = {"none": "#0ca30c", "investigate": "#fab219", "retrain": "#ec835a", "fix_data": "#d03b3b"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SIGNAL = pd.Timestamp("2026-09-08", tz="Asia/Singapore")  # first replay window with most groups in alert


def _style(ax, title, ylabel=""):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def _save(fig, name):
    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG / f"{name}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def seasonality(gold):
    g = gold.assign(month=gold.date.dt.month, elevated=gold.now_national >= THR)
    share = g.groupby("month").elevated.mean() * 100
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(share.index, share.values, color=BLUE, width=0.7)
    for m, v in share.items():
        if v >= 0.5:
            ax.text(m, v + 0.1, f"{v:.1f}%", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    _style(ax, "Share of hours with national PM2.5 ≥ 56 µg/m³, by month (Apr 2014 – Oct 2026)", "% of hours")
    _save(fig, "seasonality")


def candidates():
    e = pd.read_csv(MODELS_DIR / "hourly" / "eval_summary.csv")
    v = e[e.split == "validation_pooled"].groupby(["model", "horizon_hours"])[["mae", "f2"]].mean().reset_index()
    order = ["persistence", "ridge", "random_forest", "xgboost", "lightgbm", "xgboost_q85"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, metric, label in ((axes[0], "mae", "MAE (µg/m³), lower is better"), (axes[1], "f2", "F2 on elevated hours, higher is better")):
        for i, (h, color) in enumerate(((6, BLUE), (12, ORANGE), (24, AQUA))):
            vals = v[v.horizon_hours == h].set_index("model").reindex(order)[metric]
            ax.bar(np.arange(len(order)) + (i - 1) * 0.26, vals.values, width=0.24, color=color, label=f"+{h} h")
        ax.set_xticks(range(len(order)), [o.replace("_", "\n") for o in order], fontsize=8)
        _style(ax, label)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Pooled forward-validation scores, mean over 6 targets (selection uses F2 first)",
                 x=0.01, y=1.06, ha="left", fontsize=11, color=INK)
    _save(fig, "candidates")


def replay(db):
    runs = pd.read_sql("SELECT window_end, action FROM monitor_runs", db)
    runs["t"] = pd.to_datetime(runs.window_end, utc=True).dt.tz_convert("Asia/Singapore")
    f = pd.read_sql("""SELECT f.region, f.horizon_hours h, f.forecast_time t, f.forecast_pm25 p,
                       f.persistence_pm25 q, a.actual_pm25 y FROM forecasts f JOIN actuals a
                       ON f.forecast_time = a.observation_time AND f.region = a.region""", db)
    f["t"] = pd.to_datetime(f.t, utc=True).dt.tz_convert("Asia/Singapore")
    nat = f[(f.region == "national") & (f.h == 6)].set_index("t").y.resample("D").mean()

    fig, axes = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    axes[0].plot(nat.index, nat.values, color=BLUE, linewidth=1.2, label="National PM2.5, daily mean")
    axes[0].axhline(THR, color=INK2, linestyle="--", linewidth=0.8)
    axes[0].text(nat.index[5], THR + 3, "elevated (56)", fontsize=8, color=INK2)
    _style(axes[0], "Holdout replay: actual PM2.5 and the governance decision of each weekly monitoring window", "µg/m³")
    for action, color in STATUS.items():
        sel = runs[runs.action == action]
        if len(sel):
            axes[1].scatter(sel.t, [action] * len(sel), color=color, marker="s", s=30, label=f"{action} ({len(sel)})")
    axes[1].set_yticks(["none", "investigate", "retrain"])
    _style(axes[1], "")
    axes[1].legend(frameon=False, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.25))
    _save(fig, "replay_actions")

    el = f[f.y >= THR].copy()
    el["week"] = el.t.dt.tz_localize(None).dt.to_period("W").dt.start_time
    wk = el.groupby(["week", "h"]).apply(lambda d: pd.Series(dict(
        n=len(d), model=(d.p >= THR).mean(), persistence=(d.q >= THR).mean())), include_groups=False).reset_index()
    wk = wk[wk.n >= 12 * 6]  # weeks with real haze (>= 12 elevated hours per target on average)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    for ax, h in zip(axes, (6, 12, 24)):
        d = wk[wk.h == h]
        x = np.arange(len(d))
        ax.bar(x - 0.2, d.persistence, width=0.38, color=GREY, label="persistence")
        ax.bar(x + 0.2, d.model, width=0.38, color=BLUE, label="model")
        ax.set_xticks(x, [w.strftime("%d %b") for w in d.week], fontsize=8, rotation=45)
        _style(ax, f"+{h} h", "recall (share of haze hours flagged)" if h == 6 else "")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Haze weeks in the holdout (2026): elevated-hour recall, all 6 targets pooled",
                 x=0.01, y=1.06, ha="left", fontsize=11, color=INK)
    _save(fig, "recall_weekly")
    wk.to_csv(OUT / "recall_weekly.csv", index=False)
    return f


def retrain_test(gold, manifest):
    """Refit each selected model with data up to SIGNAL; compare on the rest of the haze episode."""
    from hazecast.hourly_model import _prepare, _labelled, _complete, _before, _fit, _predict
    gold = _prepare(gold)
    cols = manifest["feature_columns"]
    rows_out = []
    for h in (6, 12, 24):
        lab = _labelled(gold, h)
        for t in ("north", "south", "east", "west", "central", "national"):
            e = manifest["models"][str(h)][t]
            rows = _complete(lab, cols, t, h)
            test = rows.loc[rows.date >= SIGNAL]
            y = test[f"label_{t}"].to_numpy()
            el = y >= THR
            for name, est in (("old model (data to Apr 2024)", _fit(e["model"], _before(rows, pd.Timestamp(e["test_start"]), h), cols, t)),
                              ("retrained at signal", _fit(e["model"], _before(rows, SIGNAL, h), cols, t)),
                              ("persistence", None)):
                p = _predict(est, test, cols, t)
                rows_out.append(dict(h=h, target=t, model=name, mae=np.abs(p - y).mean(), n_elevated=int(el.sum()),
                                     recall=(p[el] >= THR).mean(), false_alarm=(p[~el] >= THR).mean()))
    r = pd.DataFrame(rows_out)
    r.to_csv(OUT / "retrain_test.csv", index=False)
    s = r.groupby(["h", "model"])[["mae", "recall", "false_alarm"]].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    names = ["old model (data to Apr 2024)", "retrained at signal", "persistence"]
    for ax, metric, label in zip(axes, ("recall", "false_alarm", "mae"),
                                 ("Recall on haze hours ↑", "False-alarm rate (non-haze hours) ↓", "MAE µg/m³ ↓")):
        for i, (n, color) in enumerate(zip(names, (BLUE, ORANGE, GREY))):
            vals = s[s.model == n].set_index("h").reindex([6, 12, 24])[metric]
            ax.bar(np.arange(3) + (i - 1) * 0.27, vals.values, width=0.25, color=color, label=n)
        ax.set_xticks(range(3), ["+6 h", "+12 h", "+24 h"])
        _style(ax, label)
    axes[1].legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.suptitle("Retraining at the monitor's signal (8 Sep 2026), scored on 8 Sep – 18 Oct 2026",
                 x=0.01, y=1.06, ha="left", fontsize=11, color=INK)
    _save(fig, "retrain_test")
    return s


def alert_rule(f):
    out = []
    for period, sel in (("haze, 8 Sep – 18 Oct 2026", f.t >= SIGNAL), ("whole holdout", f.t.notna())):
        g = f[sel]
        for rule, flag in (("model", g.p >= THR), ("persistence", g.q >= THR), ("model OR persistence", (g.p >= THR) | (g.q >= THR))):
            for h in (6, 12, 24):
                m, el = g.h == h, g.y >= THR
                out.append(dict(period=period, rule=rule, h=h, recall=(flag & el & m).sum() / (el & m).sum(),
                                false_alarm=(flag & ~el & m).sum() / (~el & m).sum()))
    r = pd.DataFrame(out)
    r.to_csv(OUT / "alert_rule.csv", index=False)
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gold = pd.read_parquet(GOLD_DIR / "hourly_features.parquet")
    gold["date"] = pd.to_datetime(gold["date"])
    manifest = json.loads((MODELS_DIR / "hourly" / "model_manifest.json").read_text())
    db = sqlite3.connect(MONITORING_DIR / "backtest.sqlite")
    seasonality(gold)
    candidates()
    f = replay(db)
    print(retrain_test(gold, manifest).round(3).to_string())
    print(alert_rule(f).pivot_table(index=["period", "rule"], columns="h", values=["recall", "false_alarm"]).round(3).to_string())
    print("charts ->", IMG)


if __name__ == "__main__":
    main()
