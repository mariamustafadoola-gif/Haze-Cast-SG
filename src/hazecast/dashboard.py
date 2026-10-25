"""Model performance and stability dashboard (one self-contained HTML file).

Reads a forecast ledger (forecasts, actuals, and the monitor_metrics /
monitor_runs tables hazecast.model_monitor writes) and plots every monitor
over time:

  1. governance decision per monitoring window
  2. forecast vs actual (national, +6 h)
  3. model quality: MAE and skill vs persistence per horizon, F2 on elevated hours
  4. data quality: PSI per feature (heatmap) and number of drifted features
  5. bias drift: recall gap across regions, mean signed error per region
  6. feature attribution drift: NDCG@k vs baseline

    python -m hazecast.dashboard                                   # live ledger
    python -m hazecast.dashboard --db data/monitoring/backtest.sqlite --out reports/monitoring/backtest_dashboard.html

Also served by the API at /monitoring/dashboard.
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from hazecast.config import MONITORING_DIR, PM25_ELEVATED_THRESHOLD, PSI_ALERT, SG_REGIONS
from hazecast.model_monitor import REPORTS_DIR

log = logging.getLogger(__name__)

# Reference palette (dataviz skill, references/palette.md). Light-mode steps;
# the page re-themes surfaces/ink/grid for dark mode in JS, series hues are shared.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
HORIZON_COLOR = {6: SERIES[0], 12: SERIES[1], 24: SERIES[2]}   # first three slots validate all-pairs
REGION_COLOR = dict(zip(SG_REGIONS, SERIES))                   # fixed per entity, never by rank
# Status palette (reserved, never a series colour); always shown with its text label.
STATUS = {"none": ("#0ca30c", "No action"), "investigate": ("#fab219", "Investigate"),
          "retrain": ("#ec835a", "Retrain"), "fix_data": ("#d03b3b", "Fix data")}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SEQUENTIAL = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _read(db_path: Path):
    db = sqlite3.connect(db_path)
    try:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        metrics = pd.read_sql("SELECT * FROM monitor_metrics", db) if "monitor_metrics" in tables else pd.DataFrame()
        runs = pd.read_sql("SELECT * FROM monitor_runs", db) if "monitor_runs" in tables else pd.DataFrame()
        series = pd.read_sql("""
            SELECT f.forecast_time, f.forecast_pm25, a.actual_pm25 FROM forecasts f
            LEFT JOIN actuals a ON f.forecast_time = a.observation_time AND f.region = a.region
            WHERE f.region = 'national' AND f.horizon_hours = 6 ORDER BY f.forecast_time""", db) \
            if "forecasts" in tables else pd.DataFrame()
    finally:
        db.close()
    for frame, col in ((metrics, "window_end"), (runs, "window_end"), (series, "forecast_time")):
        if not frame.empty:
            frame[col] = pd.to_datetime(frame[col], utc=True).dt.tz_convert("Asia/Singapore")
    return metrics, runs, series


def _layout(fig, title: str, ytitle: str, height: int = 340, **extra):
    fig.update_layout(**{**dict(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=INK)),
        height=height, margin=dict(l=60, r=20, t=56, b=40), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12, color=INK_2),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified"), **extra})
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID)
    fig.update_yaxes(title_text=ytitle, gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def _line(fig, x, y, name, color, dash=None, markers=False, hover_fmt=".2f"):
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines+markers" if markers else "lines",
                             line=dict(color=color, width=2, dash=dash), marker=dict(size=8),
                             hovertemplate=f"{name}: %{{y:{hover_fmt}}}<extra></extra>"))


def _metric(metrics: pd.DataFrame, monitor: str, metric: str) -> pd.DataFrame:
    return metrics.loc[(metrics.monitor == monitor) & (metrics.metric == metric)]


def figures(metrics: pd.DataFrame, runs: pd.DataFrame, series: pd.DataFrame) -> list:
    import plotly.graph_objects as go
    figs = []

    if not runs.empty:
        fig = go.Figure()
        for action, (color, label) in STATUS.items():
            sel = runs.loc[runs.action == action]
            if sel.empty:
                continue
            reasons = sel.reasons.map(lambda r: "<br>".join(json.loads(r)) or "all monitors ok")
            fig.add_trace(go.Scatter(x=sel.window_end, y=[label] * len(sel), mode="markers", name=label,
                                     marker=dict(color=color, size=11, symbol="square", line=dict(color=SURFACE, width=2)),
                                     customdata=reasons, hovertemplate="%{x|%d %b %Y}<br>%{customdata}<extra>" + label + "</extra>"))
        figs.append(_layout(fig, "Governance decision per monitoring window", "", height=230,
                            yaxis=dict(categoryorder="array", categoryarray=[v[1] for v in STATUS.values()]),
                            hovermode="closest"))

    if not series.empty:
        fig = go.Figure()
        _line(fig, series.forecast_time, series.actual_pm25, "Actual", SERIES[0], hover_fmt=".1f")
        _line(fig, series.forecast_time, series.forecast_pm25, "Forecast (+6 h)", SERIES[1], hover_fmt=".1f")
        fig.add_hline(y=PM25_ELEVATED_THRESHOLD, line=dict(color=INK_2, width=1, dash="dash"),
                      annotation_text=f"Elevated ({PM25_ELEVATED_THRESHOLD:g} µg/m³)", annotation_position="top left")
        figs.append(_layout(fig, "National PM2.5: forecast 6 hours ahead vs actual", "PM2.5 (µg/m³)", height=380))

    q = _metric(metrics, "model_quality", "mae")
    if not q.empty:
        fig = go.Figure()
        for h, g in q.groupby("horizon_hours"):
            avg = g.groupby("window_end").value.mean()
            _line(fig, avg.index, avg.values, f"+{h} h model", HORIZON_COLOR.get(h, SERIES[3]))
            p = _metric(metrics, "model_quality", "persistence_mae")
            p = p.loc[p.horizon_hours == h].groupby("window_end").value.mean()
            _line(fig, p.index, p.values, f"+{h} h persistence", HORIZON_COLOR.get(h, SERIES[3]), dash="dot")
        figs.append(_layout(fig, "MAE per window (mean over 5 regions + national)", "MAE (µg/m³)"))

        s = _metric(metrics, "model_quality", "skill")
        fig = go.Figure()
        for h, g in s.groupby("horizon_hours"):
            avg = g.groupby("window_end").value.mean()
            _line(fig, avg.index, avg.values, f"+{h} h", HORIZON_COLOR.get(h, SERIES[3]))
        fig.add_hline(y=0, line=dict(color=INK_2, width=1, dash="dash"), annotation_text="persistence",
                      annotation_position="bottom left")
        figs.append(_layout(fig, "Skill vs persistence (1 − MAE / persistence MAE); below 0 = worse than no-change",
                            "Skill", yaxis_tickformat=".0%"))

        f2 = _metric(metrics, "model_quality", "f2")
        f2 = f2.loc[f2.region == "national"]
        if not f2.empty:
            fig = go.Figure()
            for h, g in f2.groupby("horizon_hours"):
                _line(fig, g.window_end, g.value, f"+{h} h", HORIZON_COLOR.get(h, SERIES[3]), markers=True, hover_fmt=".2f")
            figs.append(_layout(fig, "F2 on elevated hours, national (only windows with elevated hours)", "F2",
                                yaxis_range=[0, 1.02]))

    psi = _metric(metrics, "data_quality", "psi")
    if not psi.empty:
        top = psi.groupby("feature").value.max().sort_values(ascending=False).head(15).index
        grid = psi.loc[psi.feature.isin(top)].pivot_table(index="feature", columns="window_end", values="value")
        grid = grid.loc[list(top)[::-1]]
        fig = go.Figure(go.Heatmap(z=grid.values, x=grid.columns, y=grid.index, zmin=0, zmax=max(1.0, PSI_ALERT * 2),
                                   colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
                                   xgap=1, ygap=1, colorbar=dict(title="PSI", thickness=12),
                                   hovertemplate="%{y}<br>%{x|%d %b %Y}<br>PSI %{z:.2f}<extra></extra>"))
        figs.append(_layout(fig, "Data drift: PSI vs training data over 30-day windows, 15 most-drifted features",
                            "", height=460, hovermode="closest"))
        n = _metric(metrics, "data_quality", "n_psi_alert")
        fig = go.Figure()
        _line(fig, n.window_end, n.value, "Features beyond calibrated PSI limit", SERIES[0], hover_fmt=".0f")
        figs.append(_layout(fig, "Number of features beyond their calibrated PSI limit (99th pct of normal "
                                 "training months)", "Features", height=280))

    gap = _metric(metrics, "bias_drift", "recall_gap")
    if not gap.empty:
        fig = go.Figure()
        for h, g in gap.groupby("horizon_hours"):
            _line(fig, g.window_end, g.value, f"+{h} h", HORIZON_COLOR.get(h, SERIES[3]), markers=True)
        figs.append(_layout(fig, "Bias: gap between best and worst region's elevated-hour recall", "Recall gap",
                            yaxis_tickformat=".0%"))
    me = _metric(metrics, "bias_drift", "mean_error")
    me = me.loc[me.horizon_hours == 6]
    if not me.empty:
        fig = go.Figure()
        for region in SG_REGIONS:
            g = me.loc[me.region == region]
            _line(fig, g.window_end, g.value, region.title(), REGION_COLOR[region], hover_fmt=".1f")
        fig.add_hline(y=0, line=dict(color=INK_2, width=1, dash="dash"))
        figs.append(_layout(fig, "Bias: mean signed error by region, +6 h (above 0 = over-forecast)", "µg/m³"))

    nd = _metric(metrics, "attribution_drift", "ndcg")
    if not nd.empty:
        fig = go.Figure()
        for h, g in nd.groupby("horizon_hours"):
            med = g.groupby("window_end").value.median()
            _line(fig, med.index, med.values, f"+{h} h median", HORIZON_COLOR.get(h, SERIES[3]), hover_fmt=".3f")
        lo = nd.groupby("window_end").value.min()
        _line(fig, lo.index, lo.values, "worst model", SERIES[7], dash="dot", hover_fmt=".3f")
        figs.append(_layout(fig, "Feature attribution drift: NDCG@10 of current vs baseline ranking "
                                 "(alert limit is calibrated per model, see docs)",
                            "NDCG", yaxis_range=[0, 1.02]))
    return figs


def _latest_table(metrics: pd.DataFrame, runs: pd.DataFrame) -> str:
    if runs.empty:
        return "<p>No monitoring runs recorded yet.</p>"
    last = runs.sort_values("window_end").iloc[-1]
    statuses = json.loads(last.statuses)
    reasons = json.loads(last.reasons) or ["all monitors within thresholds"]
    rows = "".join(f"<tr><td>{html.escape(k.replace('_', ' '))}</td><td>{html.escape(v)}</td></tr>" for k, v in statuses.items())
    q = metrics.loc[(metrics.window_end == last.window_end) & (metrics.monitor == "model_quality")]
    qt = q.pivot_table(index=["region", "horizon_hours"], columns="metric", values="value").reset_index()
    cols = [c for c in ("region", "horizon_hours", "n", "mae", "persistence_mae", "skill", "f2") if c in qt]
    qrows = "".join("<tr>" + "".join(f"<td>{_fmt(r[c])}</td>" for c in cols) + "</tr>" for _, r in qt[cols].iterrows())
    color, label = STATUS.get(last.action, (INK_2, last.action))
    return (f"<div class='decision'><span class='dot' style='background:{color}'></span>"
            f"<b>{html.escape(label)}</b> &middot; window ending {last.window_end:%d %b %Y %H:%M} SGT</div>"
            f"<ul>{''.join(f'<li>{html.escape(r)}</li>' for r in reasons)}</ul>"
            f"<table><tr><th>Monitor</th><th>Status</th></tr>{rows}</table>"
            + (f"<details><summary>Model quality in this window (table view)</summary><table><tr>"
               f"{''.join(f'<th>{c}</th>' for c in cols)}</tr>{qrows}</table></details>" if qrows else ""))


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) and abs(v) < 10 else f"{v:.1f}" if isinstance(v, float) else html.escape(str(v))


_DARK_JS = """
<script>
(function(){
  const dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
    && document.documentElement.dataset.theme !== 'light';
  if (!dark) return;
  document.querySelectorAll('.js-plotly-plot').forEach(function(el){
    Plotly.relayout(el, {paper_bgcolor:'#1a1a19', plot_bgcolor:'#1a1a19', 'font.color':'#c3c2b7',
      'title.font.color':'#f0efec', 'yaxis.gridcolor':'#33332f', 'xaxis.linecolor':'#33332f', 'yaxis.linecolor':'#33332f'});
  });
})();
</script>"""

_CSS = """
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e6e5e1;--page:#f4f3f0}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--rule:#33332f;--page:#111110}}
:root[data-theme="dark"]{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--rule:#33332f;--page:#111110}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px} .sub{color:var(--ink2);margin:0 0 20px}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:12px 14px;margin:0 0 16px}
table{border-collapse:collapse;margin:8px 0} td,th{border-bottom:1px solid var(--rule);padding:4px 10px;text-align:left;font-variant-numeric:tabular-nums}
.decision{font-size:16px;display:flex;align-items:center;gap:8px} .dot{width:12px;height:12px;border-radius:3px;display:inline-block}
"""


def build_dashboard(db_path: Path = MONITORING_DIR / "pipeline.sqlite", out_path: Path = REPORTS_DIR / "dashboard.html",
                    *, title: str = "HazeCast SG: model monitoring", subtitle: str = "") -> Path:
    metrics, runs, series = _read(Path(db_path))
    figs = figures(metrics, runs, series)
    parts = [f.to_html(full_html=False, include_plotlyjs=(i == 0), config={"displaylogo": False, "responsive": True})
             for i, f in enumerate(figs)]
    generated = datetime.now(timezone.utc).astimezone().strftime("%d %b %Y %H:%M %Z")
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>HazeCast Monitoring</title>"
            f"<style>{_CSS}</style></head><body><main><h1>{html.escape(title)}</h1>"
            f"<p class='sub'>{html.escape(subtitle)} Generated {generated} from {html.escape(Path(db_path).name)}.</p>"
            f"<div class='card'>{_latest_table(metrics, runs)}</div>"
            + "".join(f"<div class='card'>{p}</div>" for p in parts)
            + ("" if parts else "<div class='card'>No monitoring history yet.</div>")
            + f"</main>{_DARK_JS if parts else ''}</body></html>")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    log.info("Dashboard with %d charts -> %s", len(figs), out_path)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=MONITORING_DIR / "pipeline.sqlite")
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "dashboard.html")
    a = ap.parse_args()
    print(build_dashboard(a.db, a.out))
