"""Daily verification of archived forecasts against NEA 1-hour PM2.5.

Run daily in Singapore time. Scores forecast TARGET timestamps, not issue
or origin dates. Also revisits recent days for delayed/revised observations.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import logging
import math
import os
import tempfile
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from hazecast.config import DATA_DIR
from hazecast.pipeline_store import SGT, REGIONS, TARGETS, HORIZONS, connect, timestamp, iso, finite, atomic_json

log = logging.getLogger(__name__)
ENDPOINT = "https://api-open.data.gov.sg/v2/real-time/api/pm25"


def request_json(url, *, api_key=None, retries=4):
    headers = {"Accept": "application/json", "User-Agent": "HazeCast-SG-monitor/1.0"}
    if api_key:
        headers["x-api-key"] = api_key
    for attempt in range(retries):
        try:
            with urlopen(Request(url, headers=headers), timeout=30) as response:
                payload = json.load(response)
            if payload.get("code") != 0:
                raise RuntimeError(f"NEA API error: {payload.get('errorMsg', 'unknown')}")
            if not isinstance(payload.get("data"), dict):
                raise ValueError("Unexpected NEA response: missing data object")
            return payload
        except HTTPError as exc:
            if exc.code == 404:
                return {"code": 0, "data": {"items": []}, "errorMsg": "No observations"}
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            retry_header = exc.headers.get("Retry-After", "") if exc.headers else ""
            delay = float(retry_header) if retry_header.isdigit() else 2 ** (attempt + 1)
            time.sleep(min(delay, 30))
        except (URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("NEA retries exhausted")


def parse_items(items, day, *, fetched_at=None):
    """Use observation timestamp exactly. National is a derived five-region mean."""
    fetched_at = iso(fetched_at or datetime.now(timezone.utc))
    observations = {}
    for item in items:
        observed = timestamp(item["timestamp"])
        if observed.astimezone(SGT).date().isoformat() != str(day):
            continue
        if observed.minute or observed.second or observed.microsecond:
            raise ValueError("NEA PM2.5 observation is not on an hour boundary; refusing to round timestamps")
        revised = iso(item.get("updatedTimestamp") or observed)
        readings = item.get("readings", {}).get("pm25_one_hourly", {})
        values = {}
        for region in REGIONS:
            value = readings.get(region)
            values[region] = float(value) if finite(value) else None
        values["national"] = (sum(values[r] for r in REGIONS) / 5
                              if all(values[r] is not None for r in REGIONS) else None)
        for region, value in values.items():
            record = dict(observation_time=iso(observed), region=region, actual_pm25=value,
                source="derived_mean_of_five_NEA_regions" if region == "national" else "NEA_pm25_one_hourly",
                source_updated_at=revised, fetched_at=fetched_at)
            key = (record["observation_time"], region)
            if key not in observations or record["source_updated_at"] >= observations[key]["source_updated_at"]:
                observations[key] = record
    return list(observations.values())


def fetch_actuals(day, *, api_key=None, raw_dir=None, page_delay_seconds=2):
    """Fetch a full Singapore date, following every paginationToken."""
    day = date.fromisoformat(str(day)).isoformat()
    token, seen, items = None, set(), []
    fetched_at = datetime.now(timezone.utc)
    for page in range(200):
        params = {"date": day}
        if token:
            params["paginationToken"] = token
        payload = request_json(ENDPOINT + "?" + urlencode(params), api_key=api_key)
        if raw_dir is not None:
            stamp = fetched_at.strftime("%Y%m%dT%H%M%S%fZ")
            atomic_json(Path(raw_dir) / day / f"{stamp}_page{page:03d}.json", payload)
        data = payload["data"]
        batch = data.get("items", [])
        if not isinstance(batch, list):
            raise ValueError("Unexpected NEA items schema")
        items.extend(batch)
        token = data.get("paginationToken")
        if not token:
            return parse_items(items, day, fetched_at=fetched_at)
        if token in seen:
            raise RuntimeError("Repeated NEA pagination token")
        seen.add(token)
        time.sleep(page_delay_seconds)
    raise RuntimeError("Exceeded NEA pagination safety limit")


def upsert_actuals(db, records):
    with db:
        for record in records:
            db.execute('''INSERT INTO actuals VALUES
                (:observation_time,:region,:actual_pm25,:source,:source_updated_at,:fetched_at)
                ON CONFLICT(observation_time,region) DO UPDATE SET
                    actual_pm25=excluded.actual_pm25, source=excluded.source,
                    source_updated_at=excluded.source_updated_at, fetched_at=excluded.fetched_at
                WHERE excluded.source_updated_at >= actuals.source_updated_at''', record)


def metric_summary(rows):
    """Metric primitives use full-precision forecasts and each saved threshold."""
    if not rows:
        return dict(n_matched=0, mae=None, rmse=None, f2=None, n_elevated=0,
                    tp=0, fp=0, fn=0, precision=None, recall=None,
                    mean_error=None, persistence_mae=None, persistence_rmse=None,
                    persistence_f2=None, mae_skill_vs_persistence=None)
    errors = [r["forecast_pm25"] - r["actual_pm25"] for r in rows]
    actual = [r["actual_pm25"] >= r["threshold"] for r in rows]
    flagged = [r["forecast_pm25"] >= r["threshold"] for r in rows]
    base_flagged = [r["persistence_pm25"] >= r["threshold"] for r in rows]
    def counts(flags):
        tp = sum(a and p for a,p in zip(actual,flags))
        fp = sum(not a and p for a,p in zip(actual,flags))
        fn = sum(a and not p for a,p in zip(actual,flags))
        return tp,fp,fn
    tp,fp,fn = counts(flagged)
    btp,bfp,bfn = counts(base_flagged)
    positives = sum(actual)
    n = len(rows)
    mae = sum(map(abs,errors)) / n
    base_errors = [r["persistence_pm25"] - r["actual_pm25"] for r in rows]
    base_mae = sum(map(abs,base_errors)) / n
    return dict(n_matched=n, mae=mae, rmse=math.sqrt(sum(e*e for e in errors)/n),
        f2=5*tp/(5*tp+4*fn+fp) if positives else None,
        n_elevated=positives, tp=tp, fp=fp, fn=fn,
        precision=tp/(tp+fp) if tp+fp else None, recall=tp/positives if positives else None,
        mean_error=sum(errors)/n, persistence_mae=base_mae,
        persistence_rmse=math.sqrt(sum(e*e for e in base_errors)/n),
        persistence_f2=5*btp/(5*btp+4*bfn+bfp) if positives else None,
        mae_skill_vs_persistence=1-mae/base_mae if base_mae>0 else None)


def evaluate_day(db, day):
    """18 target/horizon groups, including zero-forecast or pending groups."""
    local_start = datetime.combine(date.fromisoformat(str(day)), datetime.min.time(), tzinfo=SGT)
    local_end = local_start + timedelta(days=1)
    rows = [dict(row) for row in db.execute('''
        SELECT f.*, a.actual_pm25, a.source AS actual_source,
               a.source_updated_at, a.fetched_at AS actual_fetched_at
        FROM forecasts f LEFT JOIN actuals a
          ON f.forecast_time=a.observation_time AND f.region=a.region
        WHERE f.forecast_time>=? AND f.forecast_time<?
        ORDER BY f.horizon_hours,f.region,f.forecast_time''', (iso(local_start),iso(local_end)))]
    groups = defaultdict(list)
    comparisons = []
    for row in rows:
        valid = row["actual_pm25"] is not None and finite(row["actual_pm25"])
        row["match_status"] = "matched" if valid else "pending_actual"
        row["error"] = row["forecast_pm25"] - row["actual_pm25"] if valid else None
        row["absolute_error"] = abs(row["error"]) if valid else None
        row["target_date_sgt"] = str(day)
        comparisons.append(row)
        groups[(row["region"],row["horizon_hours"])].append(row)
    summaries = []
    for region in TARGETS:
        for horizon in HORIZONS:
            group = groups[(region,horizon)]
            matched = [r for r in group if r["match_status"] == "matched"]
            n = len(group)
            summaries.append(dict(target_date_sgt=str(day), region=region, horizon_hours=horizon,
                models=sorted({r["model"] for r in group}),
                model_run_ids=sorted({r["model_run_id"] for r in group if r["model_run_id"]}),
                thresholds=sorted({r["threshold"] for r in group}),
                n_forecasts=n, n_pending_actual=n-len(matched),
                matching_coverage=len(matched)/n if n else None,
                expected_hourly_origins=24, forecast_coverage=n/24,
                **metric_summary(matched)))
    return comparisons, summaries


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd,"w",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def run_monitoring(day=None, *, lookback_days=3,
                   db_path=DATA_DIR / "monitoring" / "pipeline.sqlite",
                   output_dir=DATA_DIR / "monitoring", api_key=None,
                   mae_limit=None, rmse_limit=None, f2_min=None, min_elevated=5):
    """Revisit the requested day and preceding days, save metrics and alerts.

    Defaults to yesterday in Singapore time. Invalid/missing actuals stay
    pending. API failures are reported and cause CLI exit 2; cached observations
    can still be evaluated with their fetched_at provenance preserved.
    """
    now = datetime.now(timezone.utc)
    day = date.fromisoformat(str(day)) if day else now.astimezone(SGT).date()-timedelta(days=1)
    if day >= now.astimezone(SGT).date():
        raise ValueError("Daily monitoring expects a completed Singapore date")
    if lookback_days < 1 or min_elevated < 1:
        raise ValueError("lookback_days and min_elevated must be positive")
    for name,value in (("mae_limit",mae_limit),("rmse_limit",rmse_limit)):
        if value is not None and (not math.isfinite(value) or value<0):
            raise ValueError(f"Invalid {name}")
    if f2_min is not None and (not math.isfinite(f2_min) or not 0<=f2_min<=1):
        raise ValueError("f2_min must be between 0 and 1")
    db_path=Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError("Forecast archive does not exist; run production inference first")
    output_dir=Path(output_dir)
    errors, all_summaries, alerts = [], [], []
    api_key = api_key or os.environ.get("DATA_GOV_SG_API_KEY")
    db=connect(db_path)
    try:
        for offset in range(lookback_days):
            target_day=day-timedelta(days=offset)
            try:
                actuals=fetch_actuals(target_day,api_key=api_key,raw_dir=output_dir/"nea_raw")
                upsert_actuals(db,actuals)
            except (HTTPError,URLError,TimeoutError,ValueError,RuntimeError) as exc:
                log.error("NEA fetch failed for %s: %s",target_day,exc)
                errors.append(dict(date=str(target_day),error=str(exc)))
            comparisons,summaries=evaluate_day(db,target_day)
            all_summaries.extend(summaries)
            for s in summaries:
                reasons=[]
                if s["n_forecasts"]==0: reasons.append("no_forecasts_archived")
                elif s["n_forecasts"]<24: reasons.append("incomplete_hourly_forecast_coverage")
                if s["n_pending_actual"]:reasons.append("pending_NEA_actuals")
                if mae_limit is not None and s["mae"] is not None and s["mae"]>mae_limit:reasons.append("mae_limit_exceeded")
                if rmse_limit is not None and s["rmse"] is not None and s["rmse"]>rmse_limit:reasons.append("rmse_limit_exceeded")
                if f2_min is not None and s["n_elevated"]>=min_elevated and s["f2"]<f2_min:reasons.append("f2_below_minimum")
                if reasons:alerts.append(dict(date=str(target_day),region=s["region"],horizon_hours=s["horizon_hours"],reasons=reasons))
            report=dict(target_date_sgt=str(target_day),generated_at=iso(now),
                source=ENDPOINT,national_definition="arithmetic mean of all five valid NEA regional readings",
                summaries=summaries,api_fetch_error=next((e["error"] for e in errors if e["date"]==str(target_day)),None))
            atomic_json(output_dir/"daily"/f"{target_day}_metrics.json",report)
            write_csv(output_dir/"daily"/f"{target_day}_metrics.csv",summaries)
            write_csv(output_dir/"daily"/f"{target_day}_comparisons.csv",comparisons)
            if offset<lookback_days-1:time.sleep(2)
        status=dict(generated_at=iso(now),report_date_sgt=str(day),lookback_days=lookback_days,
            status="api_error" if errors else "attention" if alerts else "ok",
            api_errors=errors,alerts=alerts,summaries=all_summaries)
        atomic_json(output_dir/"latest_monitoring.json",status)
        log.info("Monitoring %s: %d groups, %d alerts, %d fetch errors",day,len(all_summaries),len(alerts),len(errors))
        return status
    finally:
        db.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date",dest="day",help="Forecast target day, YYYY-MM-DD in Singapore time; default yesterday")
    parser.add_argument("--lookback-days",type=int,default=3)
    parser.add_argument("--db-path",type=Path,default=DATA_DIR/"monitoring"/"pipeline.sqlite")
    parser.add_argument("--output-dir",type=Path,default=DATA_DIR/"monitoring")
    parser.add_argument("--mae-limit",type=float)
    parser.add_argument("--rmse-limit",type=float)
    parser.add_argument("--f2-min",type=float)
    parser.add_argument("--min-elevated",type=int,default=5)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    report=run_monitoring(**vars(args))
    if report["api_errors"]:raise SystemExit(2)


if __name__=="__main__":
    main()
