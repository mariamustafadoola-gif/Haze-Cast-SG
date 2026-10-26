# Getting the raw data

## Automated (recommended)

All three sources can be downloaded by script, no manual clicking:

```bash
cp .env.example .env            # add FIRMS_MAP_KEY (free: https://firms.modaps.eosdis.nasa.gov/api/map_key/)
export PYTHONPATH=src
python -m hazecast.ingest.historical --dry-run   # what is missing between 2014-04-01 and yesterday
python -m hazecast.ingest.historical             # download it (training history)
python -m hazecast.ingest.live --days 3          # the latest hours (for inference)
python -m hazecast.refresh --no-fetch            # rebuild bronze -> silver -> gold
```

`hazecast.ingest.historical` only requests what `data/raw/` doesn't already
have, so it is safe to re-run. On a clone with the Git LFS files it just
tops up the last few weeks; on a fresh clone it downloads everything:

| Source | API used | Key |
|---|---|---|
| PM2.5 | data.gov.sg dataset download (`initiate-download` / `poll-download`) for the bulk file, then `v2/real-time/api/pm25?date=` per day after it ends | none (optional `DATA_GOV_SG_API_KEY` raises the rate limit) |
| Weather | Open-Meteo archive API (default model blend, as in the manual downloads), one request per point per year; the last 2 days are re-fetched on every run because the newest hours are still being revised | none |
| Fires | NASA FIRMS area API, `VIIRS_SNPP_SP` (science quality) up to its latest date, `VIIRS_SNPP_NRT` after | `FIRMS_MAP_KEY` |

In Docker, the same code runs as Airflow DAGs: `hazecast_backfill`
(manual, history), `hazecast_hourly` (live, every hour) and
`hazecast_monitoring` (daily top-up that makes the live hours permanent).
Auto-downloaded files are git-ignored; the LFS files below are the seed.

## Manual download (reference)

The original manual steps, kept for reference. The committed Git LFS files
under `data/raw/` came from these.

## 1. PM2.5 (label + persistence feature)

Download **"Historical 1-hr PM2.5"** from
<https://data.gov.sg/datasets/d_6c0d5fc34838b12472475fdb73c0af29/view>
(the single continuous file, not the older yearly-file collection) and
place the CSV in `data/raw/pm25/`.

## 2. Weather (Singapore + 4 fire-region points)

For each point below, open the link, download the CSV, and place it in
`data/raw/weather/`. All five use the same date range so features line up.

| Point | Link |
|---|---|
| Singapore | `https://archive-api.open-meteo.com/v1/archive?latitude=1.37&longitude=103.80&start_date=2014-04-01&end_date=2026-07-31&hourly=wind_speed_10m,wind_direction_10m,precipitation&timezone=Asia%2FSingapore&format=csv` |
| Riau | `https://archive-api.open-meteo.com/v1/archive?latitude=0.5&longitude=101.4&start_date=2014-04-01&end_date=2026-07-31&hourly=wind_speed_10m,wind_direction_10m,precipitation&timezone=Asia%2FSingapore&format=csv` |
| Palembang | `https://archive-api.open-meteo.com/v1/archive?latitude=-3.0&longitude=104.8&start_date=2014-04-01&end_date=2026-07-31&hourly=wind_speed_10m,wind_direction_10m,precipitation&timezone=Asia%2FSingapore&format=csv` |
| Pontianak | `https://archive-api.open-meteo.com/v1/archive?latitude=0.0&longitude=109.3&start_date=2014-04-01&end_date=2026-07-31&hourly=wind_speed_10m,wind_direction_10m,precipitation&timezone=Asia%2FSingapore&format=csv` |
| Palangka Raya | `https://archive-api.open-meteo.com/v1/archive?latitude=-2.2&longitude=113.9&start_date=2014-04-01&end_date=2026-07-31&hourly=wind_speed_10m,wind_direction_10m,precipitation&timezone=Asia%2FSingapore&format=csv` |

## 3. Fire hotspots (NASA FIRMS)

Go to <https://firms.modaps.eosdis.nasa.gov/download/>, register for a
free account, and request:
- Source: **Suomi-NPP VIIRS**
- Area: **Indonesia**
- Format: **CSV**
- Date range: as far back as you can get (2014-01-01 onwards ideally --
  see below)

NASA emails a `.zip`. Drop it, unopened, in `data/raw/firms/` -- the
ingestion script extracts it automatically.

**Status as of this proof of concept:** both the 2019-01-01 to
2026-09-21 export and the 2014-01-01 to 2018-12-31 backfill have been
obtained and validated -- 2,181,280 detections total across Sumatra +
Kalimantan, no data gaps found in 2014-2018 (`hazecast.bronze_to_silver`
flags 73 likely-gap days, all in 2022/2024/2026). Both the 2015 and 2019
haze seasons are now evaluated in `hazecast.train` (see
`hazecast.config.HAZE_SEASONS` and `hazecast.train._usable_seasons`,
which checks feature coverage before using a season as a test fold). If
a future FIRMS request ever fails to attach to an upload, request it
split by year -- that was the workaround that ultimately wasn't even
needed once the file was placed directly in `data/raw/firms/` from disk
instead of going through a chat upload.

## After downloading

```
python -m hazecast.ingest.pm25
python -m hazecast.ingest.firms
python -m hazecast.ingest.weather
python -m hazecast.bronze_to_silver
python -m hazecast.silver_to_gold
python -m hazecast.train
python -m hazecast.score
```

Or, once Docker Compose is running, the same chain is what
`dags/hazecast_daily_dag.py` executes nightly.
