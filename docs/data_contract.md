# HazeCast SG: data contract (gold layer)

This is the agreement between the **data team** and the **AI team**.
The data team promises to deliver these two tables with exactly these columns.
The AI team builds against these columns. If a column needs to change, agree it here first.

Until the real pipeline is ready, `dummy_data/` holds tables with the same columns and made-up values.

## How the two tables join

`gold_feature_store` and `gold_label_store` join on **region** and **snapshot_date**.
Each pair is one region on one day, as seen at **06:00 Singapore time**.

## gold_feature_store (one row per region per day)

Every value is known by 06:00 on `snapshot_date`. Nothing after 06:00 is used.

| Column | Type | Meaning |
|---|---|---|
| region | string | north, south, east, west or central |
| snapshot_date | date | the day of the forecast |
| snapshot_ts | timestamp | snapshot_date at 06:00, Singapore time |
| pm25_now | float | PM2.5 reading for the hour ending 06:00 (µg/m³) |
| pm25_mean_6h | float | average PM2.5 over the 6 hours before 06:00 |
| pm25_mean_24h | float | average PM2.5 over the 24 hours before 06:00 |
| pm25_max_24h | float | highest hourly PM2.5 in the 24 hours before 06:00 |
| pm25_change_6h | float | pm25_now minus pm25_mean_6h (rising or falling) |
| fire_count_24h | int | fire hotspots in Sumatra and Kalimantan in the last 24 hours |
| fire_frp_sum_24h | float | total fire radiative power of those fires (MW) |
| fire_count_72h | int | fire hotspots in the last 72 hours |
| fire_frp_sum_72h | float | total fire radiative power in the last 72 hours (MW) |
| wind_speed_mean_24h | float | average wind speed over the last 24 hours (m/s) |
| wind_dir_sin | float | sine of the average wind direction |
| wind_dir_cos | float | cosine of the average wind direction |
| wind_from_sw_share_24h | float | share of the last 24 hours with wind from the south-west (towards Singapore from Sumatra), 0 to 1 |
| rain_sum_24h | float | total rain over the last 24 hours (mm) |
| day_of_year | int | 1 to 366 |
| month | int | 1 to 12 |
| is_sw_monsoon | int | 1 from June to September, else 0 |

Missing values are left empty (NULL). The AI team decides how to fill them, using training months only.

## gold_label_store (one row per region per day)

| Column | Type | Meaning |
|---|---|---|
| region | string | join key |
| snapshot_date | date | join key |
| label_pm25_mean_next_6h | float | average PM2.5 from 06:00 to 12:00 |
| label_pm25_mean_next_12h | float | average PM2.5 from 06:00 to 18:00 |
| label_pm25_mean_next_24h | float | average PM2.5 from 06:00 to 06:00 the next day |
| label_def | string | always "mean_pm25_next_window_ugm3" |
| is_haze_next_24h | int | 1 if label_pm25_mean_next_24h is above 55 (NEA "Elevated" band), else 0. Use it to score haze days separately |

## Agreed split (time-based)

Train 2016 to 2021, validate 2022, test 2023, out-of-time 2024.

## File locations

| | Dummy (now) | Real (when the pipeline is ready) |
|---|---|---|
| Feature store | dummy_data/gold_feature_store.parquet | datamart/gold/feature_store/ |
| Label store | dummy_data/gold_label_store.parquet | datamart/gold/label_store/ |
