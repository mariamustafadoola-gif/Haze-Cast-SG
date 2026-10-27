import logging
import pandas as pd
from hazecast.config import SILVER_DIR
from hazecast.silver_to_gold import build_gold_hourly, write_gold_hourly

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

pm25 = pd.read_parquet(SILVER_DIR / "pm25.parquet")
weather = pd.read_parquet(SILVER_DIR / "weather.parquet")
firms = pd.read_parquet(SILVER_DIR / "firms_daily.parquet")

gold = build_gold_hourly(pm25, weather, firms)
path = write_gold_hourly(gold)
log.info("hourly gold: %d rows, %d columns, %s to %s -> %s",
          len(gold), gold.shape[1], gold.date.min(), gold.date.max(), path)
