"""Train a quantile-range challenger into staging and run the promotion gate on it (check only).

    python experiments/multiq_gate_check.py data/models/hourly_staging/multiq_check_YYYYMMDD

Nothing is promoted. Writes gate_check.json next to the challenger's manifest.
"""
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from hazecast import governance
from hazecast.config import GOLD_DIR
from hazecast.hourly_model import train_final_hourly_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
out = Path(sys.argv[1])
train_final_hourly_models(pd.read_parquet(GOLD_DIR / "hourly_features.parquet"), out_dir=out)
result = governance.check_candidate(out)
(out / "gate_check.json").write_text(json.dumps(result, indent=2, default=str))
manifest = json.loads((out / "model_manifest.json").read_text())
print("QUANTILE PER HORIZON:", {h: (v["chosen"], v["proposed"], v["confidence"])
                                for h, v in manifest["quantile_selection"].items()})
print("DUAL BAR (targets passing, both seasons / holdout):",
      {h: {n: f"{v['both_seasons']}/{v['holdout']} of {v['of']}" for n, v in c["dual_bar"].items()}
       for h, c in manifest["quantile_selection"].items()})
print("GATE:", result["passed"], {k: v["passed"] for k, v in result["checks"].items()})
