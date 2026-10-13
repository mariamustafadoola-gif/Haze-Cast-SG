"""
HazeCast SG: runs the data pipeline end to end.

On main this only creates the dummy gold tables. The data team replaces it on
the data branch with the real steps: ingest, bronze, silver, gold.

    python main.py
"""

import runpy

if __name__ == "__main__":
    print("Data pipeline not built yet. Creating dummy gold tables instead.")
    runpy.run_path("scripts/make_dummy_data.py", run_name="__main__")
