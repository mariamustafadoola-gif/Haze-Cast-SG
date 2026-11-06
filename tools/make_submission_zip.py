"""Build the zip to submit: code, docs and raw data, without secrets or generated files.

Run from the repo root, with the real raw data present (`git lfs pull` first):
    python tools/make_submission_zip.py
or, without a local Python, through Docker:
    docker compose --profile dev run --rm jupyter python tools/make_submission_zip.py

Writes HazeCast_Group4_code.zip next to this repo's files. Never includes .env.
"""
from __future__ import annotations

import fnmatch
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "HazeCast_Group4_code.zip"
TOP = "HazeCast"
LFS = b"version https://git-lfs.github.com/spec/"

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints", ".pytest_cache", "mlruns",
             "mlartifacts", "logs", ".vscode", ".idea"}
SKIP_PATHS = ["data/bronze/*", "data/silver/*", "data/monitoring/*", "data/models/*", "data/gold/*.parquet",
              "data/gold/*.tmp", "reports/monitoring/*", "reports/data_quality/*", "data/raw/*/*live*",
              "new_data.zip", "mlflow.db", "airflow.db", "airflow.cfg", "*.pyc", ".DS_Store", "*.zip"]
KEEP = {".gitkeep"}


def wanted(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    if name == ".env" or (name.startswith(".env") and name != ".env.example"):
        return False                      # secrets: never
    if name in KEEP:
        return True
    return not any(fnmatch.fnmatch(rel, p) for p in SKIP_PATHS)


def main() -> int:
    files = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            rel = Path(dirpath, fn).relative_to(ROOT).as_posix()
            if wanted(rel):
                files.append(rel)
    stubs = [f for f in files if f.startswith("data/raw/") and (ROOT / f).read_bytes()[:len(LFS)] == LFS]
    if stubs and "--allow-placeholders" not in sys.argv:
        print(f"{len(stubs)} raw data files are Git LFS placeholders, not data (e.g. {stubs[0]}).")
        print("Run `git lfs pull` first, then run this script again.")
        return 1
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel in sorted(files):
            zf.write(ROOT / rel, f"{TOP}/{rel}")
    raw = sum((ROOT / f).stat().st_size for f in files if f.startswith("data/raw/"))
    print(f"Wrote {OUT.name}: {len(files)} files, raw data {raw / 1e6:.0f} MB, zip {OUT.stat().st_size / 1e6:.0f} MB")
    assert not any(f.rsplit('/', 1)[-1] == ".env" for f in files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
