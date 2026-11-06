"""A GitHub "Download ZIP" contains Git LFS placeholders instead of the raw data.
The loaders and the backfill's coverage checks must skip them, not crash."""
from pathlib import Path

import pytest

from hazecast.ingest import is_lfs_pointer, raw_files
from hazecast.ingest import historical, pm25

STUB = ("version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:d0cd5891a94201a1ff51113f3eee92ef6f474f437e609d9399db96da8a4471cc\nsize 3128166\n")


def _stub(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(STUB)
    return path


def test_detects_placeholder_and_real_file(tmp_path):
    stub = _stub(tmp_path / "a.csv")
    real = tmp_path / "b.csv"
    real.write_text("1hr_pm25,north\n1/4/2014 1:00,10\n")
    assert is_lfs_pointer(stub) and not is_lfs_pointer(real)
    assert raw_files(tmp_path / "*.csv") == [str(real)]


def test_coverage_checks_skip_placeholders(tmp_path):
    _stub(tmp_path / "pm25" / "Historical1hrPM2.5.csv")
    _stub(tmp_path / "weather" / "open-meteo-1.37N103.80E62m.csv")
    _stub(tmp_path / "firms" / "fire_archive_SV-C2_1.csv")
    assert len(historical.pm25_covered_days(tmp_path / "pm25")) == 0
    assert all(len(h) == 0 for h in historical.weather_covered_hours(tmp_path / "weather").values())
    assert len(historical.firms_covered_days(tmp_path / "firms")) == 0


def test_loader_gives_clear_error_when_only_placeholders(tmp_path):
    _stub(tmp_path / "Historical1hrPM2.5.csv")
    with pytest.raises(FileNotFoundError):
        pm25.load_pm25_bronze(tmp_path)
