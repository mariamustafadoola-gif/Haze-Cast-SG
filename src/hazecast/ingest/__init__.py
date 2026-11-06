"""Ingestion helpers shared by the source loaders."""
from __future__ import annotations

import glob
import logging
from pathlib import Path

log = logging.getLogger(__name__)
_LFS_HEADER = b"version https://git-lfs.github.com/spec/"


def is_lfs_pointer(path: str | Path) -> bool:
    """True if the file is a Git LFS placeholder rather than real data.

    GitHub's "Download ZIP" (and a clone without Git LFS) contains these small
    text stubs instead of the large raw files. Reading them as CSV fails, so
    the loaders skip them and the backfill downloads the data instead.
    """
    try:
        with open(path, "rb") as fh:
            return fh.read(len(_LFS_HEADER)) == _LFS_HEADER
    except OSError:
        return False


def raw_files(pattern: str | Path) -> list[str]:
    """Sorted files matching a glob pattern, without Git LFS placeholders."""
    files = sorted(glob.glob(str(pattern)))
    stubs = [f for f in files if is_lfs_pointer(f)]
    if stubs:
        log.warning("Skipping %d Git LFS placeholder file(s) with no data (e.g. %s); "
                    "run `git lfs pull`, or let the backfill download the history.",
                    len(stubs), Path(stubs[0]).name)
    return [f for f in files if f not in stubs]
