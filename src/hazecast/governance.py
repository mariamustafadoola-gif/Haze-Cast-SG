"""Champion/challenger promotion and rollback for the hourly models.

The retrain DAG trains a *challenger* into a staging directory; this module
decides whether it replaces the *champion* in data/models/hourly (the
directory hazecast.inference reads), following docs/MODEL_GOVERNANCE.md.

    python -m hazecast.governance promote --candidate data/models/hourly_staging/<run>
    python -m hazecast.governance rollback            # previous champion
    python -m hazecast.governance rollback --to data/models/hourly/history/model_manifest_<ts>.json
    python -m hazecast.governance status

Promotion gate (all must pass):
  1. integrity     manifest complete (18 target x horizon entries), every artifact
                   loads and gives a finite prediction on the latest complete gold row
  2. skill         on its own untouched holdout, the challenger's F2 (elevated-hour detection,
                   the selection objective) is within PROMOTION_F2_TOLERANCE of persistence's
                   for at least PROMOTION_MIN_SKILL_TARGETS of the groups with haze hours.
                   Not MAE: the q85 models are upward-biased by design and never beat
                   persistence on MAE (0 of 18 groups on the 2024-26 holdout). Persistence is
                   a floor, not a target: it cannot forecast the onset of an episode at all.
  3. no regression vs the champion's recorded holdout: MAE at most
                   PROMOTION_MAX_MAE_DEGRADATION worse in a majority of groups
     (the two holdouts are different time windows, so this is a guard-rail
     against a clearly worse model, not a head-to-head contest)
Promotion copies the challenger's artifacts, writes the manifest last (so
inference never sees a half-published model), archives the old manifest under
history/ and appends to promotion_log.jsonl. Artifacts have unique names and
are never deleted, so any archived manifest can be restored.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from hazecast import config as C
from hazecast.config import GOLD_DIR, MODELS_DIR
from hazecast.pipeline_store import TARGETS, HORIZONS, atomic_json

log = logging.getLogger(__name__)
PRODUCTION_DIR = MODELS_DIR / "hourly"
MANIFEST = "model_manifest.json"
# Files that travel with a model version (besides the per-model artifacts).
COMPANION_FILES = ("eval_summary.csv", "holdout_predictions.csv")
# All copies use shutil.copyfile (contents only): copying metadata (copy2) fails with EPERM
# when the container user overwrites a file owned by the host user on a bind mount.


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _persistence_holdout(gold: pd.DataFrame, target: str, horizon: int, test_start, threshold: float) -> dict:
    """MAE and F2 of "PM2.5 in h hours = PM2.5 now" over the challenger's holdout window."""
    from hazecast.hourly_model import build_label
    label = build_label(gold, target, horizon)
    mask = (gold.date >= pd.Timestamp(test_start)) & label.notna() & gold[f"now_{target}"].notna()
    if not mask.any():
        return dict(mae=None, f2=None)
    p, y = gold.loc[mask, f"now_{target}"].to_numpy(dtype=float), label[mask].to_numpy(dtype=float)
    actual, flagged = y >= threshold, p >= threshold
    tp, fp, fn = (actual & flagged).sum(), (~actual & flagged).sum(), (actual & ~flagged).sum()
    return dict(mae=float(np.abs(p - y).mean()), f2=float(5 * tp / (5 * tp + 4 * fn + fp)) if actual.any() else None)


def check_candidate(candidate_dir: Path, champion_dir: Path = PRODUCTION_DIR,
                    gold_path: Path = GOLD_DIR / "hourly_features.parquet") -> dict:
    import joblib
    from hazecast.hourly_model import _prepare
    candidate_dir = Path(candidate_dir)
    manifest = _load(candidate_dir / MANIFEST)
    gold = _prepare(pd.read_parquet(gold_path))
    cols = manifest["feature_columns"]
    checks, groups = {}, {}

    # 1. integrity
    problems = []
    if manifest.get("pipeline") != "hourly" or sorted(manifest.get("horizons_hours", [])) != list(HORIZONS):
        problems.append("manifest is not a 6/12/24h hourly manifest")
    complete = gold.loc[np.isfinite(gold[cols].to_numpy(dtype=float)).all(axis=1)]
    row = complete.iloc[[-1]] if len(complete) else None
    for h in HORIZONS:
        entries = manifest["models"].get(str(h), {})
        if set(entries) != set(TARGETS):
            problems.append(f"h={h}: missing targets {sorted(set(TARGETS) - set(entries))}")
            continue
        for t, e in entries.items():
            if e["model"] == "persistence" or row is None:
                continue
            try:
                pred = float(joblib.load(candidate_dir / e["artifact"]).predict(row[cols])[0])
                if not np.isfinite(pred):
                    problems.append(f"{t} h={h}: non-finite prediction")
            except Exception as exc:  # noqa: BLE001 - any load/predict failure blocks promotion
                problems.append(f"{t} h={h}: {type(exc).__name__}: {exc}")
    if row is None:
        problems.append("no complete gold row to smoke-test predictions on")
    checks["integrity"] = dict(passed=not problems, problems=problems)
    if problems:
        return dict(passed=False, checks=checks, groups=groups)

    # 2 + 3. skill vs persistence and regression vs champion
    champion = _load(champion_dir / MANIFEST) if (Path(champion_dir) / MANIFEST).exists() else None
    for h in HORIZONS:
        for t in TARGETS:
            e = manifest["models"][str(h)][t]
            test = e.get("final_test") or {}
            mae, f2 = test.get("mae"), test.get("f2")
            pers = _persistence_holdout(gold, t, h, e["test_start"], float(manifest["threshold"]))
            champ_mae = (((champion or {}).get("models", {}).get(str(h), {}).get(t) or {}).get("final_test") or {}).get("mae")
            groups[f"{t}|{h}"] = dict(
                model=e["model"], holdout_mae=mae, persistence_mae=pers["mae"], holdout_f2=f2, persistence_f2=pers["f2"],
                f2_vs_persistence=(f2 - pers["f2"]) if f2 is not None and pers["f2"] is not None else None,
                champion_mae=champ_mae,
                mae_change=(mae / champ_mae - 1) if mae is not None and champ_mae else None)
    diffs = [g["f2_vs_persistence"] for g in groups.values() if g["f2_vs_persistence"] is not None]
    share = float(np.mean([d >= -C.PROMOTION_F2_TOLERANCE for d in diffs])) if diffs else 1.0
    checks["skill"] = dict(passed=share >= C.PROMOTION_MIN_SKILL_TARGETS, share_within_f2_tolerance=share,
                           groups_with_haze=len(diffs), tolerance=C.PROMOTION_F2_TOLERANCE,
                           required=C.PROMOTION_MIN_SKILL_TARGETS,
                           note="no haze hours in holdout: skill not judged" if not diffs else None)
    changes = [g["mae_change"] for g in groups.values() if g["mae_change"] is not None]
    if champion is None or not changes:
        checks["no_regression"] = dict(passed=True, note="no champion to compare against")
    else:
        ok_share = float(np.mean([c <= C.PROMOTION_MAX_MAE_DEGRADATION for c in changes]))
        checks["no_regression"] = dict(passed=ok_share > 0.5, share_within_tolerance=ok_share,
                                       tolerance=C.PROMOTION_MAX_MAE_DEGRADATION)
    return dict(passed=all(c["passed"] for c in checks.values()), checks=checks, groups=groups)


def _archive_current(production_dir: Path) -> Path | None:
    current = production_dir / MANIFEST
    if not current.exists():
        return None
    history = production_dir / "history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = history / f"model_manifest_{stamp}.json"
    shutil.copyfile(current, dest)
    for name in COMPANION_FILES:
        if (production_dir / name).exists():
            shutil.copyfile(production_dir / name, history / f"{Path(name).stem}_{stamp}{Path(name).suffix}")
    return dest


def _log(production_dir: Path, entry: dict) -> None:
    production_dir.mkdir(parents=True, exist_ok=True)
    with open(production_dir / "promotion_log.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def _publish(manifest: dict, production_dir: Path, **provenance) -> None:
    atomic_json(production_dir / MANIFEST, {**manifest, "published_at": _now(), **provenance})


def promote(candidate_dir: Path, production_dir: Path = PRODUCTION_DIR,
            gold_path: Path = GOLD_DIR / "hourly_features.parquet", force: bool = False) -> dict:
    candidate_dir, production_dir = Path(candidate_dir), Path(production_dir)
    result = check_candidate(candidate_dir, production_dir, gold_path)
    result.update(candidate=str(candidate_dir), decided_at=_now(), forced=force and not result["passed"])
    if not result["passed"] and not force:
        log.warning("Challenger %s rejected: %s", candidate_dir,
                    {k: v["passed"] for k, v in result["checks"].items()})
        _log(production_dir, dict(action="rejected", **result))
        return dict(promoted=False, **result)
    manifest = _load(candidate_dir / MANIFEST)
    production_dir.mkdir(parents=True, exist_ok=True)
    for entries in manifest["models"].values():
        for e in entries.values():
            if e.get("artifact"):
                shutil.copyfile(candidate_dir / e["artifact"], production_dir / e["artifact"])
    archived = _archive_current(production_dir)
    for name in COMPANION_FILES:
        if (candidate_dir / name).exists():
            shutil.copyfile(candidate_dir / name, production_dir / name)
    _publish(manifest, production_dir, promoted_from=str(candidate_dir), replaced=str(archived) if archived else None)
    _log(production_dir, dict(action="promoted", archived=archived, **result))
    log.info("Promoted %s (previous manifest archived to %s)", candidate_dir, archived)
    return dict(promoted=True, archived=str(archived) if archived else None, **result)


def rollback(to: Path | None = None, production_dir: Path = PRODUCTION_DIR) -> dict:
    """Restore an archived manifest (default: the most recent one). The current one is archived first."""
    production_dir = Path(production_dir)
    history = sorted((production_dir / "history").glob("model_manifest_*.json"))
    target = Path(to) if to else (history[-1] if history else None)
    if target is None or not target.exists():
        raise FileNotFoundError("No archived manifest to roll back to")
    manifest = _load(target)
    missing = [e["artifact"] for entries in manifest["models"].values() for e in entries.values()
               if e.get("artifact") and not (production_dir / e["artifact"]).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot roll back: artifacts missing from {production_dir}: {missing[:3]}")
    archived = _archive_current(production_dir)
    _publish(manifest, production_dir, rolled_back_from=str(archived), restored=str(target))
    _log(production_dir, dict(action="rollback", restored=str(target), archived=str(archived), decided_at=_now()))
    log.info("Rolled back to %s", target)
    return dict(restored=str(target), archived=str(archived))


def status(production_dir: Path = PRODUCTION_DIR) -> dict:
    production_dir = Path(production_dir)
    if not (production_dir / MANIFEST).exists():
        return dict(champion=None)
    m = _load(production_dir / MANIFEST)
    return dict(champion=dict(published_at=m.get("published_at"), trained_at=m.get("trained_at"),
                              models={h: {t: e["model"] for t, e in entries.items()} for h, entries in m["models"].items()}),
                history=[p.name for p in sorted((production_dir / "history").glob("model_manifest_*.json"))])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("promote")
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--force", action="store_true", help="promote even if the gate fails (record why!)")
    r = sub.add_parser("rollback")
    r.add_argument("--to", type=Path)
    sub.add_parser("status")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = (promote(a.candidate, force=a.force) if a.cmd == "promote"
           else rollback(a.to) if a.cmd == "rollback" else status())
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
