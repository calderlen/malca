"""Publish completed local statistics at the standard enriched-table path."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

from malca.io.table_io import read_feature_table
from malca.products.feature_layers import ALL_FEATURE_LAYER_COLUMNS, expand_feature_layers
from malca.products.run_metadata import sha256_file
from malca.products.stage_state import assert_reusable_stage_state, read_stage_state


RECEIPT_NAME = "ENRICHED_REFRESH.json"


def refreshed_enriched_is_current(results_dir: Path) -> bool:
    receipt_path = results_dir / RECEIPT_NAME
    if not receipt_path.is_file():
        return False
    receipt = json.loads(receipt_path.read_text())
    target = results_dir / "lc_events_enriched.parquet"
    if not target.is_file() or sha256_file(target) != receipt["updated_sha256"]:
        raise ValueError("The standard enriched table no longer matches its refresh receipt")
    return True


def _atomic_copy(source: Path, destination: Path, *, replace: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copy2(source, temporary)
        if replace:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def backup_original_enriched(run_dir: Path, *, updated_hash: str | None = None) -> Path:
    results = run_dir / "results"
    target = results / "lc_events_enriched.parquet"
    backup = results / "backups/lc_events_enriched.before_period_stats.parquet"
    if backup.exists():
        target_hash = sha256_file(target)
        if target_hash != sha256_file(backup) and target_hash != updated_hash:
            if not refreshed_enriched_is_current(results):
                raise ValueError("Existing backup differs from the standard table; refusing to overwrite it")
    else:
        _atomic_copy(target, backup, replace=False)
    return backup


def promote_local_features(run_dir: Path) -> Path:
    run_dir = run_dir.expanduser().resolve()
    results = run_dir / "results"
    work = results / "local_ml_features"
    state = read_stage_state(work / "LOCAL_FEATURES_STAGE.json")
    if state is None or state["result"]["status"] not in {"success", "partial"}:
        raise ValueError("Statistics are not complete; let the current job finish before publishing")
    if state["result"]["failed"]:
        raise ValueError("Statistics have unresolved processing errors")
    assert_reusable_stage_state(state, fingerprint=state["fingerprint"])
    updated = work / "lc_events_local_features.parquet"
    updated_hash = sha256_file(updated)
    backup = backup_original_enriched(run_dir, updated_hash=updated_hash)
    old = expand_feature_layers(read_feature_table(backup))
    new = expand_feature_layers(read_feature_table(updated))
    for frame in (old, new):
        if frame["candidate_id"].isna().any() or frame["candidate_id"].duplicated().any():
            raise ValueError("Enriched tables require unique, nonmissing candidate IDs")
    if old["candidate_id"].tolist() != new["candidate_id"].tolist():
        raise ValueError("Refreshed statistics changed candidate IDs or row order")
    if len(new) != state["result"]["expected"]:
        raise ValueError("Refreshed table does not cover the expected cohort")
    if not new["stats_compute_status"].isin(["ok", "missing_period"]).all():
        raise ValueError("Refreshed table has unresolved statistics errors")
    if old["lc_path"].map(lambda p: Path(str(p)).name).tolist() != new["lc_path"].map(lambda p: Path(str(p)).name).tolist():
        raise ValueError("Refreshed statistics refer to different light curves")
    # Scientific updates are limited to the columns produced by the completed
    # measurement stage. Preserve the original events, context, and decisions.
    from malca.stv.filter import PERIODICITY_MERGE_COLS

    measured = {*PERIODICITY_MERGE_COLS, *ALL_FEATURE_LAYER_COLUMNS, "phase_period_days", "phase_source", "lc_path"}
    for column in old:
        if column.startswith("stats_") or column in measured:
            continue
        if column not in new:
            if old[column].notna().any():
                raise ValueError(f"Refreshed table lost original column {column}")
            continue
        left = old[column].astype(object).where(old[column].notna(), None)
        right = new[column].astype(object).where(new[column].notna(), None)
        if not left.eq(right).all():
            raise ValueError(f"Refreshed table changed original non-statistics column {column}")
    target = results / "lc_events_enriched.parquet"
    if sha256_file(target) != updated_hash:
        _atomic_copy(updated, target, replace=True)
    receipt = {
        "standard_table": str(target), "backup": str(backup),
        "backup_sha256": sha256_file(backup), "updated_sha256": updated_hash,
        "completed_statistics": str(updated), "candidates": len(new),
        "stage_fingerprint": state["fingerprint"]["digest"],
    }
    receipt_path = results / RECEIPT_NAME
    fd, name = tempfile.mkstemp(prefix=f".{RECEIPT_NAME}.", dir=results)
    with os.fdopen(fd, "w") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    os.replace(name, receipt_path)
    return target
