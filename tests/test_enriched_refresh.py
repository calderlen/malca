import json

import pandas as pd
import pytest

from malca.io.table_io import read_feature_table, write_feature_table
from malca.products.enriched_refresh import (
    backup_original_enriched, promote_local_features, refreshed_enriched_is_current,
)
from malca.products.feature_layers import expand_feature_layers, to_layer_first_frame
from malca.products.run_metadata import sha256_file
from malca.products.stage_state import StageResult, build_stage_fingerprint, write_stage_state


def setup_run(tmp_path, *, status="success", change_context=False, drop_row=False):
    results = tmp_path / "results"
    work = results / "local_ml_features"
    work.mkdir(parents=True)
    standard = results / "lc_events_enriched.parquet"
    source = work / "lc_events_local_features.parquet"
    old = pd.DataFrame({
        "candidate_id": ["one", "two"], "lc_path": ["/lcs/1.dat3", "/lcs/2.dat3"],
        "failed_any": False, "dip_count": [3, 4], "stats_harmonics_a0": [None, None],
    })
    new = old.assign(stats_harmonics_a0=[1.0, 2.0], stats_compute_status="ok", periodicity_period=[2., 3.])
    if change_context:
        new.loc[0, "dip_count"] = 99
    if drop_row:
        new = new.iloc[:1]
    if status == "partial":
        new.loc[1, "stats_compute_status"] = "missing_period"
        new.loc[1, "periodicity_period"] = None
    write_feature_table(to_layer_first_frame(old, run_derived=False), standard)
    write_feature_table(to_layer_first_frame(new, run_derived=False), source)
    fingerprint = build_stage_fingerprint(stage="local_ml_features", stage_version="test",
                                          candidate_ids=old.candidate_id, input_paths=[standard])
    result = StageResult("local_ml_features", status, 2,
                         succeeded=2 if status == "success" else 1 if status == "partial" else 0,
                         skipped=1 if status == "partial" else 0)
    write_stage_state(work / "LOCAL_FEATURES_STAGE.json", fingerprint=fingerprint, result=result,
                      outputs=[source] if status != "running" else [])
    return standard, source


@pytest.mark.parametrize("status", ["success", "partial"])
def test_completed_refresh_backs_up_and_publishes_standard_table(tmp_path, status):
    standard, source = setup_run(tmp_path, status=status)
    original_hash = sha256_file(standard)
    backup = backup_original_enriched(tmp_path)
    assert sha256_file(standard) == original_hash
    assert sha256_file(backup) == original_hash
    assert promote_local_features(tmp_path) == standard
    assert sha256_file(standard) == sha256_file(source)
    assert sha256_file(backup) == original_hash
    assert refreshed_enriched_is_current(standard.parent)
    assert promote_local_features(tmp_path) == standard
    assert backup_original_enriched(tmp_path) == backup
    assert sha256_file(backup) == original_hash
    frame = expand_feature_layers(read_feature_table(standard))
    assert frame.candidate_id.tolist() == ["one", "two"]
    assert frame.dip_count.tolist() == [3, 4]


@pytest.mark.parametrize("problem", ["running", "context_changed", "cohort_changed", "output_changed"])
def test_unsafe_refresh_preserves_standard_table(tmp_path, problem):
    standard, source = setup_run(tmp_path, status="running" if problem == "running" else "success",
                                  change_context=problem == "context_changed", drop_row=problem == "cohort_changed")
    original_hash = sha256_file(standard)
    if problem == "output_changed":
        source.write_bytes(b"incomplete replacement")
    with pytest.raises(ValueError):
        promote_local_features(tmp_path)
    assert sha256_file(standard) == original_hash


def test_current_refresh_check_prevents_silent_overwrite(tmp_path):
    standard, _ = setup_run(tmp_path)
    promote_local_features(tmp_path)
    standard.write_bytes(b"stale cluster merge")
    with pytest.raises(ValueError, match="refresh receipt"):
        refreshed_enriched_is_current(standard.parent)
