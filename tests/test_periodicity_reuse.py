from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from malca.io.table_io import read_feature_table, write_feature_table
from malca.products.feature_layers import expand_feature_layers, to_layer_first_frame
from malca.stv import filter as post_filter


def _historical(source_id="1"):
    return {
        "asas_sn_id": source_id, "lc_path": f"/old/run/{source_id}.dat3",
        "periodicity_period": 4.0, "periodicity_method": "pdm",
        "periodicity_bootstrap_sig": 0.02, "periodicity_score": 1.7,
        "pdm_period": 8.0, "pdm_corrected_period": 4.0, "pdm_theta": 0.2, "pdm_snr": 12.0,
        "ce_period": 4.0, "ce_entropy": 0.3, "ce_snr": 15.0,
        "lsp_period": 6.0, "lsp_power": 0.4, "lsp_bootstrap_sig": 0.05,
        "periodic_flag": True, "stats_gskew": -999.0, "failed_any": True,
    }


def _worker_result(args):
    original, resolved, n_bootstrap, significance, exclude_aliases, method, *_ = args
    stat = Path(resolved).stat()
    return {
        "lc_path": original, "resolved_path": resolved, "error": None,
        "periodicity_checkpoint_version": post_filter.PERIODICITY_CHECKPOINT_VERSION,
        "periodicity_selection_version": post_filter.PERIODICITY_SELECTION_VERSION,
        "periodicity_significance_status": "not_requested",
        "periodicity_n_bootstrap": n_bootstrap, "periodicity_significance_level": significance,
        "periodicity_exclude_aliases": exclude_aliases,
        "periodicity_input_size": stat.st_size, "periodicity_input_mtime_ns": stat.st_mtime_ns,
        "periodicity_period": 2.0, "periodicity_base_period": 2.0,
        "periodicity_harmonic_factor": 1.0, "periodicity_is_rejected": False,
        "pdm_method": method, "pdm_period": 2.0, "pdm_corrected_period": 2.0,
        "pdm_harmonic_factor": 1.0, "pdm_min_theta": 0.5, "pdm_snr": 7.0,
        "ce_period": 2.0, "ce_corrected_period": 2.0, "ce_harmonic_factor": 1.0,
        "ce_min_entropy": 0.6, "ce_snr": 8.0,
        "lsp_period": 3.0, "lsp_power": 0.3, "lsp_bootstrap_sig": 0.2,
        "lsp_is_alias": False, "lsp_is_significant": False,
    }


def test_reuse_by_id_preserves_current_rows_and_resumes_only_new_work(tmp_path, monkeypatch):
    historical = tmp_path / "historical.parquet"
    incomplete = {**_historical("2"), "pdm_snr": np.nan}
    write_feature_table(to_layer_first_frame(pd.DataFrame([_historical(), incomplete])), historical)
    original_bytes = historical.read_bytes()
    paths = [tmp_path / f"{i}.dat3" for i in (3, 1, 2)]
    for path in paths:
        path.write_text("unchanged light curve")
    df = pd.DataFrame({"asas_sn_id": ["3", "1", "2"], "lc_path": list(map(str, paths)),
                       "stats_gskew": [1.1, 2.2, 3.3], "failed_any": False,
                       "catalog_match": [False, True, False], "catalog_period": [np.nan, 99.0, np.nan]})
    calls = []

    def worker(args):
        calls.append(Path(args[0]).stem)
        return _worker_result(args)

    monkeypatch.setattr(post_filter, "_lsp_worker", worker)
    options = dict(reuse_from=historical, checkpoint_dir=tmp_path / "checkpoints", workers=1)
    out = post_filter.validate_periodicity(df, **options)
    assert calls == ["3", "2"]
    assert out["lc_path"].tolist() == df["lc_path"].tolist()
    assert out["stats_gskew"].tolist() == df["stats_gskew"].tolist()
    assert not out["failed_any"].any()
    reused = out.loc[out["asas_sn_id"].eq("1")].iloc[0]
    assert reused["pdm_snr"] == 12.0
    assert reused["ce_snr"] == 15.0
    assert reused["lsp_period"] == 6.0
    assert reused["periodicity_period"] == 4.0  # retain measurements even with catalog consensus
    assert reused["periodicity_score"] == 1.7
    assert reused["periodicity_reused_from"] == str(historical)
    assert len(reused["periodicity_reused_source_sha256"]) == 64
    saved = pd.read_parquet(tmp_path / "checkpoints/period_selection_checkpoint.parquet")
    assert set(saved["lc_path"].map(lambda p: Path(p).stem)) == {"2", "3"}
    assert set(saved["periodicity_n_bootstrap"]) == {0}
    assert historical.read_bytes() == original_bytes

    calls.clear()
    again = post_filter.validate_periodicity(df, **options)
    assert calls == []
    assert again["pdm_snr"].tolist() == out["pdm_snr"].tolist()


def test_reuse_all_rows_through_filter_and_layered_output(tmp_path, monkeypatch):
    historical = tmp_path / "historical.parquet"
    write_feature_table(to_layer_first_frame(pd.DataFrame([_historical()])), historical)

    def unexpected_worker(_):
        pytest.fail("A historical source must not enter the periodicity worker")

    monkeypatch.setattr(post_filter, "_lsp_worker", unexpected_worker)
    df = pd.DataFrame({"asas_sn_id": ["1"], "lc_path": ["/new/1.dat3"],
                       "stats_gskew": [2.2], "failed_significant_detection": [False]})
    out = post_filter.apply_filters(
        df, apply_evidence_strength=False, apply_significant_detection=False,
        apply_run_robustness=False, apply_gaia_ruwe_validation=False,
        apply_gaia_pm_validation=False, apply_periodic_catalog_validation=False,
        apply_periodicity_validation=True, periodicity_reuse_from=historical,
        periodicity_flag_only=True, show_tqdm=False,
    )
    assert len(out) == 1
    assert not out["failed_any"].iloc[0]
    assert out["pdm_snr"].iloc[0] == 12.0
    assert out["stats_gskew"].iloc[0] == 2.2
    destination = tmp_path / "result.parquet"
    write_feature_table(to_layer_first_frame(out), destination)
    reread = expand_feature_layers(read_feature_table(destination))
    assert reread["periodicity_reused_from"].iloc[0] == str(historical)
    assert reread["pdm_snr"].iloc[0] == 12.0

    rejected = post_filter.validate_periodicity(df, reuse_from=historical, flag_only=False)
    assert rejected.empty


def test_reuse_rejects_ambiguous_ids(tmp_path):
    historical = tmp_path / "duplicate.parquet"
    write_feature_table(to_layer_first_frame(pd.DataFrame([_historical(), _historical()])), historical)
    with pytest.raises(ValueError, match="unique"):
        post_filter._load_periodicity_reuse(historical)


def test_new_selection_reuse_replaces_old_fold_period_without_resampling(tmp_path, monkeypatch):
    selected_path = tmp_path / "selected.parquet"
    selected = {**_historical(), "periodicity_bootstrap_sig": np.nan,
                "periodicity_score": np.nan, "periodic_flag": False,
                "periodicity_selection_version": post_filter.PERIODICITY_SELECTION_VERSION,
                "periodicity_n_bootstrap": 0, "periodicity_significance_status": "not_requested",
                "period_native_days": 8.0, "period_corrected_days": 4.0,
                "period_for_fold_days": 4.0, "period_consensus_days": 4.0,
                "period_method": "pdm", "period_confidence": "none",
                "period_evidence_summary": '{"selection_version":"fold_fit_before_significance_v1"}'}
    write_feature_table(to_layer_first_frame(pd.DataFrame([selected])), selected_path)

    def unexpected_worker(_):
        pytest.fail("Saved new selections must bypass search and resampling")

    monkeypatch.setattr(post_filter, "_lsp_worker", unexpected_worker)
    candidates = pd.DataFrame({"asas_sn_id": ["1"], "lc_path": ["/new/1.dat3"],
                               "period_for_fold_days": [99.0], "period_confidence": ["high"]})
    result = post_filter.apply_filters(
        candidates, apply_evidence_strength=False, apply_significant_detection=False,
        apply_run_robustness=False, apply_gaia_ruwe_validation=False,
        apply_gaia_pm_validation=False, apply_periodic_catalog_validation=False,
        apply_periodicity_validation=True, periodicity_reuse_from=selected_path,
        periodicity_n_bootstrap=0, periodicity_flag_only=True, show_tqdm=False,
    )
    for key in ("period_native_days", "period_corrected_days", "period_for_fold_days",
                "period_consensus_days", "period_method", "period_confidence",
                "period_evidence_summary", "periodicity_selection_version",
                "periodicity_n_bootstrap", "periodicity_significance_status"):
        assert result[key].iloc[0] == selected[key], key
    assert not result.periodic_flag.iloc[0]
