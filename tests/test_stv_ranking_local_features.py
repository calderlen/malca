import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from malca.io.table_io import write_feature_table
from malca.products.feature_layers import to_layer_first_frame


@pytest.fixture
def ranking():
    path = Path(__file__).resolve().parents[1] / "scripts/rank_stv_dipper_candidates.py"
    spec = importlib.util.spec_from_file_location("ranking_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frames():
    review = pd.DataFrame({
        "candidate_id": ["two", "one"], "lc_path": ["/local/2.dat3", "/local/1.dat3"],
        "stats_harmonics_a0": [99., 99.], "periodicity_period": [99., 99.],
        "periodicity_bootstrap_sig": [.001, .001], "tess_flux_range": [8., 9.],
    })
    local = pd.DataFrame({
        "candidate_id": ["one", "two"], "lc_path": ["/cluster/1.dat3", "/cluster/2.dat3"],
        "stats_harmonics_a0": [1., 2.], "periodicity_period": [1.5, 2.5],
        "stats_compute_status": ["ok", "ok"], "periodicity_bootstrap_sig": np.nan,
        "tess_flux_range": np.nan,
    })
    return review, local


def test_refreshed_measurements_replace_old_values_by_id_without_erasing_context(ranking):
    review, local = frames()
    merged = ranking._attach_saved_features(review, local, refreshed_periods=True)
    assert merged.candidate_id.tolist() == ["two", "one"]
    assert merged.stats_harmonics_a0.tolist() == [2., 1.]
    assert merged.periodicity_period.tolist() == [2.5, 1.5]
    assert merged.periodicity_bootstrap_sig.isna().all()
    assert merged.tess_flux_range.tolist() == [8., 9.]
    assert review.periodicity_period.eq(99.).all()


@pytest.mark.parametrize("problem", ["dropped_candidate", "wrong_lightcurve", "stats_error", "duplicate_id"])
def test_refreshed_features_reject_incomplete_or_mismatched_inputs(ranking, problem):
    review, local = frames()
    if problem == "dropped_candidate":
        review = review.iloc[:1]
    elif problem == "wrong_lightcurve":
        local.loc[0, "lc_path"] = "/cluster/3.dat3"
    elif problem == "stats_error":
        local.loc[0, "stats_compute_status"] = "error"
    else:
        local.loc[0, "candidate_id"] = "two"
    with pytest.raises(ValueError):
        ranking._attach_saved_features(review, local, refreshed_periods=True)


@pytest.mark.parametrize("standard_table", [False, True])
def test_feature_audit_uses_new_statistics_builds_events_and_does_not_score(ranking, tmp_path, monkeypatch, standard_table):
    review, local = frames()
    lcs = tmp_path / "bundle_assets/lightcurves"
    lcs.mkdir(parents=True)
    for name in ("1.dat3", "2.dat3"):
        (lcs / name).write_text("unchanged light curve")
    (tmp_path / "results").mkdir()
    saved = tmp_path / "results/lc_events_enriched.parquet" if standard_table else tmp_path / "local.parquet"
    write_feature_table(to_layer_first_frame(local, run_derived=False), saved)
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"test model")
    columns = ["stats_harmonics_a0", "periodicity_period", "periodicity_bootstrap_sig", "tess_flux_range", "delta_mag_peak"]
    monkeypatch.setattr(ranking.joblib, "load", lambda _path: {"feature_columns": columns})
    monkeypatch.setattr(ranking, "load_review_population", lambda *_a, **_kw: review.copy())
    monkeypatch.setattr(ranking, "refreshed_enriched_is_current", lambda _path: standard_table)
    calls = []

    def events(table, *_a, **_kw):
        assert table.stats_harmonics_a0.tolist() == [2., 1.]
        assert table.periodicity_period.tolist() == [2.5, 1.5]
        calls.append(len(table))
        return table.assign(delta_mag_peak=.2, recovery_feature_state="ok")

    def no_scoring(*_a, **_kw):
        pytest.fail("Feature audit must not run the classifier")

    monkeypatch.setattr(ranking, "add_recovery_bounded_event_features", events)
    monkeypatch.setattr(ranking, "score_tuned_calibrated_head", no_scoring)
    ranking.rank(tmp_path, model_path, None, 1, local_features=None if standard_table else saved, features_only=True)
    out = tmp_path / "results/dipper_ranking"
    report = json.loads((out / "feature_coverage.json").read_text())
    assert calls == [2]
    assert report["candidates"] == 2
    assert report["feature_non_null_counts"]["delta_mag_peak"] == 2
    assert report["entirely_unavailable_features"] == ["periodicity_bootstrap_sig"]
    assert report["provisional"]
    assert report["probability_scope"] == "unvalidated_missing_feature_pattern"
    assert not (out / "all_candidates_ranked.parquet").exists()
