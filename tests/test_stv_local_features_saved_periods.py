import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from malca.io.table_io import read_feature_table, write_feature_table
from malca.products.feature_layers import expand_feature_layers, to_layer_first_frame


@pytest.fixture
def local_features():
    path = Path(__file__).resolve().parents[1] / "scripts/compute_stv_local_features.py"
    spec = importlib.util.spec_from_file_location("local_features_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_inputs(tmp_path, *, missing_period=False):
    lcs = tmp_path / "bundle_assets/lightcurves"
    lcs.mkdir(parents=True)
    for number in (1, 2):
        (lcs / f"{number}.dat3").write_text("unchanged photometry")
    source = tmp_path / "enriched.parquet"
    saved = tmp_path / "selected.parquet"
    frame = pd.DataFrame({
        "candidate_id": ["two", "one"], "asas_sn_id": ["2", "1"],
        "lc_path": ["/cluster/2.dat3", "/cluster/1.dat3"], "failed_any": False,
        "stats_variability_quasi_periodicity_q": np.nan,
        "stats_harmonics_a0": np.nan, "periodicity_period": 99.0,
    })
    periods = pd.DataFrame({
        "candidate_id": ["one", "two"], "asas_sn_id": ["1", "2"],
        "lc_path": ["/cluster/1.dat3", "/cluster/2.dat3"],
        "periodicity_period": [np.nan if missing_period else 1.5, 2.5],
        "period_for_fold_days": [np.nan if missing_period else 1.5, 2.5],
        "periodicity_status": ["error" if missing_period else "ok", "ok"],
        "periodicity_n_bootstrap": 0, "pdm_snr": 8.0,
        "periodicity_significance_status": "not_requested",
    })
    write_feature_table(to_layer_first_frame(frame, run_derived=False), source)
    write_feature_table(to_layer_first_frame(periods, run_derived=False), saved)
    return source, saved


def test_saved_periods_feed_statistics_by_id_without_new_search_and_resume(tmp_path, local_features, monkeypatch):
    source, saved = setup_inputs(tmp_path)
    calls = []

    def unexpected(*_a, **_k):
        pytest.fail("Saved periods must bypass selection and bootstrap")

    def statistics(row):
        assert row["periodicity_period"] != 99.0
        assert not any(key.startswith("stats_") for key in row)
        calls.append((row["candidate_id"], row["periodicity_period"]))
        return {"candidate_id": row["candidate_id"], "stats_compute_status": "ok",
                "stats_variability_quasi_periodicity_q": .12,
                "stats_harmonics_a0": row["periodicity_period"], "stats_lafler_kinman_delta": -.5}

    monkeypatch.setattr(local_features, "validate_periodicity", unexpected)
    monkeypatch.setattr(local_features, "_stats_task", statistics)
    options = dict(input_path=source, periods_from=saved, output_dir=tmp_path / "features", workers=1)
    output = local_features.compute(tmp_path, **options)
    result = expand_feature_layers(read_feature_table(output))
    assert calls == [("two", 2.5), ("one", 1.5)]
    assert result.candidate_id.tolist() == ["two", "one"]
    assert result.periodicity_period.tolist() == [2.5, 1.5]
    assert result.stats_harmonics_a0.tolist() == [2.5, 1.5]
    assert result.stats_variability_quasi_periodicity_q.eq(.12).all()
    assert not result.failed_any.any()
    assert result.periodicity_n_bootstrap.eq(0).all()
    local_features.compute(tmp_path, **options)
    assert len(calls) == 2
    summary = json.loads((tmp_path / "features/summary.json").read_text())
    assert not summary["period_search_performed"]

    changed = expand_feature_layers(read_feature_table(saved))
    changed.loc[0, "periodicity_period"] = 3.5
    write_feature_table(to_layer_first_frame(changed, run_derived=False), saved)
    with pytest.raises(ValueError, match="fingerprint"):
        local_features.compute(tmp_path, **options)
    assert len(calls) == 2


@pytest.mark.parametrize("problem", ["missing_id", "duplicate_id", "wrong_lightcurve"])
def test_saved_period_identity_mismatch_stops_before_computing(tmp_path, local_features, problem):
    source, saved = setup_inputs(tmp_path)
    frame = expand_feature_layers(read_feature_table(source))
    periods = expand_feature_layers(read_feature_table(saved))
    if problem == "missing_id":
        periods = periods.iloc[:1]
    elif problem == "duplicate_id":
        periods.loc[1, "candidate_id"] = "one"
    else:
        periods.loc[0, "lc_path"] = "/wrong/3.dat3"
    write_feature_table(to_layer_first_frame(periods, run_derived=False), saved)
    with pytest.raises(ValueError):
        local_features._merge_saved_periods(frame, saved)


def test_unusable_period_retains_candidate_and_does_not_block_others(tmp_path, local_features, monkeypatch):
    source, saved = setup_inputs(tmp_path, missing_period=True)
    calls = []

    def statistics(row):
        calls.append(row["candidate_id"])
        return {"candidate_id": row["candidate_id"], "stats_compute_status": "ok",
                "stats_harmonics_a0": 3.0, "stats_lafler_kinman_delta": -.5}

    monkeypatch.setattr(local_features, "_stats_task", statistics)
    output = local_features.compute(tmp_path, workers=1, input_path=source,
                                    periods_from=saved, output_dir=tmp_path / "features")
    frame = expand_feature_layers(read_feature_table(output))
    assert calls == ["two"]
    assert frame.candidate_id.tolist() == ["two", "one"]
    assert frame.stats_compute_status.tolist() == ["ok", "missing_period"]
    assert pd.isna(frame.stats_harmonics_a0.iloc[1])
    state = json.loads((output.parent / "LOCAL_FEATURES_STAGE.json").read_text())
    assert state["result"]["status"] == "partial"
    assert state["result"]["skipped"] == 1


def test_saved_period_mode_rejects_bootstrap_request(tmp_path, local_features):
    with pytest.raises(ValueError, match="n-bootstrap 0"):
        local_features.compute(tmp_path, periods_from=tmp_path / "selected.parquet", n_bootstrap=1000)
