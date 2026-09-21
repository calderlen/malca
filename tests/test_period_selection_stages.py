from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from malca.stv import filter as period_filter


def selection_record(args):
    original, resolved, _, level, exclude, method, *_ = args
    stat = Path(resolved).stat()
    return {
        "lc_path": original, "resolved_path": resolved, "error": None,
        "periodicity_checkpoint_version": period_filter.PERIODICITY_CHECKPOINT_VERSION,
        "periodicity_selection_version": period_filter.PERIODICITY_SELECTION_VERSION,
        "periodicity_significance_status": "not_requested",
        "periodicity_n_bootstrap": 0, "periodicity_significance_level": level,
        "periodicity_exclude_aliases": exclude,
        "periodicity_input_size": stat.st_size, "periodicity_input_mtime_ns": stat.st_mtime_ns,
        "periodicity_period": 3.25, "period_consensus_days": 3.25,
        "period_for_fold_days": 3.25, "period_native_days": 6.5,
        "period_corrected_days": 3.25, "period_method": "pdm", "period_confidence": "none",
        "periodicity_method": "pdm", "periodicity_base_period": 6.5,
        "periodicity_harmonic_factor": 0.5, "periodicity_is_rejected": False,
        "periodicity_bootstrap_sig": np.nan, "periodicity_is_significant": False,
        "pdm_method": method, "pdm_period": 6.5, "pdm_corrected_period": 3.25,
        "pdm_harmonic_factor": 0.5, "pdm_min_theta": 2.0, "pdm_snr": 8.0,
        "ce_period": 3.25, "ce_corrected_period": 3.25, "ce_harmonic_factor": 1.0,
        "ce_min_entropy": 0.4, "ce_snr": 8.0, "lsp_period": 3.25,
        "lsp_power": 0.8, "lsp_bootstrap_sig": np.nan,
        "lsp_is_alias": False, "lsp_is_significant": False,
    }


def test_selection_uses_no_nulls_and_can_choose_long_period_without_fap(tmp_path, monkeypatch):
    source = tmp_path / "1.dat3"
    source.write_text("loader fixture")
    time = np.linspace(0, 200, 100)
    frame = pd.DataFrame({
        "JD": time, "mag": 13 + 0.2 * np.sin(2 * np.pi * time / 30),
        "error": 0.02, "v_g_band": 0,
    })
    monkeypatch.setattr(period_filter, "load_lightcurve_df", lambda *_a, **_k: frame.copy())
    calls = []

    def pdm(*_args, **kwargs):
        calls.append(("pdm", kwargs["n_bootstrap"]))
        return {"pdm_period": 2.5, "pdm_min_theta": 3.0, "pdm_snr": 8.0}

    def ce(*_args, **kwargs):
        calls.append(("ce", kwargs["n_bootstrap"]))
        return {"ce_period": 3.5, "ce_min_entropy": 0.4, "ce_snr": 8.0}

    def ls(*_args, **kwargs):
        calls.append(("ls", kwargs["n_bootstrap"]))
        return {"ls_period_days": 4.5, "ls_power": 0.5}

    def long_ls(*_args, **kwargs):
        calls.append(("long_ls", kwargs["long_ls_kwargs"]["n_bootstrap"]))
        return {"period_consensus_days": 3.5, "period_method": "ce",
                "long_ls_period_days": 30.0, "long_ls_top_periods_days": [30.0, 40.0],
                "long_ls_max_period_days": 120.0, "long_ls_fap_bootstrap": np.nan}

    def correction(period, *_args, **_kwargs):
        objective = abs(float(period) - 30) / 30
        return {"raw_period": period, "corrected_period": period, "harmonic_factor": 1.0,
                "objective": objective, "selection_objective": objective,
                "scatter_ratio": objective, "alias_flag": False}

    monkeypatch.setattr(period_filter, "compute_pdm_stats", pdm)
    monkeypatch.setattr(period_filter, "compute_ce_stats", ce)
    monkeypatch.setattr(period_filter, "bootstrap_lomb_scargle", ls)
    monkeypatch.setattr(period_filter, "compute_period_consensus_for_lc", long_ls)
    monkeypatch.setattr(period_filter, "_correct_native_period", correction)
    monkeypatch.setattr(period_filter, "LONG_PERIOD_ENABLED", True)
    result = period_filter._period_selection_worker((str(source), str(source), 1000, .01, True, "plavchan", None))
    assert result["error"] is None
    assert calls == [("pdm", 0), ("ce", 0), ("ls", 0), ("long_ls", 0)]
    assert result["periodicity_period"] == result["period_for_fold_days"] == 30.0
    assert result["periodicity_method"] == "long_ls"
    assert result["periodicity_significance_status"] == "not_requested"
    assert not result["periodicity_is_significant"]
    assert not result["periodicity_is_rejected"]
    assert np.isnan(result["periodicity_bootstrap_sig"])


def test_optional_significance_never_replaces_selected_periods(tmp_path, monkeypatch):
    source = tmp_path / "1.dat3"
    source.write_text("fixture")
    args = (str(source), str(source), 1000, .01, True, "plavchan", None)
    selected = selection_record(args)
    measured = {**selected, "periodicity_period": 99.0, "pdm_period": 99.0,
                "period_confidence": "high", "pdm_bootstrap_sig": .001,
                "ce_bootstrap_sig": .003, "lsp_bootstrap_sig": .2,
                "long_ls_fap_bootstrap": .4, "long_ls_period_days": 30.0}
    monkeypatch.setattr(period_filter, "_lsp_worker", lambda _args: measured)
    monkeypatch.setattr(period_filter, "LONG_PERIOD_ENABLED", True)
    result = period_filter._period_significance_worker((args, selected))
    for key in ("periodicity_period", "period_consensus_days", "period_for_fold_days",
                "period_native_days", "period_corrected_days", "pdm_period", "period_confidence"):
        assert result[key] == selected[key]
    assert result["periodicity_bootstrap_sig"] == pytest.approx(.004)
    assert result["periodicity_significance_status"] == "complete"
    assert result["periodicity_is_significant"]
    assert result["periodicity_significance_scope"] == "source_searches_bonferroni"


def test_saved_selection_can_gain_significance_without_new_searches(tmp_path, monkeypatch):
    sources = [tmp_path / f"{i}.dat3" for i in range(2)]
    for path in sources:
        path.write_text("fixture")
    frame = pd.DataFrame({"lc_path": list(map(str, sources))})
    checkpoint = tmp_path / "checkpoint/period_selection_checkpoint.parquet"
    search_calls = []
    significance_calls = []

    def select(args):
        search_calls.append(args[0])
        return selection_record(args)

    def significance(task):
        args, selected = task
        saved = pd.read_parquet(checkpoint)
        assert set(saved.lc_path) == set(frame.lc_path)
        assert saved.periodicity_period.eq(3.25).all()
        assert saved.period_for_fold_days.eq(3.25).all()
        significance_calls.append(args[0])
        return {**selected, "periodicity_n_bootstrap": args[2],
                "periodicity_significance_status": "complete", "periodicity_bootstrap_sig": .004}

    monkeypatch.setattr(period_filter, "_period_selection_worker", select)
    monkeypatch.setattr(period_filter, "_period_significance_worker", significance)
    options = dict(checkpoint_dir=checkpoint.parent, skip_if_consensus=False)
    initial = period_filter.validate_periodicity(frame, **options)
    assert len(search_calls) == 2 and not significance_calls
    assert initial.periodicity_significance_status.eq("not_requested").all()
    assert not initial.periodic_flag.any()
    tested = period_filter.validate_periodicity(frame, n_bootstrap=1000, **options)
    assert len(search_calls) == 2 and len(significance_calls) == 2
    assert tested.periodicity_period.tolist() == initial.periodicity_period.tolist()
    assert tested.period_consensus_days.tolist() == initial.period_consensus_days.tolist()
    again = period_filter.validate_periodicity(frame, n_bootstrap=1000, **options)
    assert len(search_calls) == len(significance_calls) == 2
    assert again.periodicity_significance_status.eq("complete").all()


@pytest.mark.parametrize("failure", ["test_error", "changed_input"])
def test_failed_new_significance_clears_old_detection_but_keeps_period(tmp_path, monkeypatch, failure):
    source = tmp_path / "1.dat3"
    source.write_text("fixture")
    args = (str(source), str(source), 2000, .01, True, "plavchan", None)
    selected = {**selection_record(args), "periodicity_n_bootstrap": 1000,
                "periodicity_bootstrap_sig": .001, "periodicity_is_significant": True,
                "periodicity_is_rejected": True, "pdm_bootstrap_sig": .001,
                "pdm_is_significant": True, "periodicity_significance_status": "complete"}
    measured = ({"error": "Null test failed"} if failure == "test_error" else
                {**selected, "periodicity_input_size": selected["periodicity_input_size"] + 1})
    monkeypatch.setattr(period_filter, "_lsp_worker", lambda _args: measured)
    result = period_filter._period_significance_worker((args, selected))
    assert result["periodicity_significance_status"] == "error"
    assert result["periodicity_period"] == result["period_for_fold_days"] == 3.25
    assert result["periodicity_n_bootstrap"] == 2000
    assert np.isnan(result["periodicity_bootstrap_sig"])
    assert np.isnan(result["pdm_bootstrap_sig"])
    assert not result["periodicity_is_significant"]
    assert not result["periodicity_is_rejected"]
    assert not result["pdm_is_significant"]


def test_interrupted_significance_leaves_every_selected_period_checkpointed(tmp_path, monkeypatch):
    source = tmp_path / "1.dat3"
    source.write_text("fixture")
    monkeypatch.setattr(period_filter, "_period_selection_worker", selection_record)

    def interrupt(_task):
        raise KeyboardInterrupt

    monkeypatch.setattr(period_filter, "_period_significance_worker", interrupt)
    with pytest.raises(KeyboardInterrupt):
        period_filter.validate_periodicity(
            pd.DataFrame({"lc_path": [str(source)]}), n_bootstrap=1000,
            checkpoint_dir=tmp_path, skip_if_consensus=False,
        )
    saved = pd.read_parquet(tmp_path / "period_selection_checkpoint.parquet")
    assert saved.periodicity_period.tolist() == [3.25]
    assert saved.periodicity_significance_status.tolist() == ["not_requested"]


def test_interruption_preserves_completed_significance_between_checkpoint_batches(tmp_path, monkeypatch):
    sources = [tmp_path / f"{i}.dat3" for i in range(2)]
    for path in sources:
        path.write_text("fixture")
    frame = pd.DataFrame({"lc_path": list(map(str, sources))})
    monkeypatch.setattr(period_filter, "_period_selection_worker", selection_record)
    calls = []

    def significance(task):
        args, selected = task
        calls.append(args[0])
        if len(calls) == 2:
            raise KeyboardInterrupt
        return {**selected, "periodicity_n_bootstrap": args[2],
                "periodicity_significance_status": "complete"}

    monkeypatch.setattr(period_filter, "_period_significance_worker", significance)
    options = dict(n_bootstrap=1000, checkpoint_dir=tmp_path, skip_if_consensus=False)
    with pytest.raises(KeyboardInterrupt):
        period_filter.validate_periodicity(frame, **options)
    saved = pd.read_parquet(tmp_path / "period_selection_checkpoint.parquet").set_index("lc_path")
    assert saved.loc[str(sources[0]), "periodicity_significance_status"] == "complete"
    assert saved.loc[str(sources[1]), "periodicity_significance_status"] == "not_requested"

    def unexpected_selection(_args):
        raise AssertionError("Resume must not rerun selected periods")

    monkeypatch.setattr(period_filter, "_period_selection_worker", unexpected_selection)
    resumed = period_filter.validate_periodicity(frame, **options)
    assert calls == [str(sources[0]), str(sources[1]), str(sources[1])]
    assert resumed.periodicity_significance_status.eq("complete").all()
