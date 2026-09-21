from __future__ import annotations

import sqlite3
import weakref
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from malca.catalogs import gaia_fetch
from malca.catalogs.gaia_banyan_backfill import load_review_cohort
from malca.enrichment.banyan import BANYAN_BATCH_SIZE, compute_banyan_membership
from malca.enrichment.characterize import (
    _module_completed,
    gaia_enrichment_needed_mask,
    merge_gaia_catalog_rows,
)
from malca.review.store import init_db


def _complete_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "gaia_id": "1",
        "source_id": "1",
        "ra": 10.0,
        "dec": 20.0,
        "phot_g_mean_mag": 12.0,
        "phot_bp_mean_mag": 12.5,
        "phot_rp_mean_mag": 11.5,
        "parallax": 10.0,
        "parallax_error": 0.2,
        "pmra": 5.0,
        "pmra_error": 0.1,
        "pmdec": -3.0,
        "pmdec_error": 0.1,
        "radial_velocity": 15.0,
        "radial_velocity_error": 1.0,
    }
    row.update(overrides)
    return row


def test_banyan_adapter_uses_real_api_shape_and_records_provenance() -> None:
    calls: list[dict[str, object]] = []

    def fake_membership(**kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        return pd.DataFrame(
            {
                ("ALL", "FIELD"): [0.8],
                ("ALL", "FIELD_MS"): [0.1],
                ("ALL", "BETA_PIC"): [0.1],
                ("YA_PROB", "Global"): [0.1],
                ("BEST_YA", "Global"): ["BETA_PIC"],
            }
        )

    out = compute_banyan_membership(
        pd.DataFrame([_complete_row()]),
        association_threshold=0.05,
        membership_func=fake_membership,
    )

    assert len(calls) == 1
    assert calls[0]["use_plx"] is True
    assert calls[0]["use_rv"] is True
    assert out.loc[0, "banyan_status"] == "ok"
    assert out.loc[0, "banyan_input_mode"] == "pm+plx+rv"
    assert np.isclose(out.loc[0, "banyan_field_prob"], 0.9)
    assert out.loc[0, "banyan_best_assoc"] == "BETA_PIC"
    assert np.isclose(out.loc[0, "banyan_best_assoc_prob"], 0.1)
    assert out.loc[0, "banyan_adapter_version"]
    assert out.loc[0, "banyan_version"]


def test_banyan_adapter_explains_missing_pm_errors_without_calling_package() -> None:
    called = False

    def fail_if_called(**_kwargs: object) -> pd.DataFrame:
        nonlocal called
        called = True
        raise AssertionError("ineligible row must not call BANYAN")

    row = _complete_row(pmra_error=np.nan)
    out = compute_banyan_membership(
        pd.DataFrame([row]),
        membership_func=fail_if_called,
    )

    assert not called
    assert out.loc[0, "banyan_status"] == "missing_proper_motion_error"
    assert pd.isna(out.loc[0, "banyan_field_prob"])


def test_banyan_adapter_falls_back_to_pm_only_for_nonphysical_parallax() -> None:
    calls: list[dict[str, object]] = []

    def fake_membership(**kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        return pd.DataFrame(
            {
                ("ALL", "FIELD"): [1.0],
                ("YA_PROB", "Global"): [0.0],
                ("BEST_YA", "Global"): ["FIELD"],
            }
        )

    row = _complete_row(parallax=-1.0, radial_velocity=np.nan, radial_velocity_error=np.nan)
    out = compute_banyan_membership(pd.DataFrame([row]), membership_func=fake_membership)

    assert len(calls) == 1
    assert "plx" not in calls[0]
    assert "rv" not in calls[0]
    assert out.loc[0, "banyan_input_mode"] == "pm"
    assert out.loc[0, "banyan_status"] == "ok"


def _fake_probabilities(ra: np.ndarray) -> pd.DataFrame:
    probability = ra / 360.0
    return pd.DataFrame({
        ("ALL", "FIELD"): 1.0 - probability,
        ("ALL", "BETA_PIC"): probability,
        ("YA_PROB", "Global"): probability,
        ("BEST_YA", "Global"): "BETA_PIC",
    })


def test_banyan_batches_preserve_modes_indices_and_results() -> None:
    rows = [
        _complete_row(ra=10.0 + i, parallax=plx, radial_velocity=rv)
        for i in range(3)
        for plx, rv in [(np.nan, np.nan), (10.0, np.nan), (np.nan, 15.0), (10.0, 15.0)]
    ]
    rows.append(_complete_row(pmra_error=np.nan))
    frame = pd.DataFrame(rows, index=[90, 20, 70, 40, 10, 60, 30, 80, 50, 100, 120, 110, 0])
    original = frame.copy(deep=True)
    calls = []
    previous = None

    def membership(**kwargs):
        nonlocal previous
        # Full outputs must be released before the next allocation.
        assert previous is None or previous() is None
        calls.append((len(kwargs["ra"]), "plx" in kwargs, "rv" in kwargs))
        result = _fake_probabilities(kwargs["ra"])
        previous = weakref.ref(result)
        return result

    batched = compute_banyan_membership(frame, membership_func=membership, batch_size=2)
    assert calls == [(n, plx, rv) for plx, rv in [(False, False), (True, False), (False, True), (True, True)] for n in [2, 1]]
    assert previous() is None
    whole = compute_banyan_membership(frame, membership_func=membership, batch_size=100)
    pd.testing.assert_frame_equal(
        batched.drop(columns="banyan_updated_at"), whole.drop(columns="banyan_updated_at"),
    )
    pd.testing.assert_frame_equal(frame, original)
    assert batched["banyan_status"].tolist() == ["ok"] * 12 + ["missing_proper_motion_error"]


def test_banyan_default_caps_calls_and_continues_after_failed_batch() -> None:
    frame = pd.DataFrame([_complete_row(ra=float(i % 360)) for i in range(2 * BANYAN_BATCH_SIZE + 1)])
    sizes = []

    def membership(**kwargs):
        sizes.append(len(kwargs["ra"]))
        if len(sizes) == 2:
            raise RuntimeError("batch failed")
        return _fake_probabilities(kwargs["ra"])

    out = compute_banyan_membership(frame, membership_func=membership)
    assert sizes == [BANYAN_BATCH_SIZE, BANYAN_BATCH_SIZE, 1]
    assert out["banyan_status"].tolist() == ["ok"] * BANYAN_BATCH_SIZE + ["calculation_error"] * BANYAN_BATCH_SIZE + ["ok"]
    assert out.iloc[BANYAN_BATCH_SIZE:2 * BANYAN_BATCH_SIZE]["banyan_error"].eq("batch failed").all()
    assert out.iloc[BANYAN_BATCH_SIZE:2 * BANYAN_BATCH_SIZE]["banyan_field_prob"].isna().all()


@pytest.mark.parametrize("size", [0, -1])
def test_banyan_rejects_invalid_batch_size(size: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        compute_banyan_membership(pd.DataFrame([_complete_row()]), batch_size=size)


@pytest.mark.parametrize("plx,rv", [(np.nan, np.nan), (10.0, np.nan), (np.nan, 15.0), (10.0, 15.0)])
def test_banyan_batches_match_installed_solver(plx: float, rv: float) -> None:
    banyan = pytest.importorskip("banyan_sigma")
    # A small hypothesis set exercises the real solver in all four input modes.
    membership = partial(banyan.membership_probability, hypotheses=["ABDMG", "FIELD"])
    frame = pd.DataFrame([
        _complete_row(ra=10.0 + i, pmra=5.0 + i, parallax=plx, radial_velocity=rv)
        for i in range(3)
    ], index=[103, 22, 11])
    batched = compute_banyan_membership(frame, membership_func=membership, batch_size=2)
    whole = compute_banyan_membership(frame, membership_func=membership, batch_size=3)
    assert batched["banyan_status"].eq("ok").all(), batched["banyan_error"].tolist()
    pd.testing.assert_frame_equal(
        batched.drop(columns="banyan_updated_at"), whole.drop(columns="banyan_updated_at"),
        rtol=1e-12, atol=1e-14,
    )


def test_gaia_completeness_is_row_specific_and_merge_prefers_gaia_id() -> None:
    frame = pd.DataFrame(
        [
            _complete_row(source_id="999", pmra_error=np.nan, pmdec_error=np.nan),
            _complete_row(gaia_id="2", source_id="2"),
        ]
    )
    needed = gaia_enrichment_needed_mask(frame)
    assert needed.tolist() == [True, False]

    gaia_row = pd.DataFrame(
        [
            {
                "source_id": "1",
                "pmra_error": 0.2,
                "pmdec_error": 0.3,
                "gaia_fetch_schema_version": "2",
            }
        ]
    )
    out = merge_gaia_catalog_rows(frame, gaia_row)

    assert np.isclose(out.loc[0, "pmra_error"], 0.2)
    assert np.isclose(out.loc[0, "pmdec_error"], 0.3)
    assert bool(out.loc[0, "gaia_banyan_input_complete"])
    assert out.loc[0, "gaia_enrichment_status"] == "complete"
    assert out.loc[1, "gaia_enrichment_status"] == "existing_complete"


def test_banyan_reuse_matches_identity_and_preserves_current_fields() -> None:
    from malca.enrichment.banyan import BANYAN_OUTPUT_COLUMNS

    original = pd.DataFrame([
        _complete_row(source_id=str(i), gaia_id=str(i), ra=40.0 + i)
        for i in range(1, 4)
    ])
    saved = compute_banyan_membership(
        original, membership_func=lambda **kw: _fake_probabilities(kw["ra"]),
    )
    current = original.iloc[[2, 0]].copy()
    current.index = [103, 22]
    current["new_measurement"] = [7.0, 8.0]
    calls = []

    def membership(**kwargs):
        calls.append(kwargs)
        return _fake_probabilities(kwargs["ra"])

    result = compute_banyan_membership(current, previous_results=saved, membership_func=membership)
    assert not calls
    pd.testing.assert_frame_equal(result[current.columns], current)
    expected = saved.iloc[[2, 0]][list(BANYAN_OUTPUT_COLUMNS)].set_axis(current.index)
    pd.testing.assert_frame_equal(result[list(BANYAN_OUTPUT_COLUMNS)], expected)


@pytest.mark.parametrize("changed", [
    "ra", "dec", "pmra", "pmdec", "pmra_error", "pmdec_error",
    "parallax", "parallax_error", "radial_velocity", "radial_velocity_error",
    "banyan_version", "banyan_adapter_version", "banyan_status", "banyan_field_prob",
])
def test_banyan_reuse_recalculates_only_changed_or_failed_rows(changed: str) -> None:
    current = pd.DataFrame([
        _complete_row(source_id="1", gaia_id="1", ra=40.0),
        _complete_row(source_id="2", gaia_id="2", ra=50.0),
    ])
    saved = compute_banyan_membership(
        current, membership_func=lambda **kw: _fake_probabilities(kw["ra"]),
    )
    if changed in {"banyan_version", "banyan_adapter_version", "banyan_status"}:
        saved.loc[1, changed] = "stale_or_failed"
    elif changed == "banyan_field_prob":
        saved.loc[1, changed] = np.nan
    else:
        current.loc[1, changed] += 0.01
    calls = []

    def membership(**kwargs):
        calls.extend(kwargs["ra"])
        return _fake_probabilities(kwargs["ra"])

    result = compute_banyan_membership(current, previous_results=saved, membership_func=membership)
    assert calls == [current.loc[1, "ra"]]
    assert result["banyan_status"].eq("ok").all()
    assert result.loc[0, "banyan_updated_at"] == saved.loc[0, "banyan_updated_at"]


def test_banyan_reuse_handles_new_ambiguous_and_newly_eligible_sources() -> None:
    current = pd.DataFrame([
        _complete_row(source_id=str(i), gaia_id=str(i), ra=40.0 + i)
        for i in range(1, 5)
    ])
    old = current.iloc[:3].copy()
    old.loc[2, "pmra_error"] = np.nan
    saved = compute_banyan_membership(
        old, membership_func=lambda **kw: _fake_probabilities(kw["ra"]),
    )
    saved = pd.concat([saved, saved.iloc[[1]]], ignore_index=True)
    calls = []

    def membership(**kwargs):
        calls.extend(kwargs["ra"])
        return _fake_probabilities(kwargs["ra"])

    result = compute_banyan_membership(current, previous_results=saved, membership_func=membership)
    assert calls == [42.0, 43.0, 44.0]
    assert result["banyan_status"].eq("ok").all()


def test_banyan_reuse_applies_current_association_threshold() -> None:
    frame = pd.DataFrame([_complete_row(ra=40.0)])
    calls = []

    def membership(**kwargs):
        calls.append(1)
        return _fake_probabilities(kwargs["ra"])

    saved = compute_banyan_membership(frame, membership_func=membership, association_threshold=0.05)
    higher = compute_banyan_membership(
        frame, previous_results=saved, membership_func=membership, association_threshold=0.2,
    )
    assert len(calls) == 1
    assert higher.loc[0, "banyan_best_assoc"] == ""
    lower = compute_banyan_membership(
        frame, previous_results=higher, membership_func=membership, association_threshold=0.05,
    )
    assert len(calls) == 2
    assert lower.loc[0, "banyan_best_assoc"] == "BETA_PIC"


def test_banyan_reuse_preserves_probability_roundoff() -> None:
    frame = pd.DataFrame([_complete_row(ra=40.0)])
    saved = compute_banyan_membership(
        frame, membership_func=lambda **kw: _fake_probabilities(kw["ra"]),
    )
    saved.loc[0, "banyan_ya_prob"] = np.nextafter(1.0, 2.0)

    def unexpected_solver(**kwargs):
        pytest.fail("floating-point roundoff must not invalidate a saved result")

    result = compute_banyan_membership(frame, previous_results=saved, membership_func=unexpected_solver)
    assert result.loc[0, "banyan_ya_prob"] == saved.loc[0, "banyan_ya_prob"]
    assert result.loc[0, "banyan_updated_at"] == saved.loc[0, "banyan_updated_at"]


@pytest.mark.filterwarnings("error::FutureWarning")
def test_characterize_banyan_reuses_completed_layer_first_product(tmp_path: Path, monkeypatch) -> None:
    import malca.enrichment.banyan as banyan
    from malca.enrichment.characterize import query_banyan_sigma
    from malca.io.table_io import write_feature_table

    frame = pd.DataFrame([
        _complete_row(ra=40.0),
        _complete_row(source_id="2", gaia_id="2", pmra=np.nan),
    ])
    saved = compute_banyan_membership(
        frame, membership_func=lambda **kw: _fake_probabilities(kw["ra"]),
    )
    product = tmp_path / "lc_events_characterized.parquet"
    write_feature_table(saved, product)

    def unexpected_solver():
        pytest.fail("matching saved results must not invoke the solver")

    monkeypatch.setattr(banyan, "_membership_callable", unexpected_solver)
    result = query_banyan_sigma(frame, reuse_from=product)
    assert result.loc[0, "banyan_status"] == "ok"
    assert result.loc[1, "banyan_status"] == "missing_proper_motion"
    assert result.loc[0, "banyan_updated_at"] == saved.loc[0, "banyan_updated_at"]


def test_banyan_checkpoint_reruns_when_new_gaia_inputs_become_eligible() -> None:
    old = compute_banyan_membership(
        pd.DataFrame([_complete_row(pmra_error=np.nan, pmdec_error=np.nan)]),
        membership_func=lambda **_kwargs: pd.DataFrame(),
    )
    old["char_status_banyan"] = "ok"
    assert _module_completed(old, "banyan")

    old["pmra_error"] = 0.1
    old["pmdec_error"] = 0.1
    assert not _module_completed(old, "banyan")


def test_gaia_fetch_refreshes_requested_legacy_rows_and_preserves_other_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache = tmp_path / "gaia.parquet"
    pd.DataFrame(
        [
            {"source_id": "1", "pmra": 1.0},
            {"source_id": "2", "pmra": 2.0},
        ]
    ).to_parquet(cache, index=False)
    monkeypatch.setattr(
        gaia_fetch,
        "canonicalize_gaia_ids",
        lambda values, **_kwargs: pd.DataFrame(
            {"source_id": list(values), "gaia_id_mapping_status": ["dr3"] * len(values)}
        ),
    )
    monkeypatch.setattr(gaia_fetch.pyvo.dal, "TAPService", lambda _url: object())

    def fake_fetch(_service: object, chunk_ids: list[str]) -> pd.DataFrame:
        return gaia_fetch._mark_current_fetch_rows(
            pd.DataFrame(
                [
                    {
                        "source_id": source_id,
                        "ra": 10.0,
                        "dec": 20.0,
                        "pmra": 1.0,
                        "pmra_error": 0.1,
                        "pmdec": 2.0,
                        "pmdec_error": 0.1,
                    }
                    for source_id in chunk_ids
                ]
            )
        )

    monkeypatch.setattr(gaia_fetch, "_fetch_chunk", fake_fetch)
    out = gaia_fetch.fetch_gaia_catalog(["1"], output_path=cache)

    assert set(out["source_id"].dropna().astype(str)) == {"1", "2"}
    refreshed = out[out["source_id"].astype(str).eq("1")].iloc[0]
    preserved = out[out["source_id"].astype(str).eq("2")].iloc[0]
    assert refreshed["gaia_fetch_schema_version"] == gaia_fetch.GAIA_FETCH_SCHEMA_VERSION
    assert np.isclose(refreshed["pmra_error"], 0.1)
    assert pd.isna(preserved["gaia_fetch_schema_version"])


def test_reviewed_dipper_cohort_selection(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        init_db(conn)
        conn.execute(
            """
            INSERT INTO candidates(candidate_id, payload_json, imported_at)
            VALUES (?, ?, ?), (?, ?, ?)
            """,
            (
                "dip", '{"gaia_id":"1"}', "2026-01-01T00:00:00Z",
                "other", '{"gaia_id":"2"}', "2026-01-01T00:00:00Z",
            ),
        )
        conn.execute(
            """
            INSERT INTO reviews(
                candidate_id, status, workflow_status, morphology_primary, updated_at
            )
            VALUES ('dip', 'reviewed', 'reviewed', 'dimming_event', '2026-01-01T00:00:00Z'),
                   ('other', 'reviewed', 'reviewed', 'periodic', '2026-01-01T00:00:00Z')
            """
        )

    cohort = load_review_cohort(db_path)
    assert cohort["candidate_id"].tolist() == ["dip"]
    assert str(cohort.loc[0, "gaia_id"]) == "1"
