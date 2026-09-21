"""Stable MALCA adapter for the GitHub BANYAN Sigma implementation."""

from __future__ import annotations

import importlib.metadata
import json
import math
import operator
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from malca.catalogs.gaia_ids import parse_gaia_source_id


BANYAN_ADAPTER_VERSION = "2"
BANYAN_BATCH_SIZE = 256
BANYAN_INPUT_COLUMNS = (
    "ra", "dec", "pmra", "pmdec", "pmra_error", "pmdec_error",
    "parallax", "parallax_error", "radial_velocity", "radial_velocity_error",
)
BANYAN_OUTPUT_COLUMNS = (
    "banyan_field_prob",
    "banyan_ya_prob",
    "banyan_best_assoc",
    "banyan_best_assoc_prob",
    "banyan_probabilities_json",
    "banyan_input_mode",
    "banyan_status",
    "banyan_error",
    "banyan_version",
    "banyan_adapter_version",
    "banyan_updated_at",
)


def _finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _package_version() -> str:
    try:
        return importlib.metadata.version("banyan-sigma")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _column(frame: pd.DataFrame, *tokens: str) -> pd.Series | None:
    wanted = tuple(token.upper() for token in tokens)
    for column in frame.columns:
        if isinstance(column, tuple):
            parts = tuple(str(part).upper() for part in column)
        else:
            parts = (str(column).upper(),)
        if all(token in parts for token in wanted):
            return frame[column]
    return None


def _association_probability_columns(frame: pd.DataFrame) -> dict[str, pd.Series]:
    probabilities: dict[str, pd.Series] = {}
    for column in frame.columns:
        if not isinstance(column, tuple) or len(column) < 2:
            continue
        first, second = str(column[0]).upper(), str(column[1]).upper()
        if first == "ALL" and "FIELD" not in second and second not in {"GLOBAL", "METRICS"}:
            probabilities[second] = pd.to_numeric(frame[column], errors="coerce")
    return probabilities


def _parse_membership_output(result: pd.DataFrame, *, association_threshold: float) -> pd.DataFrame:
    if not isinstance(result, pd.DataFrame) or result.empty:
        raise RuntimeError("BANYAN Sigma returned no rows")

    association_columns = _association_probability_columns(result)
    field_columns = []
    for column in result.columns:
        parts = tuple(str(part).upper() for part in column) if isinstance(column, tuple) else (str(column).upper(),)
        if "ALL" in parts and any("FIELD" in part for part in parts):
            field_columns.append(pd.to_numeric(result[column], errors="coerce"))
    field_prob = (
        pd.concat(field_columns, axis=1).sum(axis=1, min_count=1)
        if field_columns
        else pd.Series(np.nan, index=result.index, dtype=float)
    )

    ya_prob_raw = _column(result, "GLOBAL", "YA_PROB")
    if ya_prob_raw is None:
        ya_prob_raw = _column(result, "YA_PROB")
    ya_prob = (
        pd.to_numeric(ya_prob_raw, errors="coerce")
        if ya_prob_raw is not None
        else 1.0 - field_prob
    )
    best_ya_raw = _column(result, "GLOBAL", "BEST_YA")
    if best_ya_raw is None:
        best_ya_raw = _column(result, "BEST_YA")
    best_ya = (
        best_ya_raw.fillna("").astype(str)
        if best_ya_raw is not None
        else pd.Series("", index=result.index, dtype=object)
    )

    parsed_rows: list[dict[str, object]] = []
    for index in result.index:
        association = str(best_ya.loc[index]).strip().upper()
        assoc_prob = math.nan
        if association and association != "FIELD" and association in association_columns:
            value = association_columns[association].loc[index]
            assoc_prob = float(value) if pd.notna(value) and math.isfinite(float(value)) else math.nan
        named = association if math.isfinite(assoc_prob) and assoc_prob > association_threshold else ""
        probability_map = {
            name: float(values.loc[index])
            for name, values in association_columns.items()
            if pd.notna(values.loc[index]) and math.isfinite(float(values.loc[index]))
        }
        parsed_rows.append(
            {
                "banyan_field_prob": float(field_prob.loc[index]),
                "banyan_ya_prob": float(ya_prob.loc[index]),
                "banyan_best_assoc": named,
                "banyan_best_assoc_prob": assoc_prob,
                "banyan_probabilities_json": json.dumps(
                    probability_map, sort_keys=True, separators=(",", ":")
                ),
            }
        )
    return pd.DataFrame(parsed_rows, index=result.index)


def _membership_callable() -> Callable[..., pd.DataFrame]:
    try:
        import banyan_sigma
    except Exception as exc:  # pragma: no cover - environment-specific import failure
        raise RuntimeError(f"Could not import banyan_sigma: {exc}") from exc
    func = getattr(banyan_sigma, "membership_probability", None)
    if not callable(func):
        raise RuntimeError("banyan_sigma does not expose membership_probability()")
    return func


def load_banyan_results(path: Path) -> pd.DataFrame:
    """Read only the identity, solver inputs, and BANYAN outputs of a product."""
    from malca.io.table_io import read_feature_table
    from malca.products.feature_layers import feature_layer_for_column, parse_layer_value

    columns = ["source_id", "gaia_id", *BANYAN_INPUT_COLUMNS, *BANYAN_OUTPUT_COLUMNS]
    layers = {column: feature_layer_for_column(column) for column in columns}
    saved = read_feature_table(path, columns=list(dict.fromkeys(
        layer or column for column, layer in layers.items()
    )))
    parsed = {
        layer: saved[layer].map(parse_layer_value)
        for layer in set(layers.values()) if layer is not None
    }
    return pd.DataFrame({
        column: parsed[layer].map(lambda row, key=column: row.get(key, pd.NA))
        if layer is not None else saved[column]
        for column, layer in layers.items()
    }, index=saved.index)


def _reuse_banyan_results(
    out: pd.DataFrame,
    previous: pd.DataFrame,
    *,
    association_threshold: float,
) -> pd.Series:
    """Copy successful results only for identical identities, inputs, and versions."""
    reusable = pd.Series(False, index=out.index)
    if previous.empty or not set(BANYAN_OUTPUT_COLUMNS).issubset(previous.columns):
        return reusable

    def identities(frame: pd.DataFrame) -> pd.Series:
        keys = pd.Series(None, index=frame.index, dtype=object)
        for column in ("source_id", "gaia_id"):
            if column in frame.columns:
                keys = keys.fillna(frame[column].map(parse_gaia_source_id))
        return keys

    old_keys = identities(previous)
    unique = old_keys.notna() & ~old_keys.duplicated(keep=False)
    saved = previous.loc[unique].copy()
    saved.index = old_keys.loc[unique]
    keys = identities(out)
    saved = saved.reindex(keys)
    saved.index = out.index
    reusable = (
        keys.notna()
        & saved["banyan_status"].fillna("").eq("ok")
        & out["banyan_status"].eq("pending")
    )
    for column in ("banyan_version", "banyan_adapter_version", "banyan_input_mode"):
        reusable &= saved[column].fillna("").astype(str).eq(out[column].fillna("").astype(str))
    for column in BANYAN_INPUT_COLUMNS:
        current, old = _finite_series(out, column), _finite_series(saved, column)
        reusable &= current.eq(old) | (current.isna() & old.isna())
    for column in ("banyan_field_prob", "banyan_ya_prob"):
        # Summed solver probabilities can round just outside [0, 1].
        reusable &= _finite_series(saved, column).between(-1e-12, 1.0 + 1e-12)

    # The threshold only controls the reported association label. If a lower
    # threshold needs a name that the old output omitted, recompute that row.
    named = _finite_series(saved, "banyan_best_assoc_prob") > association_threshold
    labels = saved["banyan_best_assoc"].fillna("").astype(str).str.strip()
    reusable &= ~named | labels.ne("")
    reusable = reusable.fillna(False)
    saved.loc[~named, "banyan_best_assoc"] = ""
    for column in BANYAN_OUTPUT_COLUMNS:
        values = saved[column]
        if pd.api.types.is_numeric_dtype(out[column]):
            values = pd.to_numeric(values, errors="coerce").astype(float)
        out.loc[reusable, column] = values.loc[reusable]
    return reusable


def compute_banyan_membership(
    candidates: pd.DataFrame,
    *,
    association_threshold: float = 0.1,
    membership_func: Callable[..., pd.DataFrame] | None = None,
    batch_size: int = BANYAN_BATCH_SIZE,
    show_progress: bool = False,
    previous_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return BANYAN results, bounding each library call to *batch_size* stars.

    Summarize and release each batch's full per-association output before
    starting the next call; the library's internal batching retains those
    outputs for the whole input group. Compatible successful rows from
    *previous_results* are reused without replacing current candidate fields.
    """
    batch_size = operator.index(batch_size)
    if batch_size < 1:
        raise ValueError("BANYAN batch_size must be a positive integer")
    out = candidates.copy()
    now = pd.Timestamp.now(tz="UTC").isoformat()
    package_version = _package_version()
    defaults: dict[str, object] = {
        "banyan_field_prob": np.nan,
        "banyan_ya_prob": np.nan,
        "banyan_best_assoc": "",
        "banyan_best_assoc_prob": np.nan,
        "banyan_probabilities_json": "{}",
        "banyan_input_mode": "",
        "banyan_status": "missing_inputs",
        "banyan_error": "",
        "banyan_version": package_version,
        "banyan_adapter_version": BANYAN_ADAPTER_VERSION,
        "banyan_updated_at": now,
    }
    for column, value in defaults.items():
        out[column] = value
    if out.empty:
        return out

    ra = _finite_series(out, "ra")
    dec = _finite_series(out, "dec")
    pmra = _finite_series(out, "pmra")
    pmdec = _finite_series(out, "pmdec")
    epmra = _finite_series(out, "pmra_error")
    epmdec = _finite_series(out, "pmdec_error")
    plx = _finite_series(out, "parallax")
    eplx = _finite_series(out, "parallax_error")
    rv = _finite_series(out, "radial_velocity")
    erv = _finite_series(out, "radial_velocity_error")

    coordinates_ok = ra.notna() & dec.notna() & ra.between(0, 360, inclusive="left") & dec.between(-90, 90)
    motion_ok = pmra.notna() & pmdec.notna()
    motion_error_ok = epmra.notna() & epmdec.notna() & (epmra > 0) & (epmdec > 0)
    eligible = coordinates_ok & motion_ok & motion_error_ok

    out.loc[~coordinates_ok, "banyan_status"] = "missing_coordinates"
    out.loc[coordinates_ok & ~motion_ok, "banyan_status"] = "missing_proper_motion"
    out.loc[coordinates_ok & motion_ok & ~motion_error_ok, "banyan_status"] = "missing_proper_motion_error"

    parallax_ok = plx.notna() & eplx.notna() & (plx > 0) & (eplx > 0)
    rv_ok = rv.notna() & erv.notna() & (erv > 0)
    out.loc[eligible, "banyan_input_mode"] = "pm"
    out.loc[eligible & parallax_ok, "banyan_input_mode"] = "pm+plx"
    out.loc[eligible & rv_ok, "banyan_input_mode"] = "pm+rv"
    out.loc[eligible & parallax_ok & rv_ok, "banyan_input_mode"] = "pm+plx+rv"
    out.loc[eligible, "banyan_status"] = "pending"

    if previous_results is not None:
        reused = _reuse_banyan_results(
            out, previous_results, association_threshold=float(association_threshold),
        )
        if show_progress:
            print(f"BANYAN cache hit: {int(reused.sum())}/{int(eligible.sum())} eligible sources")
        eligible &= ~reused

    if not eligible.any():
        return out
    try:
        func = membership_func or _membership_callable()
    except Exception as exc:
        out.loc[eligible, "banyan_status"] = "package_api_mismatch"
        out.loc[eligible, "banyan_error"] = str(exc)
        return out

    with tqdm(
        total=int(eligible.sum()), desc="BANYAN membership", unit="source",
        disable=not show_progress,
    ) as progress:
        for mode in ("pm", "pm+plx", "pm+rv", "pm+plx+rv"):
            mode_indices = out.index[eligible & out["banyan_input_mode"].eq(mode)]
            for start in range(0, len(mode_indices), batch_size):
                indices = mode_indices[start:start + batch_size]
                kwargs: dict[str, object] = {
                    "ra": ra.loc[indices].to_numpy(dtype=float),
                    "dec": dec.loc[indices].to_numpy(dtype=float),
                    "pmra": pmra.loc[indices].to_numpy(dtype=float),
                    "pmdec": pmdec.loc[indices].to_numpy(dtype=float),
                    "epmra": epmra.loc[indices].to_numpy(dtype=float),
                    "epmdec": epmdec.loc[indices].to_numpy(dtype=float),
                }
                if "plx" in mode:
                    kwargs.update(
                        plx=plx.loc[indices].to_numpy(dtype=float),
                        eplx=eplx.loc[indices].to_numpy(dtype=float),
                        use_plx=True,
                    )
                if "rv" in mode:
                    kwargs.update(
                        rv=rv.loc[indices].to_numpy(dtype=float),
                        erv=erv.loc[indices].to_numpy(dtype=float),
                        use_rv=True,
                    )
                try:
                    result = func(**kwargs)
                    if len(result) != len(indices):
                        raise RuntimeError("BANYAN Sigma returned an unexpected number of rows")
                    parsed = _parse_membership_output(
                        result, association_threshold=float(association_threshold)
                    )
                    parsed.index = indices
                    for column in parsed.columns:
                        out.loc[indices, column] = parsed[column]
                    finite = pd.to_numeric(
                        out.loc[indices, "banyan_field_prob"], errors="coerce"
                    ).notna()
                    good_indices = finite.index[finite]
                    bad_indices = finite.index[~finite]
                    out.loc[good_indices, "banyan_status"] = "ok"
                    out.loc[bad_indices, "banyan_status"] = "calculation_error"
                    out.loc[bad_indices, "banyan_error"] = "BANYAN returned no finite field probability"
                except Exception as exc:
                    out.loc[indices, "banyan_status"] = "calculation_error"
                    out.loc[indices, "banyan_error"] = str(exc)
                finally:
                    # Do not retain the previous batch while the next call allocates.
                    result = parsed = None
                    progress.update(len(indices))

    return out
