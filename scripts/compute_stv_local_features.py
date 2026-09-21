"""Compute local statistics using saved periods or a new period search."""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from malca.config import POST_FILTER_PDM_METHOD
from malca.core.stats import _enrich_row_worker
from malca.io.table_io import read_feature_table, write_feature_table, write_parquet_table
from malca.products.feature_layers import expand_feature_layers, to_layer_first_frame
from malca.products.stage_state import (
    StageResult, assert_reusable_stage_state, build_stage_fingerprint,
    read_stage_state, write_stage_state,
)
from malca.stv.filter import PERIODICITY_MERGE_COLS, validate_periodicity


REPO = Path(__file__).resolve().parents[1]
PERIOD_COLUMNS = (
    "periodicity_period", "pdm_corrected_period", "ce_corrected_period",
    "pdm_period", "ce_period", "period_consensus_days",
    "pre_periodicity_selected_period", "phase_period_days",
)


def _stats_task(row: dict) -> dict:
    # Supply only identity and periods: _enrich_row_worker preserves existing
    # stats keys, including NaNs, so passing the old stats would prevent filling them.
    path = Path(row["lc_path"])
    result = _enrich_row_worker((row, path.stem, str(path), True, path.suffix.lstrip(".")))
    return {"candidate_id": row["candidate_id"],
            **{key: value for key, value in result.items() if key.startswith("stats_")}}


def _merge_saved_periods(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Attach measurements by candidate ID without changing the scoring cohort."""
    selected = expand_feature_layers(read_feature_table(path))
    if "candidate_id" not in selected or selected["candidate_id"].isna().any():
        raise ValueError("Saved periods require nonmissing candidate IDs")
    selected["candidate_id"] = selected["candidate_id"].astype(str)
    if selected["candidate_id"].duplicated().any():
        raise ValueError("Duplicate candidate IDs in saved periods")
    requested = set(frame["candidate_id"])
    provided = set(selected["candidate_id"])
    if requested != provided:
        raise ValueError(f"Saved-period cohort differs: {len(requested - provided)} missing, "
                         f"{len(provided - requested)} extra candidates")
    selected = selected.set_index("candidate_id").loc[frame["candidate_id"]]
    if "lc_path" not in selected or selected["lc_path"].isna().any():
        raise ValueError("Saved periods require light-curve paths")
    selected_names = selected["lc_path"].map(lambda value: Path(str(value)).name).tolist()
    current_names = frame["lc_path"].map(lambda value: Path(str(value)).name).tolist()
    if selected_names != current_names:
        raise ValueError("Saved period IDs refer to different light curves")
    if "periodicity_period" not in selected or "periodicity_status" not in selected:
        raise ValueError("Saved periods lack adopted periods or calculation status")
    values = pd.to_numeric(selected["periodicity_period"], errors="coerce")
    invalid = values.isna() | ~np.isfinite(values) | values.le(0) | selected["periodicity_status"].eq("error").fillna(False)
    if invalid.any():
        print(f"{int(invalid.sum()):,} sources have no usable selected period; retaining their rows with missing-period flags", flush=True)
    merged = frame.copy()
    for column in (*PERIODICITY_MERGE_COLS, "phase_period_days", "phase_source"):
        if column in selected:
            merged[column] = selected[column].to_numpy()
    return merged


def compute(run_dir: Path, *, workers: int = 4, n_bootstrap: int = 0,
            input_path: Path | None = None, output_dir: Path | None = None,
            periods_from: Path | None = None) -> Path:
    run_dir = run_dir.expanduser().resolve()
    source = (input_path or run_dir / "results/lc_events_enriched_all.parquet").expanduser().resolve()
    out = (output_dir or run_dir / "results/local_ml_features").expanduser().resolve()
    if workers < 1 or n_bootstrap < 0:
        raise ValueError("workers must be positive and n_bootstrap nonnegative")
    if periods_from is not None:
        periods_from = periods_from.expanduser().resolve()
        if n_bootstrap != 0:
            raise ValueError("--periods-from reuses measurements; use --n-bootstrap 0")
    print(f"Reading saved cluster features: {source}", flush=True)
    frame = expand_feature_layers(read_feature_table(source))
    decisions = frame["failed_any"].astype("boolean")
    if decisions.isna().any():
        raise ValueError("Missing saved acceptance decisions")
    frame = frame.loc[~decisions].copy().reset_index(drop=True)
    if frame.empty or frame["candidate_id"].isna().any() or frame["candidate_id"].duplicated().any():
        raise ValueError("Expected a nonempty passing cohort with unique candidate IDs")
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["lc_path"] = frame["lc_path"].map(
        lambda value: str(run_dir / "bundle_assets/lightcurves" / Path(str(value)).name))
    if not frame["lc_path"].map(lambda value: Path(value).is_file()).all():
        raise FileNotFoundError("Some native light curves are missing; finish bundle import first")
    if periods_from is not None:
        frame = _merge_saved_periods(frame, periods_from)

    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "LOCAL_FEATURES_STAGE.json"
    fingerprint = build_stage_fingerprint(
        stage="local_ml_features", stage_version="3", candidate_ids=frame["candidate_id"],
        input_paths=[source, *([periods_from] if periods_from else []), *frame["lc_path"].tolist()],
        settings={"n_bootstrap": n_bootstrap, "pdm_method": POST_FILTER_PDM_METHOD,
                  "periods_from": str(periods_from) if periods_from else None,
                  "compute_ls": True, "flag_only": True, "skip_if_consensus": False},
        code_base=REPO,
        code_paths=("scripts/compute_stv_local_features.py", "malca/core/stats.py",
                    "malca/core/utils.py", "malca/core/periodogram.py", "malca/core/period_consensus.py",
                    "malca/stv/filter.py", "malca/products/feature_layers.py"),
    )
    period_path = out / "lc_events_periodicity.parquet"
    checkpoint = out / "stats_CHECKPOINT.parquet"
    output = out / "lc_events_local_features.parquet"
    state = read_stage_state(state_path)
    if state is not None or any(out.glob("*.parquet")):
        assert_reusable_stage_state(state, fingerprint=fingerprint)
    write_stage_state(state_path, fingerprint=fingerprint,
                      result=StageResult("local_ml_features", "running", len(frame)))

    period_search_performed = False
    periods = expand_feature_layers(read_feature_table(period_path)) if period_path.exists() else None
    if periods_from is not None:
        periods = frame
        print(f"Step 1/2: reusing {len(periods):,} saved periods from {periods_from}", flush=True)
        if not period_path.exists():
            write_feature_table(to_layer_first_frame(periods, run_derived=False), period_path)
    elif periods is None or periods["periodicity_status"].eq("error").any():
        print(f"Step 1/2: measuring periods for {len(frame):,} candidates; no catalog queries", flush=True)
        period_search_performed = True
        periods = validate_periodicity(
            frame, n_bootstrap=n_bootstrap, pdm_method=POST_FILTER_PDM_METHOD,
            flag_only=True, skip_if_consensus=False, workers=workers,
            checkpoint_dir=out / "periodicity_checkpoint", lightcurve_bundle_dir=run_dir / "bundle_assets/lightcurves",
            show_tqdm=True, verbose=True,
        )
        if periods["candidate_id"].tolist() != frame["candidate_id"].tolist():
            raise ValueError("Period measurements changed the candidate cohort")
        write_feature_table(to_layer_first_frame(periods, run_derived=False), period_path)
    else:
        print("Step 1/2: reusing saved period measurements", flush=True)

    cached = pd.read_parquet(checkpoint) if checkpoint.exists() else pd.DataFrame()
    records = {str(row["candidate_id"]): row for row in cached.to_dict("records")}
    if records and (len(records) != len(cached) or not set(records) <= set(frame["candidate_id"])):
        raise ValueError("Invalid statistics checkpoint candidate IDs")
    tasks = []
    for row in periods.to_dict("records"):
        adopted_period = pd.to_numeric(row.get("periodicity_period"), errors="coerce")
        if pd.isna(adopted_period) or not np.isfinite(adopted_period) or adopted_period <= 0 or str(row.get("periodicity_status", "")) == "error":
            records[row["candidate_id"]] = {
                "candidate_id": row["candidate_id"],
                "stats_compute_status": "missing_period",
                "stats_compute_error": "No usable adopted period; phase-dependent statistics unavailable",
            }
            continue
        previous = records.get(row["candidate_id"], {})
        if previous.get("stats_compute_status") == "ok":
            continue
        tasks.append({key: row[key] for key in ("candidate_id", "lc_path", *PERIOD_COLUMNS) if key in row})
    print(f"Step 2/2: computing statistics with measured periods; {len(tasks):,} remaining", flush=True)

    def consume(iterator) -> None:
        try:
            for count, result in enumerate(tqdm(iterator, total=len(tasks), desc="Period-dependent statistics"), 1):
                records[str(result["candidate_id"])] = result
                if count % 100 == 0:
                    write_parquet_table(pd.DataFrame(records.values()), checkpoint)
        finally:
            if records:
                write_parquet_table(pd.DataFrame(records.values()), checkpoint)

    if workers == 1:
        consume(map(_stats_task, tasks))
    elif tasks:
        with Pool(processes=workers, maxtasksperchild=50) as pool:
            consume(pool.imap_unordered(_stats_task, tasks, chunksize=1))

    statistics = pd.DataFrame(records.values())
    if len(statistics) != len(periods):
        raise ValueError("Statistics completion does not cover every candidate")
    stats_columns = [c for c in statistics if c.startswith("stats_")]
    combined = periods.drop(columns=stats_columns, errors="ignore").merge(
        statistics, on="candidate_id", how="left", validate="one_to_one")
    if combined["failed_any"].astype("boolean").any():
        raise ValueError("Local measurement unexpectedly changed acceptance decisions")
    write_feature_table(to_layer_first_frame(combined, run_derived=False), output)
    unavailable = combined["stats_compute_status"].eq("missing_period")
    errors = combined["stats_compute_status"].ne("ok") & ~unavailable
    error_cols = [c for c in ("candidate_id", "lc_path", "periodicity_status", "stats_compute_status", "stats_compute_error") if c in combined]
    combined.loc[errors | unavailable, error_cols].to_csv(out / "errors.csv", index=False)
    summary = {
        "candidates": len(combined), "errors": int(errors.sum()),
        "missing_periods": int(unavailable.sum()),
        "populated": {c: int(pd.to_numeric(combined[c], errors="coerce").notna().sum()) if c in combined else 0
                      for c in ("pdm_snr", "stats_lafler_kinman_delta")},
        "output": str(output), "external_fetches": False,
        "periods_from": str(periods_from) if periods_from else None,
        "period_search_performed": period_search_performed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_stage_state(state_path, fingerprint=fingerprint,
                      result=StageResult("local_ml_features", "error" if errors.any() else ("partial" if unavailable.any() else "success"),
                                         len(combined), succeeded=int((~errors & ~unavailable).sum()),
                                         failed=int(errors.sum()), skipped=int(unavailable.sum())),
                      outputs=[output])
    print(json.dumps(summary, indent=2), flush=True)
    if errors.any():
        raise RuntimeError(f"{int(errors.sum())} candidates have processing errors; see {out / 'errors.csv'}. Rerun to retry.")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-bootstrap", type=int, default=0,
                        help="Optional significance resamples after period selection (default: 0)")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--periods-from", type=Path,
                        help="Reuse the completed period table by candidate ID; skip period search and bootstrap")
    args = parser.parse_args()
    compute(args.run_dir, workers=args.workers, n_bootstrap=args.n_bootstrap,
            input_path=args.input, output_dir=args.output_dir, periods_from=args.periods_from)
