"""Rank a local STV cohort, with an explicit provisional stage-1 preview option."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from malca.io.table_io import read_feature_table
from malca.products.feature_layers import expand_feature_layers
from malca.products.enriched_refresh import refreshed_enriched_is_current
from malca.meta_analysis.ml.candidate_features import add_recovery_bounded_event_features
from malca.meta_analysis.ml.july1_review_training import load_review_population
from malca.meta_analysis.ml.july1_tuned_calibrated_four_class_training import score_tuned_calibrated_head
from malca.stv.filter import PERIODICITY_MERGE_COLS


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO / (
    "output/runs/dat3-full-extended_2026-07-01-v4/results/"
    "lightgbm_probability_tuned_20260802T232447Z/parent_four_class/model.joblib"
)


def _attach_saved_features(table: pd.DataFrame, enriched: pd.DataFrame, *, refreshed_periods: bool) -> pd.DataFrame:
    """Join saved measurements by ID, preserving downstream catalog features."""
    table, enriched = table.copy(), enriched.copy()
    for frame in (table, enriched):
        if frame["candidate_id"].isna().any():
            raise ValueError("Missing candidate IDs in saved features or scoring cohort")
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        if frame["candidate_id"].duplicated().any():
            raise ValueError("Duplicate candidate IDs in saved features or scoring cohort")
    table = table.set_index("candidate_id", drop=False)
    enriched = enriched.set_index("candidate_id", drop=False)
    if not set(table.index) <= set(enriched.index):
        raise ValueError("Some Review candidates have no saved statistics")
    if refreshed_periods:
        if set(table.index) != set(enriched.index):
            raise ValueError("Review and local-feature cohorts differ; reconcile candidate IDs before scoring")
        if not enriched["stats_compute_status"].isin(["ok", "missing_period"]).all():
            raise ValueError("Local statistics have unresolved processing errors")
        current_names = table["lc_path"].map(lambda value: Path(str(value)).name)
        saved_names = enriched["lc_path"].reindex(table.index).map(lambda value: Path(str(value)).name)
        if not current_names.equals(saved_names):
            raise ValueError("Saved features refer to different light curves")
    period_columns = {*PERIODICITY_MERGE_COLS, "phase_period_days", "phase_source"}
    if refreshed_periods:
        # Layered files can omit columns that are entirely null. Such omissions
        # must clear stale measurements, including significance from an older run.
        for column in table.columns:
            if (column.startswith("stats_") or column in period_columns) and column not in enriched:
                table[column] = np.nan
    for column in enriched.columns:
        if column.startswith("stats_") or (refreshed_periods and column in period_columns) or column not in table:
            table[column] = enriched[column].reindex(table.index)
    return table.reset_index(drop=True)


def rank(run_dir: Path, model_path: Path, new_candidates: Path | None, workers: int,
         *, stage1_preview: bool = False, local_features: Path | None = None,
         features_only: bool = False) -> None:
    run_dir = run_dir.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    if stage1_preview and local_features is not None:
        raise ValueError("--local-features requires the enriched Review cohort, not --stage1-preview")
    output_name = "dipper_ranking_stage1_preview" if stage1_preview else "dipper_ranking"
    out_dir = run_dir / "results" / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = joblib.load(model_path)
    stats_name = "lc_events_enriched_all.parquet" if stage1_preview else "lc_events_enriched.parquet"
    statistics_path = local_features.expanduser().resolve() if local_features is not None else run_dir / "results" / stats_name
    enriched = expand_feature_layers(read_feature_table(statistics_path))
    if stage1_preview:
        print("Provisional stage-1 preview: using saved features; no fetches or new LC measurements.", flush=True)
        decisions = enriched["failed_any"].astype("boolean")
        if decisions.isna().any():
            raise ValueError("Missing stage-1 filter decisions")
        table = enriched.loc[~decisions].copy()
    else:
        table = load_review_population(run_dir / "review/review.db", keep_morphology_secondary_json=False)
    if table.empty or table["candidate_id"].duplicated().any():
        raise ValueError("Expected a nonempty candidate cohort with unique candidate IDs")
    table["candidate_id"] = table["candidate_id"].astype(str)

    refreshed = local_features is not None or (not stage1_preview and refreshed_enriched_is_current(run_dir / "results"))
    table = _attach_saved_features(table, enriched, refreshed_periods=refreshed)
    local_lcs = run_dir / "bundle_assets/lightcurves"
    table["lc_path"] = table["lc_path"].map(lambda value: str(local_lcs / Path(str(value)).name))
    if not table["lc_path"].map(lambda value: Path(value).is_file()).all():
        raise FileNotFoundError("Some candidates lack local native light curves; import the full passer bundle")
    table["asas_sn_id"] = table["lc_path"].map(lambda value: Path(value).stem)
    if not stage1_preview:
        table = add_recovery_bounded_event_features(
            table, run_dir / "results/ml_feature_cache/recovery_bounded_event_features.parquet", workers=workers,
        )
    missing = [column for column in model["feature_columns"] if column not in table]
    coverage = {column: int(table[column].notna().sum()) for column in model["feature_columns"] if column in table}
    unavailable = [c for c in model["feature_columns"] if not coverage.get(c, 0)]
    provisional = stage1_preview or bool(unavailable)
    report = {
        "model": str(model_path), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "candidates": len(table), "missing_columns": missing, "feature_non_null_counts": coverage,
        "recovery_states": table["recovery_feature_state"].value_counts().to_dict() if "recovery_feature_state" in table else {},
        "input_stage": "stage1_preview" if stage1_preview else "enriched_review_cohort",
        "statistics_from": str(statistics_path),
        "features_only": features_only,
        "provisional": provisional,
        "entirely_unavailable_features": unavailable,
        "probability_scope": ("unvalidated_stage1_missing_feature_pattern" if stage1_preview else
                              "unvalidated_missing_feature_pattern" if unavailable else "reviewed_candidate_distribution"),
    }
    (out_dir / "feature_coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    if features_only:
        print(f"Prepared and audited {len(table):,} candidates; {len(unavailable)}/{len(model['feature_columns'])} model features entirely unavailable. No classifier scores computed. See {out_dir / 'feature_coverage.json'}", flush=True)
        return
    if missing and not stage1_preview:
        raise ValueError(f"Missing model features; see {out_dir / 'feature_coverage.json'}")
    if not stage1_preview and table["recovery_feature_state"].isin(["error", "no_lightcurve"]).any():
        raise RuntimeError("Recovery-feature errors remain; see the feature cache before scoring")
    for prefix in ("stats_", "dip_", "jump_"):
        columns = [column for column in coverage if column.startswith(prefix)]
        if columns and not any(coverage[column] for column in columns):
            raise ValueError(f"Entire model feature family {prefix} is empty")
    period_measurements = ("periodicity_period", "periodicity_scatter_ratio", "pdm_snr", "ce_snr")
    if not stage1_preview and not any(coverage.get(column, 0) for column in period_measurements):
        raise ValueError("Period measurements are missing; enable apply_periodicity_validation in the home config")

    print(f"Scoring {len(table):,} candidates; {len(report['entirely_unavailable_features'])}/{len(model['feature_columns'])} model features entirely unavailable.", flush=True)
    scores = score_tuned_calibrated_head(model, table)
    ranked = table[["candidate_id", "asas_sn_id", "lc_path"]].merge(scores, on="candidate_id", validate="one_to_one")
    if not np.isfinite(ranked["prob_dimming_event"]).all():
        raise ValueError("Nonfinite dimmer probabilities")
    ranked = ranked.sort_values(["prob_dimming_event", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    if provisional:
        ranked["score_status"] = "provisional_stage1_missing_features" if stage1_preview else "provisional_missing_features"
    if new_candidates is not None:
        new = pd.read_csv(new_candidates.expanduser(), dtype={"asas_sn_id": str})
        if new["asas_sn_id"].duplicated().any():
            raise ValueError("Duplicate source IDs in new-candidate list")
        selected = ranked.merge(new[["asas_sn_id", "july1_status"]], on="asas_sn_id", validate="one_to_one")
        selected = selected.sort_values("rank").rename(columns={"rank": "rank_all_candidates"})
        selected.insert(0, "rank", np.arange(1, len(selected) + 1))
        selected.to_csv(out_dir / "new_candidates_ranked.csv", index=False)
        absent_name = "new_candidates_absent_from_stage1_cohort.csv" if stage1_preview else "new_candidates_absent_from_home_cohort.csv"
        new.loc[~new["asas_sn_id"].isin(ranked["asas_sn_id"])].to_csv(out_dir / absent_name, index=False)
        print(f"Ranked {len(selected)}/{len(new)} new cluster candidates present in the scoring cohort")
    ranked.to_csv(out_dir / "all_candidates_ranked.csv", index=False)
    ranked.to_parquet(out_dir / "all_candidates_ranked.parquet", index=False)
    print(f"Ranked {len(ranked)} candidates by prob_dimming_event: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--new-candidates", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--local-features", type=Path,
                        help="Use refreshed local statistics and periods while preserving Review catalog features")
    parser.add_argument("--features-only", action="store_true",
                        help="Prepare cached event features and report model-input coverage without scoring")
    parser.add_argument("--stage1-preview", action="store_true",
                        help="Score imported cluster passers immediately; missing features yield provisional scores")
    args = parser.parse_args()
    rank(args.run_dir, args.model, args.new_candidates, args.workers, stage1_preview=args.stage1_preview,
         local_features=args.local_features, features_only=args.features_only)
