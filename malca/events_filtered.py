#!/usr/bin/env python3
"""
Wrapper script to run events.py on pre-filtered light curves.

Workflow:
1. Build/load manifest (source_id → lc_dir mapping)
2. Apply pre-filters (sparse, periodic, multi-camera)
3. Construct file paths for kept sources
4. Pass to events.py

Usage:
    python -m malca.events_filtered --mag-bin 13_13.5 [events.py args...]
"""
from __future__ import annotations

import argparse
from datetime import datetime
import shlex
import subprocess
import sys
from pathlib import Path
import pandas as pd
import tempfile

from malca.manifest import build_manifest_dataframe
from malca.pre_filter import apply_pre_filters


def safe_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write parquet atomically to avoid corruption on interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def parse_output_format(events_args: list[str]) -> str:
    """Find --output-format value in events args if provided."""
    for i, arg in enumerate(events_args):
        if arg == "--output-format" and i + 1 < len(events_args):
            return str(events_args[i + 1]).lower()
    return "csv"


def default_run_dir(base_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_root / "runs" / timestamp


def clear_existing_output(path: Path | None, fmt: str, quiet: bool = False) -> None:
    if path is None or (not path.exists()):
        return
    try:
        if fmt == "parquet_chunk" and path.is_dir():
            removed_any = False
            for child in path.glob("chunk_*.parquet*"):
                child.unlink()
                removed_any = True
            if removed_any:
                if not quiet:
                    print(f"Overwriting existing output chunks in {path}")
        else:
            path.unlink()
            if not quiet:
                print(f"Overwriting existing output file: {path}")
    except Exception as e:
        if not quiet:
            print(f"Warning: could not remove existing output {path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run events.py on pre-filtered light curves",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="All other arguments are passed directly to events.py"
    )

    # Manifest/pre-filter args
    parser.add_argument("--mag-bin", required=True, nargs="+", help="Magnitude bin(s) to process")
    parser.add_argument("--index-root", type=Path, default=Path("/data/poohbah/1/assassin/rowan.90/lcsv2"),
                        help="Index root directory (contains mag_bin/index*.csv)")
    parser.add_argument("--lc-root", type=Path, default=Path("/data/poohbah/1/assassin/rowan.90/lcsv2"),
                        help="Light curve root directory (contains mag_bin/lc*_cal/)")
    parser.add_argument("--manifest-file", type=Path, default=None,
                        help="Manifest file (default: lc_manifest_{mag_bin}.parquet)")
    parser.add_argument("--filtered-file", type=Path, default=None,
                        help="Filtered manifest file (default: lc_filtered_{mag_bin}.parquet)")
    parser.add_argument("--force-manifest", action="store_true",
                        help="Force rebuild manifest even if exists")
    parser.add_argument("--force-filter", action="store_true",
                        help="Force re-run pre-filters even if filtered file exists")

    # Pre-filter args
    parser.add_argument("--min-time-span", type=float, default=100.0, help="Min time span (days)")
    parser.add_argument("--min-points-per-day", type=float, default=0.05, help="Min cadence")
    parser.add_argument("--min-cameras", type=int, default=2, help="Min cameras required")
    parser.add_argument("--skip-sparse", action="store_true", help="Skip sparse LC filter")
    parser.add_argument("--skip-multi-camera", action="store_true", help="Skip multi-camera filter")
    parser.add_argument("--skip-vsx", action="store_true", help="Skip VSX crossmatch/tagging")
    parser.add_argument("--vsx-max-sep", type=float, default=3.0, help="Max separation for VSX match (arcsec)")
    parser.add_argument("--vsx-mode", type=str, default="filter", choices=["tag", "filter"],
                        help="VSX handling: tag adds sep_arcsec/class columns, filter removes matches (default: filter)")
    parser.add_argument("--vsx-crossmatch", type=Path, default=Path("input/vsx/asassn_x_vsx_matches_20250919_2252.csv"), help="Path to pre-crossmatched VSX CSV (with asas_sn_id, sep_arcsec, class)")
    parser.add_argument("--workers", type=int, default=10, help="Workers for parallel processing")
    parser.add_argument("--stats-chunk-size", type=int, default=5000, help="Rows per checkpoint save during stats computation")
    parser.add_argument("--batch-size", type=int, default=2000, help="Max light curves per events.py call")

    # events.py args
    parser.add_argument("--trigger-mode", type=str, default="posterior_prob", choices=["logbf", "posterior_prob"], help="Triggering mode")
    parser.add_argument("--logbf-threshold-dip", type=float, default=5.0, help="Per-point dip trigger threshold")
    parser.add_argument("--logbf-threshold-jump", type=float, default=5.0, help="Per-point jump trigger threshold")
    parser.add_argument("--significance-threshold", type=float, default=99.99997, help="Posterior probability threshold (if trigger-mode=posterior_prob)")
    parser.add_argument("--p-points", type=int, default=12, help="Number of points in the logit-spaced p grid")
    parser.add_argument("--mag-points", type=int, default=12, help="Number of points in the magnitude grid")
    parser.add_argument("--run-min-points", type=int, default=2, help="Min triggered points in a run")
    parser.add_argument("--run-allow-gap-points", type=int, default=5, help="Allow up to this many missing indices inside a run")
    parser.add_argument("--run-max-gap-days", type=float, default=None, help="Break runs if JD gap exceeds this")
    parser.add_argument("--run-min-duration-days", type=float, default=0.0, help="Require run duration >= this (default: 0.0 = disabled)")
    parser.add_argument("--no-event-prob", action="store_true", help="Skip LOO event responsibilities")
    parser.add_argument("--p-min-dip", type=float, default=None, help="Minimum dip fraction for p-grid")
    parser.add_argument("--p-max-dip", type=float, default=None, help="Maximum dip fraction for p-grid")
    parser.add_argument("--p-min-jump", type=float, default=None, help="Minimum jump fraction for p-grid")
    parser.add_argument("--p-max-jump", type=float, default=None, help="Maximum jump fraction for p-grid")
    parser.add_argument("--baseline-func", type=str, default="gp", choices=["gp", "gp_masked", "trend"], help="Baseline function")
    # Baseline kwargs (GP kernel parameters)
    parser.add_argument("--baseline-s0", type=float, default=0.0005, help="GP kernel S0 parameter (default: 0.0005)")
    parser.add_argument("--baseline-w0", type=float, default=0.0031415926535897933, help="GP kernel w0 parameter (default: pi/1000)")
    parser.add_argument("--baseline-q", type=float, default=0.7, help="GP kernel Q parameter (default: 0.7)")
    parser.add_argument("--baseline-jitter", type=float, default=0.006, help="GP jitter term (default: 0.006)")
    parser.add_argument("--baseline-sigma-floor", type=float, default=None, help="Minimum sigma floor (default: None)")
    # Magnitude grid bounds (override auto-detection)
    parser.add_argument("--mag-min-dip", type=float, default=None, help="Min magnitude for dip grid (overrides auto)")
    parser.add_argument("--mag-max-dip", type=float, default=None, help="Max magnitude for dip grid (overrides auto)")
    parser.add_argument("--mag-min-jump", type=float, default=None, help="Min magnitude for jump grid (overrides auto)")
    parser.add_argument("--mag-max-jump", type=float, default=None, help="Max magnitude for jump grid (overrides auto)")
    parser.add_argument("--no-sigma-eff", action="store_true", help="Do not replace errors with sigma_eff")
    parser.add_argument("--allow-missing-sigma-eff", action="store_true", help="Do not error if baseline omits sigma_eff")
    parser.add_argument("--min-mag-offset", type=float, default=0.1, help="Require |event_mag - baseline_mag| > threshold")
    parser.add_argument("--output", type=str, default=None, help="Output path for results (default: <out_dir>/lc_events_results.csv)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory for all outputs (default: output/runs/<timestamp>)")
    parser.add_argument("--output-format", type=str, default="csv", choices=["csv", "parquet", "parquet_chunk", "duckdb"], help="Output format")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Write results in chunks of this many rows")

    parser.add_argument("-o", "--overwrite", action="store_true",
                        help="Overwrite checkpoint log and existing output if present (start fresh).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output (default: quiet).")

    args = parser.parse_args()

    # Build events.py args from parsed arguments
    events_args = []
    if args.verbose:
        events_args.append("--verbose")
    events_args.extend(["--workers", str(args.workers)])
    events_args.extend(["--trigger-mode", args.trigger_mode])
    events_args.extend(["--logbf-threshold-dip", str(args.logbf_threshold_dip)])
    events_args.extend(["--logbf-threshold-jump", str(args.logbf_threshold_jump)])
    events_args.extend(["--significance-threshold", str(args.significance_threshold)])
    events_args.extend(["--p-points", str(args.p_points)])
    events_args.extend(["--mag-points", str(args.mag_points)])
    events_args.extend(["--run-min-points", str(args.run_min_points)])
    events_args.extend(["--run-allow-gap-points", str(args.run_allow_gap_points)])
    if args.run_max_gap_days is not None:
        events_args.extend(["--run-max-gap-days", str(args.run_max_gap_days)])
    if args.run_min_duration_days is not None:
        events_args.extend(["--run-min-duration-days", str(args.run_min_duration_days)])
    if args.no_event_prob:
        events_args.append("--no-event-prob")
    if args.p_min_dip is not None:
        events_args.extend(["--p-min-dip", str(args.p_min_dip)])
    if args.p_max_dip is not None:
        events_args.extend(["--p-max-dip", str(args.p_max_dip)])
    if args.p_min_jump is not None:
        events_args.extend(["--p-min-jump", str(args.p_min_jump)])
    if args.p_max_jump is not None:
        events_args.extend(["--p-max-jump", str(args.p_max_jump)])
    events_args.extend(["--baseline-func", args.baseline_func])
    # Baseline kwargs
    events_args.extend(["--baseline-s0", str(args.baseline_s0)])
    events_args.extend(["--baseline-w0", str(args.baseline_w0)])
    events_args.extend(["--baseline-q", str(args.baseline_q)])
    events_args.extend(["--baseline-jitter", str(args.baseline_jitter)])
    if args.baseline_sigma_floor is not None:
        events_args.extend(["--baseline-sigma-floor", str(args.baseline_sigma_floor)])
    # Magnitude grid bounds
    if args.mag_min_dip is not None:
        events_args.extend(["--mag-min-dip", str(args.mag_min_dip)])
    if args.mag_max_dip is not None:
        events_args.extend(["--mag-max-dip", str(args.mag_max_dip)])
    if args.mag_min_jump is not None:
        events_args.extend(["--mag-min-jump", str(args.mag_min_jump)])
    if args.mag_max_jump is not None:
        events_args.extend(["--mag-max-jump", str(args.mag_max_jump)])
    if args.no_sigma_eff:
        events_args.append("--no-sigma-eff")
    if args.allow_missing_sigma_eff:
        events_args.append("--allow-missing-sigma-eff")
    events_args.extend(["--min-mag-offset", str(args.min_mag_offset)])
    events_args.extend(["--output-format", args.output_format])
    events_args.extend(["--chunk-size", str(args.chunk_size)])

    def log(message: str) -> None:
        if args.verbose:
            print(message)

    # Determine file names
    mag_bin_tag = args.mag_bin[0] if len(args.mag_bin) == 1 else "multi"

    # IMPORTANT: never write to filesystem root (/output). Default to a writable directory.
    events_format = parse_output_format(events_args)
    base_output_root = Path("/home/lenhart.106/code/malca/output")
    if args.out_dir is not None:
        out_dir = Path(args.out_dir).expanduser()
    elif args.filtered_file is not None:
        out_dir = Path(args.filtered_file).expanduser().parent
    elif args.manifest_file is not None:
        out_dir = Path(args.manifest_file).expanduser().parent
    elif args.output is not None:
        out_dir = Path(args.output).expanduser().parent
    else:
        out_dir = default_run_dir(base_output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests_dir = out_dir / "manifests"
    prefilter_dir = out_dir / "prefilter"
    paths_dir = out_dir / "paths"
    results_dir = out_dir / "results"
    for d in (manifests_dir, prefilter_dir, paths_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    if args.output is None:
        events_output = results_dir / "lc_events_results.csv"
    else:
        events_output = Path(args.output).expanduser()
        if args.out_dir is not None and not events_output.is_absolute():
            events_output = out_dir / events_output
        elif args.out_dir is None and not events_output.is_absolute():
            events_output = out_dir / events_output

    events_args.extend(["--output", str(events_output)])

    manifest_file = Path(args.manifest_file).expanduser() if args.manifest_file else (manifests_dir / f"lc_manifest_{mag_bin_tag}.parquet")
    filtered_file = Path(args.filtered_file).expanduser() if args.filtered_file else (prefilter_dir / f"lc_filtered_{mag_bin_tag}.parquet")
    stats_checkpoint_file = prefilter_dir / f"lc_stats_checkpoint_{mag_bin_tag}.parquet"

    # Save run parameters to JSON for full reproducibility
    import json
    run_start_time = datetime.now()

    run_params_file = out_dir / "run_params.json"
    try:
        orig_argv = getattr(sys, "orig_argv", None)
        cmd = shlex.join(orig_argv) if orig_argv else shlex.join([sys.executable] + sys.argv)

        run_params = {
            "timestamp": run_start_time.isoformat(),
            "command": cmd,
            "mag_bin": args.mag_bin,
            # Pre-filter parameters
            "min_time_span": args.min_time_span,
            "min_points_per_day": args.min_points_per_day,
            "min_cameras": args.min_cameras,
            "skip_sparse": args.skip_sparse,
            "skip_multi_camera": args.skip_multi_camera,
            "skip_vsx": args.skip_vsx,
            "vsx_max_sep": args.vsx_max_sep,
            "vsx_mode": args.vsx_mode,
            "vsx_crossmatch": str(args.vsx_crossmatch),
            # Detection parameters
            "trigger_mode": args.trigger_mode,
            "logbf_threshold_dip": args.logbf_threshold_dip,
            "logbf_threshold_jump": args.logbf_threshold_jump,
            "significance_threshold": args.significance_threshold,
            "p_points": args.p_points,
            "p_min_dip": args.p_min_dip,
            "p_max_dip": args.p_max_dip,
            "p_min_jump": args.p_min_jump,
            "p_max_jump": args.p_max_jump,
            "mag_points": args.mag_points,
            "mag_min_dip": args.mag_min_dip,
            "mag_max_dip": args.mag_max_dip,
            "mag_min_jump": args.mag_min_jump,
            "mag_max_jump": args.mag_max_jump,
            # Baseline parameters
            "baseline_func": args.baseline_func,
            "baseline_s0": args.baseline_s0,
            "baseline_w0": args.baseline_w0,
            "baseline_q": args.baseline_q,
            "baseline_jitter": args.baseline_jitter,
            "baseline_sigma_floor": args.baseline_sigma_floor,
            # Run parameters
            "run_min_points": args.run_min_points,
            "run_allow_gap_points": args.run_allow_gap_points,
            "run_max_gap_days": args.run_max_gap_days,
            "run_min_duration_days": args.run_min_duration_days,
            "no_event_prob": args.no_event_prob,
            "no_sigma_eff": args.no_sigma_eff,
            "allow_missing_sigma_eff": args.allow_missing_sigma_eff,
            "min_mag_offset": args.min_mag_offset,
            # System parameters
            "workers": args.workers,
            "batch_size": args.batch_size,
            "output_format": args.output_format,
            # File paths
            "index_root": str(args.index_root),
            "lc_root": str(args.lc_root),
            "out_dir": str(out_dir),
            "manifest_file": str(manifest_file),
            "filtered_file": str(filtered_file),
            "events_output": str(events_output),
        }

        with open(run_params_file, "w") as f:
            json.dump(run_params, f, indent=2, default=str)

    except Exception as e:
        if args.verbose:
            print(f"Warning: could not write run_params.json: {e}")

    # Write a simple run log with the command and key paths.
    run_log = out_dir / "run.log"
    try:
        events_cmd_preview = shlex.join([sys.executable, "-m", "malca.events", *events_args, "--", "<paths_file>"])
        run_log.write_text(
            "\n".join([
                f"timestamp: {run_start_time.isoformat()}",
                f"command: {cmd}",
                f"events_cmd: {events_cmd_preview}",
                f"out_dir: {out_dir}",
                f"run_params: {run_params_file}",
                f"manifests_dir: {manifests_dir}",
                f"prefilter_dir: {prefilter_dir}",
                f"paths_dir: {paths_dir}",
                f"results_dir: {results_dir}",
                f"results_output: {events_output}",
                f"manifest_file: {manifest_file}",
                f"filtered_file: {filtered_file}",
                f"stats_checkpoint: {stats_checkpoint_file}",
                f"rejected_pre_filter: {prefilter_dir / f'rejected_pre_filter_{mag_bin_tag}.csv'}",
            ]) + "\n"
        )
    except Exception as e:
        if args.verbose:
            print(f"Warning: could not write run log: {e}")

    # Step 1: Build or load manifest
    if args.force_manifest or not manifest_file.exists():
        log(f"Building manifest for mag_bin={args.mag_bin}...")
        df_manifest = build_manifest_dataframe(
            args.index_root,
            args.lc_root,
            mag_bins=args.mag_bin,
            id_column="asas_sn_id",
            show_progress=args.verbose
        )

        # Only keep sources where .dat2 or .csv files exist
        df_manifest = df_manifest[df_manifest["dat_exists"]].reset_index(drop=True)

        log(f"Saving manifest to {manifest_file} ({len(df_manifest)} sources)")
        safe_write_parquet(df_manifest, manifest_file)
    else:
        log(f"Loading existing manifest from {manifest_file}")
        df_manifest = pd.read_parquet(manifest_file)
        log(f"Loaded {len(df_manifest)} sources")

    # Step 2: Apply pre-filters

    if args.force_filter or not filtered_file.exists():
        log(f"\nApplying pre-filters with {args.workers} workers...")

        # Use lc_dir as the directory path for pre_filter compatibility (path/<id>.dat2)
        df_to_filter = df_manifest.rename(columns={"lc_dir": "path"}).copy()

        df_filtered = apply_pre_filters(
            df_to_filter,
            apply_sparse=not args.skip_sparse,
            min_time_span=args.min_time_span,
            min_points_per_day=args.min_points_per_day,
            apply_vsx=not args.skip_vsx,
            vsx_max_sep_arcsec=args.vsx_max_sep,
            vsx_mode=args.vsx_mode,
            vsx_crossmatch_csv=args.vsx_crossmatch,
            apply_multi_camera=not args.skip_multi_camera,
            min_cameras=args.min_cameras,
            n_workers=args.workers,
            show_tqdm=args.verbose,
            rejected_log_csv=str(prefilter_dir / f"rejected_pre_filter_{mag_bin_tag}.csv"),
            stats_checkpoint=str(stats_checkpoint_file),
            stats_chunk_size=args.stats_chunk_size,
        )

        log(f"\nKept {len(df_filtered)}/{len(df_manifest)} sources after pre-filtering")
        log(f"Saving filtered manifest to {filtered_file}")
        safe_write_parquet(df_filtered, filtered_file)
    else:
        log(f"\nLoading existing filtered manifest from {filtered_file}")
        df_filtered = pd.read_parquet(filtered_file)
        log(f"Loaded {len(df_filtered)} filtered sources")

    # Step 3: Construct file paths (use full dat_path for events.py input)
    file_col = "dat_path" if "dat_path" in df_filtered.columns else "path"

    vsx_tags_file = None
    if not args.skip_vsx and "sep_arcsec" in df_filtered.columns and "class" in df_filtered.columns:
        vsx_tags_dir = prefilter_dir / "vsx_tags"
        vsx_tags_dir.mkdir(parents=True, exist_ok=True)
        vsx_tags_file = vsx_tags_dir / f"vsx_tags_{mag_bin_tag}.csv"
        tag_df = df_filtered[[file_col, "sep_arcsec", "class"]].rename(columns={file_col: "path"})
        tag_df.to_csv(vsx_tags_file, index=False)
        events_args.extend(["--metadata-csv", str(vsx_tags_file)])
        if run_log.exists():
            try:
                with run_log.open("a") as f:
                    f.write(f"vsx_tags: {vsx_tags_file}\n")
                    f.write(f"events_metadata_csv: {vsx_tags_file}\n")
            except Exception as e:
                if args.verbose:
                    print(f"Warning: could not update run log with vsx_tags: {e}")

    file_paths = df_filtered[file_col].tolist()

    if not file_paths:
        log("\nNo sources to process after filtering!")
        return

    # Step 4: Call events.py with the filtered paths in batches, with resume support
    log(f"\nPreparing to run events.py on {len(file_paths)} light curves...")

    # Write paths to temp file for events.py to consume
    paths_file = paths_dir / f"filtered_paths_{mag_bin_tag}.txt"
    with open(paths_file, "w") as f:
        for path in file_paths:
            f.write(f"{path}\n")
    if run_log.exists():
        try:
            with run_log.open("a") as f:
                f.write(f"paths_file: {paths_file}\n")
        except Exception as e:
            if args.verbose:
                print(f"Warning: could not update run log with paths_file: {e}")

    # Resume logic: skip paths already recorded in events checkpoint log if present
    base_output = events_output or (results_dir / "lc_events_results.csv")
    suffix_map = {"csv": ".csv", "parquet": ".parquet", "parquet_chunk": None, "duckdb": ".duckdb"}
    ext = suffix_map.get(events_format)
    if ext and base_output.suffix.lower() != ext:
        base_output = base_output.with_suffix(ext)
    checkpoint_log = base_output.with_name(f"{base_output.stem}_PROCESSED.txt")
    processed_paths: set[str] = set()
    if checkpoint_log.exists() and args.overwrite:
        try:
            with open(checkpoint_log, "w"):
                pass
            log(f"Overwriting checkpoint log: {checkpoint_log}")
        except Exception as e:
            log(f"Warning: could not overwrite checkpoint log {checkpoint_log}: {e}")

    if args.overwrite:
        clear_existing_output(base_output, events_format, quiet=not args.verbose)

    if checkpoint_log.exists() and not args.overwrite:
        try:
            with open(checkpoint_log, "r") as f:
                processed_paths = {line.strip() for line in f if line.strip()}
            log(f"Checkpoint detected, skipping {len(processed_paths)} already-processed paths")
        except Exception as e:
            log(f"Warning: could not read checkpoint log {checkpoint_log}: {e}")

    remaining = [p for p in file_paths if str(p) not in processed_paths]
    if not remaining:
        log("All paths already processed according to checkpoint. Exiting.")
        return

    # Batch and run
    batch_size = max(1, args.batch_size)
    total_batches = (len(remaining) + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(len(remaining), start + batch_size)
        batch_paths = remaining[start:end]

        log(f"\nRunning batch {batch_idx + 1}/{total_batches} ({len(batch_paths)} LCs)...")

        events_cmd = [
            sys.executable, "-m", "malca.events",
            *events_args,
            "--",
            *batch_paths,
        ]

        # Execute
        try:
            result = subprocess.run(events_cmd, check=False)
            if result.returncode != 0:
                print(f"events.py returned non-zero exit ({result.returncode}); stopping.")
                sys.exit(result.returncode)
        except Exception as e:
            print(f"\nError running events.py: {e}")
            print(f"\nFiltered paths saved to: {paths_file}")
            print(f"You can manually run events.py with these paths")
            sys.exit(1)

        # Append processed paths to checkpoint log safely
        checkpoint_log.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_log, "a") as f:
            for p in batch_paths:
                f.write(f"{p}\n")

    log("\nAll batches completed.")

    # Generate run summary with results statistics
    run_end_time = datetime.now()
    run_summary_file = out_dir / "run_summary.json"
    try:
        summary = {
            "run_info": {
                "start_time": run_start_time.isoformat(),
                "end_time": run_end_time.isoformat(),
                "duration_seconds": (run_end_time - run_start_time).total_seconds(),
            },
            "manifest_stats": {
                "total_sources": len(df_manifest),
                "filtered_sources": len(df_filtered),
                "kept_fraction": len(df_filtered) / len(df_manifest) if len(df_manifest) > 0 else 0.0,
            },
        }

        # Pre-filter rejection breakdown
        rejected_log = prefilter_dir / f"rejected_pre_filter_{mag_bin_tag}.csv"
        if rejected_log.exists():
            try:
                df_rejected = pd.read_csv(rejected_log)
                if "reason" in df_rejected.columns:
                    rejection_counts = df_rejected["reason"].value_counts().to_dict()
                    summary["pre_filter_rejections"] = {
                        "total_rejected": len(df_rejected),
                        "by_reason": rejection_counts,
                    }
            except Exception as e:
                if args.verbose:
                    print(f"Warning: could not parse rejection log: {e}")

        # Detection results statistics
        results_files = []
        if events_format == "csv":
            if base_output.exists():
                results_files = [base_output]
        elif events_format == "parquet":
            if base_output.exists():
                results_files = [base_output]
        elif events_format == "parquet_chunk":
            chunk_dir = base_output.parent if base_output.suffix else base_output
            results_files = sorted(chunk_dir.glob("chunk_*.parquet"))
        elif events_format == "duckdb":
            # DuckDB results aren't easily parsed here
            pass

        if results_files:
            try:
                if events_format == "csv":
                    df_results = pd.read_csv(results_files[0])
                else:  # parquet or parquet_chunk
                    df_results = pd.concat([pd.read_parquet(f) for f in results_files], ignore_index=True)

                detection_stats = {
                    "total_detections": len(df_results),
                    "unique_sources": df_results["path"].nunique() if "path" in df_results.columns else None,
                }

                # Count significant detections
                if "dip_significant" in df_results.columns:
                    detection_stats["dip_significant"] = int(df_results["dip_significant"].sum())
                if "jump_significant" in df_results.columns:
                    detection_stats["jump_significant"] = int(df_results["jump_significant"].sum())

                # Event type counts
                if "event_type" in df_results.columns:
                    detection_stats["by_event_type"] = df_results["event_type"].value_counts().to_dict()

                summary["detection_stats"] = detection_stats

            except Exception as e:
                if args.verbose:
                    print(f"Warning: could not parse detection results: {e}")

        # VSX statistics if available
        if vsx_tags_file and vsx_tags_file.exists():
            try:
                df_vsx = pd.read_csv(vsx_tags_file)
                summary["vsx_stats"] = {
                    "sources_with_vsx_match": len(df_vsx),
                    "vsx_classes": df_vsx["class"].value_counts().to_dict() if "class" in df_vsx.columns else None,
                }
            except Exception as e:
                if args.verbose:
                    print(f"Warning: could not parse VSX tags: {e}")

        # Write summary
        with open(run_summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        log(f"\nRun summary saved to {run_summary_file}")

    except Exception as e:
        if args.verbose:
            print(f"Warning: could not write run summary: {e}")


if __name__ == "__main__":
    main()
