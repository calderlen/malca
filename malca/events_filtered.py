#!/usr/bin/env python3
"""
Wrapper script to run events.py on pre-filtered light curves.

Workflow:
1. Build/load manifest (source_id → lc_dir mapping)
2. Apply pre-filters (sparse, periodic, multi-camera)
3. Construct file paths for kept sources
4. Pass to events.py

Usage:
    python -m events_filtered --mag-bin 13_13.5 [events.py args...]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import tempfile

from manifest import build_manifest_dataframe
from pre_filter import apply_pre_filters


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


def parse_output_path(events_args: list[str]) -> Path | None:
    """Find --output value in events args if provided."""
    for i, arg in enumerate(events_args):
        if arg == "--output" and i + 1 < len(events_args):
            return Path(events_args[i + 1]).expanduser()
    return None


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
    parser.add_argument("--skip-vsx", action="store_true", help="Skip VSX known variable filter")
    parser.add_argument("--vsx-max-sep", type=float, default=3.0, help="Max separation for VSX match (arcsec)")
    parser.add_argument("--vsx-catalog", type=Path, default=Path("input/vsx/vsx_cleaned.csv"), help="Path to VSX catalog CSV")
    parser.add_argument("--n-workers", type=int, default=16, help="Workers for pre-filter stats")
    parser.add_argument("--batch-size", type=int, default=2000, help="Max light curves per events.py call to limit arg size and allow resume")

    # Parse known args, rest go to events.py
    args, events_args = parser.parse_known_args()

    # Determine file names
    mag_bin_tag = args.mag_bin[0] if len(args.mag_bin) == 1 else "multi"

    # IMPORTANT: never write to filesystem root (/output). Default to a writable directory.
    events_output = parse_output_path(events_args)
    if args.filtered_file is not None:
        out_dir = Path(args.filtered_file).expanduser().parent
    elif args.manifest_file is not None:
        out_dir = Path(args.manifest_file).expanduser().parent
    elif events_output is not None:
        out_dir = events_output.parent
    else:
        out_dir = Path.home() / "malca_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = Path(args.manifest_file).expanduser() if args.manifest_file else (out_dir / f"lc_manifest_{mag_bin_tag}.parquet")
    filtered_file = Path(args.filtered_file).expanduser() if args.filtered_file else (out_dir / f"lc_filtered_{mag_bin_tag}.parquet")

    # Step 1: Build or load manifest
    if args.force_manifest or not manifest_file.exists():
        print(f"Building manifest for mag_bin={args.mag_bin}...")
        df_manifest = build_manifest_dataframe(
            args.index_root,
            args.lc_root,
            mag_bins=args.mag_bin,
            id_column="asas_sn_id",
            show_progress=True
        )

        # Only keep sources where .dat2 or .csv files exist
        df_manifest = df_manifest[df_manifest["dat_exists"]].reset_index(drop=True)

        print(f"Saving manifest to {manifest_file} ({len(df_manifest)} sources)")
        safe_write_parquet(df_manifest, manifest_file)
    else:
        print(f"Loading existing manifest from {manifest_file}")
        df_manifest = pd.read_parquet(manifest_file)
        print(f"Loaded {len(df_manifest)} sources")

    # Step 2: Apply pre-filters
    if args.force_filter or not filtered_file.exists():
        print(f"\nApplying pre-filters with {args.n_workers} workers...")

        # Use lc_dir as the directory path for pre_filter compatibility (path/<id>.dat2)
        df_to_filter = df_manifest.rename(columns={"lc_dir": "path"}).copy()

        df_filtered = apply_pre_filters(
            df_to_filter,
            apply_sparse=not args.skip_sparse,
            min_time_span=args.min_time_span,
            min_points_per_day=args.min_points_per_day,
            apply_vsx=not args.skip_vsx,
            vsx_max_sep_arcsec=args.vsx_max_sep,
            vsx_catalog_csv=args.vsx_catalog,
            apply_multi_camera=not args.skip_multi_camera,
            min_cameras=args.min_cameras,
            n_workers=args.n_workers,
            show_tqdm=True,
            rejected_log_csv=str(out_dir / f"rejected_pre_filter_{mag_bin_tag}.csv")
        )

        print(f"\nKept {len(df_filtered)}/{len(df_manifest)} sources after pre-filtering")
        print(f"Saving filtered manifest to {filtered_file}")
        safe_write_parquet(df_filtered, filtered_file)
    else:
        print(f"\nLoading existing filtered manifest from {filtered_file}")
        df_filtered = pd.read_parquet(filtered_file)
        print(f"Loaded {len(df_filtered)} filtered sources")

    # Step 3: Construct file paths (use full dat_path for events.py input)
    file_col = "dat_path" if "dat_path" in df_filtered.columns else "path"

    file_paths = df_filtered[file_col].tolist()

    if not file_paths:
        print("\nNo sources to process after filtering!")
        return

    # Step 4: Call events.py with the filtered paths in batches, with resume support
    print(f"\nPreparing to run events.py on {len(file_paths)} light curves...")

    # Write paths to temp file for events.py to consume
    paths_file = out_dir / f"filtered_paths_{mag_bin_tag}.txt"
    with open(paths_file, "w") as f:
        for path in file_paths:
            f.write(f"{path}\n")

    # Resume logic: skip paths already recorded in events checkpoint log if present
    base_output = events_output or (out_dir / "lc_events_results.csv")
    checkpoint_log = base_output.with_name(f"{base_output.stem}_PROCESSED.txt")
    processed_paths: set[str] = set()
    if checkpoint_log.exists():
        try:
            with open(checkpoint_log, "r") as f:
                processed_paths = {line.strip() for line in f if line.strip()}
            print(f"Checkpoint detected, skipping {len(processed_paths)} already-processed paths")
        except Exception as e:
            print(f"Warning: could not read checkpoint log {checkpoint_log}: {e}")

    remaining = [p for p in file_paths if str(p) not in processed_paths]
    if not remaining:
        print("All paths already processed according to checkpoint. Exiting.")
        return

    # Batch and run
    batch_size = max(1, args.batch_size)
    total_batches = (len(remaining) + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(len(remaining), start + batch_size)
        batch_paths = remaining[start:end]

        print(f"\nRunning batch {batch_idx + 1}/{total_batches} ({len(batch_paths)} LCs)...")

        events_cmd = [
            "python", "-m", "events",
            *events_args,
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

    print("\nAll batches completed.")


if __name__ == "__main__":
    main()
