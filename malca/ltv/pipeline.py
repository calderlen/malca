"""
LTV Full Pipeline Integration — Optimized for Scale.

Combines all LTV modules into a complete slowly varying source detection pipeline.

CRITICAL OPTIMIZATION: Pipeline stages run in optimal order:
1. Vectorized filters FIRST (instant, reduce 17M → ~36K)
2. Gaia TAP filters (batch queries on reduced set)
3. Catalog crossmatches (parallel on filtered candidates only)
4. NEOWISE extraction (parallel on filtered candidates only)
5. Extinction correction (vectorized)

This ordering is crucial for performance:
- Running crossmatch before filtering = weeks
- Running crossmatch after filtering = hours
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from malca.ltv.filter import (
    apply_all_filters,
    filter_slope_threshold,
    filter_max_diff_threshold,
    filter_south_pole,
    filter_high_proper_motion,
    filter_bright_star_artifacts,
)
from malca.ltv.crossmatch import (
    crossmatch_all_catalogs,
    crossmatch_gaia_dr3,
    crossmatch_vsx,
    crossmatch_milliquas,
    query_simbad_classification,
)
from malca.ltv.neowise import extract_neowise_trends


# =============================================================================
# EXTINCTION CORRECTION
# =============================================================================

def apply_extinction_correction(
    df: pd.DataFrame,
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Apply dust extinction correction using dustmaps3d.
    Vectorized operation — fast on any size.
    """
    from malca.characterize import get_dust_extinction
    
    if verbose:
        print("Applying extinction correction...")
    
    df = get_dust_extinction(df)
    
    if verbose:
        n_with_av = (df["A_v_3d"] > 0).sum() if "A_v_3d" in df.columns else 0
        print(f"[apply_extinction_correction] {n_with_av}/{len(df)} sources have A_V > 0")
    
    return df


# =============================================================================
# FULL PIPELINE
# =============================================================================

def run_full_pipeline(
    df: pd.DataFrame,
    *,
    # Filtering thresholds
    min_slope: float = 0.03,
    min_diff: float = 0.3,
    min_dec: float = -88.0,
    max_pm: float = 100.0,
    # Pipeline stages
    run_filters: bool = True,
    run_crossmatch: bool = True,
    run_neowise: bool = True,
    run_extinction: bool = True,
    # Crossmatch options
    include_gaia_dr3: bool = True,
    include_gaia_alerts: bool = True,
    include_vsx: bool = True,
    include_milliquas: bool = True,
    include_simbad: bool = True,
    match_radius_arcsec: float = 3.0,
    # Parallel processing
    n_workers: int = 8,
    chunk_size: int = 5000,
    # Output
    log_csv: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the complete LTV pipeline — optimized for 17M+ sources.
    
    CRITICAL: Pipeline runs in optimal order to minimize API calls:
    1. Vectorized filters (instant on 17M sources)
    2. Gaia TAP filters (batch on reduced set)
    3. Catalog crossmatches (parallel on filtered candidates)
    4. NEOWISE extraction (parallel on filtered candidates)
    5. Extinction correction (vectorized)
    
    For 17M sources with paper thresholds (~0.4% pass rate):
    - Filtering: ~36K candidates remain
    - Crossmatch: ~hours (not weeks) because of reduced set
    """
    n0 = len(df)
    
    if verbose:
        print("=" * 60)
        print("LTV FULL PIPELINE — Optimized for Scale")
        print("=" * 60)
        print(f"Input: {n0:,} sources")
        print(f"Workers: {n_workers}, Chunk size: {chunk_size}")
        print()
    
    # =========================================================================
    # Stage 1: Filtering (MUST run first)
    # =========================================================================
    if run_filters:
        if verbose:
            print("-" * 60)
            print("STAGE 1: FILTERING")
            print("-" * 60)
        
        df = apply_all_filters(
            df,
            min_slope=min_slope,
            min_diff=min_diff,
            min_dec=min_dec,
            max_pm=max_pm,
            query_gaia=True,
            chunk_size=chunk_size,
            n_workers=n_workers,
            verbose=verbose,
            log_csv=log_csv,
        )
        
        if verbose:
            reduction = (1 - len(df)/n0) * 100
            print(f"\n→ After filtering: {len(df):,} candidates ({reduction:.1f}% reduction)")
            print()
    
    if df.empty:
        if verbose:
            print("No sources remaining after filtering")
        return df
    
    # =========================================================================
    # Stage 2: Catalog crossmatch (only on filtered candidates)
    # =========================================================================
    if run_crossmatch:
        if verbose:
            print("-" * 60)
            print("STAGE 2: CATALOG CROSSMATCH")
            print("-" * 60)
            print(f"Crossmatching {len(df):,} filtered candidates")
            print()
        
        df = crossmatch_all_catalogs(
            df,
            include_gaia_dr3=include_gaia_dr3,
            include_gaia_alerts=include_gaia_alerts,
            include_vsx=include_vsx,
            include_milliquas=include_milliquas,
            include_simbad=include_simbad,
            match_radius_arcsec=match_radius_arcsec,
            n_workers=n_workers,
            verbose=verbose,
        )
        print()
    
    # =========================================================================
    # Stage 3: NEOWISE extraction (only on filtered candidates)
    # =========================================================================
    if run_neowise:
        if verbose:
            print("-" * 60)
            print("STAGE 3: NEOWISE EXTRACTION")
            print("-" * 60)
        
        df = extract_neowise_trends(
            df,
            n_workers=n_workers,
            verbose=verbose,
        )
        print()
    
    # =========================================================================
    # Stage 4: Extinction correction
    # =========================================================================
    if run_extinction:
        if verbose:
            print("-" * 60)
            print("STAGE 4: EXTINCTION CORRECTION")
            print("-" * 60)
        
        df = apply_extinction_correction(df, verbose=verbose)
        print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    if verbose:
        print("=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Final: {len(df):,} candidates ({len(df)/n0*100:.2f}% of input)")
        print()
        
        # Summary statistics
        if "vsx_name" in df.columns:
            n_vsx = df["vsx_name"].notna().sum()
            pct = n_vsx/len(df)*100 if len(df) > 0 else 0
            print(f"Previously classified (VSX): {n_vsx:,} ({pct:.1f}%)")
        
        if "milliquas_name" in df.columns:
            n_agn = df["milliquas_name"].notna().sum()
            print(f"AGN (MILLIQUAS): {n_agn:,}")
        
        if "ls_fap" in df.columns:
            n_periodic = (df["ls_fap"] < 0.1).sum()
            pct = n_periodic/len(df)*100 if len(df) > 0 else 0
            print(f"Periodic sources (FAP < 0.1): {n_periodic:,} ({pct:.1f}%)")
        
        if "neowise_n_epochs" in df.columns:
            n_neowise = (df["neowise_n_epochs"] > 0).sum()
            pct = n_neowise/len(df)*100 if len(df) > 0 else 0
            print(f"With NEOWISE data: {n_neowise:,} ({pct:.1f}%)")
        
        print("=" * 60)
    
    return df


# =============================================================================
# CLI SUBCOMMAND
# =============================================================================

def add_pipeline_args(parser):
    """Add pipeline arguments to argparse."""
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV/Parquet from ltv.core processing",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV/Parquet for pipeline results",
    )
    parser.add_argument(
        "--min-slope",
        type=float,
        default=0.03,
        help="Minimum |Slope| threshold (mag/yr)",
    )
    parser.add_argument(
        "--min-diff",
        type=float,
        default=0.3,
        help="Minimum |max diff| threshold (mag)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=8,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Chunk size for batch queries",
    )
    parser.add_argument(
        "--skip-filters",
        action="store_true",
        help="Skip filtering stage",
    )
    parser.add_argument(
        "--skip-crossmatch",
        action="store_true",
        help="Skip catalog crossmatch stage",
    )
    parser.add_argument(
        "--skip-neowise",
        action="store_true",
        help="Skip NEOWISE extraction stage",
    )
    parser.add_argument(
        "--skip-extinction",
        action="store_true",
        help="Skip extinction correction stage",
    )
    parser.add_argument(
        "--log-rejections",
        type=str,
        default=None,
        help="Log rejected sources to this CSV",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress",
    )
    return parser


def run_pipeline_cli(args):
    """Run pipeline from CLI arguments."""
    # Load input
    input_path = Path(args.input)
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    
    print(f"Loaded {len(df):,} sources from {input_path}")
    
    # Run pipeline
    df = run_full_pipeline(
        df,
        min_slope=args.min_slope,
        min_diff=args.min_diff,
        run_filters=not args.skip_filters,
        run_crossmatch=not args.skip_crossmatch,
        run_neowise=not args.skip_neowise,
        run_extinction=not args.skip_extinction,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
        log_csv=args.log_rejections,
        verbose=args.verbose,
    )
    
    # Save output
    output_path = Path(args.output)
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    
    print(f"Saved {len(df):,} candidates to {output_path}")
    return df
