"""
Post-filters that run AFTER events.py.
Most filters depend only on the output columns from events.py; the camera
median validation also reads per-camera stats from .raw2 files via path.

Filters:
7. filter_posterior_strength - require strong Bayes factors
8. filter_event_probability - require high event probabilities
9. filter_run_robustness - require sufficient run count and points
10. filter_morphology - require specific morphology with good BIC
11. validate_camera_medians - require camera median agreement from .raw2

Required input columns (from events.py):
    dip_bayes_factor, jump_bayes_factor,
    dip_max_log_bf_local, jump_max_log_bf_local,
    dip_max_event_prob, jump_max_event_prob,
    dip_run_count, jump_run_count,
    dip_max_run_points, jump_max_run_points,
    dip_max_run_cameras, jump_max_run_cameras,
    dip_best_morph, jump_best_morph,
    dip_best_delta_bic, jump_best_delta_bic,
    path (for logging and camera median validation)
"""

from __future__ import annotations
from pathlib import Path
from time import perf_counter
import re
import pandas as pd
import numpy as np
from tqdm.auto import tqdm


def log_rejections(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    filter_name: str,
    log_csv: str | Path | None,
) -> None:
    """
    Log rejected candidates to a CSV file.
    Assumes 'path' column exists (from events.py).
    """
    if log_csv is None:
        return

    before_ids = set(df_before["path"].astype(str))
    after_ids = set(df_after["path"].astype(str))
    rejected = sorted(before_ids - after_ids)
    if not rejected:
        return

    log_path = Path(log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df_log = pd.DataFrame({"path": rejected, "filter": filter_name})
    df_log.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)


def filter_posterior_strength(
    df: pd.DataFrame,
    *,
    min_bayes_factor: float = 10.0,
    require_finite_local_bf: bool = True,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Require dip_bayes_factor or jump_bayes_factor > threshold.
    Optionally require dip_max_log_bf_local or jump_max_log_bf_local to be finite.
    """
    n0 = len(df)
    pbar = tqdm(total=2, desc="filter_posterior_strength", leave=False) if show_tqdm else None

    # At least one of dip or jump BF must exceed threshold
    mask = (df["dip_bayes_factor"].fillna(0) > min_bayes_factor) | \
           (df["jump_bayes_factor"].fillna(0) > min_bayes_factor)

    # Require finite local BF if requested
    if require_finite_local_bf:
        is_finite_dip = df["dip_max_log_bf_local"].notna() & np.isfinite(df["dip_max_log_bf_local"])
        is_finite_jump = df["jump_max_log_bf_local"].notna() & np.isfinite(df["jump_max_log_bf_local"])
        mask &= (is_finite_dip | is_finite_jump)

    out = df.loc[mask].reset_index(drop=True)

    if pbar:
        pbar.update(1)

    if show_tqdm and verbose:
        tqdm.write(f"[filter_posterior_strength] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_posterior_strength", rejected_log_csv)

    if pbar:
        pbar.update(1)
        pbar.close()

    return out


# =============================================================================
# Filter 8: Event probability
# ============================================================================

def filter_event_probability(
    df: pd.DataFrame,
    *,
    min_event_prob: float = 0.5,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Keep light curves with dip_max_event_prob or jump_max_event_prob > threshold.
    """
    n0 = len(df)
    pbar = tqdm(total=2, desc="filter_event_probability", leave=False) if show_tqdm else None

    mask = (df["dip_max_event_prob"].fillna(0) > min_event_prob) | \
           (df["jump_max_event_prob"].fillna(0) > min_event_prob)

    out = df.loc[mask].reset_index(drop=True)

    if pbar:
        pbar.update(1)

    if show_tqdm and verbose:
        tqdm.write(f"[filter_event_probability] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_event_probability", rejected_log_csv)

    if pbar:
        pbar.update(1)
        pbar.close()

    return out


# =============================================================================
# Filter 9: Run robustness
# =============================================================================

def filter_run_robustness(
    df: pd.DataFrame,
    *,
    min_run_count: int = 1,
    min_run_points: int = 2,
    min_run_cameras: int = 2,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Require dip_run_count or jump_run_count >= min_run_count.
    Require dip_max_run_points or jump_max_run_points >= min_run_points.
    Require dip_max_run_cameras or jump_max_run_cameras >= min_run_cameras.
    """
    n0 = len(df)
    pbar = tqdm(total=2, desc="filter_run_robustness", leave=False) if show_tqdm else None

    # Check run counts
    dip_count_ok = df["dip_run_count"].fillna(0) >= min_run_count
    jump_count_ok = df["jump_run_count"].fillna(0) >= min_run_count

    # Check run points
    dip_points_ok = df["dip_max_run_points"].fillna(0) >= min_run_points
    jump_points_ok = df["jump_max_run_points"].fillna(0) >= min_run_points

    # Check run cameras
    dip_cams_ok = df["dip_max_run_cameras"].fillna(0) >= min_run_cameras
    jump_cams_ok = df["jump_max_run_cameras"].fillna(0) >= min_run_cameras

    dip_ok = dip_count_ok & dip_points_ok & dip_cams_ok
    jump_ok = jump_count_ok & jump_points_ok & jump_cams_ok

    mask = dip_ok | jump_ok
    out = df.loc[mask].reset_index(drop=True)

    if pbar:
        pbar.update(1)

    if show_tqdm and verbose:
        tqdm.write(f"[filter_run_robustness] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_run_robustness", rejected_log_csv)

    if pbar:
        pbar.update(1)
        pbar.close()

    return out


# =============================================================================
# Filter 10: Morphology
# =============================================================================

def filter_morphology(
    df: pd.DataFrame,
    *,
    dip_morphology: str = "gaussian",
    jump_morphology: str = "paczynski",
    min_delta_bic: float = 10.0,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Keep runs whose best morphology is 'gaussian' for dips or 'paczynski' for jumps,
    with dip_best_delta_bic/jump_best_delta_bic >= threshold to reject noise-like runs.
    """
    n0 = len(df)
    pbar = tqdm(total=2, desc="filter_morphology", leave=False) if show_tqdm else None

    # Check morphology for dips
    dip_morph_ok = (df["dip_best_morph"].fillna("").str.lower() == dip_morphology.lower()) & \
                   (df["dip_best_delta_bic"].fillna(0) >= min_delta_bic)

    # Check morphology for jumps
    jump_morph_ok = (df["jump_best_morph"].fillna("").str.lower() == jump_morphology.lower()) & \
                    (df["jump_best_delta_bic"].fillna(0) >= min_delta_bic)

    mask = dip_morph_ok | jump_morph_ok
    out = df.loc[mask].reset_index(drop=True)

    if pbar:
        pbar.update(1)

    if show_tqdm and verbose:
        tqdm.write(f"[filter_morphology] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_morphology", rejected_log_csv)

    if pbar:
        pbar.update(1)
        pbar.close()

    return out


# =============================================================================
# Validation filters (expensive checks, run after event detection)
# =============================================================================

RAW_STATS_COLUMNS = [
    "camera",
    "median",
    "sig1_low",
    "sig1_high",
    "p90_low",
    "p90_high",
]


def _parse_mag_bin_range(mag_bin: str | None) -> tuple[float, float] | None:
    if not mag_bin:
        return None
    token = mag_bin.strip().replace("-", "_")
    parts = token.split("_")
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def _mag_bin_range_from_path(path: Path) -> tuple[float, float] | None:
    match = re.search(r"(\d+(?:\.\d+)?_\d+(?:\.\d+)?)", str(path))
    if not match:
        return None
    return _parse_mag_bin_range(match.group(1))


def _find_raw_stats_path(path: Path) -> Path | None:
    path = Path(path)
    if path.suffix.lower() == ".raw2":
        return path if path.exists() else None
    candidate = path.with_suffix(".raw2")
    return candidate if candidate.exists() else None


def _read_raw_camera_stats(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        names=RAW_STATS_COLUMNS,
        comment="#",
        header=None,
    )
    for col in ("median", "sig1_low", "sig1_high", "p90_low", "p90_high"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["median"].notna()].reset_index(drop=True)
    return df


def validate_camera_medians(
    df: pd.DataFrame,
    *,
    max_median_spread: float = 0.3,
    mag_bin: str | None = None,
    mag_tolerance: float = 0.5,
    allow_missing_raw: bool = False,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Validate candidates using per-camera median magnitudes from .raw2 files.

    Rejects candidates where camera medians disagree strongly, or where the
    camera medians fall outside the expected magnitude bin (with tolerance).
    """
    if "path" not in df.columns:
        raise KeyError("validate_camera_medians requires a 'path' column")

    mag_bin_range = _parse_mag_bin_range(mag_bin)

    min_medians = []
    max_medians = []
    mean_medians = []
    median_spreads = []
    n_cameras = []
    valid_flags = []
    reasons = []
    mag_min_bounds = []
    mag_max_bounds = []
    keep_flags = []

    paths = df["path"].astype(str).tolist()
    iterator = enumerate(paths)
    if show_tqdm:
        iterator = tqdm(iterator, total=len(paths), desc="validate_camera_medians", leave=False)

    for _, path_str in iterator:
        path = Path(path_str)
        raw_path = _find_raw_stats_path(path)

        if raw_path is None:
            min_medians.append(np.nan)
            max_medians.append(np.nan)
            mean_medians.append(np.nan)
            median_spreads.append(np.nan)
            n_cameras.append(0)
            valid_flags.append(False)
            reasons.append("missing_raw")
            mag_min_bounds.append(np.nan)
            mag_max_bounds.append(np.nan)
            keep_flags.append(bool(allow_missing_raw))
            continue

        try:
            stats = _read_raw_camera_stats(raw_path)
        except Exception:
            stats = pd.DataFrame()

        if stats.empty or "median" not in stats.columns:
            min_medians.append(np.nan)
            max_medians.append(np.nan)
            mean_medians.append(np.nan)
            median_spreads.append(np.nan)
            n_cameras.append(0)
            valid_flags.append(False)
            reasons.append("empty_raw_stats")
            mag_min_bounds.append(np.nan)
            mag_max_bounds.append(np.nan)
            keep_flags.append(bool(allow_missing_raw))
            continue

        medians = stats["median"].to_numpy(dtype=float)
        med_min = float(np.nanmin(medians)) if medians.size else np.nan
        med_max = float(np.nanmax(medians)) if medians.size else np.nan
        med_mean = float(np.nanmean(medians)) if medians.size else np.nan
        spread = float(med_max - med_min) if np.isfinite(med_min) and np.isfinite(med_max) else np.nan

        row_range = mag_bin_range or _mag_bin_range_from_path(path)
        if row_range is not None:
            mag_min = float(row_range[0] - mag_tolerance)
            mag_max = float(row_range[1] + mag_tolerance)
        else:
            mag_min = np.nan
            mag_max = np.nan

        row_valid = True
        row_reasons = []

        if np.isfinite(spread) and max_median_spread is not None:
            if spread > max_median_spread:
                row_valid = False
                row_reasons.append(f"spread>{max_median_spread}")

        if np.isfinite(med_min) and np.isfinite(med_max) and np.isfinite(mag_min) and np.isfinite(mag_max):
            if med_min < mag_min or med_max > mag_max:
                row_valid = False
                row_reasons.append("outside_mag_bin")

        min_medians.append(med_min)
        max_medians.append(med_max)
        mean_medians.append(med_mean)
        median_spreads.append(spread)
        n_cameras.append(int(stats["camera"].nunique()))
        valid_flags.append(row_valid)
        reasons.append(";".join(row_reasons) if row_reasons else "")
        mag_min_bounds.append(mag_min)
        mag_max_bounds.append(mag_max)
        keep_flags.append(row_valid)

    df_out = df.copy()
    df_out["camera_median_min"] = min_medians
    df_out["camera_median_max"] = max_medians
    df_out["camera_median_mean"] = mean_medians
    df_out["camera_median_spread"] = median_spreads
    df_out["camera_median_n_cameras"] = n_cameras
    df_out["camera_median_valid"] = valid_flags
    df_out["camera_median_reason"] = reasons
    df_out["camera_median_mag_min"] = mag_min_bounds
    df_out["camera_median_mag_max"] = mag_max_bounds

    out = df_out.loc[keep_flags].reset_index(drop=True)
    if show_tqdm and verbose:
        tqdm.write(f"[validate_camera_medians] kept {len(out)}/{len(df_out)}")
    log_rejections(df_out, out, "validate_camera_medians", rejected_log_csv)
    return out

def validate_periodicity(
    df: pd.DataFrame,
    *,
    max_power: float = 0.5,
    n_bootstrap: int = 1000,
    significance_level: float = 0.01,
    exclude_alias_periods: bool = True,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Detailed periodicity validation on candidates (like ZTF paper Section 4.5).

    Uses bootstrap Lomb-Scargle periodogram with significance testing to identify:
    - Eclipsing binaries (short periods ~1 day)
    - Rotating variables (periods ~30 days)
    - Other periodic contamination

    Much more expensive than pre-filter LSP - uses bootstrap for significance.
    Only run on detected candidates, not all sources.

    Parameters
    ----------
    df : pd.DataFrame
        Candidates from events.py (must have 'path' column)
    max_power : float
        Maximum LSP power to keep (default 0.5)
    n_bootstrap : int
        Number of bootstrap iterations for significance (default 1000)
    significance_level : float
        Significance threshold (default 0.01 = 1% like paper)
    exclude_alias_periods : bool
        Exclude known alias periods (sidereal day, lunar month, ZTF cadence)
    show_tqdm : bool
        Show progress
    rejected_log_csv : str | Path | None
        Log file for rejected candidates

    Returns
    -------
    pd.DataFrame
        Candidates without strong periodic signals

    Notes
    -----
    This implements the paper's approach:
    - Bootstrap LSP for significance levels
    - Phase-fold at best period
    - Exclude alias frequencies
    - Flag coherent periodic structure

    TODO: Implement full bootstrap LSP with light curve reading
    For now, returns input unchanged (placeholder).
    """
    n0 = len(df)

    if show_tqdm and verbose:
        tqdm.write(f"[validate_periodicity] TODO: Implement bootstrap LSP - currently placeholder")
        tqdm.write(f"[validate_periodicity] kept {len(df)}/{n0} (no filtering applied)")

    # TODO: Implement:
    # 1. Read light curves from 'path' column
    # 2. Compute LSP for each
    # 3. Bootstrap for significance levels
    # 4. Identify significant periods
    # 5. Exclude alias periods (1 day, 29.6 days, etc.)
    # 6. Phase-fold and check for coherent structure
    # 7. Flag/reject strongly periodic sources

    return df


def validate_gaia_ruwe(
    df: pd.DataFrame,
    *,
    gaia_catalog: pd.DataFrame | Path | None = None,
    max_ruwe: float = 1.4,
    flag_only: bool = True,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Validate candidates using Gaia RUWE (Renormalized Unit Weight Error).

    RUWE > 1.4 indicates potential companion (binary contamination).
    Paper identifies 5/81 candidates with high RUWE.

    Parameters
    ----------
    df : pd.DataFrame
        Candidates (must have ra_deg, dec_deg columns)
    gaia_catalog : pd.DataFrame | Path | None
        Gaia DR3 catalog with RUWE values, or path to CSV
        If None, attempts to query Gaia archive (slow)
    max_ruwe : float
        RUWE threshold (default 1.4, from paper)
    flag_only : bool
        If True, add 'ruwe' and 'high_ruwe_flag' columns but don't reject
        If False, reject sources with RUWE > max_ruwe
    show_tqdm : bool
        Show progress
    rejected_log_csv : str | Path | None
        Log file for rejected candidates

    Returns
    -------
    pd.DataFrame
        Candidates with RUWE information added

    Notes
    -----
    Paper approach:
    - RUWE ~ 1 consistent with single stars
    - RUWE > 1.4 indicates binarity
    - 5/81 candidates flagged (potential companions)
    - Still need follow-up (imaging, RV) to confirm

    TODO: Implement Gaia crossmatch
    For now, returns input unchanged (placeholder).
    """
    n0 = len(df)

    if show_tqdm and verbose:
        tqdm.write(f"[validate_gaia_ruwe] TODO: Implement Gaia RUWE crossmatch - currently placeholder")
        tqdm.write(f"[validate_gaia_ruwe] kept {len(df)}/{n0} (no filtering applied)")

    # TODO: Implement:
    # 1. Crossmatch to Gaia DR3 using ra_deg, dec_deg
    # 2. Add 'ruwe' column from Gaia
    # 3. Flag high RUWE sources (> 1.4)
    # 4. Optionally reject (if flag_only=False)

    return df


def validate_periodic_catalog(
    df: pd.DataFrame,
    *,
    catalog_path: Path | None = None,
    max_sep_arcsec: float = 3.0,
    flag_only: bool = True,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Crossmatch candidates to known periodic variable catalogs.

    Paper crossmatched to Chen+2020 ZTF periodic catalog and found
    14 matches, including 1 compelling BY Dra variable.

    Parameters
    ----------
    df : pd.DataFrame
        Candidates (must have ra_deg, dec_deg columns)
    catalog_path : Path | None
        Path to periodic catalog CSV (e.g., Chen+2020, ASAS-SN variables)
        Must have: ra, dec, period, class columns
    max_sep_arcsec : float
        Maximum separation for crossmatch (default 3 arcsec)
    flag_only : bool
        If True, add 'periodic_catalog_match' flag but don't reject
        If False, reject catalog matches
    show_tqdm : bool
        Show progress
    rejected_log_csv : str | Path | None
        Log file for rejected candidates

    Returns
    -------
    pd.DataFrame
        Candidates with periodic catalog match flags

    Notes
    -----
    Paper found:
    - 14/81 candidates matched Chen+2020 ZTF periodic catalog
    - Visual inspection showed most lacked coherent phase-folded structure
    - 1 compelling match: BY Dra variable with 2.57d period

    Suggests many catalog "periodic" sources aren't strongly periodic
    at the level detected by events.py Bayesian fitting.

    TODO: Implement catalog crossmatch
    For now, returns input unchanged (placeholder).
    """
    n0 = len(df)

    if show_tqdm and verbose:
        tqdm.write(f"[validate_periodic_catalog] TODO: Implement periodic catalog crossmatch - currently placeholder")
        tqdm.write(f"[validate_periodic_catalog] kept {len(df)}/{n0} (no filtering applied)")

    # TODO: Implement:
    # 1. Load periodic catalog (Chen+2020, ASAS-SN vars, etc.)
    # 2. Crossmatch by coordinates
    # 3. Add 'catalog_period', 'catalog_class' columns
    # 4. Flag matches
    # 5. Optionally reject (if flag_only=False)

    return df


# =============================================================================
# Main orchestration
# =============================================================================

def apply_post_filters(
    df: pd.DataFrame,
    *,
    # Filter 7: posterior strength
    apply_posterior_strength: bool = True,
    min_bayes_factor: float = 10.0,
    require_finite_local_bf: bool = True,
    # Filter 8: event probability
    apply_event_probability: bool = True,
    min_event_prob: float = 0.5,
    # Filter 9: run robustness
    apply_run_robustness: bool = True,
    min_run_count: int = 1,
    min_run_points: int = 2,
    min_run_cameras: int = 2,
    # Filter 10: morphology
    apply_morphology: bool = False,
    dip_morphology: str = "gaussian",
    jump_morphology: str = "paczynski",
    min_delta_bic: float = 10.0,
    # Validation: camera medians
    apply_camera_median_validation: bool = False,
    max_camera_median_spread: float = 0.3,
    mag_bin: str | None = None,
    mag_tolerance: float = 0.5,
    allow_missing_raw: bool = False,
    # General
    show_tqdm: bool = True,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = "rejected_post_filter.csv",
) -> pd.DataFrame:
    """
    Apply post-filters after running events.py.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe from events.py
    apply_* : bool
        Whether to apply each filter
    show_tqdm : bool
        Show progress bars
    verbose : bool
        Print per-filter summaries and totals
    rejected_log_csv : str | Path | None
        Path to log rejected candidates

    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    df_filtered = df.copy()
    n_start = len(df_filtered)

    filters = []

    if apply_posterior_strength:
        filters.append(("posterior_strength", filter_posterior_strength, {
            "min_bayes_factor": min_bayes_factor,
            "require_finite_local_bf": require_finite_local_bf,
            "show_tqdm": show_tqdm,
            "verbose": verbose,
            "rejected_log_csv": rejected_log_csv,
        }))

    if apply_event_probability:
        filters.append(("event_probability", filter_event_probability, {
            "min_event_prob": min_event_prob,
            "show_tqdm": show_tqdm,
            "verbose": verbose,
            "rejected_log_csv": rejected_log_csv,
        }))

    if apply_run_robustness:
        filters.append(("run_robustness", filter_run_robustness, {
            "min_run_count": min_run_count,
            "min_run_points": min_run_points,
            "min_run_cameras": min_run_cameras,
            "show_tqdm": show_tqdm,
            "verbose": verbose,
            "rejected_log_csv": rejected_log_csv,
        }))

    if apply_morphology:
        filters.append(("morphology", filter_morphology, {
            "dip_morphology": dip_morphology,
            "jump_morphology": jump_morphology,
            "min_delta_bic": min_delta_bic,
            "show_tqdm": show_tqdm,
            "verbose": verbose,
            "rejected_log_csv": rejected_log_csv,
        }))

    if apply_camera_median_validation:
        filters.append(("camera_medians", validate_camera_medians, {
            "max_median_spread": max_camera_median_spread,
            "mag_bin": mag_bin,
            "mag_tolerance": mag_tolerance,
            "allow_missing_raw": allow_missing_raw,
            "show_tqdm": show_tqdm,
            "verbose": verbose,
            "rejected_log_csv": rejected_log_csv,
        }))

    # Apply filters sequentially
    total_steps = len(filters)
    if total_steps > 0:
        with tqdm(total=total_steps, desc="apply_post_filters", leave=True, disable=not show_tqdm) as pbar:
            for label, func, kwargs in filters:
                n_before = len(df_filtered)
                start = perf_counter()
                df_filtered = func(df_filtered, **kwargs)
                elapsed = perf_counter() - start
                n_after = len(df_filtered)

                if verbose:
                    pbar.set_postfix_str(f"{label}: {n_before} → {n_after} ({elapsed:.2f}s)")
                else:
                    pbar.set_postfix_str("")
                pbar.update(1)

    n_end = len(df_filtered)
    if show_tqdm and verbose:
        tqdm.write(f"\n[apply_post_filters] Total: {n_start} → {n_end} ({n_end/n_start*100:.1f}% kept)")

    return df_filtered.reset_index(drop=True)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply post-filters to events.py results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python -m malca.post_filter --input results.csv --output results_filtered.csv
  python -m malca.post_filter --input results.csv --output results_filtered.csv --min-bayes-factor 20 --skip-morphology
"""
    )

    # I/O
    parser.add_argument("--input", type=Path, required=True, help="Input CSV/Parquet from events.py")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV/Parquet path")

    # Filter toggles (all enabled by default except morphology)
    parser.add_argument("--skip-posterior-strength", action="store_true", help="Skip posterior strength filter")
    parser.add_argument("--skip-event-probability", action="store_true", help="Skip event probability filter")
    parser.add_argument("--skip-run-robustness", action="store_true", help="Skip run robustness filter")
    parser.add_argument("--apply-morphology", action="store_true", help="Apply morphology filter (off by default)")
    parser.add_argument("--apply-camera-median-validation", action="store_true",
                        help="Apply camera median validation using .raw2 files (off by default)")

    # Posterior strength parameters
    parser.add_argument("--min-bayes-factor", type=float, default=10.0,
                        help="Minimum Bayes factor for posterior strength filter (default: 10)")
    parser.add_argument("--allow-infinite-local-bf", action="store_true",
                        help="Allow infinite local BF (default: require finite)")

    # Event probability parameters
    parser.add_argument("--min-event-prob", type=float, default=0.5,
                        help="Minimum event probability (default: 0.5)")

    # Run robustness parameters
    parser.add_argument("--min-run-count", type=int, default=1,
                        help="Minimum number of runs (default: 1)")
    parser.add_argument("--min-run-points", type=int, default=2,
                        help="Minimum points per run (default: 2)")
    parser.add_argument("--min-run-cameras", type=int, default=2,
                        help="Minimum cameras per run (default: 2)")

    # Morphology parameters
    parser.add_argument("--dip-morphology", type=str, default="gaussian",
                        choices=["gaussian", "paczynski"],
                        help="Required morphology for dips (default: gaussian)")
    parser.add_argument("--jump-morphology", type=str, default="paczynski",
                        choices=["gaussian", "paczynski"],
                        help="Required morphology for jumps (default: paczynski)")
    parser.add_argument("--min-delta-bic", type=float, default=10.0,
                        help="Minimum delta BIC for morphology filter (default: 10)")

    # Camera median validation parameters
    parser.add_argument("--max-camera-median-spread", type=float, default=0.3,
                        help="Max allowed spread between camera medians (default: 0.3)")
    parser.add_argument("--mag-bin", type=str, default=None,
                        help="Expected magnitude bin (e.g., 13_13.5). If omitted, try to parse from path.")
    parser.add_argument("--mag-tolerance", type=float, default=0.5,
                        help="Allowed magnitude tolerance beyond mag bin (default: 0.5)")
    parser.add_argument("--allow-missing-raw", action="store_true",
                        help="Do not reject candidates missing .raw2 stats")

    # General options
    parser.add_argument("--no-tqdm", action="store_true", help="Disable progress bars")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print per-filter summaries (default: off)")
    parser.add_argument("--no-reject-log", action="store_true", help="Disable rejection logging")
    parser.add_argument("--reject-log", type=Path, default=None,
                        help="Path to rejection log CSV (default: rejected_post_filter.csv)")

    args = parser.parse_args()

    # Load input
    input_path = args.input.expanduser()
    if input_path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    print(f"Loaded {len(df)} rows from {input_path}")

    # Determine reject log path
    if args.no_reject_log:
        reject_log = None
    elif args.reject_log:
        reject_log = args.reject_log.expanduser()
    else:
        reject_log = args.output.parent / "rejected_post_filter.csv"

    # Apply filters
    df_filtered = apply_post_filters(
        df,
        # Filter toggles
        apply_posterior_strength=not args.skip_posterior_strength,
        apply_event_probability=not args.skip_event_probability,
        apply_run_robustness=not args.skip_run_robustness,
        apply_morphology=args.apply_morphology,
        apply_camera_median_validation=args.apply_camera_median_validation,
        # Posterior strength
        min_bayes_factor=args.min_bayes_factor,
        require_finite_local_bf=not args.allow_infinite_local_bf,
        # Event probability
        min_event_prob=args.min_event_prob,
        # Run robustness
        min_run_count=args.min_run_count,
        min_run_points=args.min_run_points,
        min_run_cameras=args.min_run_cameras,
        # Morphology
        dip_morphology=args.dip_morphology,
        jump_morphology=args.jump_morphology,
        min_delta_bic=args.min_delta_bic,
        # Camera median validation
        max_camera_median_spread=args.max_camera_median_spread,
        mag_bin=args.mag_bin,
        mag_tolerance=args.mag_tolerance,
        allow_missing_raw=args.allow_missing_raw,
        # General
        show_tqdm=not args.no_tqdm,
        verbose=args.verbose,
        rejected_log_csv=reject_log,
    )

    # Save output
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() in (".parquet", ".pq"):
        df_filtered.to_parquet(output_path, index=False)
    else:
        df_filtered.to_csv(output_path, index=False)

    print(f"\nWrote {len(df_filtered)} filtered rows to {output_path}")
    print(f"Kept {len(df_filtered)}/{len(df)} ({len(df_filtered)/len(df)*100:.1f}%)")


if __name__ == "__main__":
    main()
