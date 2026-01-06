"""
Pre-filters that run BEFORE events.py.
These filters depend only on raw light curve data (dat2 files) and external catalogs.

Filters:
1. filter_sparse_lightcurves - remove LCs with insufficient time span or cadence
2. filter_periodic_candidates - remove strongly periodic sources
3. filter_vsx_match - remove known variable stars from VSX
4. filter_bright_nearby_stars - remove sources contaminated by nearby stars
5. filter_multi_camera - remove single-camera detections

Input format:
    DataFrame with columns: asas_sn_id (or id/source_id), path (to directory containing dat2 files)

    Filters will compute required stats from dat2 files on-the-fly:
    - time_span_days, points_per_day (computed from JD column)
    - ls_max_power, best_period (computed via Lomb-Scargle periodogram)
    - vsx_match_sep_arcsec, vsx_class (computed via VSX crossmatch using data in input/vsx/)
    - bns_separation_arcsec, bns_delta_mag (computed via ASAS-SN catalog crossmatch)
    - n_cameras (counted from camera# column)

    If any of these columns already exist in the input DataFrame, they will be used
    instead of recomputing.
"""

from __future__ import annotations
from pathlib import Path
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from tqdm.auto import tqdm

from utils import (
    read_lc_dat2,
    get_id_col,
    compute_time_stats,
    compute_periodogram,
    compute_n_cameras,
)
from vsx_crossmatch import propagate_asassn_coords, vsx_coords


def _compute_stats_for_row(asas_sn_id: str, dir_path: str, compute_time: bool, compute_period: bool, compute_cameras: bool) -> dict:
    """
    Helper function for parallel processing. Computes requested stats for a single light curve.
    Returns a dict with requested stats.
    """
    result = {"asas_sn_id": asas_sn_id}

    try:
        df_g, df_v = read_lc_dat2(asas_sn_id, dir_path)
        df_lc = pd.concat([df_g, df_v], ignore_index=True) if not df_g.empty or not df_v.empty else pd.DataFrame()

        if compute_time:
            time_stats = compute_time_stats(df_lc)
            result.update(time_stats)

        if compute_period:
            period_stats = compute_periodogram(df_lc)
            result.update(period_stats)

        if compute_cameras:
            result["n_cameras"] = compute_n_cameras(df_lc)

    except Exception as e:
        # If there's an error, return default values
        if compute_time:
            result["time_span_days"] = 0.0
            result["points_per_day"] = 0.0
        if compute_period:
            result["ls_max_power"] = 0.0
            result["best_period"] = np.nan
        if compute_cameras:
            result["n_cameras"] = 0

    return result


def _compute_stats_parallel(df: pd.DataFrame, id_col: str, path_col: str, compute_time: bool = False,
                            compute_period: bool = False, compute_cameras: bool = False,
                            n_workers: int = 4, show_tqdm: bool = False) -> pd.DataFrame:
    """
    Compute stats for all rows in parallel using ProcessPoolExecutor.
    Returns a copy of df with new columns added.
    """
    df_with_stats = df.copy()

    # Initialize columns
    if compute_time:
        df_with_stats["time_span_days"] = 0.0
        df_with_stats["points_per_day"] = 0.0
    if compute_period:
        df_with_stats["ls_max_power"] = 0.0
        df_with_stats["best_period"] = np.nan
    if compute_cameras:
        df_with_stats["n_cameras"] = 0

    # Prepare tasks
    tasks = []
    for idx, row in df.iterrows():
        asas_sn_id = str(row[id_col])
        dir_path = str(row[path_col])
        tasks.append((idx, asas_sn_id, dir_path))

    # Process in parallel
    pbar = tqdm(total=len(tasks), desc="Computing stats (parallel)", leave=False, disable=not show_tqdm)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_compute_stats_for_row, asas_sn_id, dir_path, compute_time, compute_period, compute_cameras): idx
            for idx, asas_sn_id, dir_path in tasks
        }

        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()

            if compute_time:
                df_with_stats.loc[idx, "time_span_days"] = result["time_span_days"]
                df_with_stats.loc[idx, "points_per_day"] = result["points_per_day"]
            if compute_period:
                df_with_stats.loc[idx, "ls_max_power"] = result["ls_max_power"]
                df_with_stats.loc[idx, "best_period"] = result["best_period"]
            if compute_cameras:
                df_with_stats.loc[idx, "n_cameras"] = result["n_cameras"]

            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()

    return df_with_stats


def log_rejections(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    filter_name: str,
    log_csv: str | Path | None,
) -> None:
    """
    Log rejected candidates to a CSV file.
    """
    if log_csv is None:
        return

    # Try to find an ID column
    id_col = None
    for candidate in ["path", "asas_sn_id", "id", "source_id"]:
        if candidate in df_before.columns:
            id_col = candidate
            break

    if id_col is None:
        return

    before_ids = set(df_before[id_col].astype(str))
    after_ids = set(df_after[id_col].astype(str))
    rejected = sorted(before_ids - after_ids)
    if not rejected:
        return

    log_path = Path(log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df_log = pd.DataFrame({id_col: rejected, "filter": filter_name})
    df_log.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)


def filter_sparse_lightcurves(
    df: pd.DataFrame,
    *,
    min_time_span: float = 100.0,
    min_points_per_day: float = 0.05,
    show_tqdm: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Remove candidates with:
    - less than min_time_span days of observation (default 100)
    - less than min_points_per_day on average (default 0.05 = 1 point per 20 days)

    If time_span_days and points_per_day columns exist, use them.
    Otherwise, read dat2 files and compute on-the-fly.
    """
    n0 = len(df)

    # Check if we need to compute stats
    need_compute = ("time_span_days" not in df.columns) or ("points_per_day" not in df.columns)

    if need_compute:
        id_col = get_id_col(df)
        path_col = "path" if "path" in df.columns else None

        if path_col is None:
            raise ValueError("Need 'path' column to read dat2 files")

        df_with_stats = df.copy()
        df_with_stats["time_span_days"] = 0.0
        df_with_stats["points_per_day"] = 0.0

        pbar = tqdm(total=len(df), desc="filter_sparse_lightcurves (computing stats)", leave=False, disable=not show_tqdm)
        for idx, row in df.iterrows():
            asas_sn_id = str(row[id_col])
            dir_path = str(row[path_col])

            df_g, df_v = read_lc_dat2(asas_sn_id, dir_path)
            df_lc = pd.concat([df_g, df_v], ignore_index=True) if not df_g.empty or not df_v.empty else pd.DataFrame()

            stats = compute_time_stats(df_lc)
            df_with_stats.loc[idx, "time_span_days"] = stats["time_span_days"]
            df_with_stats.loc[idx, "points_per_day"] = stats["points_per_day"]

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        df = df_with_stats

    # Apply filter
    mask = (df["time_span_days"] >= min_time_span) & \
           (df["points_per_day"] >= min_points_per_day)
    out = df.loc[mask].reset_index(drop=True)

    if show_tqdm:
        tqdm.write(f"[filter_sparse_lightcurves] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_sparse_lightcurves", rejected_log_csv)

    return out


def filter_periodic_candidates(
    df: pd.DataFrame,
    *,
    max_power: float = 0.5,
    min_period: float | None = None,
    max_period: float | None = None,
    show_tqdm: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Remove candidates with strong periodicity.

    If ls_max_power and best_period columns exist, use them.
    Otherwise, read dat2 files and compute Lomb-Scargle periodogram on-the-fly.
    """
    n0 = len(df)

    # Check if we need to compute periodogram
    need_compute = ("ls_max_power" not in df.columns) or ("best_period" not in df.columns)

    if need_compute:
        id_col = get_id_col(df)
        path_col = "path" if "path" in df.columns else None

        if path_col is None:
            raise ValueError("Need 'path' column to read dat2 files")

        df_with_stats = df.copy()
        df_with_stats["ls_max_power"] = 0.0
        df_with_stats["best_period"] = np.nan

        pbar = tqdm(total=len(df), desc="filter_periodic_candidates (computing periodogram)", leave=False, disable=not show_tqdm)
        for idx, row in df.iterrows():
            asas_sn_id = str(row[id_col])
            dir_path = str(row[path_col])

            df_g, df_v = read_lc_dat2(asas_sn_id, dir_path)
            df_lc = pd.concat([df_g, df_v], ignore_index=True) if not df_g.empty or not df_v.empty else pd.DataFrame()

            stats = compute_periodogram(df_lc)
            df_with_stats.loc[idx, "ls_max_power"] = stats["ls_max_power"]
            df_with_stats.loc[idx, "best_period"] = stats["best_period"]

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        df = df_with_stats

    # Apply filter
    mask = df["ls_max_power"] <= max_power

    if min_period is not None:
        mask &= (df["best_period"] >= min_period) | df["best_period"].isna()
    if max_period is not None:
        mask &= (df["best_period"] <= max_period) | df["best_period"].isna()

    out = df.loc[mask].reset_index(drop=True)

    if show_tqdm:
        tqdm.write(f"[filter_periodic_candidates] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_periodic_candidates", rejected_log_csv)

    return out


def filter_vsx_match(
    df: pd.DataFrame,
    *,
    max_sep_arcsec: float = 3.0,
    exclude_classes: list[str] | None = None,
    vsx_catalog_csv: str | Path = "results_crossmatch/vsx_cleaned_20250926_1557.csv",
    show_tqdm: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Filter candidates based on VSX crossmatch results.
    Remove candidates that match known variables with specified classes.

    If vsx_match_sep_arcsec and vsx_class columns exist, use them.
    Otherwise, perform crossmatch using VSX catalog from results_crossmatch/vsx_cleaned_*.csv

    Required columns for crossmatch: ra_deg, dec_deg, pm_ra, pm_dec
    Raises ValueError if required columns or catalog file are missing.
    """
    n0 = len(df)

    # Check if we need to compute crossmatch
    need_compute = ("vsx_match_sep_arcsec" not in df.columns) or ("vsx_class" not in df.columns)

    if need_compute:
        # Load VSX catalog
        vsx_path = Path(vsx_catalog_csv)
        vsx_df = pd.read_csv(vsx_path)
        vsx_df = vsx_df.dropna(subset=["ra", "dec"]).reset_index(drop=True)

        # Verify input has required columns
        required_cols = ["ra_deg", "dec_deg", "pm_ra", "pm_dec"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for VSX crossmatch: {missing_cols}")

        # Perform crossmatch
        df_with_vsx = df.copy()
        coords_asassn = propagate_asassn_coords(df)
        coords_vsx = vsx_coords(vsx_df)

        idx_vsx, sep2d, _ = coords_asassn.match_to_catalog_sky(coords_vsx)

        df_with_vsx["vsx_match_sep_arcsec"] = sep2d.arcsec
        df_with_vsx["vsx_class"] = pd.Series(index=df_with_vsx.index, dtype=object)

        mask_matched = sep2d < (max_sep_arcsec * u.arcsec)
        if "class" in vsx_df.columns:
            df_with_vsx.loc[mask_matched, "vsx_class"] = vsx_df.loc[idx_vsx[mask_matched], "class"].values

        df = df_with_vsx

    # Apply filter
    has_match = df["vsx_match_sep_arcsec"].fillna(999) <= max_sep_arcsec
    is_excluded_type = df["vsx_class"].fillna("").isin(exclude_classes) if exclude_classes else False
    mask = ~(has_match & is_excluded_type)
    out = df.loc[mask].reset_index(drop=True)

    if show_tqdm:
        tqdm.write(f"[filter_vsx_match] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_vsx_match", rejected_log_csv)

    return out


def filter_bright_nearby_stars(
    df: pd.DataFrame,
    *,
    max_separation_arcsec: float = 5.0,
    max_mag_diff: float = 3.0,
    asassn_catalog_csv: str | Path = "results_crossmatch/asassn_index_masked_concat_cleaned_20250926_1557.csv",
    show_tqdm: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Filter out candidates with bright nearby stars that could contaminate photometry.

    If bns_separation_arcsec and bns_delta_mag columns exist, use them.
    Otherwise, perform catalog crossmatch using ASAS-SN index from results_crossmatch/asassn_index_*.csv

    Required columns for crossmatch: ra_deg, dec_deg, and a magnitude column
    (gaia_mag, pstarrs_g_mag, mag, mean_mag, or median_mag)

    For each candidate, finds all catalog sources within max_separation_arcsec and checks
    if any are brighter by less than max_mag_diff. Such candidates are rejected.

    The ASAS-SN index file contains: asas_sn_id, ra_deg, dec_deg, pm_ra, pm_dec,
    gaia_mag, pstarrs_g_mag, pstarrs_r_mag, and other photometric columns.

    Raises ValueError if required columns or catalog file are missing.
    """
    n0 = len(df)

    # Check if we need to compute crossmatch
    need_compute = ("bns_separation_arcsec" not in df.columns) or ("bns_delta_mag" not in df.columns)

    if need_compute:
        # Load ASAS-SN catalog
        catalog_path = Path(asassn_catalog_csv)
        catalog = pd.read_csv(catalog_path)

        # Verify required columns in input
        required_cols = ["ra_deg", "dec_deg"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for BNS filter: {missing_cols}")

        # Find magnitude column from input
        mag_col = None
        for col in ["gaia_mag", "pstarrs_g_mag", "mag", "mean_mag", "median_mag", "photometry_median_mag"]:
            if col in df.columns:
                mag_col = col
                break

        if mag_col is None:
            raise ValueError(f"No magnitude column found in input DataFrame. Expected one of: gaia_mag, pstarrs_g_mag, mag, mean_mag, median_mag")

        # Build coordinate arrays
        coords_candidates = SkyCoord(
            ra=df["ra_deg"].values * u.deg,
            dec=df["dec_deg"].values * u.deg,
        )

        coords_catalog = SkyCoord(
            ra=catalog["ra_deg"].values * u.deg,
            dec=catalog["dec_deg"].values * u.deg,
        )

        # For each candidate, find minimum separation and mag diff to nearby sources
        df_with_bns = df.copy()
        df_with_bns["bns_separation_arcsec"] = 999.0
        df_with_bns["bns_delta_mag"] = 999.0

        pbar = tqdm(total=len(df), desc="filter_bright_nearby_stars (crossmatch)", leave=False, disable=not show_tqdm)

        for idx, row in df.iterrows():
            coord = coords_candidates[idx]
            candidate_mag = row[mag_col]

            if not np.isfinite(candidate_mag):
                if pbar:
                    pbar.update(1)
                continue

            # Find all catalog sources within max_separation_arcsec
            seps = coord.separation(coords_catalog)
            nearby_mask = seps < (max_separation_arcsec * u.arcsec)

            if not nearby_mask.any():
                if pbar:
                    pbar.update(1)
                continue

            # Get nearby catalog sources
            nearby_catalog = catalog[nearby_mask].copy()
            nearby_seps = seps[nearby_mask].arcsec

            # Find magnitude column in catalog
            catalog_mag_col = None
            for col in ["gaia_mag", "pstarrs_g_mag", "pstarrs_r_mag", "mag", "mean_mag", "median_mag", "g_mag", "v_mag"]:
                if col in nearby_catalog.columns:
                    catalog_mag_col = col
                    break

            if catalog_mag_col is None:
                raise ValueError(f"No magnitude column found in catalog. Expected one of: gaia_mag, pstarrs_g_mag, pstarrs_r_mag, mag")

            # Compute mag differences (positive means catalog star is brighter)
            mag_diffs = candidate_mag - nearby_catalog[catalog_mag_col].values

            # Find the closest bright nearby star
            bright_mask = (mag_diffs > 0) & np.isfinite(mag_diffs)
            if bright_mask.any():
                min_idx = np.argmin(nearby_seps[bright_mask])
                df_with_bns.loc[idx, "bns_separation_arcsec"] = nearby_seps[bright_mask][min_idx]
                df_with_bns.loc[idx, "bns_delta_mag"] = mag_diffs[bright_mask][min_idx]

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        df = df_with_bns

    # Apply filter
    has_bns = (df["bns_separation_arcsec"].fillna(999) <= max_separation_arcsec) & \
              (df["bns_delta_mag"].fillna(999) <= max_mag_diff)
    mask = ~has_bns
    out = df.loc[mask].reset_index(drop=True)

    if show_tqdm:
        tqdm.write(f"[filter_bright_nearby_stars] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_bright_nearby_stars", rejected_log_csv)

    return out


def filter_multi_camera(
    df: pd.DataFrame,
    *,
    min_cameras: int = 2,
    show_tqdm: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Remove candidates that were only detected on one camera.

    If n_cameras column exists, use it.
    Otherwise, read dat2 files and count unique cameras on-the-fly.
    """
    n0 = len(df)

    # Check if we need to compute n_cameras
    need_compute = "n_cameras" not in df.columns

    if need_compute:
        id_col = get_id_col(df)
        path_col = "path" if "path" in df.columns else None

        if path_col is None:
            raise ValueError("Need 'path' column to read dat2 files")

        df_with_cameras = df.copy()
        df_with_cameras["n_cameras"] = 0

        pbar = tqdm(total=len(df), desc="filter_multi_camera (counting cameras)", leave=False, disable=not show_tqdm)
        for idx, row in df.iterrows():
            asas_sn_id = str(row[id_col])
            dir_path = str(row[path_col])

            df_g, df_v = read_lc_dat2(asas_sn_id, dir_path)
            df_lc = pd.concat([df_g, df_v], ignore_index=True) if not df_g.empty or not df_v.empty else pd.DataFrame()

            n_cams = compute_n_cameras(df_lc)
            df_with_cameras.loc[idx, "n_cameras"] = n_cams

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        df = df_with_cameras

    # Apply filter
    out = df.loc[df["n_cameras"] >= min_cameras].reset_index(drop=True)

    if show_tqdm:
        tqdm.write(f"[filter_multi_camera] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_multi_camera", rejected_log_csv)

    return out


def apply_pre_filters(
    df: pd.DataFrame,
    *,
    # Filter 1: sparse lightcurves
    apply_sparse: bool = True,
    min_time_span: float = 100.0,
    min_points_per_day: float = 0.05,
    # Filter 2: periodic candidates
    apply_periodic: bool = True,
    max_power: float = 0.5,
    min_period: float | None = None,
    max_period: float | None = None,
    # Filter 3: VSX crossmatch
    apply_vsx: bool = False,
    vsx_max_sep_arcsec: float = 3.0,
    vsx_exclude_classes: list[str] | None = None,
    vsx_catalog_csv: str | Path = "results_crossmatch/vsx_cleaned_20250926_1557.csv",
    # Filter 4: bright nearby stars
    apply_bns: bool = False,
    bns_max_separation_arcsec: float = 5.0,
    bns_max_mag_diff: float = 3.0,
    asassn_catalog_csv: str | Path = "results_crossmatch/asassn_index_masked_concat_cleaned_20250926_1557.csv",
    # Filter 5: multi camera
    apply_multi_camera: bool = True,
    min_cameras: int = 2,
    # General
    n_workers: int = 1,
    show_tqdm: bool = True,
    rejected_log_csv: str | Path | None = "rejected_pre_filter.csv",
) -> pd.DataFrame:
    """
    Apply pre-filters before running events.py.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with ID and path columns
    apply_* : bool
        Whether to apply each filter
    n_workers : int
        Number of parallel workers for computing stats (default 1 = sequential).
        Filters 1, 2, and 5 (sparse, periodic, multi_camera) can benefit from parallelization.
    show_tqdm : bool
        Show progress bars
    rejected_log_csv : str | Path | None
        Path to log rejected candidates

    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    df_filtered = df.copy()
    n_start = len(df_filtered)

    # Pre-compute stats in parallel if requested and needed
    if n_workers > 1 and "path" in df_filtered.columns:
        id_col = get_id_col(df_filtered)

        need_time = apply_sparse and (("time_span_days" not in df_filtered.columns) or ("points_per_day" not in df_filtered.columns))
        need_period = apply_periodic and (("ls_max_power" not in df_filtered.columns) or ("best_period" not in df_filtered.columns))
        need_cameras = apply_multi_camera and ("n_cameras" not in df_filtered.columns)

        if need_time or need_period or need_cameras:
            if show_tqdm:
                tqdm.write(f"[apply_pre_filters] Pre-computing stats with {n_workers} workers")
            df_filtered = _compute_stats_parallel(
                df_filtered, id_col, "path",
                compute_time=need_time,
                compute_period=need_period,
                compute_cameras=need_cameras,
                n_workers=n_workers,
                show_tqdm=show_tqdm
            )

    filters = []

    if apply_sparse:
        filters.append(("sparse", filter_sparse_lightcurves, {
            "min_time_span": min_time_span,
            "min_points_per_day": min_points_per_day,
            "show_tqdm": show_tqdm,
            "rejected_log_csv": rejected_log_csv,
        }))

    if apply_periodic:
        filters.append(("periodic", filter_periodic_candidates, {
            "max_power": max_power,
            "min_period": min_period,
            "max_period": max_period,
            "show_tqdm": show_tqdm,
            "rejected_log_csv": rejected_log_csv,
        }))

    if apply_vsx:
        filters.append(("vsx_match", filter_vsx_match, {
            "max_sep_arcsec": vsx_max_sep_arcsec,
            "exclude_classes": vsx_exclude_classes,
            "vsx_catalog_csv": vsx_catalog_csv,
            "show_tqdm": show_tqdm,
            "rejected_log_csv": rejected_log_csv,
        }))

    if apply_bns:
        filters.append(("bright_nearby_stars", filter_bright_nearby_stars, {
            "max_separation_arcsec": bns_max_separation_arcsec,
            "max_mag_diff": bns_max_mag_diff,
            "asassn_catalog_csv": asassn_catalog_csv,
            "show_tqdm": show_tqdm,
            "rejected_log_csv": rejected_log_csv,
        }))

    if apply_multi_camera:
        filters.append(("multi_camera", filter_multi_camera, {
            "min_cameras": min_cameras,
            "show_tqdm": show_tqdm,
            "rejected_log_csv": rejected_log_csv,
        }))

    # Apply filters sequentially
    total_steps = len(filters)
    if total_steps > 0:
        with tqdm(total=total_steps, desc="apply_pre_filters", leave=True, disable=not show_tqdm) as pbar:
            for label, func, kwargs in filters:
                n_before = len(df_filtered)
                start = perf_counter()
                df_filtered = func(df_filtered, **kwargs)
                elapsed = perf_counter() - start
                n_after = len(df_filtered)

                pbar.set_postfix_str(f"{label}: {n_before} → {n_after} ({elapsed:.2f}s)")
                pbar.update(1)

    n_end = len(df_filtered)
    if show_tqdm:
        tqdm.write(f"\n[apply_pre_filters] Total: {n_start} → {n_end} ({n_end/n_start*100:.1f}% kept)")

    return df_filtered.reset_index(drop=True)
