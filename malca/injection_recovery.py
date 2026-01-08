"""
Injection-recovery testing for dipper detection pipeline.

Implements approach similar to ZTF paper Section 3.5:
1. Select control sample of clean light curves
2. Inject synthetic dips using skew-normal model
3. Run through detection pipeline
4. Measure detection efficiency vs amplitude/duration

This validates the completeness and contamination of the pipeline
and characterizes sensitivity to different dip morphologies.
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from scipy.stats import skewnorm
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from utils import read_lc_dat2, read_lc_csv


def skewnormal_dip(
    t: np.ndarray,
    t_center: float,
    duration: float,
    amplitude: float,
    skewness: float = 0.0,
    offset: float = 0.0,
) -> np.ndarray:
    """
    Generate skew-normal dip profile.

    Parameters
    ----------
    t : np.ndarray
        Time array
    t_center : float
        Center time of dip (mean μ)
    duration : float
        Duration of dip (related to σ)
    amplitude : float
        Depth of dip in magnitudes
    skewness : float
        Skewness parameter α (default 0 = symmetric Gaussian)
        Positive = tail to right, Negative = tail to left
    offset : float
        Baseline offset C0

    Returns
    -------
    np.ndarray
        Magnitude perturbation (positive = fainter)

    Notes
    -----
    Paper uses skew-normal with α ∈ [-0.5, 0.5] to model
    both symmetric and asymmetric dips.
    """
    # Convert duration to standard deviation
    # FWHM ≈ 2.355 * σ for Gaussian, use similar scaling
    sigma = duration / 2.355

    # Skew-normal PDF scaled to amplitude
    loc = t_center
    scale = sigma

    # Compute skew-normal (scipy parameterization: a=skewness)
    dip = amplitude * skewnorm.pdf(t, a=skewness, loc=loc, scale=scale)

    # Normalize to peak at amplitude
    if dip.max() > 0:
        dip = dip * (amplitude / dip.max())

    return dip + offset


def estimate_magnitude_error_polynomial(
    lc_sample: list[pd.DataFrame],
    order: int = 5,
    mag_col: str = "mag",
    err_col: str = "error",
) -> np.poly1d:
    """
    Fit polynomial to approximate mag-dependent errors.

    Paper uses 5th-order polynomial for magnitude vs error.

    Parameters
    ----------
    lc_sample : list[pd.DataFrame]
        Sample of light curves to fit
    order : int
        Polynomial order (default 5, from paper)
    mag_col : str
        Magnitude column name
    err_col : str
        Error column name

    Returns
    -------
    np.poly1d
        Polynomial function: mag → σ_mag
    """
    all_mags = []
    all_errs = []

    for df in lc_sample:
        if df.empty:
            continue
        mag = df[mag_col].values
        err = df[err_col].values

        # Keep finite values
        mask = np.isfinite(mag) & np.isfinite(err) & (err > 0)
        all_mags.extend(mag[mask])
        all_errs.extend(err[mask])

    if len(all_mags) < 10:
        # Fallback: constant error
        return np.poly1d([0.1])

    # Fit polynomial: err = p(mag)
    coeffs = np.polyfit(all_mags, all_errs, order)
    return np.poly1d(coeffs)


def inject_dip(
    df_lc: pd.DataFrame,
    t_center: float,
    duration: float,
    amplitude: float,
    skewness: float = 0.0,
    mag_err_poly: np.poly1d | None = None,
    mag_col: str = "mag",
    time_col: str = "JD",
    err_col: str = "error",
) -> pd.DataFrame:
    """
    Inject synthetic dip into light curve.

    Adds noise as: f(μ,σ,α,C0) + N(0,σ_mag) + N(0,σ_σmag)

    Parameters
    ----------
    df_lc : pd.DataFrame
        Light curve to inject into
    t_center : float
        Center time of dip
    duration : float
        Duration (FWHM) in days
    amplitude : float
        Depth in magnitudes
    skewness : float
        Skew-normal α parameter
    mag_err_poly : np.poly1d | None
        Polynomial for magnitude-dependent errors
        If None, uses constant scatter from LC stddev
    mag_col : str
        Magnitude column
    time_col : str
        Time column
    err_col : str
        Error column

    Returns
    -------
    pd.DataFrame
        Light curve with injected dip
    """
    df_out = df_lc.copy()

    if df_out.empty:
        return df_out

    # Compute baseline statistics
    mag_baseline = df_out[mag_col].median()
    sigma_mag = df_out[mag_col].std()

    if mag_err_poly is not None:
        # Use magnitude-dependent errors
        avg_err_from_poly = mag_err_poly(mag_baseline)
        sigma_sigma_mag = df_out[err_col].std()
    else:
        # Use constant scatter
        avg_err_from_poly = sigma_mag
        sigma_sigma_mag = 0.0

    # Generate dip profile
    t = df_out[time_col].values
    dip_profile = skewnormal_dip(t, t_center, duration, amplitude, skewness)

    # Add observational noise (paper Equation 3)
    noise_intrinsic = np.random.normal(0, sigma_mag, size=len(t))
    noise_error = np.random.normal(0, sigma_sigma_mag, size=len(t))

    # Apply: mag_new = mag_baseline + dip + noise
    df_out[mag_col] = mag_baseline + dip_profile + noise_intrinsic + noise_error

    return df_out


def select_control_sample(
    manifest_df: pd.DataFrame,
    n_sample: int = 10000,
    reject_candidates: pd.DataFrame | None = None,
    min_points: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Select clean light curves for injection testing.

    Paper selects 10,000 stars that did NOT meet δ > 3 criterion
    to get unbiased control set without significant dips.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        Full manifest with all sources
    n_sample : int
        Number to sample (default 10000, from paper)
    reject_candidates : pd.DataFrame | None
        DataFrame of known candidates to exclude
    min_points : int
        Minimum number of detections required
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Clean control sample
    """
    df = manifest_df.copy()

    # Exclude candidates if provided
    if reject_candidates is not None and "asas_sn_id" in reject_candidates.columns:
        exclude_ids = set(reject_candidates["asas_sn_id"].astype(str))
        df = df[~df["asas_sn_id"].astype(str).isin(exclude_ids)]

    # Require minimum points (if column exists)
    if "n_points" in df.columns:
        df = df[df["n_points"] >= min_points]

    # Random sample
    rng = np.random.default_rng(seed)
    if len(df) < n_sample:
        return df.reset_index(drop=True)

    indices = rng.choice(len(df), size=n_sample, replace=False)
    return df.iloc[indices].reset_index(drop=True)


def run_injection_recovery(
    control_sample: pd.DataFrame,
    detection_func: Callable[[pd.DataFrame], dict],
    amplitude_grid: np.ndarray | None = None,
    duration_grid: np.ndarray | None = None,
    n_injections_per_grid: int = 100,
    skewness_range: tuple[float, float] = (-0.5, 0.5),
    lc_reader: str = "dat2",
    mag_err_order: int = 5,
    seed: int = 42,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Run full injection-recovery test.

    Paper approach:
    - 100,000 amplitude × duration combinations
    - Random sampling from 10,000 control LCs
    - Skewness α ∈ [-0.5, 0.5]
    - Single dip per LC

    Parameters
    ----------
    control_sample : pd.DataFrame
        Clean light curves (from select_control_sample)
    detection_func : Callable
        Function that takes LC DataFrame and returns detection dict
        Must return: {'detected': bool, 'bayes_factor': float, ...}
    amplitude_grid : np.ndarray | None
        Amplitude values to test (magnitudes)
        If None, uses default: np.linspace(0.1, 2.0, 100)
    duration_grid : np.ndarray | None
        Duration values to test (days)
        If None, uses default: np.logspace(0.5, 2.5, 100)
    n_injections_per_grid : int
        Number of injections per grid point (default 100)
        Total injections = len(amplitude) × len(duration) × n_injections_per_grid
    skewness_range : tuple[float, float]
        Range for random skewness (default [-0.5, 0.5] from paper)
    lc_reader : str
        Light curve format: "dat2" or "csv"
    mag_err_order : int
        Polynomial order for mag-error relation (default 5)
    seed : int
        Random seed
    show_progress : bool
        Show progress bar

    Returns
    -------
    pd.DataFrame
        Results with columns: amplitude, duration, skewness, detected, bayes_factor, etc.
    """
    rng = np.random.default_rng(seed)

    # Default grids
    if amplitude_grid is None:
        amplitude_grid = np.linspace(0.1, 2.0, 100)  # 0.1 to 2 mag
    if duration_grid is None:
        duration_grid = np.logspace(0.5, 2.5, 100)  # ~3 to ~300 days

    # Load sample light curves for error polynomial
    print("Loading control sample light curves...")
    lc_sample = []
    for idx, row in control_sample.iterrows():
        if idx >= 100:  # Use first 100 for polynomial fit
            break
        asas_sn_id = str(row["asas_sn_id"])
        path = str(row.get("path", row.get("lc_dir", "")))

        try:
            if lc_reader == "dat2":
                df_g, df_v = read_lc_dat2(asas_sn_id, path)
                df = pd.concat([df_g, df_v], ignore_index=True)
            elif lc_reader == "csv":
                df_g, df_v = read_lc_csv(asas_sn_id, path)
                df = pd.concat([df_g, df_v], ignore_index=True)
            else:
                continue

            if not df.empty:
                lc_sample.append(df)
        except Exception:
            continue

    # Fit magnitude-error polynomial
    print(f"Fitting {mag_err_order}th-order polynomial to magnitude errors...")
    mag_err_poly = estimate_magnitude_error_polynomial(lc_sample, order=mag_err_order)

    # Generate grid
    total_injections = len(amplitude_grid) * len(duration_grid) * n_injections_per_grid
    print(f"Running {total_injections:,} injection tests...")

    results = []
    pbar = tqdm(total=total_injections, disable=not show_progress)

    for amplitude in amplitude_grid:
        for duration in duration_grid:
            for _ in range(n_injections_per_grid):
                # Randomly sample control LC
                idx = rng.integers(0, len(control_sample))
                row = control_sample.iloc[idx]
                asas_sn_id = str(row["asas_sn_id"])
                path = str(row.get("path", row.get("lc_dir", "")))

                try:
                    # Load light curve
                    if lc_reader == "dat2":
                        df_g, df_v = read_lc_dat2(asas_sn_id, path)
                        df = pd.concat([df_g, df_v], ignore_index=True)
                    elif lc_reader == "csv":
                        df_g, df_v = read_lc_csv(asas_sn_id, path)
                        df = pd.concat([df_g, df_v], ignore_index=True)
                    else:
                        pbar.update(1)
                        continue

                    if df.empty or len(df) < 10:
                        pbar.update(1)
                        continue

                    # Random injection parameters
                    t_min, t_max = df["JD"].min(), df["JD"].max()
                    t_center = rng.uniform(t_min + duration, t_max - duration)
                    skewness = rng.uniform(*skewness_range)

                    # Inject dip
                    df_injected = inject_dip(
                        df, t_center, duration, amplitude, skewness, mag_err_poly
                    )

                    # Run detection
                    detection_result = detection_func(df_injected)

                    # Store results
                    result = {
                        "amplitude": amplitude,
                        "duration": duration,
                        "skewness": skewness,
                        "t_center": t_center,
                        "asas_sn_id": asas_sn_id,
                        **detection_result,
                    }
                    results.append(result)

                except Exception as e:
                    # Log failure but continue
                    results.append({
                        "amplitude": amplitude,
                        "duration": duration,
                        "skewness": 0.0,
                        "detected": False,
                        "error": str(e),
                    })

                pbar.update(1)

    pbar.close()

    return pd.DataFrame(results)


def compute_detection_efficiency(
    results_df: pd.DataFrame,
    amplitude_bins: int = 20,
    duration_bins: int = 20,
    detected_col: str = "detected",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute detection efficiency grid.

    Paper produces 2D heatmap: amplitude vs duration → efficiency.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from run_injection_recovery
    amplitude_bins : int
        Number of amplitude bins
    duration_bins : int
        Number of duration bins
    detected_col : str
        Column indicating detection (boolean)

    Returns
    -------
    amp_centers : np.ndarray
        Amplitude bin centers
    dur_centers : np.ndarray
        Duration bin centers
    efficiency_grid : np.ndarray
        2D array of detection efficiencies (0 to 1)
    """
    # Create bins
    amp_edges = np.linspace(
        results_df["amplitude"].min(),
        results_df["amplitude"].max(),
        amplitude_bins + 1
    )
    dur_edges = np.logspace(
        np.log10(results_df["duration"].min()),
        np.log10(results_df["duration"].max()),
        duration_bins + 1
    )

    amp_centers = (amp_edges[:-1] + amp_edges[1:]) / 2
    dur_centers = np.sqrt(dur_edges[:-1] * dur_edges[1:])  # Geometric mean for log scale

    # Compute efficiency per bin
    efficiency_grid = np.zeros((amplitude_bins, duration_bins))

    for i in range(amplitude_bins):
        for j in range(duration_bins):
            mask = (
                (results_df["amplitude"] >= amp_edges[i]) &
                (results_df["amplitude"] < amp_edges[i + 1]) &
                (results_df["duration"] >= dur_edges[j]) &
                (results_df["duration"] < dur_edges[j + 1])
            )

            if mask.sum() > 0:
                efficiency_grid[i, j] = results_df.loc[mask, detected_col].mean()
            else:
                efficiency_grid[i, j] = np.nan

    return amp_centers, dur_centers, efficiency_grid


def plot_detection_efficiency(
    amp_centers: np.ndarray,
    dur_centers: np.ndarray,
    efficiency_grid: np.ndarray,
    output_path: Path | str | None = None,
    title: str = "Detection Efficiency",
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    """
    Plot detection efficiency heatmap (like paper Figure 5).

    Parameters
    ----------
    amp_centers : np.ndarray
        Amplitude bin centers
    dur_centers : np.ndarray
        Duration bin centers
    efficiency_grid : np.ndarray
        2D efficiency array
    output_path : Path | str | None
        Save path (if None, shows plot)
    title : str
        Plot title
    vmin, vmax : float
        Colorbar limits
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    im = ax.pcolormesh(
        dur_centers, amp_centers, efficiency_grid,
        cmap="viridis", vmin=vmin, vmax=vmax, shading="auto"
    )

    ax.set_xscale("log")
    ax.set_xlabel("Duration (days)", fontsize=14)
    ax.set_ylabel("Amplitude (mag)", fontsize=14)
    ax.set_title(title, fontsize=16)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Detection Efficiency", fontsize=14)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved to {output_path}")
    else:
        plt.show()


def compute_auxiliary_statistics(df_lc: pd.DataFrame, mag_col: str = "mag") -> dict:
    """
    Compute auxiliary time-series statistics.

    Paper mentions: skewness, Von Neumann ratio

    Parameters
    ----------
    df_lc : pd.DataFrame
        Light curve
    mag_col : str
        Magnitude column

    Returns
    -------
    dict
        Statistics: skewness, von_neumann_ratio, etc.
    """
    from scipy.stats import skew

    mag = df_lc[mag_col].values
    mag_finite = mag[np.isfinite(mag)]

    if len(mag_finite) < 3:
        return {
            "skewness": np.nan,
            "von_neumann_ratio": np.nan,
        }

    # Skewness
    skewness_val = skew(mag_finite)

    # Von Neumann ratio: measures autocorrelation
    # VNR = Σ(x_{i+1} - x_i)^2 / Σ(x_i - mean)^2
    # VNR ~ 2 for uncorrelated noise, < 2 for positive correlation
    diff_sq = np.diff(mag_finite) ** 2
    dev_sq = (mag_finite - mag_finite.mean()) ** 2

    if dev_sq.sum() > 0:
        von_neumann_ratio = diff_sq.sum() / dev_sq.sum()
    else:
        von_neumann_ratio = np.nan

    return {
        "skewness": skewness_val,
        "von_neumann_ratio": von_neumann_ratio,
    }
