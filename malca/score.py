"""
Event scoring metric for ASAS-SN light curves (dips and microlensing).

Implements a heuristic event score:

    S = (1 / (ln(N + 1) * N)) * sum_i (delta_i / (2 * FWHM_i) * Ndet_i * (1 / chi2_i))

where each event i is measured from the light curve. Supports:
    - Dips (symmetric Gaussian-like decreases in brightness)
    - Microlensing (symmetric Paczyński curve brightening events)

The reported score is log10(S).

This module provides:
    - compute_event_score(df_lc, ..., event_type='dip'|'microlensing')
    - score_lightcurve_path(path, ..., event_type=...)
    - CLI for batch scoring
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from malca.utils import gaussian
from malca.stats import robust_sigma
from malca.utils import read_lc_dat2


@dataclass
class EventStats:
    """Statistics for a single detected event (dip or microlensing)."""
    t0: float  # Time of peak
    delta: float  # Amplitude (mag units, always positive)
    fwhm_days: float  # Full width at half maximum
    n_det: int  # Number of detections in event
    chi2: float  # Chi-squared of fit
    valid: bool  # Passes quality cuts
    event_type: str  # 'dip' or 'microlensing'


def paczynski(t: np.ndarray, A0: float, t0: float, tE: float, baseline: float) -> np.ndarray:
    """
    Paczyński microlensing light curve in magnitudes.

    Parameters
    ----------
    t : array
        Time values
    A0 : float
        Peak magnification amplitude (dimensionless, A0 > 1)
    t0 : float
        Time of peak magnification
    tE : float
        Einstein crossing time (characteristic timescale in days)
    baseline : float
        Baseline magnitude (unmagnified)

    Returns
    -------
    mag : array
        Magnitudes at times t

    Notes
    -----
    The standard Paczyński curve is:
        u(t) = sqrt(u0^2 + ((t - t0) / tE)^2)
        A(t) = (u^2 + 2) / (u * sqrt(u^2 + 4))

    where u0 is the minimum impact parameter related to A0:
        A0 = (u0^2 + 2) / (u0 * sqrt(u0^2 + 4))

    We solve for u0 from A0, then compute magnitudes:
        m(t) = baseline - 2.5 * log10(A(t))
    """
    # Solve for u0 from peak magnification A0
    # For A0 >> 1, u0 ≈ 1/A0; for A0 = 1, u0 = infinity (no lensing)
    if A0 <= 1.0:
        return np.full_like(t, baseline, dtype=float)

    # Newton-Raphson to solve A0 = (u0^2 + 2) / (u0 * sqrt(u0^2 + 4))
    u0 = 1.0 / A0  # Initial guess
    for _ in range(10):
        sqrt_term = np.sqrt(u0**2 + 4)
        A_curr = (u0**2 + 2) / (u0 * sqrt_term)
        dA_du = -(u0**2 + 4 - 2) / (u0**2 * sqrt_term)
        u0 = u0 - (A_curr - A0) / dA_du
        if abs(A_curr - A0) < 1e-6:
            break

    # Compute u(t)
    u = np.sqrt(u0**2 + ((t - t0) / tE)**2)

    # Compute magnification A(t)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))

    # Convert to magnitudes
    mag = baseline - 2.5 * np.log10(A)
    return mag


def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous runs of True values in boolean mask."""
    runs: list[tuple[int, int]] = []
    if mask.size == 0:
        return runs
    in_run = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_run:
            in_run = True
            start = i
        elif not val and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, mask.size - 1))
    return runs


def _half_max_width(
    jd: np.ndarray,
    mag: np.ndarray,
    peak_idx: int,
    baseline: float,
    delta: float,
    magnitude_dips: bool
) -> float:
    """
    Compute FWHM by finding half-maximum crossings.

    For dips: magnitude_dips=True, half_level = baseline + 0.5 * delta
    For brightening: magnitude_dips=False, half_level = baseline - 0.5 * delta
    """
    if delta <= 0:
        return 0.0
    if magnitude_dips:
        half_level = baseline + 0.5 * delta
        above = mag >= half_level
    else:
        half_level = baseline - 0.5 * delta
        above = mag <= half_level

    # Find nearest crossings around peak
    left = peak_idx
    while left > 0 and above[left]:
        left -= 1
    right = peak_idx
    while right < len(mag) - 1 and above[right]:
        right += 1

    # Linear interpolation at crossings
    def interp(i0, i1):
        if i0 == i1:
            return jd[i0]
        y0, y1 = mag[i0], mag[i1]
        if y1 == y0:
            return jd[i0]
        frac = (half_level - y0) / (y1 - y0)
        return jd[i0] + frac * (jd[i1] - jd[i0])

    if left == peak_idx or right == peak_idx:
        return float(jd[right] - jd[left]) if right > left else 0.0
    t_left = interp(left, left + 1)
    t_right = interp(right - 1, right)
    width = float(t_right - t_left)
    return max(width, 0.0)


def _fit_gaussian(
    jd: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    t0: float,
    baseline: float,
    amp: float,
    sigma_guess: float
) -> tuple[float, float]:
    """
    Fit Gaussian profile to event.

    Returns (FWHM, chi2).
    """
    if sigma_guess <= 0:
        sigma_guess = max((jd.max() - jd.min()) / 6.0, 0.1)
    try:
        popt, _ = curve_fit(
            gaussian,
            jd,
            mag,
            p0=[amp, t0, sigma_guess, baseline],
            sigma=err,
            maxfev=2000,
        )
        resid = mag - gaussian(jd, *popt)
        chi2 = float(np.nansum((resid / err) ** 2))
        sigma = float(abs(popt[2]))
        fwhm = 2.3548 * sigma
        return fwhm, chi2
    except Exception:
        return 0.0, np.nan


def _fit_paczynski(
    jd: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    t0: float,
    baseline: float,
    delta_mag: float,
    tE_guess: float
) -> tuple[float, float]:
    """
    Fit Paczyński microlensing curve to brightening event.

    Parameters
    ----------
    jd, mag, err : arrays
        Time, magnitude, and error arrays
    t0 : float
        Initial guess for time of peak
    baseline : float
        Baseline magnitude (unmagnified)
    delta_mag : float
        Peak brightening amplitude (positive, in mag)
    tE_guess : float
        Initial guess for Einstein crossing time (days)

    Returns
    -------
    fwhm : float
        Full width at half maximum (days)
    chi2 : float
        Chi-squared of fit
    """
    if tE_guess <= 0:
        tE_guess = max((jd.max() - jd.min()) / 4.0, 1.0)

    # Convert delta_mag to peak magnification A0
    # delta_mag = 2.5 * log10(A0), so A0 = 10^(delta_mag / 2.5)
    A0_guess = 10.0 ** (delta_mag / 2.5)
    if A0_guess <= 1.0:
        return 0.0, np.nan

    try:
        popt, _ = curve_fit(
            paczynski,
            jd,
            mag,
            p0=[A0_guess, t0, tE_guess, baseline],
            sigma=err,
            bounds=([1.01, jd.min(), 0.1, baseline - 5], [100, jd.max(), 1000, baseline + 5]),
            maxfev=5000,
        )

        # Compute chi2
        resid = mag - paczynski(jd, *popt)
        chi2 = float(np.nansum((resid / err) ** 2))

        # Estimate FWHM from fitted parameters
        # For Paczyński, FWHM ≈ 2 * tE * sqrt((A0 + sqrt(A0^2 - 1)) / A0)
        # Approximation: FWHM ≈ 2.4 * tE for typical A0
        tE_fit = popt[2]
        fwhm = float(2.4 * tE_fit)

        return fwhm, chi2
    except Exception:
        return 0.0, np.nan


def compute_event_score(
    df_lc: pd.DataFrame,
    *,
    event_type: Literal['dip', 'microlensing'] = 'dip',
    sigma_threshold: float = 1.0,
    edge_sigma: float = 0.5,
    min_fwhm_days: float = 1.5,
    min_delta_mag: float = 0.05,
) -> tuple[float, list[EventStats]]:
    """
    Compute event score for a single light curve DataFrame in log10 space.

    Parameters
    ----------
    df_lc : DataFrame
        Light curve with columns 'JD', 'mag', 'error'
    event_type : {'dip', 'microlensing'}
        Type of event to search for:
        - 'dip': Symmetric magnitude increases (dippers)
        - 'microlensing': Symmetric magnitude decreases (Paczyński curves)
    sigma_threshold : float
        Detection threshold in units of robust sigma
    edge_sigma : float
        Edge detection threshold for event boundaries
    min_fwhm_days : float
        Minimum FWHM to consider event valid
    min_delta_mag : float
        Minimum amplitude to consider event valid

    Returns
    -------
    score : float
        Log10 event score (higher = more significant events; -inf if no valid events)
    events : list[EventStats]
        List of detected events with statistics
    """
    if df_lc.empty:
        return -np.inf, []
    df = df_lc.copy()
    for col in ["JD", "mag", "error"]:
        if col not in df.columns:
            return -np.inf, []
    df = df[np.isfinite(df["JD"]) & np.isfinite(df["mag"]) & np.isfinite(df["error"])].copy()
    if df.empty:
        return -np.inf, []
    df = df.sort_values("JD").reset_index(drop=True)

    jd = df["JD"].to_numpy(float)
    mag = df["mag"].to_numpy(float)
    err = df["error"].to_numpy(float)

    baseline = float(np.nanmedian(mag))
    sigma = float(robust_sigma(mag))
    if not np.isfinite(sigma) or sigma <= 0:
        return -np.inf, []

    # Detect events based on type
    if event_type == 'dip':
        # Dips: magnitude increases (fainter)
        magnitude_dips = True
        event_mask = mag >= (baseline + sigma_threshold * sigma)
        edge_level = baseline + edge_sigma * sigma
    elif event_type == 'microlensing':
        # Microlensing: magnitude decreases (brighter)
        magnitude_dips = False
        event_mask = mag <= (baseline - sigma_threshold * sigma)
        edge_level = baseline - edge_sigma * sigma
    else:
        raise ValueError(f"Unknown event_type: {event_type}")

    runs = _find_runs(event_mask)
    if not runs:
        return -np.inf, []

    events: list[EventStats] = []
    for start, end in runs:
        seg = slice(start, end + 1)

        # Find peak
        if magnitude_dips:  # Dip: maximum magnitude
            peak_idx = int(start + np.nanargmax(mag[seg]))
            delta = float(mag[peak_idx] - baseline)
        else:  # Microlensing: minimum magnitude
            peak_idx = int(start + np.nanargmin(mag[seg]))
            delta = float(baseline - mag[peak_idx])

        if not np.isfinite(delta) or delta <= 0:
            continue

        # Expand edges until back within edge_sigma
        left = peak_idx
        while left > 0 and ((mag[left] > edge_level) if magnitude_dips else (mag[left] < edge_level)):
            left -= 1
        right = peak_idx
        while right < len(mag) - 1 and ((mag[right] > edge_level) if magnitude_dips else (mag[right] < edge_level)):
            right += 1

        window = slice(left, right + 1)
        n_det = int(window.stop - window.start)
        if n_det <= 0:
            continue

        # Compute FWHM
        fwhm = _half_max_width(jd, mag, peak_idx, baseline, delta, magnitude_dips)
        if fwhm <= 0:
            fwhm = float(jd[right] - jd[left]) if right > left else 0.0

        # Fit model and refine FWHM
        if event_type == 'dip':
            # Fit Gaussian
            amp = float(delta if magnitude_dips else -delta)
            fwhm_fit, chi2 = _fit_gaussian(
                jd[window], mag[window], err[window],
                jd[peak_idx], baseline, amp,
                fwhm / 2.3548 if fwhm > 0 else 0.0
            )
        else:  # microlensing
            # Fit Paczyński curve
            tE_guess = fwhm / 2.4 if fwhm > 0 else 10.0
            fwhm_fit, chi2 = _fit_paczynski(
                jd[window], mag[window], err[window],
                jd[peak_idx], baseline, delta, tE_guess
            )

        if fwhm_fit > 0:
            fwhm = fwhm_fit

        valid = bool(
            np.isfinite(chi2)
            and chi2 > 0
            and fwhm >= min_fwhm_days
            and delta >= min_delta_mag
        )
        events.append(EventStats(
            t0=float(jd[peak_idx]),
            delta=delta,
            fwhm_days=fwhm,
            n_det=n_det,
            chi2=chi2,
            valid=valid,
            event_type=event_type
        ))

    if not events:
        return -np.inf, []

    # Compute score
    N = len(events)
    terms = []
    for evt in events:
        if not evt.valid:
            continue
        terms.append((evt.delta / (2.0 * evt.fwhm_days)) * evt.n_det * (1.0 / evt.chi2))

    if N <= 0 or not terms:
        return -np.inf, events

    score = float(np.sum(terms)) / (np.log(N + 1.0) * N)
    if not np.isfinite(score) or score <= 0:
        return -np.inf, events
    return float(np.log10(score)), events


def _load_lightcurve_path(path: Path) -> pd.DataFrame:
    """Load light curve from .dat2 file."""
    path = Path(path)
    if path.suffix != ".dat2":
        return pd.DataFrame()
    dfg, dfv = read_lc_dat2(path.stem, str(path.parent))
    return pd.concat([dfg, dfv], ignore_index=True) if not (dfg.empty and dfv.empty) else pd.DataFrame()


def score_lightcurve_path(path: Path, **kwargs) -> tuple[float, list[EventStats]]:
    """Score a single light curve file."""
    df = _load_lightcurve_path(path)
    return compute_event_score(df, **kwargs)


def attach_event_scores(
    df_results: pd.DataFrame,
    *,
    event_type: Literal['dip', 'microlensing'] = 'dip',
    only_significant: bool = True,
    sigma_threshold: float = 1.0,
    edge_sigma: float = 0.5,
    min_fwhm_days: float = 1.5,
    min_delta_mag: float = 0.05,
) -> pd.DataFrame:
    """
    Attach event scores to an events.py results DataFrame.

    Expects a 'path' column that points to the .dat2 light curve.

    Parameters
    ----------
    df_results : DataFrame
        Results from events.py
    event_type : {'dip', 'microlensing'}
        Type of events to score
    only_significant : bool
        Only score rows where dip_significant=True
    sigma_threshold : float
        Detection threshold (robust sigma)
    edge_sigma : float
        Edge detection threshold
    min_fwhm_days : float
        Minimum FWHM for valid events
    min_delta_mag : float
        Minimum amplitude for valid events

    Returns
    -------
    df_scored : DataFrame
        Input DataFrame with added columns:
        - dipper_score or microlens_score: Log10 event score (-inf if no valid events)
        - dipper_n_events or microlens_n_events: Number of detected events
        - dipper_n_valid_events or microlens_n_valid_events: Number of events passing quality cuts
    """
    df = df_results.copy()
    scores = []

    for _, row in df.iterrows():
        if only_significant and not bool(row.get("dip_significant", False)):
            scores.append((-np.inf, 0, 0))
            continue
        path = Path(str(row.get("path", "")))
        score, events = score_lightcurve_path(
            path,
            event_type=event_type,
            sigma_threshold=sigma_threshold,
            edge_sigma=edge_sigma,
            min_fwhm_days=min_fwhm_days,
            min_delta_mag=min_delta_mag,
        )
        scores.append((score, len(events), int(sum(1 for e in events if e.valid))))

    prefix = 'dipper' if event_type == 'dip' else 'microlens'
    df[f"{prefix}_score"] = [s[0] for s in scores]
    df[f"{prefix}_n_events"] = [s[1] for s in scores]
    df[f"{prefix}_n_valid_events"] = [s[2] for s in scores]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute event scores for events.py results (dips or microlensing)."
    )
    parser.add_argument("--events", type=Path, required=True, help="events.py results CSV/Parquet")
    parser.add_argument("--output", type=Path, default=Path("/home/lenhart.106/code/malca/output/event_scores.csv"), help="Output CSV path")
    parser.add_argument(
        "--event-type",
        type=str,
        choices=['dip', 'microlensing'],
        default='dip',
        help="Type of events to score (default: dip)"
    )
    parser.add_argument("--only-significant", action="store_true", help="Score only dip_significant rows (default)")
    parser.add_argument("--include-all", action="store_true", help="Score all rows (overrides --only-significant)")
    parser.add_argument("--sigma-threshold", type=float, default=1.0, help="Detection threshold (robust sigma)")
    parser.add_argument("--edge-sigma", type=float, default=0.5, help="Edge detection threshold")
    parser.add_argument("--min-fwhm-days", type=float, default=1.5, help="Minimum FWHM (days)")
    parser.add_argument("--min-delta-mag", type=float, default=0.05, help="Minimum amplitude (mag)")
    args = parser.parse_args()

    events_path = args.events.expanduser()
    if events_path.suffix.lower() in (".parquet", ".pq"):
        df_events = pd.read_parquet(events_path)
    else:
        df_events = pd.read_csv(events_path)

    only_significant = True
    if args.include_all:
        only_significant = False
    elif args.only_significant:
        only_significant = True

    df_scored = attach_event_scores(
        df_events,
        event_type=args.event_type,
        only_significant=only_significant,
        sigma_threshold=args.sigma_threshold,
        edge_sigma=args.edge_sigma,
        min_fwhm_days=args.min_fwhm_days,
        min_delta_mag=args.min_delta_mag,
    )

    out_path = args.output.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_scored.to_csv(out_path, index=False)

    prefix = 'dipper' if args.event_type == 'dip' else 'microlens'
    score_values = df_scored[f"{prefix}_score"].to_numpy()
    n_scored = int(np.isfinite(score_values).sum())
    print(f"Scored {n_scored}/{len(df_scored)} {args.event_type} candidates")
    print(f"Wrote {len(df_scored)} rows to {out_path}")


if __name__ == "__main__":
    main()
