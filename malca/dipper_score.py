"""
Dipper scoring metric adapted for ASAS-SN light curves.

Implements a heuristic dipper score:

    S = (1 / (ln(N + 1) * N)) * sum_i (delta_i / (2 * FWHM_i) * Ndet_i * (1 / chi2_i))

where each dip i is measured from the light curve. This module provides:
    - compute_dipper_score(df_lc, ...)
    - score_lightcurve_path(path, ...)
    - CLI for batch scoring
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from malca.utils import gaussian
from malca.stats import robust_sigma
from malca.utils import read_lc_dat2


@dataclass
class DipStats:
    t0: float
    delta: float
    fwhm_days: float
    n_det: int
    chi2: float
    valid: bool


def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
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


def _half_max_width(jd: np.ndarray, mag: np.ndarray, peak_idx: int, baseline: float, delta: float, magnitude_dips: bool) -> float:
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
    # Linear interpolation at crossings if possible
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


def _fit_gaussian(jd: np.ndarray, mag: np.ndarray, err: np.ndarray, t0: float, baseline: float, amp: float, sigma_guess: float) -> tuple[float, float]:
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


def compute_dipper_score(
    df_lc: pd.DataFrame,
    *,
    dip_sigma: float = 1.0,
    edge_sigma: float = 0.5,
    min_fwhm_days: float = 1.5,
    min_delta_mag: float = 0.05,
    magnitude_dips: bool = True,
) -> tuple[float, list[DipStats]]:
    """
    Compute the dipper score for a single light curve DataFrame.
    """
    if df_lc.empty:
        return 0.0, []
    df = df_lc.copy()
    for col in ["JD", "mag", "error"]:
        if col not in df.columns:
            return 0.0, []
    df = df[np.isfinite(df["JD"]) & np.isfinite(df["mag"]) & np.isfinite(df["error"])].copy()
    if df.empty:
        return 0.0, []
    df = df.sort_values("JD").reset_index(drop=True)

    jd = df["JD"].to_numpy(float)
    mag = df["mag"].to_numpy(float)
    err = df["error"].to_numpy(float)

    baseline = float(np.nanmedian(mag))
    sigma = float(robust_sigma(mag))
    if not np.isfinite(sigma) or sigma <= 0:
        return 0.0, []

    if magnitude_dips:
        dip_mask = mag >= (baseline + dip_sigma * sigma)
        edge_level = baseline + edge_sigma * sigma
    else:
        dip_mask = mag <= (baseline - dip_sigma * sigma)
        edge_level = baseline - edge_sigma * sigma

    runs = _find_runs(dip_mask)
    if not runs:
        return 0.0, []

    dips: list[DipStats] = []
    for start, end in runs:
        seg = slice(start, end + 1)
        if magnitude_dips:
            peak_idx = int(start + np.nanargmax(mag[seg]))
            delta = float(mag[peak_idx] - baseline)
        else:
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

        fwhm = _half_max_width(jd, mag, peak_idx, baseline, delta, magnitude_dips)
        if fwhm <= 0:
            fwhm = float(jd[right] - jd[left]) if right > left else 0.0

        amp = float(delta if magnitude_dips else -delta)
        fwhm_fit, chi2 = _fit_gaussian(jd[window], mag[window], err[window], jd[peak_idx], baseline, amp, fwhm / 2.3548 if fwhm > 0 else 0.0)
        if fwhm_fit > 0:
            fwhm = fwhm_fit

        valid = bool(
            np.isfinite(chi2)
            and chi2 > 0
            and fwhm >= min_fwhm_days
            and delta >= min_delta_mag
        )
        dips.append(DipStats(t0=float(jd[peak_idx]), delta=delta, fwhm_days=fwhm, n_det=n_det, chi2=chi2, valid=valid))

    if not dips:
        return 0.0, []

    N = len(dips)
    terms = []
    for d in dips:
        if not d.valid:
            continue
        terms.append((d.delta / (2.0 * d.fwhm_days)) * d.n_det * (1.0 / d.chi2))

    if N <= 0 or not terms:
        return 0.0, dips

    score = float(np.sum(terms)) / (np.log(N + 1.0) * N)
    return score, dips


def _load_lightcurve_path(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix != ".dat2":
        return pd.DataFrame()
    dfg, dfv = read_lc_dat2(path.stem, str(path.parent))
    return pd.concat([dfg, dfv], ignore_index=True) if not (dfg.empty and dfv.empty) else pd.DataFrame()


def score_lightcurve_path(path: Path, **kwargs) -> tuple[float, list[DipStats]]:
    df = _load_lightcurve_path(path)
    return compute_dipper_score(df, **kwargs)


def attach_dipper_scores(
    df_results: pd.DataFrame,
    *,
    only_significant: bool = True,
    dip_sigma: float = 1.0,
    edge_sigma: float = 0.5,
    min_fwhm_days: float = 1.5,
    min_delta_mag: float = 0.05,
) -> pd.DataFrame:
    """
    Attach dipper scores to an events.py results DataFrame.

    Expects a 'path' column that points to the .dat2 light curve.
    """
    df = df_results.copy()
    scores = []
    for _, row in df.iterrows():
        if only_significant and not bool(row.get("dip_significant", False)):
            scores.append((0.0, 0, 0))
            continue
        path = Path(str(row.get("path", "")))
        score, dips = score_lightcurve_path(
            path,
            dip_sigma=dip_sigma,
            edge_sigma=edge_sigma,
            min_fwhm_days=min_fwhm_days,
            min_delta_mag=min_delta_mag,
        )
        scores.append((score, len(dips), int(sum(1 for d in dips if d.valid))))
    df["dipper_score"] = [s[0] for s in scores]
    df["dipper_n_dips"] = [s[1] for s in scores]
    df["dipper_n_valid_dips"] = [s[2] for s in scores]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute dipper scores for events.py results.")
    parser.add_argument("--events", type=Path, required=True, help="events.py results CSV/Parquet")
    parser.add_argument("--output", type=Path, default=Path("/output/dipper_scores.csv"), help="Output CSV path")
    parser.add_argument("--only-significant", action="store_true", help="Score only dip_significant rows (default)")
    parser.add_argument("--include-all", action="store_true", help="Score all rows (overrides --only-significant)")
    parser.add_argument("--dip-sigma", type=float, default=1.0)
    parser.add_argument("--edge-sigma", type=float, default=0.5)
    parser.add_argument("--min-fwhm-days", type=float, default=1.5)
    parser.add_argument("--min-delta-mag", type=float, default=0.05)
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

    df_scored = attach_dipper_scores(
        df_events,
        only_significant=only_significant,
        dip_sigma=args.dip_sigma,
        edge_sigma=args.edge_sigma,
        min_fwhm_days=args.min_fwhm_days,
        min_delta_mag=args.min_delta_mag,
    )

    out_path = args.output.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_scored.to_csv(out_path, index=False)
    print(f"Wrote {len(df_scored)} rows to {out_path}")


if __name__ == "__main__":
    main()
