import numpy as np
import pandas as pd
import scipy
from astropy.stats import biweight_location, biweight_scale

def clean_lc(df):
    mask = np.ones(len(df), dtype=bool)

    if "saturated" in df.columns:
        mask &= (df["saturated"] == 0)

    mask &= df["JD"].notna() & df["mag"].notna()
    if "error" in df.columns:
        mask &= df["error"].notna() & (df["error"] > 0.) & (df["error"] < 1.)
    df = df.loc[mask]

    df = df.sort_values("JD").reset_index(drop=True)
    return df

def year_to_jd(year):
    jd_epoch = 2449718.5                                
    year_epoch = 1995
    days_in_year = 365.25
                                                                  
    return (year - year_epoch) * days_in_year + (jd_epoch - 2450000.0)


def jd_to_year(jd):
    jd_epoch = 2449718.5                                
    year_epoch = 1995
    days_in_year = 365.25
                                                               
    return year_epoch + ((jd + 2450000.0) - jd_epoch) / days_in_year


def peak_search_residual_baseline(
    df,
    prominence=0.17,
    distance=25,
    height=0.3,
    width=2,
    apply_box_filter=True,
    max_dips=10,
    max_std=0.15,
    max_peaks_per_time=0.015,
):
    """
    Peak finder that prefers per-camera-baseline residuals when available.
    Falls back to (mag - mean) if 'resid' is absent.
    """

    mag = np.asarray(df["mag"], float)
    jd = np.asarray(df["JD"], float)
    meanmag = float(np.nanmean(mag)) if mag.size else np.nan

    if "resid" in df.columns:
        resid = np.asarray(df["resid"], float)
        values = np.nan_to_num(resid, nan=0.0)
    else:
        values = mag - meanmag

    peak, _ = scipy.signal.find_peaks(
        values,
        prominence=prominence,
        distance=distance,
        height=height,
        width=width,
    )

    n_peaks = len(peak)

    if apply_box_filter:
        jd_span = float(jd[-1] - jd[0]) if jd.size > 1 else 0.0
        peaks_per_time = (n_peaks / jd_span) if jd_span > 0 else np.inf
        std_mag = float(np.nanstd(values))

        if (
            n_peaks == 0
            or n_peaks >= max_dips
            or peaks_per_time > max_peaks_per_time
            or std_mag > max_std
        ):
            return pd.Series(dtype=int, name="peaks"), meanmag, 0

    return pd.Series(peak, name="peaks"), meanmag, n_peaks


def peak_search_biweight_delta(
    df,
    sigma_threshold=3.0,
    distance=25,
    width=1,
    prominence=0.0,
    apply_box_filter=False,
    max_dips=10,
    max_peaks_per_time=0.015,
    max_std_sigma=2.5,
    *,
    mag_col="mag",
    t_col="JD",
    err_col="error",
    biweight_c=6.0,
    eps=1e-6,
):
    """
    Tzanidakis et al. (2025) biweight magnitude deviation
    """

    mag = np.asarray(df[mag_col], float) if mag_col in df.columns else np.array([], float)
    jd = np.asarray(df[t_col], float) if t_col in df.columns else np.array([], float)

    if err_col in df.columns:
        err = np.asarray(df[err_col], float)
    else:
        err = np.full_like(mag, np.nan, dtype=float)

    finite_m = np.isfinite(mag)

                       
    if finite_m.any():
        R = float(biweight_location(mag[finite_m], c=biweight_c))
        S = float(biweight_scale(mag[finite_m], c=biweight_c))
    else:
        R = np.nan
        S = 0.0

    if not np.isfinite(S) or S < 0:
        S = 0.0

    err2 = np.where(np.isfinite(err), err**2, 0.0)
    denom = np.sqrt(err2 + S**2)
    denom = np.where(denom > 0, denom, eps)

                                          
    delta = (mag - R) / denom
    values = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

    peak, _ = scipy.signal.find_peaks(
        values,
        height=sigma_threshold,
        distance=distance,
        width=width,
        prominence=prominence,
    )

    n_peaks = len(peak)

    if apply_box_filter:
                                                      
        jd_span = float(jd[-1] - jd[0]) if jd.size > 1 else 0.0
        peaks_per_time = (n_peaks / jd_span) if jd_span > 0 else np.inf
        std_sig = float(np.nanstd(values))

        if (
            n_peaks == 0
            or n_peaks >= max_dips
            or peaks_per_time > max_peaks_per_time
            or std_sig > max_std_sigma
        ):
            return pd.Series(dtype=int, name="peaks"), R, 0

    return pd.Series(peak, name="peaks"), R, n_peaks
def empty_metrics(prefix):
    vals = {
        'n_dip_runs': 0,
        'n_jump_runs': 0,
        'n_dip_points': 0,
        'n_jump_points': 0,
        'most_recent_dip': np.nan,
        'most_recent_jump': np.nan,
        'max_depth': np.nan,
        'max_height': np.nan,
        'max_dip_duration': np.nan,
        'max_jump_duration': np.nan,
        'dip_fraction': np.nan,
        'jump_fraction': np.nan,
    }
    out = {f'{prefix}_{k}': vals[k] for k in vals}
    out[f'{prefix}_is_dip_dominated'] = False
    return out
