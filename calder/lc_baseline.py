import numpy as np
import pandas as pd
#from celerite2 import terms, GaussianProcess
#from scipy.optimize import minimize

JD_OFFSET = 2458000.0
JD_MIN_REL = -2000.0


def _relative_times(jd_array):
    jd_array = np.asarray(jd_array, dtype=float)
    rel = jd_array - JD_OFFSET
    return jd_array, rel


def _valid_time_mask(rel_times):
    return np.isfinite(rel_times) & (rel_times >= JD_MIN_REL)


def global_mean_baseline(
    df,
    t_col="JD",         
    mag_col="mag",
    err_col="error",
):
    df_out = df.copy()
    for col in ("baseline", "resid", "sigma_resid"):
        if col not in df_out.columns:
            df_out[col] = np.nan

    t = df_out.loc[:, t_col].to_numpy(dtype=float)
    _, rel = _relative_times(t)
    time_mask = _valid_time_mask(rel)
    m = df_out.loc[:, mag_col].to_numpy(dtype=float)
    e = df_out.loc[:, err_col].to_numpy(dtype=float)

    baseline = np.full_like(m, np.nan, dtype=float)
    resid = np.full_like(m, np.nan, dtype=float)

    good = np.isfinite(m) & time_mask
    if good.any():
        mean_mag = float(np.mean(m[good]))
        baseline[good] = mean_mag
        resid[good] = m[good] - mean_mag

    resid_good = np.isfinite(resid)
    if resid_good.any():
        resid_vals = resid[resid_good]
        med_resid = float(np.median(resid_vals))
        mad = float(1.4826 * np.median(np.abs(resid_vals - med_resid)))
    else:
        med_resid = np.nan
        mad = np.nan

    e_good = time_mask & np.isfinite(e)
    e_med = float(np.median(e[e_good])) if e_good.any() else np.nan

    mad_num = mad if np.isfinite(mad) else 0.0
    e_med_num = e_med if np.isfinite(e_med) else 0.0
    robust_std = float(np.sqrt(mad_num**2 + e_med_num**2))
    robust_std = max(robust_std, 1e-6)

    sigma_resid = np.full_like(resid, np.nan)
    sigma_resid[resid_good] = resid[resid_good] / robust_std

    df_out.loc[:, "baseline"] = baseline
    df_out.loc[:, "resid"] = resid
    df_out.loc[:, "sigma_resid"] = sigma_resid

    return df_out

def global_median_baseline(
    df,
    t_col="JD",         
    mag_col="mag",
    err_col="error",
):
    df_out = df.copy()
    for col in ("baseline", "resid", "sigma_resid"):
        if col not in df_out.columns:
            df_out[col] = np.nan

    t = df_out.loc[:, t_col].to_numpy(dtype=float)
    _, rel = _relative_times(t)
    time_mask = _valid_time_mask(rel)
    m = df_out.loc[:, mag_col].to_numpy(dtype=float)
    e = df_out.loc[:, err_col].to_numpy(dtype=float)

    baseline = np.full_like(m, np.nan, dtype=float)
    resid = np.full_like(m, np.nan, dtype=float)

    good = np.isfinite(m) & time_mask
    if good.any():
        median_mag = float(np.median(m[good]))
        baseline[good] = median_mag
        resid[good] = m[good] - median_mag

    resid_good = np.isfinite(resid)
    if resid_good.any():
        resid_vals = resid[resid_good]
        med_resid = float(np.median(resid_vals))
        mad = float(1.4826 * np.median(np.abs(resid_vals - med_resid)))
    else:
        med_resid = np.nan
        mad = np.nan

    e_good = time_mask & np.isfinite(e)
    e_med = float(np.median(e[e_good])) if e_good.any() else np.nan

    mad_num = mad if np.isfinite(mad) else 0.0
    e_med_num = e_med if np.isfinite(e_med) else 0.0
    robust_std = float(np.sqrt(mad_num**2 + e_med_num**2))
    robust_std = max(robust_std, 1e-6)

    sigma_resid = np.full_like(resid, np.nan)
    sigma_resid[resid_good] = resid[resid_good] / robust_std

    df_out.loc[:, "baseline"] = baseline
    df_out.loc[:, "resid"] = resid
    df_out.loc[:, "sigma_resid"] = sigma_resid

    return df_out


# rolling time window helper functions for subsequent baselines

def rolling_time_median(jd, mag, days=300.0, min_points=10, min_days=30.0, past_only=True):
    """
    Rolling median in time using pure numpy optimization (searchsorted).
    If past_only=True, uses [t0 - days, t0] (one-sided) to avoid future-leakage into ongoing dips.
    Halves 'days' down to min_days until >= min_points exist.
    """

    jd = np.asarray(jd, dtype=float)
    mag = np.asarray(mag, dtype=float)
    out = np.full_like(mag, np.nan, dtype=float)

    _, rel = _relative_times(jd)
    valid_mask = _valid_time_mask(rel)
    valid_indices = np.nonzero(valid_mask)[0]
    if valid_indices.size == 0:
        return out

    jd_valid = rel[valid_mask]
    mag_valid = mag[valid_mask]
    order = np.argsort(jd_valid)
    jd_valid = jd_valid[order]
    mag_valid = mag_valid[order]
    valid_indices = valid_indices[order]

    for i, t0 in enumerate(jd_valid):
        window = float(days)

        while window >= float(min_days):
            if past_only:
                lo_val, hi_val = t0 - window, t0
            else:
                half = window / 2.0
                lo_val, hi_val = t0 - half, t0 + half
                
            idx_start = np.searchsorted(jd_valid, lo_val, side='left')
            idx_end = np.searchsorted(jd_valid, hi_val, side='right')
            
            # Slice the arrays (very fast view, no copy)
            vals = mag_valid[idx_start:idx_end]
            
            # Check for NaNs
            # (In pure numpy, we must filter them explicitly before median)
            finite_vals = vals[np.isfinite(vals)]
            
            if len(finite_vals) >= int(min_points):
                out[valid_indices[i]] = np.median(finite_vals)
                break
            
            window /= 2.0
            
    return out

def rolling_time_mad(jd, resid, days=200.0, min_points=10, min_days=20.0, past_only=True, add_err=None):
    """
    Rolling robust scatter (MAD) using pure NumPy optimization.
    1.4826 * median(|resid - median(resid)|)
    """
    jd = np.asarray(jd, dtype=float)
    resid = np.asarray(resid, dtype=float)
    out = np.full_like(resid, np.nan, dtype=float)

    _, rel = _relative_times(jd)
    valid_mask = _valid_time_mask(rel)
    valid_indices = np.nonzero(valid_mask)[0]
    if valid_indices.size == 0:
        return out

    jd_valid = rel[valid_mask]
    resid_valid = resid[valid_mask]
    order = np.argsort(jd_valid)
    jd_valid = jd_valid[order]
    resid_valid = resid_valid[order]
    valid_indices = valid_indices[order]
    
    # Handle add_err (scalar or array)
    # We prep this outside the loop to avoid overhead
    if add_err is not None:
        if np.ndim(add_err) > 0:
            err_is_array = True
            err_array = np.asarray(add_err, dtype=float)
            err_array = err_array[valid_mask][order]
        else:
            err_is_array = False
            err_scalar = float(add_err)
    else:
        err_is_array = False
    
    for i, t0 in enumerate(jd_valid):
        window = float(days)
        
        while window >= float(min_days):
            if past_only:
                lo_val, hi_val = t0 - window, t0
            else:
                half = window / 2.0
                lo_val, hi_val = t0 - half, t0 + half
            
            # BINARY SEARCH OPTIMIZATION
            idx_start = np.searchsorted(jd_valid, lo_val, side='left')
            idx_end = np.searchsorted(jd_valid, hi_val, side='right')
            
            vals = resid_valid[idx_start:idx_end]
            finite_vals = vals[np.isfinite(vals)]
            
            if len(finite_vals) >= int(min_points):
                med = np.median(finite_vals)
                # MAD calculation
                mad = 1.4826 * np.median(np.abs(finite_vals - med))
                
                if add_err is not None:
                    if err_is_array:
                        err_here = err_array[i]
                    else:
                        err_here = err_scalar
                    mad = np.sqrt(mad**2 + err_here**2)
                
                out[valid_indices[i]] = max(mad, 1e-6)
                break
            
            window /= 2.0
            
    return out

def per_camera_mean_baseline(
    df,
    t_col="JD",         
    mag_col="mag",
    err_col="error",
    cam_col="camera#",
):
    df_out = df.copy()
    for col in ("baseline", "resid", "sigma_resid"):
        if col not in df_out.columns:
            df_out[col] = np.nan

    for _, sub in df_out.groupby(cam_col, group_keys=False):
        idx = sub.index

        t = df_out.loc[idx, t_col].to_numpy(dtype=float)
        _, rel = _relative_times(t)
        time_mask = _valid_time_mask(rel)
        m = df_out.loc[idx, mag_col].to_numpy(dtype=float)
        e = df_out.loc[idx, err_col].to_numpy(dtype=float)

        baseline = np.full_like(m, np.nan, dtype=float)
        resid = np.full_like(m, np.nan, dtype=float)

        good = np.isfinite(m) & time_mask
        if good.any():
            cam_mean = float(np.mean(m[good]))
            baseline[good] = cam_mean
            resid[good] = m[good] - cam_mean

        resid_good = np.isfinite(resid)
        if resid_good.any():
            resid_vals = resid[resid_good]
            med_resid = float(np.median(resid_vals))
            mad = float(1.4826 * np.median(np.abs(resid_vals - med_resid)))
        else:
            med_resid = np.nan
            mad = np.nan

        e_good = time_mask & np.isfinite(e)
        e_med = float(np.median(e[e_good])) if e_good.any() else np.nan

        mad_num = mad if np.isfinite(mad) else 0.0
        e_med_num = e_med if np.isfinite(e_med) else 0.0
        robust_std = float(np.sqrt(mad_num**2 + e_med_num**2))
        robust_std = max(robust_std, 1e-6)

        sigma_resid = np.full_like(resid, np.nan, dtype=float)
        sigma_resid[resid_good] = resid[resid_good] / robust_std

        df_out.loc[idx, "baseline"] = baseline
        df_out.loc[idx, "resid"] = resid
        df_out.loc[idx, "sigma_resid"] = sigma_resid

    return df_out

def per_camera_median_baseline(
    df,
    days=300.0,
    min_points=10,
    t_col="JD",
    mag_col="mag",
    err_col="error",
    cam_col="camera#",
):
    """
    returns a df that mirrors input df but with three extra float columns: (1) baseline, a rolling 300-day median mag computed within each camera group; (2) resid, residual mag-baseline per-camera; (3) sigma_resid, residual divided by (MAD+mag_error) in quadrature, yielding a per-point significance
    """
    # work on a copy; initialize df_outputs
    df_out = df.copy()
    for col in ("baseline", "resid", "sigma_resid"):
        if col not in df_out.columns:
            df_out[col] = np.nan

    # group by camera and fill columns
    for _, sub in df_out.groupby(cam_col, group_keys=False):
        idx = sub.index

        t = df_out.loc[idx, t_col].to_numpy(dtype=float)
        _, rel = _relative_times(t)
        time_mask = _valid_time_mask(rel)
        m = df_out.loc[idx, mag_col].to_numpy(dtype=float)
        e = df_out.loc[idx, err_col].to_numpy(dtype=float)

        base = rolling_time_median(t, m, days=days, min_points=min_points)
        resid = m - base

        # robust scatter
        resid_good = np.isfinite(resid)
        if resid_good.any():
            resid_vals = resid[resid_good]
            med_resid = float(np.median(resid_vals))
            mad = float(1.4826 * np.median(np.abs(resid_vals - med_resid)))
        else:
            med_resid = np.nan
            mad = np.nan

        e_good = time_mask & np.isfinite(e)
        e_med = float(np.median(e[e_good])) if e_good.any() else np.nan

        mad_num = mad if np.isfinite(mad) else 0.0
        e_med_num = e_med if np.isfinite(e_med) else 0.0
        robust_std = float(np.sqrt(mad_num**2 + e_med_num**2))
        robust_std = max(robust_std, 1e-6)  # avoid 0/NaN

        sigma_resid = resid / robust_std

        df_out.loc[idx, "baseline"] = base
        df_out.loc[idx, "resid"] = resid
        df_out.loc[idx, "sigma_resid"] = sigma_resid

    return df_out

def per_camera_trend_baseline(
        df,
        days_short=50., 
        days_long=800.0, 
        min_points=10, 
        last_window_guard=120.0, 
        t_col="JD", 
        mag_col="mag", 
        err_col="error", 
        cam_col="camera#"):
    """
    Multi-scale, one-sided (past-only) rolling-median baseline per camera to avoid
    future-leakage; rolling MAD for local significance; late-window guard for right-censored dips.
    """

    df_out = df.copy()
    for col in ("baseline", "resid", "sigma_resid"):
        if col not in df_out.columns:
            df_out[col] = np.nan

    # process per camera, sorted by time
    for _, sub in df_out.groupby(cam_col, group_keys=False):
        idx = sub.sort_values(t_col).index
        t = df_out.loc[idx, t_col].to_numpy(float)
        _, rel = _relative_times(t)
        time_mask = _valid_time_mask(rel)
        m = df_out.loc[idx, mag_col].to_numpy(float)
        e = df_out.loc[idx, err_col].to_numpy(float)

        # multi-scale, past-only baselines
        base_s = rolling_time_median(t, m, days=days_short, min_points=min_points, past_only=True)
        base_l = rolling_time_median(t, m, days=days_long, min_points=min_points, past_only=True)

        # choose baseline adaptively: prefer long if |base_s - base_l| is small (slow trend), else short
        choose_short = np.isfinite(base_s) & np.isfinite(base_l) & (np.abs(base_s - base_l) > 0.05) # 0.05 mag heuristic
        baseline = np.where(choose_short & np.isfinite(base_s), base_s,
        np.where(np.isfinite(base_l), base_l, base_s))

        resid = m - baseline

        # rolling local robust scatter; add median(err) in quadrature
        finite_e = e[time_mask & np.isfinite(e)]
        e_med = float(np.median(finite_e)) if finite_e.size else 0.0
        sigma_loc = rolling_time_mad(t, resid, days=days_short, min_points=max(8, min_points//2),
        past_only=True, add_err=e_med)

        # late-window guard: within last N days, allow right-censored dips (don’t inflate sigma by requiring return)
        near_end = np.zeros(len(t), dtype=bool)
        if np.any(time_mask):
            tmax = np.nanmax(rel[time_mask])
            valid_rel = rel[time_mask]
            near_end_vals = (tmax - valid_rel) <= float(last_window_guard)
            near_end_indices = np.nonzero(time_mask)[0]
            near_end[near_end_indices[near_end_vals]] = True
        
        # If sigma_loc missing near the end, fall back to global robust + e_med
        if np.isnan(sigma_loc[near_end]).any():
            r_good = np.isfinite(resid)
            if r_good.any():
                med_r = np.nanmedian(resid[r_good])
                mad_r = 1.4826 * np.nanmedian(np.abs(resid[r_good] - med_r))
                robust = float(np.sqrt(max(mad_r, 0.0)**2 + max(e_med, 0.0)**2))
                sigma_loc[near_end & ~np.isfinite(sigma_loc)] = max(robust, 1e-6)

        sigma_loc = np.where(np.isfinite(sigma_loc), sigma_loc, 1e-6)
        sigma_resid = resid / sigma_loc

        df_out.loc[idx, "baseline"] = baseline
        df_out.loc[idx, "resid"] = resid
        df_out.loc[idx, "sigma_resid"] = sigma_resid

    return df_out
