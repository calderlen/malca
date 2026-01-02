import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import glob
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.optimize import curve_fit
from tqdm import tqdm
import warnings
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", message=".*Covariance of the parameters could not be estimated.*")
warnings.filterwarnings("ignore", message=".*overflow encountered in.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in.*", category=RuntimeWarning)

from lc_utils import read_lc_dat2
from baseline import per_camera_gp_baseline
from df_utils import clean_lc


MAG_BINS = ['12_12.5', '12.5_13', '13_13.5', '13.5_14', '14_14.5', '14.5_15']

DEFAULT_BASELINE_KWARGS = dict(
    S0=0.0005,
    w0=0.0031415926535897933,
    q=0.7,
    jitter=0.006,
    sigma_floor=None,
    add_sigma_eff_col=True,
)

def gaussian(t, amp, t0, sigma, baseline):
    """
    gaussian kernel + baseline term
    """
    return baseline + amp * np.exp(-0.5 * ((t - t0) / sigma) ** 2)

def paczynski(t, amp, t0, tE, baseline):
    """
    paczynski kernel + baseline term
    """
    tE = np.maximum(np.abs(tE), 1e-5)
    return baseline + amp / np.sqrt(1.0 + ((t - t0) / tE) ** 2)


def log_gaussian(x, mu, sigma):
    """
    ln p = -1/2 * ((x-mu)/sigma)^2 - ln(sigma) - 1/2 ln(2pi)
    """
    x = np.asarray(x, float)
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    sigma = np.clip(sigma, 1e-12, np.inf)
    z = (x - mu) / sigma
    return -0.5 * z**2 - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)


def logit_spaced_grid(p_min=1e-4, p_max=1.0 - 1e-4, n=80):
    """
    probability grid that is uniform in logit space, with minimum of 1e-12 and maximum of 1-1e-12
    """
    p_min = float(np.clip(p_min, 1e-12, 1 - 1e-12))
    p_max = float(np.clip(p_max, 1e-12, 1 - 1e-12))
    q_min = np.log(p_min / (1.0 - p_min))
    q_max = np.log(p_max / (1.0 - p_max))
    q = np.linspace(q_min, q_max, int(n))
    return 1.0 / (1.0 + np.exp(-q))


def default_mag_grid(baseline_mag: float, mags: np.ndarray, kind: str, n=60):
    """
    
    """
    mags_finite = mags[np.isfinite(mags)]
    if len(mags_finite) == 0:
        raise ValueError("No finite magnitude values for grid construction")
    lo, hi = np.nanpercentile(mags, [5, 95])
    if not (np.isfinite(lo) and np.isfinite(hi)):
        med = np.nanmedian(mags)
        lo, hi = med - 0.5, med + 0.5
    spread = max(hi - lo, 0.05)

    if kind == "dip":
        start = baseline_mag + 0.02
        stop = max(baseline_mag + 0.02, hi + 0.5 * spread)
    elif kind == "jump":
        start = min(baseline_mag - 0.02, lo - 0.5 * spread)
        stop = baseline_mag - 0.02
    else:
        raise ValueError("kind must be 'dip' or 'jump'")

    if start == stop:
        stop = start + (0.1 if kind == "dip" else -0.1)

    return np.linspace(start, stop, int(n))


def robust_median_dt_days(jd: np.ndarray) -> float:
    """
    
    """
    jd = np.asarray(jd, float)
    if jd.size < 2:
        return np.nan
    dt = np.diff(jd)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return np.nan
    return float(np.nanmedian(dt))



def bic(resid, err, n_params):
    """
    bayesian information criterion = chi2 + k * ln(n)
    """
    err = np.clip(err, 1e-9, np.inf)
    chi2 = np.nansum((resid / err) ** 2)
    n_points = len(resid)
    if n_points == 0:
        return np.inf
    return chi2 + n_params * np.log(n_points)


def classify_run_morphology(jd, mag, err, run_idx, kind="dip"):
    """
    fits gaussian vs paczynski vs noise kernel to an individual run, returns a dict with kernel chosen and params
    """
    pad = 5
    start_i = int(max(0, run_idx[0] - pad))
    end_i = int(min(len(jd), run_idx[-1] + pad + 1))
    
    t_seg = jd[start_i:end_i]
    y_seg = mag[start_i:end_i]
    e_seg = err[start_i:end_i]
    
    if len(t_seg) < 4:
        return {
            "morphology": "none", "bic": np.nan, "delta_bic_null": 0.0, "params": {}
        }

    baseline_guess = np.nanmedian(y_seg)
    abs_diff = np.abs(y_seg - baseline_guess)
    peak_local_idx = np.argmax(abs_diff)
    
    t0_guess = t_seg[peak_local_idx]
    amp_guess = y_seg[peak_local_idx] - baseline_guess
    sigma_guess = max((t_seg[-1] - t_seg[0]) / 4.0, 0.01)
    
    resid_null = y_seg - baseline_guess
    bic_null = bic(resid_null, e_seg, 1)
    
    best_bic = bic_null
    best_model = "noise"
    best_params = {}

    try:
        popt_g, _ = curve_fit(
            gaussian, t_seg, y_seg, 
            p0=[amp_guess, t0_guess, sigma_guess, baseline_guess],
            sigma=e_seg, maxfev=2000
        )
        resid_g = y_seg - gaussian(t_seg, *popt_g)
        bic_g = bic(resid_g, e_seg, 4)
        
        is_valid = (popt_g[0] > 0) if kind == "dip" else (popt_g[0] < 0)
        
        if is_valid and bic_g < (best_bic - 10):
            best_bic = bic_g
            best_model = "gaussian"
            best_params = {
                "amp": popt_g[0], "t0": popt_g[1], 
                "sigma": popt_g[2], "baseline": popt_g[3]
            }
    except Exception:
        pass

    if kind == "jump":
        try:
            amp_p_guess = -abs(amp_guess) 
            
            popt_p, _ = curve_fit(
                paczynski, t_seg, y_seg,
                p0=[amp_p_guess, t0_guess, sigma_guess, baseline_guess],
                sigma=e_seg, maxfev=2000
            )
            resid_p = y_seg - paczynski(t_seg, *popt_p)
            bic_p = bic(resid_p, e_seg, 4)

            is_valid_p = (popt_p[0] < 0)

            if is_valid_p and bic_p < (best_bic - 10): 
                best_bic = bic_p
                best_model = "paczynski"
                best_params = {
                    "amp": popt_p[0], "t0": popt_p[1], 
                    "tE": popt_p[2], "baseline": popt_p[3]
                }
        except Exception:
            pass

    return {
        "morphology": best_model,
        "bic": float(best_bic),
        "delta_bic_null": float(bic_null - best_bic),
        "params": best_params
    }



def build_runs(
    trig_idx: np.ndarray,
    jd: np.ndarray,
    *,
    allow_gap_points: int = 1,
    max_gap_days: float | None = None,
):
    """
    build runs from clustered triggered points
    """
    jd = np.asarray(jd, float)
    trig_idx = np.asarray(trig_idx, dtype=int)
    trig_idx = trig_idx[(trig_idx >= 0) & (trig_idx < jd.size)]
    if trig_idx.size == 0:
        return []

    trig_idx = np.unique(trig_idx)
    trig_idx.sort()

    cad = robust_median_dt_days(jd)
    if max_gap_days is None:
        if np.isfinite(cad):
            max_gap_days = max(5.0 * cad, 5.0)
        else:
            max_gap_days = 5.0
    max_gap_days = float(max_gap_days)

    max_index_step = int(allow_gap_points) + 1

    runs = []
    cur = [int(trig_idx[0])]
    for k in range(1, trig_idx.size):
        i_prev = cur[-1]
        i = int(trig_idx[k])

        idx_step = i - i_prev
        dt = jd[i] - jd[i_prev]

        if (idx_step <= max_index_step) and np.isfinite(dt) and (dt <= max_gap_days):
            cur.append(i)
        else:
            runs.append(np.asarray(cur, dtype=int))
            cur = [i]
    runs.append(np.asarray(cur, dtype=int))
    return runs


def filter_runs(
    runs,
    jd: np.ndarray,
    score_vec: np.ndarray,
    *,
    min_points: int = 3,
    min_duration_days: float | None = None,
    per_point_threshold: float | None = None,
    sum_threshold: float | None = None,
):
    """
    filter runs by minimum points, minimum duration, per_point_threshold, sum_threshold; returns dict of kept runs' starting and ending indices/JDs, number of points
    """
    jd = np.asarray(jd, float)
    score_vec = np.asarray(score_vec, float)

    cad = robust_median_dt_days(jd)
    if min_duration_days is None:
        if np.isfinite(cad):
            min_duration_days = max(2.0 * cad, 2.0)
        else:
            min_duration_days = 2.0
    min_duration_days = float(min_duration_days)

    kept = []
    summaries = []

    for r in runs:
        r = np.asarray(r, dtype=int)
        if r.size == 0:
            continue

        n = int(r.size)
        dur = float(jd[r[-1]] - jd[r[0]]) if n >= 2 else 0.0
        vals = score_vec[r]
        run_max = float(np.nanmax(vals)) if np.isfinite(vals).any() else np.nan
        run_sum = float(np.nansum(vals)) if np.isfinite(vals).any() else np.nan

        ok = True
        if n < int(min_points):
            ok = False
        if dur < min_duration_days:
            ok = False
        if (per_point_threshold is not None) and (not (np.isfinite(run_max) and run_max >= float(per_point_threshold))):
            ok = False
        if (sum_threshold is not None) and (not (np.isfinite(run_sum) and run_sum >= float(sum_threshold))):
            ok = False

        summaries.append(
            dict(
                start_idx=int(r[0]),
                end_idx=int(r[-1]),
                n_points=n,
                start_jd=float(jd[r[0]]),
                end_jd=float(jd[r[-1]]),
                duration_days=dur,
                run_max=run_max,
                run_sum=run_sum,
                kept=bool(ok),
            )
        )

        if ok:
            kept.append(r)

    return kept, summaries


def summarize_kept_runs(kept_runs, jd: np.ndarray, score_vec: np.ndarray):
    """
    
    """
    jd = np.asarray(jd, float)
    score_vec = np.asarray(score_vec, float)

    if not kept_runs:
        return dict(
            n_runs=0,
            max_run_points=0,
            max_run_duration=np.nan,
            max_run_sum=np.nan,
            max_run_max=np.nan,
        )

    max_pts = 0
    max_dur = -np.inf
    max_sum = -np.inf
    max_max = -np.inf

    for r in kept_runs:
        r = np.asarray(r, int)
        max_pts = max(max_pts, int(r.size))
        if r.size >= 2:
            max_dur = max(max_dur, float(jd[r[-1]] - jd[r[0]]))
        else:
            max_dur = max(max_dur, 0.0)

        vals = score_vec[r]
        if np.isfinite(vals).any():
            max_sum = max(max_sum, float(np.nansum(vals)))
            max_max = max(max_max, float(np.nanmax(vals)))

    return dict(
        n_runs=int(len(kept_runs)),
        max_run_points=int(max_pts),
        max_run_duration=float(max_dur) if np.isfinite(max_dur) else np.nan,
        max_run_sum=float(max_sum) if np.isfinite(max_sum) else np.nan,
        max_run_max=float(max_max) if np.isfinite(max_max) else np.nan,
    )


def bayesian_excursion_significance(
    df: pd.DataFrame,
    *,
    kind: str = "dip",
    mag_col: str = "mag",
    err_col: str = "error",

    baseline_func=per_camera_gp_baseline,
    baseline_kwargs: dict | None = None,
    df_base: pd.DataFrame | None = None,

    use_sigma_eff: bool = True,
    require_sigma_eff: bool = False,

    p_min: float | None = None,
    p_max: float | None = None,
    p_points: int = 80,
    mag_grid: np.ndarray | None = None,

    trigger_mode: str = "logbf",
    logbf_threshold: float = 5.0,
    significance_threshold: float = 99.99997,

    run_min_points: int = 3,
    run_allow_gap_points: int = 1,
    run_max_gap_days: float | None = None,
    run_min_duration_days: float | None = None,
    run_sum_threshold: float | None = None,
    run_sum_multiplier: float = 2.5,

    compute_event_prob: bool = True,
):
    """
    Returns a dict including:
      - log_bf_local (N,)
      - event_probability (N,) if compute_event_prob
      - event_indices (after run gating)
      - significant (after run gating)
      - run diagnostics
      - global bayes_factor
    """
    df = clean_lc(df)
    jd = np.asarray(df["JD"], float)
    mags = np.asarray(df[mag_col], float)

    mags_finite = np.isfinite(mags).sum()
    mags_total = len(mags)
    if mags_finite == 0:
        raise ValueError(
            f"All magnitudes are NaN/inf after reading: "
            f"total={mags_total}, finite={mags_finite}, "
            f"NaN={np.isnan(mags).sum()}, inf={np.isinf(mags).sum()}"
        )

    if err_col in df.columns:
        errs = np.asarray(df[err_col], float)
    else:
        errs = np.full_like(mags, 0.05)
    
    errs_finite = np.isfinite(errs).sum()
    errs_positive = (errs > 0).sum() if errs_finite > 0 else 0
    if errs_finite == 0:
        raise ValueError(
            f"All errors are NaN/inf: "
            f"total={len(errs)}, finite={errs_finite}, "
            f"NaN={np.isnan(errs).sum()}, inf={np.isinf(errs).sum()}"
        )
    if errs_positive == 0:
        raise ValueError(
            f"All errors are non-positive: "
            f"total={len(errs)}, finite={errs_finite}, positive={errs_positive}, "
            f"min={np.nanmin(errs) if errs_finite > 0 else 'N/A'}"
        )

    if baseline_kwargs is None:
        baseline_kwargs = dict(DEFAULT_BASELINE_KWARGS)

    used_sigma_eff = False

    if df_base is None and baseline_func is not None:
        df_base = baseline_func(df, **baseline_kwargs)

    if df_base is None:
        if not np.isfinite(mags).any():
            raise ValueError("All magnitude values are NaN/inf")
        baseline_mags = np.full_like(mags, np.nanmedian(mags))
        baseline_sources = np.full(len(mags), "global_median", dtype=object)
    else:
        if "baseline" in df_base.columns:
            baseline_mags = np.asarray(df_base["baseline"], float)
        else:
            baseline_mags = np.asarray(df_base[mag_col], float)
        if "baseline_source" in df_base.columns:
            baseline_sources = np.asarray(df_base["baseline_source"], dtype=object)
        else:
            baseline_sources = np.full(len(df_base), "unknown", dtype=object)

        if use_sigma_eff and ("sigma_eff" in df_base.columns):
            errs_new = np.asarray(df_base["sigma_eff"], float)
            errs_new_finite = np.isfinite(errs_new).sum()
            errs_new_positive = (errs_new > 0).sum() if errs_new_finite > 0 else 0
            if errs_new_finite == 0:
                raise ValueError(
                    f"Baseline returned all NaN/inf sigma_eff: "
                    f"total={len(errs_new)}, finite={errs_new_finite}, "
                    f"NaN={np.isnan(errs_new).sum()}, inf={np.isinf(errs_new).sum()}"
                )
            if errs_new_positive == 0:
                raise ValueError(
                    f"Baseline returned all non-positive sigma_eff: "
                    f"total={len(errs_new)}, finite={errs_new_finite}, positive={errs_new_positive}, "
                    f"min={np.nanmin(errs_new) if errs_new_finite > 0 else 'N/A'}"
                )
            errs = errs_new
            used_sigma_eff = True
        elif require_sigma_eff:
            raise RuntimeError("require_sigma_eff=True but baseline did not return 'sigma_eff'")

    baseline_finite = np.isfinite(baseline_mags).sum()
    if baseline_finite == 0:
        raise ValueError(
            f"Baseline function returned all NaN/inf values: "
            f"total={len(baseline_mags)}, finite={baseline_finite}, "
            f"NaN={np.isnan(baseline_mags).sum()}, inf={np.isinf(baseline_mags).sum()}, "
            f"baseline_func={baseline_func.__name__ if baseline_func else 'None'}"
        )
    
    errs_finite_final = np.isfinite(errs).sum()
    errs_positive_final = (errs > 0).sum() if errs_finite_final > 0 else 0
    if errs_finite_final == 0:
        raise ValueError(
            f"All errors are NaN/inf after baseline: "
            f"total={len(errs)}, finite={errs_finite_final}, "
            f"NaN={np.isnan(errs).sum()}, inf={np.isinf(errs).sum()}"
        )
    if errs_positive_final == 0:
        raise ValueError(
            f"All errors are non-positive after baseline: "
            f"total={len(errs)}, finite={errs_finite_final}, positive={errs_positive_final}, "
            f"min={np.nanmin(errs) if errs_finite_final > 0 else 'N/A'}"
        )
    
    total_points = len(mags)
    valid_mask = (
        np.isfinite(mags)
        & np.isfinite(errs)
        & (errs > 0)
        & np.isfinite(baseline_mags)
    )
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        raise ValueError(
            "No valid points after baseline/error filtering: "
            f"total={total_points}, finite_mags={np.isfinite(mags).sum()}, "
            f"finite_errs={np.isfinite(errs).sum()}, positive_errs={(errs > 0).sum()}, "
            f"finite_baseline={np.isfinite(baseline_mags).sum()}"
        )
    if n_valid < total_points:
        mags = mags[valid_mask]
        errs = errs[valid_mask]
        baseline_mags = baseline_mags[valid_mask]
        baseline_sources = baseline_sources[valid_mask]
        jd = jd[valid_mask]

    baseline_mag = float(np.nanmedian(baseline_mags))

    if p_min is None and p_max is None:
        if kind == "dip":
            p_min, p_max = 0.5, 1.0 - 1e-4
        elif kind == "jump":
            p_min, p_max = 1e-4, 0.5
        else:
            raise ValueError("kind must be 'dip' or 'jump'")

    p_grid = logit_spaced_grid(p_min=p_min, p_max=p_max, n=p_points)

    if mag_grid is None:
        mag_grid = default_mag_grid(baseline_mag, mags, kind, n=60)
    else:
        mag_grid = np.asarray(mag_grid, float)

    M = int(len(mag_grid))
    N = int(len(mags))

    if kind == "dip":
        log_Pb_vec = log_gaussian(mags, baseline_mags, errs)
        log_Pb_grid = np.broadcast_to(log_Pb_vec, (M, N))
        log_Pf_grid = log_gaussian(mags[None, :], mag_grid[:, None], errs)
        excursion_component = "faint"

    elif kind == "jump":
        log_Pb_grid = log_gaussian(mags[None, :], mag_grid[:, None], errs)
        log_Pf_vec = log_gaussian(mags, baseline_mags, errs)
        log_Pf_grid = np.broadcast_to(log_Pf_vec, (M, N))
        excursion_component = "bright"
        
        if not np.isfinite(log_Pf_vec).any():
            raise ValueError("All baseline likelihood values are NaN/inf")
        if not np.isfinite(log_Pb_grid).any():
            raise ValueError("All excursion likelihood values are NaN/inf")

    else:
        raise ValueError("kind must be 'dip' or 'jump'")

    valid_points = (np.isfinite(log_Pb_grid).any(axis=0)) | (np.isfinite(log_Pf_grid).any(axis=0))
    n_valid_points = int(valid_points.sum())
    total_points = log_Pb_grid.shape[1]
    if n_valid_points == 0:
        raise ValueError(
            "No valid likelihood contributions after baseline: "
            f"total={total_points}, baseline_finite={np.isfinite(log_Pb_grid).sum()}, "
            f"excursion_finite={np.isfinite(log_Pf_grid).sum()}"
        )
    if n_valid_points < total_points:
        mags = mags[valid_points]
        errs = errs[valid_points]
        baseline_mags = baseline_mags[valid_points]
        baseline_sources = baseline_sources[valid_points]
        jd = jd[valid_points]
        log_Pb_grid = log_Pb_grid[:, valid_points]
        log_Pf_grid = log_Pf_grid[:, valid_points]
        if kind == "dip":
            log_Pb_vec = log_Pb_vec[valid_points]
        else:
            log_Pf_vec = log_Pf_vec[valid_points]
        N = n_valid_points

    if kind == "dip":
        loglik_baseline_only = float(np.sum(log_Pb_vec))
        log_px_baseline = log_Pb_vec
        log_px_event = logsumexp(log_Pf_grid, axis=0) - np.log(M)
    else:
        loglik_baseline_only = float(np.sum(log_Pf_vec))
        log_px_baseline = log_Pf_vec
        log_px_event = logsumexp(log_Pb_grid, axis=0) - np.log(M)

    log_bf_local = log_px_event - log_px_baseline

    max_log_bf_local = float(np.nanmax(log_bf_local)) if np.isfinite(log_bf_local).any() else np.nan

    log_p = np.log(p_grid)
    log_1mp = np.log1p(-p_grid)

    log_Pb_weighted = log_p[None, :, None] + log_Pb_grid[:, None, :]
    log_Pf_weighted = log_1mp[None, :, None] + log_Pf_grid[:, None, :]

    log_Pb_weighted = np.where(np.isfinite(log_Pb_weighted), log_Pb_weighted, -np.inf)
    log_Pf_weighted = np.where(np.isfinite(log_Pf_weighted), log_Pf_weighted, -np.inf)

    log_mix = np.logaddexp(log_Pb_weighted, log_Pf_weighted)
    
    log_mix_finite = np.isfinite(log_mix).sum()
    log_mix_total = log_mix.size
    if log_mix_finite == 0:
        raise ValueError(
            f"All log_mix values are NaN/inf: "
            f"total={log_mix_total}, finite={log_mix_finite}, "
            f"log_Pb_weighted_finite={np.isfinite(log_Pb_weighted).sum()}/{log_Pb_weighted.size}, "
            f"log_Pf_weighted_finite={np.isfinite(log_Pf_weighted).sum()}/{log_Pf_weighted.size}"
        )
    
    loglik = np.sum(log_mix, axis=2)

    loglik_finite = np.isfinite(loglik).sum()
    loglik_total = loglik.size
    loglik_inf_neg = np.isinf(loglik) & (loglik < 0)
    loglik_inf_neg_count = loglik_inf_neg.sum()
    
    if loglik_finite == 0:
        if loglik_inf_neg_count == loglik_total:
            raise ValueError(
                f"All loglik values are -inf (all inputs were invalid): "
                f"total={loglik_total}, finite={loglik_finite}, -inf={loglik_inf_neg_count}, "
                f"This indicates all data points or baseline values were invalid."
            )
        else:
            raise ValueError(
                f"All loglik values are NaN/inf before normalization: "
                f"total={loglik_total}, finite={loglik_finite}, "
                f"NaN={np.isnan(loglik).sum()}, -inf={loglik_inf_neg_count}, +inf={np.isinf(loglik).sum() - loglik_inf_neg_count}"
            )
    
    loglik_sum = logsumexp(loglik)
    if not np.isfinite(loglik_sum):
        raise ValueError(
            f"logsumexp(loglik) is NaN/inf: "
            f"loglik_sum={loglik_sum}, loglik_finite={loglik_finite}/{loglik_total}, "
            f"loglik_min={np.nanmin(loglik) if loglik_finite > 0 else 'N/A'}, "
            f"loglik_max={np.nanmax(loglik) if loglik_finite > 0 else 'N/A'}"
        )
    
    log_post_norm = loglik - loglik_sum
    
    log_post_finite = np.isfinite(log_post_norm).sum()
    if log_post_finite == 0:
        raise ValueError(
            f"All log_posterior values are NaN/inf after normalization: "
            f"total={log_post_norm.size}, finite={log_post_finite}, "
            f"loglik_finite={loglik_finite}/{loglik_total}, loglik_sum={loglik_sum}, "
            f"loglik_range=[{np.nanmin(loglik) if loglik_finite > 0 else 'N/A'}, {np.nanmax(loglik) if loglik_finite > 0 else 'N/A'}]"
        )
    
    best_m_idx, best_p_idx = np.unravel_index(np.nanargmax(log_post_norm), log_post_norm.shape)
    best_mag_event = float(mag_grid[int(best_m_idx)])
    best_p = float(p_grid[int(best_p_idx)])

    K = loglik.size
    log_evidence_mixture = float(logsumexp(loglik) - np.log(K))
    bayes_factor = float(log_evidence_mixture - loglik_baseline_only)

    if compute_event_prob:
        event_prob = np.zeros(N, dtype=float)
        for j in range(N):
            loglik_excl = loglik - log_mix[:, :, j]

            bright_num = loglik_excl + log_p[None, :] + log_Pb_grid[:, j][:, None]
            faint_num = loglik_excl + log_1mp[None, :] + log_Pf_grid[:, j][:, None]

            log_bright = logsumexp(bright_num)
            log_faint = logsumexp(faint_num)
            log_norm = logsumexp(np.array([log_bright, log_faint]))

            bright_prob = float(np.exp(log_bright - log_norm))
            faint_prob = float(np.exp(log_faint - log_norm))

            if excursion_component == "faint":
                event_prob[j] = faint_prob
            else:
                event_prob[j] = bright_prob
    else:
        event_prob = None

    if trigger_mode == "logbf":
        per_point_thr = float(logbf_threshold)
        score_vec = np.asarray(log_bf_local, float)
        raw_idx = np.nonzero(np.isfinite(score_vec) & (score_vec >= per_point_thr))[0]
        trigger_threshold_used = per_point_thr
        trigger_value_max = max_log_bf_local

        if run_sum_threshold is None:
            run_sum_threshold_eff = float(run_sum_multiplier) * per_point_thr
        else:
            run_sum_threshold_eff = float(run_sum_threshold)

    elif trigger_mode == "posterior_prob":
        if event_prob is None:
            raise RuntimeError("trigger_mode='posterior_prob' requires compute_event_prob=True")

        thr_prob = significance_threshold / 100.0 if significance_threshold > 1.0 else float(significance_threshold)
        score_vec = np.asarray(event_prob, float)
        raw_idx = np.nonzero(np.isfinite(score_vec) & (score_vec >= thr_prob))[0]
        trigger_threshold_used = thr_prob
        trigger_value_max = float(np.nanmax(score_vec)) if score_vec.size else np.nan

        if run_sum_threshold is None:
            run_sum_threshold_eff = float(run_min_points) * thr_prob
        else:
            run_sum_threshold_eff = float(run_sum_threshold)

    else:
        raise ValueError("trigger_mode must be 'logbf' or 'posterior_prob'")

    kept_runs = []
    run_summaries = []

    if raw_idx.size == 0:
        event_indices = np.array([], dtype=int)
        significant = False
        run_stats = summarize_kept_runs([], jd, score_vec)
    else:
        runs = build_runs(
            raw_idx,
            jd,
            allow_gap_points=int(run_allow_gap_points),
            max_gap_days=run_max_gap_days,
        )

        kept_runs, initial_summaries = filter_runs(
            runs,
            jd,
            score_vec,
            min_points=int(run_min_points),
            min_duration_days=run_min_duration_days,
            per_point_threshold=trigger_threshold_used,
            sum_threshold=run_sum_threshold_eff,
        )

        final_summaries = []
        for i, r in enumerate(kept_runs):
            summary = initial_summaries[i]
            morph_res = classify_run_morphology(jd, mags, errs, r, kind=kind)
            summary.update(morph_res)
            final_summaries.append(summary)
        
        run_summaries = final_summaries

        if kept_runs:
            event_indices = np.unique(np.concatenate(kept_runs)).astype(int)
            significant = True
        else:
            event_indices = np.array([], dtype=int)
            significant = False

        run_stats = summarize_kept_runs(kept_runs, jd, score_vec)

    return dict(
        kind=str(kind),
        baseline_mag=float(baseline_mag),
        best_mag_event=float(best_mag_event),
        best_p=float(best_p),

        log_bf_local=log_bf_local,
        max_log_bf_local=float(max_log_bf_local) if np.isfinite(max_log_bf_local) else np.nan,
        event_probability=event_prob,
        used_sigma_eff=bool(used_sigma_eff),

        trigger_mode=str(trigger_mode),
        trigger_threshold=float(trigger_threshold_used),
        trigger_max=float(trigger_value_max) if np.isfinite(trigger_value_max) else np.nan,
        event_indices=event_indices,
        significant=bool(significant),

        run_sum_threshold=float(run_sum_threshold_eff),
        run_summaries=run_summaries,
        **run_stats,

        bayes_factor=float(bayes_factor),
        log_evidence_mixture=float(log_evidence_mixture),
        log_evidence_baseline=float(loglik_baseline_only),
        baseline_source=",".join(sorted({str(x) for x in baseline_sources if isinstance(x, (str, bytes)) and len(str(x)) > 0})) or "unknown",

        p_grid=p_grid,
        mag_grid=mag_grid,
    )


def run_bayesian_significance(
    df: pd.DataFrame,
    *,
    baseline_func=per_camera_gp_baseline,
    baseline_kwargs: dict | None = None,

    p_points: int = 80,
    p_min_dip: float | None = None,
    p_max_dip: float | None = None,
    p_min_jump: float | None = None,
    p_max_jump: float | None = None,
    mag_grid_dip: np.ndarray | None = None,
    mag_grid_jump: np.ndarray | None = None,

    trigger_mode: str = "logbf",
    logbf_threshold_dip: float = 5.0,
    logbf_threshold_jump: float = 5.0,
    significance_threshold: float = 99.99997,

    run_min_points: int = 3,
    run_allow_gap_points: int = 1,
    run_max_gap_days: float | None = None,
    run_min_duration_days: float | None = None,
    run_sum_threshold: float | None = None,
    run_sum_multiplier: float = 2.5,

    use_sigma_eff: bool = True,
    require_sigma_eff: bool = True,

    compute_event_prob: bool = True,
):
    """
    compute baseline one then reuse it for dip and jump scoring
    """
    df = clean_lc(df)

    if baseline_kwargs is None:
        baseline_kwargs = dict(DEFAULT_BASELINE_KWARGS)

    df_base = baseline_func(df, **baseline_kwargs) if baseline_func is not None else None

    dip = bayesian_excursion_significance(
        df,
        kind="dip",
        baseline_func=None,
        baseline_kwargs=baseline_kwargs,
        df_base=df_base,
        use_sigma_eff=use_sigma_eff,
        require_sigma_eff=require_sigma_eff,
        p_min=p_min_dip,
        p_max=p_max_dip,
        p_points=p_points,
        mag_grid=mag_grid_dip,
        trigger_mode=trigger_mode,
        logbf_threshold=logbf_threshold_dip,
        significance_threshold=significance_threshold,
        run_min_points=run_min_points,
        run_allow_gap_points=run_allow_gap_points,
        run_max_gap_days=run_max_gap_days,
        run_min_duration_days=run_min_duration_days,
        run_sum_threshold=run_sum_threshold,
        run_sum_multiplier=run_sum_multiplier,
        compute_event_prob=compute_event_prob,
    )

    jump = bayesian_excursion_significance(
        df,
        kind="jump",
        baseline_func=None,
        baseline_kwargs=baseline_kwargs,
        df_base=df_base,
        use_sigma_eff=use_sigma_eff,
        require_sigma_eff=require_sigma_eff,
        p_min=p_min_jump,
        p_max=p_max_jump,
        p_points=p_points,
        mag_grid=mag_grid_jump,
        trigger_mode=trigger_mode,
        logbf_threshold=logbf_threshold_jump,
        significance_threshold=significance_threshold,
        run_min_points=run_min_points,
        run_allow_gap_points=run_allow_gap_points,
        run_max_gap_days=run_max_gap_days,
        run_min_duration_days=run_min_duration_days,
        run_sum_threshold=run_sum_threshold,
        run_sum_multiplier=run_sum_multiplier,
        compute_event_prob=compute_event_prob,
    )

    return dict(dip=dip, jump=jump)



def _process_one(
    path: str,
    *,
    trigger_mode: str,
    logbf_threshold_dip: float,
    logbf_threshold_jump: float,
    significance_threshold: float,
    p_points: int,

    run_min_points: int,
    run_allow_gap_points: int,
    run_max_gap_days: float | None,
    run_min_duration_days: float | None,
    run_sum_threshold: float | None,
    run_sum_multiplier: float,

    compute_event_prob: bool,
):
    """
    
    """
    import os
    from df_plot import read_skypatrol_csv
    path = str(path)
    
    if os.path.isfile(path) and (path.endswith('.csv') or path.endswith('-light-curves.csv')):
        try:
            result = read_skypatrol_csv(path)
            if isinstance(result, tuple):
                if len(result) == 2:
                    df = pd.concat([result[0], result[1]], ignore_index=True) if not (result[0].empty and result[1].empty) else pd.DataFrame()
                else:
                    raise ValueError(f"read_skypatrol_csv returned unexpected tuple length: {len(result)}")
            else:
                df = result
        except ValueError as e:
            raise ValueError(f"Error reading {path}: {e}")
    elif os.path.isfile(path) and path.endswith('.dat2'):
        dir_path = os.path.dirname(path) or '.'
        basename = os.path.basename(path)
        asassn_id = basename.replace('.dat2', '')
        try:
            dfg, dfv = read_lc_dat2(asassn_id, dir_path)
            df = pd.concat([dfg, dfv], ignore_index=True) if not (dfg.empty and dfv.empty) else pd.DataFrame()
        except Exception as e:
            raise ValueError(f"Error reading .dat2 file {path}: {e}")
    else:
        import glob
        csv_files = sorted(glob.glob(os.path.join(path, '*-light-curves.csv')))
        if csv_files:
            df = read_skypatrol_csv(csv_files[0])
        else:
            raise ValueError(f"Cannot read light curve from path (not a CSV file and no CSVs found in directory): {path}")

    if df.empty:
        raise ValueError(f"Empty dataframe read from {path}")
    
    required_cols = ["JD", "mag", "error"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}")
    
    valid_mask = (
        np.isfinite(df["JD"]) & 
        np.isfinite(df["mag"]) & 
        np.isfinite(df["error"]) &
        (df["error"] > 0) &
        (df["error"] < 10)
    )
    
    df = df[valid_mask].copy()
    
    if len(df) < 10:
        raise ValueError(f"Insufficient valid data points ({len(df)} < 10) in {path}")
    
    res = run_bayesian_significance(
        df,
        trigger_mode=trigger_mode,
        logbf_threshold_dip=logbf_threshold_dip,
        logbf_threshold_jump=logbf_threshold_jump,
        significance_threshold=significance_threshold,
        p_points=p_points,

        run_min_points=run_min_points,
        run_allow_gap_points=run_allow_gap_points,
        run_max_gap_days=run_max_gap_days,
        run_min_duration_days=run_min_duration_days,
        run_sum_threshold=run_sum_threshold,
        run_sum_multiplier=run_sum_multiplier,

        compute_event_prob=compute_event_prob,
        use_sigma_eff=True,
        require_sigma_eff=True,
    )

    dip = res["dip"]
    jump = res["jump"]

    def get_best_morph_info(run_list):
        """
        
        """
        if not run_list:
            return "none", 0.0, 0.0
        best = sorted(run_list, key=lambda x: x['run_sum'], reverse=True)[0]
        
        morph = best.get('morphology', 'none')
        delta_bic = best.get('delta_bic_null', 0.0)
        
        params = best.get('params', {})
        if morph == 'gaussian':
            main_param = params.get('sigma', np.nan)
        elif morph == 'paczynski':
            main_param = params.get('tE', np.nan)
        else:
            main_param = np.nan
            
        return morph, float(delta_bic), float(main_param)

    dip_morph, dip_dbic, dip_param = get_best_morph_info(dip["run_summaries"])
    jump_morph, jump_dbic, jump_param = get_best_morph_info(jump["run_summaries"])

    return dict(
        path=str(path),

        dip_significant=bool(dip["significant"]),
        jump_significant=bool(jump["significant"]),

        dip_best_morph=str(dip_morph),
        dip_best_delta_bic=float(dip_dbic),
        dip_best_width_param=float(dip_param),
        
        jump_best_morph=str(jump_morph),
        jump_best_delta_bic=float(jump_dbic),
        jump_best_width_param=float(jump_param),

        dip_count=int(len(dip["event_indices"])),
        jump_count=int(len(jump["event_indices"])),

        dip_run_count=int(dip.get("n_runs", 0)),
        jump_run_count=int(jump.get("n_runs", 0)),

        dip_max_run_points=int(dip.get("max_run_points", 0)),
        jump_max_run_points=int(jump.get("max_run_points", 0)),
        dip_max_run_duration=float(dip.get("max_run_duration", np.nan)),
        jump_max_run_duration=float(jump.get("max_run_duration", np.nan)),
        dip_max_run_sum=float(dip.get("max_run_sum", np.nan)),
        jump_max_run_sum=float(jump.get("max_run_sum", np.nan)),
        dip_max_run_max=float(dip.get("max_run_max", np.nan)),
        jump_max_run_max=float(jump.get("max_run_max", np.nan)),

        dip_max_log_bf_local=float(dip.get("max_log_bf_local", np.nan)),
        jump_max_log_bf_local=float(jump.get("max_log_bf_local", np.nan)),

        dip_bayes_factor=float(dip["bayes_factor"]),
        jump_bayes_factor=float(jump["bayes_factor"]),

        dip_best_p=float(dip["best_p"]),
        jump_best_p=float(jump["best_p"]),

        used_sigma_eff=bool(dip.get("used_sigma_eff", False) and jump.get("used_sigma_eff", False)),
        baseline_source=str(dip.get("baseline_source", jump.get("baseline_source", "unknown"))),
        trigger_mode=str(trigger_mode),
        dip_trigger_threshold=float(dip.get("trigger_threshold", np.nan)),
        jump_trigger_threshold=float(jump.get("trigger_threshold", np.nan)),
        dip_run_sum_threshold=float(dip.get("run_sum_threshold", np.nan)),
        jump_run_sum_threshold=float(jump.get("run_sum_threshold", np.nan)),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run Bayesian excursion scoring on light curves in parallel."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Paths to light-curve files (optional if using --mag-bin)",
    )
    parser.add_argument(
        "--mag-bin",
        dest="mag_bins",
        action="append",
        choices=MAG_BINS,
        help="Process all light curves in this magnitude bin (can specify multiple times). "
             "Light curves are found in <lc-path>/<mag_bin>/lc*_cal/*.dat2",
    )
    parser.add_argument(
        "--lc-path",
        type=str,
        default="/data/poohbah/1/assassin/rowan.90/lcsv2",
        help="Base path to light curve directories (default: /data/poohbah/1/assassin/rowan.90/lcsv2)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (mp.cpu_count() or 1) - 20),
        help="Number of worker processes (default: cpu_count-20)",
    )

    parser.add_argument(
        "--trigger-mode",
        type=str,
        default="logbf",
        choices=["logbf", "posterior_prob"],
        help="Triggering mode: 'logbf' (default) or 'posterior_prob' (slow)",
    )
    parser.add_argument(
        "--logbf-threshold-dip",
        type=float,
        default=5.0,
        help="Per-point dip trigger: log BF_j >= this (default 5.0 ~ BF>150)",
    )
    parser.add_argument(
        "--logbf-threshold-jump",
        type=float,
        default=5.0,
        help="Per-point jump trigger: log BF_j >= this (default 5.0 ~ BF>150)",
    )
    parser.add_argument(
        "--significance-threshold",
        type=float,
        default=99.99997,
        help="Only used if --trigger-mode posterior_prob (default ~ 5-sigma)",
    )
    parser.add_argument(
        "--p-points",
        type=int,
        default=80,
        help="Number of points in the logit-spaced p grid (default 80)",
    )

    parser.add_argument(
        "--run-min-points",
        type=int,
        default=3,
        help="Min triggered points in a run (default 3)",
    )
    parser.add_argument(
        "--run-allow-gap-points",
        type=int,
        default=1,
        help="Allow up to this many missing indices inside a run (default 1)",
    )
    parser.add_argument(
        "--run-max-gap-days",
        type=float,
        default=None,
        help="Break runs if JD gap exceeds this (default cadence-adaptive)",
    )
    parser.add_argument(
        "--run-min-duration-days",
        type=float,
        default=None,
        help="Require run duration >= this (default cadence-adaptive)",
    )
    parser.add_argument(
        "--run-sum-threshold",
        type=float,
        default=None,
        help="Require run sum-score >= this (default derived from per-point threshold)",
    )
    parser.add_argument(
        "--run-sum-multiplier",
        type=float,
        default=2.5,
        help="If --run-sum-threshold is None: sum_thr = multiplier * per_point_thr (logbf mode)",
    )

    parser.add_argument(
        "--no-event-prob",
        action="store_true",
        help="Skip LOO event responsibilities (faster). Required unless using posterior_prob triggering.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="/home/lenhart.106/Documents/asassn-variability/outputs/lc_excursions_bayes_results.parquet",
        help="Parquet path for results. If --mag-bin is set, the bin name is appended to the filename.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Write Parquet in chunks of this many results (default: 10000). Set to 0 to disable chunking; use None for auto.",
    )

    args = parser.parse_args()

    if args.trigger_mode == "posterior_prob" and args.no_event_prob:
        raise SystemExit("posterior_prob triggering requires event_prob; remove --no-event-prob")

    compute_event_prob = (not args.no_event_prob)

    expanded_inputs = []
    
    if args.mag_bins:
        lc_path = args.lc_path
        for mag_bin in args.mag_bins:
            mag_bin_dir = os.path.join(lc_path, mag_bin)
            if not os.path.exists(mag_bin_dir):
                print(f"Warning: mag_bin directory does not exist: {mag_bin_dir}", flush=True)
                continue
            
            lc_dirs = sorted(glob.glob(os.path.join(mag_bin_dir, "lc*_cal")))
            if not lc_dirs:
                print(f"Warning: No lc*_cal directories found in {mag_bin_dir}", flush=True)
                continue
            
            print(f"Found {len(lc_dirs)} lc*_cal directories in {mag_bin}...", flush=True)
            
            for lc_dir in lc_dirs:
                dat2_files = sorted(glob.glob(os.path.join(lc_dir, "*.dat2")))
                if dat2_files:
                    expanded_inputs.extend(dat2_files)
                    print(f"  {os.path.basename(lc_dir)}: {len(dat2_files)} .dat2 files", flush=True)
            
            print(f"Total .dat2 files in {mag_bin}: {len([f for f in expanded_inputs if mag_bin in f])}", flush=True)
    
    for pattern in args.inputs:
        if '*' in pattern or '?' in pattern or '[' in pattern:
            matches = glob.glob(pattern)
            if matches:
                expanded_inputs.extend(sorted(matches))
            else:
                print(f"Warning: glob pattern '{pattern}' matched no files", flush=True)
        else:
            expanded_inputs.append(pattern)
    
    seen = set()
    expanded_inputs = [x for x in expanded_inputs if not (x in seen or seen.add(x))]
    
    if not expanded_inputs:
        raise SystemExit("No input files found. Specify --mag-bin or provide input file paths.")
    
    print(f"Processing {len(expanded_inputs)} light curve file(s)...", flush=True)

    results = []
    errors = []
    
    if args.chunk_size is None:
        if len(expanded_inputs) < 10000:
            chunk_size = 500
        elif len(expanded_inputs) < 100000:
            chunk_size = 1000
        else:
            chunk_size = 5000
        print(f"Auto-selected chunk size: {chunk_size} (based on {len(expanded_inputs)} files)", flush=True)
    elif args.chunk_size > 0:
        chunk_size = args.chunk_size
    else:
        chunk_size = None

    total_written = 0
    writer = None
    output_path = Path(args.output) if args.output else None
    if output_path and args.mag_bins:
        # Append mag-bin to the filename for clarity; use first bin if multiple
        bin_tag = args.mag_bins[0] if len(args.mag_bins) == 1 else "multi"
        output_path = output_path.with_name(f"{output_path.stem}_{bin_tag}{output_path.suffix}")
        args.output = str(output_path)

    def _write_chunk(chunk_results, is_final=False):
        """Write a chunk of results to Parquet."""
        if not chunk_results or not args.output:
            return
        nonlocal total_written, writer
        df_chunk = pd.DataFrame(chunk_results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="brotli")
        writer.write_table(table)

        total_written += len(chunk_results)
        if is_final:
            if writer is not None:
                writer.close()
            print(f"Wrote {total_written} total rows to {args.output}", flush=True)
        else:
            print(f"Wrote chunk: {len(chunk_results)} rows (total: {total_written})", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                _process_one,
                path,
                trigger_mode=args.trigger_mode,
                logbf_threshold_dip=args.logbf_threshold_dip,
                logbf_threshold_jump=args.logbf_threshold_jump,
                significance_threshold=args.significance_threshold,
                p_points=args.p_points,
                run_min_points=args.run_min_points,
                run_allow_gap_points=args.run_allow_gap_points,
                run_max_gap_days=args.run_max_gap_days,
                run_min_duration_days=args.run_min_duration_days,
                run_sum_threshold=args.run_sum_threshold,
                run_sum_multiplier=args.run_sum_multiplier,
                compute_event_prob=compute_event_prob,
            ): path
            for path in expanded_inputs
        }

        for fut in tqdm(as_completed(futs), total=len(futs), desc="LCs", unit="lc"):
            path = futs[fut]
            try:
                result = fut.result()
                results.append(result)
                
                if chunk_size and len(results) >= chunk_size:
                    _write_chunk(results)
                    results = []
            except Exception as e:
                import traceback
                tb_str = traceback.format_exc()
                errors.append(dict(path=str(path), error=repr(e), traceback=tb_str))
                print(f"ERROR processing {path}: {e}", flush=True)
                if "too many values to unpack" in str(e):
                    print(f"Full traceback:\n{tb_str}", flush=True)

    if results:
        _write_chunk(results, is_final=True)
    elif args.output and total_written == 0:
        pass
    else:
        for row in results:
            print(
                f"{row['path']}\t"
                f"mode={row['trigger_mode']}\t"
                f"dip_sig={row['dip_significant']} ({row['dip_best_morph']}) dip_dBIC={row['dip_best_delta_bic']:.1f}\t"
                f"jump_sig={row['jump_significant']} ({row['jump_best_morph']}) jump_dBIC={row['jump_best_delta_bic']:.1f}"
            )

    if errors:
        print(f"Completed with {len(errors)} failures.", flush=True)


if __name__ == "__main__":
    main()
