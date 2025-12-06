from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.stats import biweight_location, biweight_scale
from scipy.signal import find_peaks

from lc_utils import read_lc_dat, read_lc_raw, match_index_to_lc
from lc_baseline import per_camera_trend_baseline
from lc_dips import clean_lc

MAG_BINS = ["12_12.5", "12.5_13", "13_13.5", "13.5_14", "14_14.5", "14.5_15"]


def fit_paczynski_peaks(delta, jd, err, peak_idx, sigma_threshold):
    """
    Fit Paczynski-like microlensing curves to biweight-delta peaks and compute a heuristic score.

    Model: amp / sqrt(1 + ((t - t0) / tE)^2)
    FWHM for this model is 2 * sqrt(3) * tE.
    """
    out = {
        "total_score": 0.0,
        "best_score": 0.0,
        "scores": [],
        "fwhm": [],
        "delta_peak": [],
        "n_det": [],
        "chi2_red": [],
    }

    if len(peak_idx) == 0 or delta.size == 0:
        return out

    delta = np.asarray(delta, float)
    jd = np.asarray(jd, float)
    err = np.asarray(err, float) if err is not None else np.full_like(delta, np.nan)

    def paczynski(t, amp, t0, tE):
        tE = np.maximum(tE, 1e-6)  # enforce positive timescale
        return amp / np.sqrt(1.0 + ((t - t0) / tE) ** 2)

    scores = []
    fwhm_list = []
    delta_peak_list = []
    n_det_list = []
    chi2_list = []

    for p in peak_idx:
        p = int(p)
        delta_peak = float(delta[p]) if 0 <= p < len(delta) else 0.0

        # find window where delta returns to <= 0.5 sigma_threshold on both sides
        left = p
        while left > 0 and delta[left] > 0.5 * sigma_threshold:
            left -= 1
        right = p
        nmax = len(delta) - 1
        while right < nmax and delta[right] > 0.5 * sigma_threshold:
            right += 1

        window = slice(left, right + 1)
        t_win = jd[window]
        d_win = delta[window]
        n_det = len(d_win)
        if n_det < 3:
            # fallback, cannot fit
            fwhm = 0.0
            chi2_red = np.inf
            score = 0.0
        else:
            amp0 = max(delta_peak, 0.0)
            t0_0 = float(jd[p])
            window_span = max(t_win.max() - t_win.min(), 1e-3)
            tE0 = max(window_span / (2.0 * np.sqrt(3.0)), 0.5)
            half_width = window_span / 2.0
            try:
                popt, _ = curve_fit(
                    paczynski,
                    t_win,
                    d_win,
                    p0=[amp0, t0_0, tE0],
                    bounds=(
                        [0.0, t_win.min() - half_width, 1e-3],
                        [np.inf, t_win.max() + half_width, np.inf],
                    ),
                    maxfev=4000,
                )
                amp, t0_fit, tE = popt
                tE = abs(tE)
                model = paczynski(t_win, amp, t0_fit, tE)
                resid = d_win - model
                dof = max(n_det - 3, 1)
                chi2 = float(np.nansum(resid**2))
                chi2_red = chi2 / dof
                fwhm = float(2.0 * np.sqrt(3.0) * tE)
                # paper score term: (delta/2) * FWHM * N_det * (1 / chi2_red)
                score = float(
                    (max(delta_peak, 0.0) / 2.0)
                    * max(fwhm, 0.0)
                    * max(n_det, 1)
                    / max(chi2_red, 1e-6)
                )
            except Exception:
                fwhm = 0.0
                chi2_red = np.inf
                score = 0.0

        scores.append(score)
        fwhm_list.append(fwhm)
        delta_peak_list.append(delta_peak)
        n_det_list.append(n_det)
        chi2_list.append(chi2_red)

    N = len(scores)
    if N > 0:
        denom = float(np.log(N + 1) ** N)
        denom = denom if np.isfinite(denom) and denom > 0 else 1.0
        total_score = float(np.nansum(scores) / denom)
        best_score = float(np.nanmax(scores))
    else:
        total_score = 0.0
        best_score = 0.0

    out.update(
        {
            "total_score": total_score,
            "best_score": best_score,
            "scores": scores,
            "fwhm": fwhm_list,
            "delta_peak": delta_peak_list,
            "n_det": n_det_list,
            "chi2_red": chi2_list,
        }
    )
    return out

def _compute_biweight_delta_peaks(df, *, mag_col="mag", t_col="JD", err_col="error", biweight_c=6.0, eps=1e-6):
    """
    Peak-oriented version of biweight delta where brightenings are positive.

    Uses (R - mag) so that microlensing-like peaks (smaller magnitudes) produce
    positive delta values suitable for peak finding.
    """
    mag = np.asarray(df[mag_col], float) if mag_col in df.columns else np.array([], float)
    jd = np.asarray(df[t_col], float) if t_col in df.columns else np.array([], float)
    if err_col in df.columns:
        err = np.asarray(df[err_col], float)
    else:
        err = np.full_like(mag, np.nan, dtype=float)

    finite_m = np.isfinite(mag)
    R = float(biweight_location(mag[finite_m], c=biweight_c)) if finite_m.any() else np.nan
    S = float(biweight_scale(mag[finite_m], c=biweight_c)) if finite_m.any() else 0.0
    if not np.isfinite(S) or S < 0:
        S = 0.0

    err2 = np.where(np.isfinite(err), err**2, 0.0)
    denom = np.sqrt(err2 + S**2)
    denom = np.where(denom > 0, denom, eps)
    delta = (R - mag) / denom
    return delta, jd, R


def process_record_microlensing(
    record: dict,
    *,
    peak_kwargs: dict | None = None,
) -> dict:
    """
    Process a single light curve record looking for microlensing-like peaks
    using the biweight-delta (brightenings positive) series.
    """
    peak_kwargs = dict(peak_kwargs or {})
    sigma_threshold = float(peak_kwargs.get("sigma_threshold", 3.0))

    asn = record["asas_sn_id"]
    dfg, dfv = read_lc_dat(asn, record["lc_dir"])
    raw_df = read_lc_raw(asn, record["lc_dir"])

    jd_first = np.nan
    jd_last = np.nan

    def _analyze_band(df_band):
        if df_band.empty or len(df_band) < 30:
            return {
                "n": 0,
                "R": np.nan,
                "peaks_idx": [],
                "peaks_jd": [],
                "fit": {
                    "total_score": 0.0,
                    "best_score": 0.0,
                    "scores": [],
                    "fwhm": [],
                    "delta_peak": [],
                    "n_det": [],
                    "chi2_red": [],
                },
            }

        df_band = clean_lc(df_band).sort_values("JD").reset_index(drop=True)
        delta, jd, R = _compute_biweight_delta_peaks(df_band, **peak_kwargs)
        peaks, _ = find_peaks(
            np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0),
            height=sigma_threshold,
            distance=int(peak_kwargs.get("distance", 50)),
        )

        fit = fit_paczynski_peaks(
            delta,
            jd,
            df_band["error"].to_numpy(float) if "error" in df_band.columns else None,
            peaks,
            sigma_threshold,
        )

        return {
            "n": len(peaks),
            "R": R,
            "peaks_idx": np.asarray(peaks, dtype=int).tolist(),
            "peaks_jd": df_band["JD"].values[np.asarray(peaks, dtype=int)].tolist() if len(peaks) else [],
            "fit": fit,
        }

    g_res = _analyze_band(dfg)
    v_res = _analyze_band(dfv)

    if not dfg.empty:
        jd_first = float(dfg["JD"].iloc[0])
        jd_last = float(dfg["JD"].iloc[-1])
    if not dfv.empty:
        if np.isnan(jd_first):
            jd_first = float(dfv["JD"].iloc[0])
        if np.isnan(jd_last):
            jd_last = float(dfv["JD"].iloc[-1])

    row = {
        "mag_bin": record.get("mag_bin"),
        "asas_sn_id": asn,
        "index_num": record.get("index_num"),
        "index_csv": record.get("index_csv"),
        "lc_dir": record.get("lc_dir"),
        "dat_path": record.get("dat_path"),
        "raw_path": os.path.join(record.get("lc_dir"), f"{asn}.raw"),
        "g_n_peaks": g_res["n"],
        "g_biweight_R": g_res["R"],
        "g_peaks_idx": g_res["peaks_idx"],
        "g_peaks_jd": g_res["peaks_jd"],
        "g_total_score": g_res["fit"]["total_score"],
        "g_best_score": g_res["fit"]["best_score"],
        "g_scores": g_res["fit"]["scores"],
        "g_fwhm": g_res["fit"]["fwhm"],
        "g_delta_peak": g_res["fit"]["delta_peak"],
        "g_n_det": g_res["fit"]["n_det"],
        "g_chi2_red": g_res["fit"]["chi2_red"],
        "v_n_peaks": v_res["n"],
        "v_biweight_R": v_res["R"],
        "v_peaks_idx": v_res["peaks_idx"],
        "v_peaks_jd": v_res["peaks_jd"],
        "v_total_score": v_res["fit"]["total_score"],
        "v_best_score": v_res["fit"]["best_score"],
        "v_scores": v_res["fit"]["scores"],
        "v_fwhm": v_res["fit"]["fwhm"],
        "v_delta_peak": v_res["fit"]["delta_peak"],
        "v_n_det": v_res["fit"]["n_det"],
        "v_chi2_red": v_res["fit"]["chi2_red"],
        "jd_first": jd_first,
        "jd_last": jd_last,
        "n_rows_g": int(len(dfg)) if not dfg.empty else 0,
        "n_rows_v": int(len(dfv)) if not dfv.empty else 0,
    }

    if not raw_df.empty:
        scatter_vals = (raw_df["sig1_high"] - raw_df["sig1_low"]).to_numpy(dtype=float)
        finite_scatter = scatter_vals[np.isfinite(scatter_vals)]
        row["raw_robust_scatter"] = float(np.nanmedian(finite_scatter)) if finite_scatter.size > 0 else np.nan
    else:
        row["raw_robust_scatter"] = np.nan

    return row


def microlensing_peak_finder(
    *,
    index_path="/data/poohbah/1/assassin/lenhart/code/calder/lcsv2_masked/",
    lc_path="/data/poohbah/1/assassin/rowan.90/lcsv2",
    mag_bins=MAG_BINS,
    id_column="asas_sn_id",
    out_dir="./results_microlensing",
    out_format="csv",
    n_workers=8,
    chunk_size=250000,
    max_inflight=None,
    peak_kwargs: dict | None = None,
    collected_rows: list | None = None,
) -> pd.DataFrame | None:
    """
    Run microlensing peak search across magnitude bins, writing CSV/Parquet.
    """
    peak_kwargs = dict(peak_kwargs or {})
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"microlensing_peaks.{out_format}")
    if max_inflight is None:
        max_inflight = max(4, n_workers)

    rows_buffer: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for b in mag_bins:
            pbar = tqdm(total=1, desc=f"bin {b}", leave=False)

            def flush_if_needed(force: bool = False):
                if not rows_buffer:
                    return
                if len(rows_buffer) >= chunk_size or force:
                    if out_format == "parquet":
                        pd.DataFrame(rows_buffer).to_parquet(out_path, index=False)
                    else:
                        mode = "a" if os.path.exists(out_path) else "w"
                        header = not os.path.exists(out_path)
                        pd.DataFrame(rows_buffer).to_csv(out_path, index=False, mode=mode, header=header)
                    rows_buffer.clear()

            pending = set()
            scheduled = 0

            def drain_some(all_pending):
                done_now = 0
                done = []
                for fut in as_completed(all_pending, timeout=0.1):
                    done.append(fut)
                for fut in done:
                    row = fut.result()
                    rows_buffer.append(row)
                    if collected_rows is not None:
                        collected_rows.append(row)
                    all_pending.remove(fut)
                    done_now += 1
                    flush_if_needed()
                return done_now

            record_iter = match_index_to_lc(
                index_path=index_path,
                lc_path=lc_path,
                mag_bins=[b],
                id_column=id_column,
            )

            for rec in record_iter:
                if not rec.get("found", False):
                    continue
                pending.add(
                    ex.submit(
                        process_record_microlensing,
                        rec,
                        peak_kwargs=peak_kwargs,
                    )
                )
                scheduled += 1
                pbar.total = scheduled
                pbar.refresh()
                if len(pending) >= max_inflight:
                    done = drain_some(pending)
                    pbar.update(done)

            while pending:
                done = drain_some(pending)
                pbar.update(done)

            pbar.close()
            flush_if_needed(force=True)

    if collected_rows is not None:
        return pd.DataFrame(collected_rows)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run microlensing peak finder across bins.")
    parser.add_argument("--mag-bin", dest="mag_bins", action="append", choices=MAG_BINS, help="Specify bins to run; omit to process all.")
    parser.add_argument("--out-dir", default="./results_microlensing")
    parser.add_argument("--format", choices=("parquet", "csv"), default="csv")
    parser.add_argument("--n-workers", type=int, default=8, help="Parallel processes")
    parser.add_argument("--chunk-size", type=int, default=250000, help="Rows per CSV flush")
    args = parser.parse_args()
    bins = args.mag_bins or MAG_BINS
    microlensing_peak_finder(
        mag_bins=bins,
        out_dir=args.out_dir,
        out_format=args.format,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
    )

