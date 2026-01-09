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

import argparse
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import skewnorm
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from malca.utils import read_lc_dat2
from malca.events import run_bayesian_significance
from malca.baseline import (
    per_camera_gp_baseline,
    per_camera_gp_baseline_masked,
    per_camera_trend_baseline,
)


_GLOBAL: dict[str, object] = {}


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
        Center time of dip (mean mu)
    duration : float
        Duration of dip (related to sigma)
    amplitude : float
        Depth of dip in magnitudes
    skewness : float
        Skewness parameter alpha (default 0 = symmetric Gaussian)
        Positive = tail to right, Negative = tail to left
    offset : float
        Baseline offset C0

    Returns
    -------
    np.ndarray
        Magnitude perturbation (positive = fainter)
    """
    sigma = duration / 2.355
    dip = amplitude * skewnorm.pdf(t, a=skewness, loc=t_center, scale=sigma)
    if dip.size and dip.max() > 0:
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
    """
    all_mags = []
    all_errs = []

    for df in lc_sample:
        if df.empty:
            continue
        mag = df[mag_col].values
        err = df[err_col].values
        mask = np.isfinite(mag) & np.isfinite(err) & (err > 0)
        all_mags.extend(mag[mask])
        all_errs.extend(err[mask])

    if len(all_mags) < 10:
        return np.poly1d([0.1])

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
    """
    df_out = df_lc.copy()
    if df_out.empty:
        return df_out

    mag_baseline = df_out[mag_col].median()
    sigma_mag = df_out[mag_col].std()

    if mag_err_poly is not None:
        sigma_sigma_mag = df_out[err_col].std()
    else:
        sigma_sigma_mag = 0.0

    t = df_out[time_col].values
    dip_profile = skewnormal_dip(t, t_center, duration, amplitude, skewness)

    noise_intrinsic = np.random.normal(0, sigma_mag, size=len(t))
    noise_error = np.random.normal(0, sigma_sigma_mag, size=len(t))

    df_out[mag_col] = mag_baseline + dip_profile + noise_intrinsic + noise_error
    return df_out


def _get_id_col(df: pd.DataFrame) -> str:
    for col in ("asas_sn_id", "source_id", "id"):
        if col in df.columns:
            return col
    raise KeyError("Manifest is missing a usable ID column (expected asas_sn_id/source_id/id).")


def _resolve_lc_dir(row: pd.Series) -> Path | None:
    if "lc_dir" in row and pd.notna(row["lc_dir"]):
        return Path(str(row["lc_dir"]))
    if "dat_path" in row and pd.notna(row["dat_path"]):
        return Path(str(row["dat_path"])).parent
    if "path" in row and pd.notna(row["path"]):
        p = Path(str(row["path"]))
        return p if p.is_dir() else p.parent
    return None


def _load_lc(asas_sn_id: str, lc_dir: Path) -> pd.DataFrame:
    df_g, df_v = read_lc_dat2(asas_sn_id, str(lc_dir))
    if df_g.empty and df_v.empty:
        return pd.DataFrame()
    return pd.concat([df_g, df_v], ignore_index=True)


def select_control_sample(
    manifest_df: pd.DataFrame,
    n_sample: int = 10000,
    reject_candidates: pd.DataFrame | None = None,
    min_points: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Select clean light curves for injection testing.
    """
    df = manifest_df.copy()

    if reject_candidates is not None and "asas_sn_id" in reject_candidates.columns:
        exclude_ids = set(reject_candidates["asas_sn_id"].astype(str))
        df = df[~df["asas_sn_id"].astype(str).isin(exclude_ids)]

    if "n_points" in df.columns:
        df = df[df["n_points"] >= min_points]

    rng = np.random.default_rng(seed)
    if len(df) < n_sample:
        return df.reset_index(drop=True)

    indices = rng.choice(len(df), size=n_sample, replace=False)
    return df.iloc[indices].reset_index(drop=True)


def _build_detection_kwargs(args: argparse.Namespace) -> dict:
    baseline_map = {
        "gp": per_camera_gp_baseline,
        "gp_masked": per_camera_gp_baseline_masked,
        "trend": per_camera_trend_baseline,
    }
    use_sigma_eff = not args.no_sigma_eff
    require_sigma_eff = use_sigma_eff and (not args.allow_missing_sigma_eff)
    return dict(
        trigger_mode=args.trigger_mode,
        logbf_threshold_dip=args.logbf_threshold_dip,
        logbf_threshold_jump=args.logbf_threshold_jump,
        significance_threshold=args.significance_threshold,
        p_points=args.p_points,
        p_min_dip=args.p_min_dip,
        p_max_dip=args.p_max_dip,
        p_min_jump=args.p_min_jump,
        p_max_jump=args.p_max_jump,
        run_min_points=args.run_min_points,
        run_allow_gap_points=args.run_allow_gap_points,
        run_max_gap_days=args.run_max_gap_days,
        run_min_duration_days=args.run_min_duration_days,
        run_sum_threshold=args.run_sum_threshold,
        run_sum_multiplier=args.run_sum_multiplier,
        compute_event_prob=(not args.no_event_prob),
        use_sigma_eff=use_sigma_eff,
        require_sigma_eff=require_sigma_eff,
        baseline_func=baseline_map.get(args.baseline_func, per_camera_gp_baseline),
    )


def _default_detection_func(df: pd.DataFrame, detection_kwargs: dict) -> dict:
    res = run_bayesian_significance(df, **detection_kwargs)
    dip = res["dip"]
    jump = res["jump"]
    return dict(
        detected=bool(dip["significant"]),
        dip_significant=bool(dip["significant"]),
        jump_significant=bool(jump["significant"]),
        dip_bayes_factor=float(dip["bayes_factor"]),
        jump_bayes_factor=float(jump["bayes_factor"]),
        dip_best_p=float(dip["best_p"]),
        jump_best_p=float(jump["best_p"]),
    )


def _trial_indices_to_params(
    trial_index: int,
    n_injections_per_grid: int,
    n_durations: int,
) -> tuple[int, int, int]:
    n_per_amp = n_durations * n_injections_per_grid
    amp_idx = trial_index // n_per_amp
    rem = trial_index % n_per_amp
    dur_idx = rem // n_injections_per_grid
    inj_idx = rem % n_injections_per_grid
    return amp_idx, dur_idx, inj_idx


def _simulate_trial(
    trial_index: int,
    *,
    control_ids: np.ndarray,
    control_dirs: np.ndarray,
    amplitude_grid: np.ndarray,
    duration_grid: np.ndarray,
    n_injections_per_grid: int,
    skew_min: float,
    skew_max: float,
    mag_err_poly: np.poly1d | None,
    detection_kwargs: dict,
    seed: int,
) -> dict:
    amp_idx, dur_idx, inj_idx = _trial_indices_to_params(
        trial_index, n_injections_per_grid, len(duration_grid)
    )
    if amp_idx >= len(amplitude_grid):
        return dict(trial_index=trial_index, detected=False, error="trial_index_out_of_range")

    amplitude = float(amplitude_grid[amp_idx])
    duration = float(duration_grid[dur_idx])

    rng = np.random.default_rng(seed + int(trial_index))
    control_idx = int(rng.integers(0, len(control_ids)))
    asas_sn_id = str(control_ids[control_idx])
    lc_dir = Path(str(control_dirs[control_idx]))

    try:
        df = _load_lc(asas_sn_id, lc_dir)
        if df.empty or len(df) < 10:
            return dict(
                trial_index=trial_index,
                amp_index=amp_idx,
                dur_index=dur_idx,
                inj_index=inj_idx,
                amplitude=amplitude,
                duration=duration,
                asas_sn_id=asas_sn_id,
                detected=False,
                error="empty_or_short_lc",
            )

        t_min = float(df["JD"].min())
        t_max = float(df["JD"].max())
        if not np.isfinite(t_min) or not np.isfinite(t_max) or (t_max - t_min <= 2 * duration):
            return dict(
                trial_index=trial_index,
                amp_index=amp_idx,
                dur_index=dur_idx,
                inj_index=inj_idx,
                amplitude=amplitude,
                duration=duration,
                asas_sn_id=asas_sn_id,
                detected=False,
                error="invalid_time_range",
            )

        t_center = rng.uniform(t_min + duration, t_max - duration)
        skewness = rng.uniform(skew_min, skew_max)

        df_injected = inject_dip(df, t_center, duration, amplitude, skewness, mag_err_poly)
        detection_result = _default_detection_func(df_injected, detection_kwargs)

        return dict(
            trial_index=trial_index,
            amp_index=amp_idx,
            dur_index=dur_idx,
            inj_index=inj_idx,
            amplitude=amplitude,
            duration=duration,
            skewness=float(skewness),
            t_center=float(t_center),
            asas_sn_id=asas_sn_id,
            **detection_result,
        )
    except Exception as exc:
        return dict(
            trial_index=trial_index,
            amp_index=amp_idx,
            dur_index=dur_idx,
            inj_index=inj_idx,
            amplitude=amplitude,
            duration=duration,
            asas_sn_id=asas_sn_id,
            detected=False,
            error=str(exc),
        )


def _init_worker(
    control_ids: np.ndarray,
    control_dirs: np.ndarray,
    amplitude_grid: np.ndarray,
    duration_grid: np.ndarray,
    n_injections_per_grid: int,
    skew_min: float,
    skew_max: float,
    mag_err_poly: np.poly1d | None,
    detection_kwargs: dict,
    seed: int,
) -> None:
    _GLOBAL["control_ids"] = control_ids
    _GLOBAL["control_dirs"] = control_dirs
    _GLOBAL["amplitude_grid"] = amplitude_grid
    _GLOBAL["duration_grid"] = duration_grid
    _GLOBAL["n_injections_per_grid"] = n_injections_per_grid
    _GLOBAL["skew_min"] = skew_min
    _GLOBAL["skew_max"] = skew_max
    _GLOBAL["mag_err_poly"] = mag_err_poly
    _GLOBAL["detection_kwargs"] = detection_kwargs
    _GLOBAL["seed"] = seed


def _process_trial_batch(trial_indices: list[int]) -> list[dict]:
    results = []
    for trial_index in trial_indices:
        results.append(
            _simulate_trial(
                trial_index,
                control_ids=_GLOBAL["control_ids"],
                control_dirs=_GLOBAL["control_dirs"],
                amplitude_grid=_GLOBAL["amplitude_grid"],
                duration_grid=_GLOBAL["duration_grid"],
                n_injections_per_grid=_GLOBAL["n_injections_per_grid"],
                skew_min=float(_GLOBAL["skew_min"]),
                skew_max=float(_GLOBAL["skew_max"]),
                mag_err_poly=_GLOBAL["mag_err_poly"],
                detection_kwargs=_GLOBAL["detection_kwargs"],
                seed=int(_GLOBAL["seed"]),
            )
        )
    return results


class CsvWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.columns = None
        if self.path.exists() and self.path.stat().st_size > 0:
            try:
                self.columns = pd.read_csv(self.path, nrows=0).columns.tolist()
            except Exception:
                self.columns = None

    def write_chunk(self, chunk_results: list[dict]) -> None:
        if not chunk_results:
            return
        df_chunk = pd.DataFrame(chunk_results)
        if self.columns is None:
            self.columns = list(df_chunk.columns)
        df_chunk = df_chunk.reindex(columns=self.columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = not self.path.exists() or self.path.stat().st_size == 0
        df_chunk.to_csv(self.path, mode="a", header=header, index=False)

    def close(self) -> None:
        return


def _write_checkpoint(path: Path, last_index: int) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(str(int(last_index)))
    tmp_path.replace(path)


def _read_checkpoint(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
        if text:
            return int(text)
    except Exception:
        return None
    return None


def run_injection_recovery(
    control_sample: pd.DataFrame,
    *,
    detection_kwargs: dict,
    amplitude_grid: np.ndarray | None = None,
    duration_grid: np.ndarray | None = None,
    n_injections_per_grid: int = 100,
    skewness_range: tuple[float, float] = (-0.5, 0.5),
    mag_err_order: int = 5,
    mag_err_sample: int = 100,
    seed: int = 42,
    workers: int = 1,
    task_size: int = 50,
    checkpoint_interval: int = 1000,
    chunk_size: int = 1000,
    output_path: Path | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = True,
    overwrite: bool = False,
    max_trials: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame | None:
    """
    Run injection-recovery with optional parallelism and checkpointing.
    """
    rng = np.random.default_rng(seed)

    if amplitude_grid is None:
        amplitude_grid = np.linspace(0.1, 2.0, 100)
    if duration_grid is None:
        duration_grid = np.logspace(0.5, 2.5, 100)

    amp_grid = np.asarray(amplitude_grid, float)
    dur_grid = np.asarray(duration_grid, float)

    total_trials = int(len(amp_grid) * len(dur_grid) * int(n_injections_per_grid))
    if max_trials is not None:
        total_trials = int(min(total_trials, max_trials))

    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists() and overwrite and not resume:
            output_path.unlink()
        if output_path.exists() and not resume and not overwrite:
            raise SystemExit(f"Output exists: {output_path} (use --overwrite or --no-resume)")

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
    elif output_path is not None:
        checkpoint_path = output_path.with_name(f"{output_path.stem}_PROCESSED.txt")

    if checkpoint_path and checkpoint_path.exists() and overwrite and not resume:
        checkpoint_path.unlink()

    start_index = 0
    if resume and checkpoint_path and checkpoint_path.exists():
        last = _read_checkpoint(checkpoint_path)
        if last is not None:
            start_index = int(last) + 1

    if start_index >= total_trials:
        print("All trials already completed per checkpoint.")
        return None

    id_col = _get_id_col(control_sample)
    control_ids = control_sample[id_col].astype(str).to_numpy()
    control_dirs = []
    for _, row in control_sample.iterrows():
        lc_dir = _resolve_lc_dir(row)
        if lc_dir is None:
            control_dirs.append("")
        else:
            control_dirs.append(str(lc_dir))
    control_dirs = np.asarray(control_dirs, dtype=object)

    if len(control_ids) == 0:
        raise SystemExit("Control sample is empty.")

    print("Loading control sample light curves for error polynomial...")
    lc_sample = []
    for idx, row in control_sample.iterrows():
        if idx >= mag_err_sample:
            break
        asas_sn_id = str(row[id_col])
        lc_dir = _resolve_lc_dir(row)
        if lc_dir is None:
            continue
        try:
            df = _load_lc(asas_sn_id, lc_dir)
            if not df.empty:
                lc_sample.append(df)
        except Exception:
            continue

    print(f"Fitting {mag_err_order}th-order polynomial to magnitude errors...")
    mag_err_poly = estimate_magnitude_error_polynomial(lc_sample, order=mag_err_order)

    writer = CsvWriter(output_path) if output_path else None
    results: list[dict] = []

    pbar = tqdm(total=total_trials, initial=start_index, disable=not show_progress)
    skew_min, skew_max = float(skewness_range[0]), float(skewness_range[1])

    def flush_results(is_final: bool = False) -> None:
        nonlocal results
        if not results:
            return
        if writer is not None:
            writer.write_chunk(results)
        if is_final and writer is not None:
            writer.close()
        results = []

    if workers <= 1:
        for trial_index in range(start_index, total_trials):
            res = _simulate_trial(
                trial_index,
                control_ids=control_ids,
                control_dirs=control_dirs,
                amplitude_grid=amp_grid,
                duration_grid=dur_grid,
                n_injections_per_grid=n_injections_per_grid,
                skew_min=skew_min,
                skew_max=skew_max,
                mag_err_poly=mag_err_poly,
                detection_kwargs=detection_kwargs,
                seed=seed,
            )
            results.append(res)
            pbar.update(1)
            if chunk_size and len(results) >= chunk_size:
                flush_results()
            if checkpoint_path and (trial_index + 1) % checkpoint_interval == 0:
                flush_results()
                _write_checkpoint(checkpoint_path, trial_index)
        flush_results(is_final=True)
        if checkpoint_path:
            _write_checkpoint(checkpoint_path, total_trials - 1)
        pbar.close()
        return None if output_path else pd.DataFrame(results)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(
            control_ids,
            control_dirs,
            amp_grid,
            dur_grid,
            n_injections_per_grid,
            skew_min,
            skew_max,
            mag_err_poly,
            detection_kwargs,
            seed,
        ),
    ) as ex:
        for batch_start in range(start_index, total_trials, checkpoint_interval):
            batch_end = min(batch_start + checkpoint_interval, total_trials)
            batch_indices = list(range(batch_start, batch_end))
            tasks = [batch_indices[i:i + task_size] for i in range(0, len(batch_indices), task_size)]

            futures = {ex.submit(_process_trial_batch, task): task for task in tasks}
            for fut in as_completed(futures):
                batch_results = fut.result()
                results.extend(batch_results)
                pbar.update(len(batch_results))
                if chunk_size and len(results) >= chunk_size:
                    flush_results()

            flush_results()
            if checkpoint_path:
                _write_checkpoint(checkpoint_path, batch_end - 1)

    flush_results(is_final=True)
    if checkpoint_path:
        _write_checkpoint(checkpoint_path, total_trials - 1)
    pbar.close()
    return None if output_path else pd.DataFrame(results)


def compute_detection_efficiency(
    results_df: pd.DataFrame,
    amplitude_bins: int = 20,
    duration_bins: int = 20,
    detected_col: str = "detected",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute detection efficiency grid.
    """
    amp_edges = np.linspace(
        results_df["amplitude"].min(),
        results_df["amplitude"].max(),
        amplitude_bins + 1,
    )
    dur_edges = np.logspace(
        np.log10(results_df["duration"].min()),
        np.log10(results_df["duration"].max()),
        duration_bins + 1,
    )

    amp_centers = (amp_edges[:-1] + amp_edges[1:]) / 2
    dur_centers = np.sqrt(dur_edges[:-1] * dur_edges[1:])

    efficiency_grid = np.zeros((amplitude_bins, duration_bins))

    for i in range(amplitude_bins):
        for j in range(duration_bins):
            mask = (
                (results_df["amplitude"] >= amp_edges[i])
                & (results_df["amplitude"] < amp_edges[i + 1])
                & (results_df["duration"] >= dur_edges[j])
                & (results_df["duration"] < dur_edges[j + 1])
            )
            efficiency_grid[i, j] = results_df.loc[mask, detected_col].mean() if mask.sum() else np.nan

    return amp_centers, dur_centers, efficiency_grid


def plot_detection_efficiency(
    amp_centers: np.ndarray,
    dur_centers: np.ndarray,
    efficiency_grid: np.ndarray,
    output_path: Path | str | None = None,
    title: str = "Detection Efficiency",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """
    Plot detection efficiency heatmap.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(
        dur_centers,
        amp_centers,
        efficiency_grid,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        shading="auto",
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
    """
    from scipy.stats import skew

    mag = df_lc[mag_col].values
    mag_finite = mag[np.isfinite(mag)]

    if len(mag_finite) < 3:
        return {"skewness": np.nan, "von_neumann_ratio": np.nan}

    skewness_val = skew(mag_finite)
    diff_sq = np.diff(mag_finite) ** 2
    dev_sq = (mag_finite - mag_finite.mean()) ** 2
    von_neumann_ratio = diff_sq.sum() / dev_sq.sum() if dev_sq.sum() > 0 else np.nan
    return {"skewness": skewness_val, "von_neumann_ratio": von_neumann_ratio}


def _build_grid_linear(min_val: float, max_val: float, steps: int) -> np.ndarray:
    return np.linspace(float(min_val), float(max_val), int(steps))


def _build_grid_log(min_val: float, max_val: float, steps: int) -> np.ndarray:
    return np.logspace(np.log10(float(min_val)), np.log10(float(max_val)), int(steps))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run injection-recovery tests for dip detection.")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest parquet path.")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--control-sample-size",
        "--control-sample",
        "--control-n",
        dest="control_sample_size",
        type=int,
        default=10000,
        help="Number of control LCs to sample.",
    )
    parser.add_argument("--min-points", type=int, default=50, help="Minimum points in control sample if available.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--amp-min", type=float, default=0.1)
    parser.add_argument("--amp-max", type=float, default=2.0)
    parser.add_argument("--amp-steps", type=int, default=100)
    parser.add_argument("--dur-min", type=float, default=10.0)
    parser.add_argument("--dur-max", type=float, default=300.0)
    parser.add_argument("--dur-steps", type=int, default=100)
    parser.add_argument("--n-injections-per-grid", type=int, default=100)
    parser.add_argument("--skew-min", type=float, default=-0.5)
    parser.add_argument("--skew-max", type=float, default=0.5)
    parser.add_argument("--mag-err-order", type=int, default=5)
    parser.add_argument("--mag-err-sample", type=int, default=100)

    parser.add_argument("--workers", type=int, default=10, help="Parallel workers.")
    parser.add_argument("--task-size", type=int, default=50, help="Trials per worker task.")
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Trials per checkpoint update.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Rows per output flush.")
    parser.add_argument("--max-trials", type=int, default=None, help="Limit total trials (debug).")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume even if checkpoint exists.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output/checkpoint.")

    parser.add_argument("--trigger-mode", type=str, default="posterior_prob", choices=["logbf", "posterior_prob"])
    parser.add_argument("--logbf-threshold-dip", type=float, default=5.0)
    parser.add_argument("--logbf-threshold-jump", type=float, default=5.0)
    parser.add_argument("--significance-threshold", type=float, default=99.99997)
    parser.add_argument("--p-points", type=int, default=15)
    parser.add_argument("--p-min-dip", type=float, default=None)
    parser.add_argument("--p-max-dip", type=float, default=None)
    parser.add_argument("--p-min-jump", type=float, default=None)
    parser.add_argument("--p-max-jump", type=float, default=None)
    parser.add_argument("--run-min-points", type=int, default=3)
    parser.add_argument("--run-allow-gap-points", type=int, default=1)
    parser.add_argument("--run-max-gap-days", type=float, default=None)
    parser.add_argument("--run-min-duration-days", type=float, default=None)
    parser.add_argument("--run-sum-threshold", type=float, default=None)
    parser.add_argument("--run-sum-multiplier", type=float, default=2.5)
    parser.add_argument("--baseline-func", type=str, default="gp", choices=["gp", "gp_masked", "trend"])
    parser.add_argument("--no-event-prob", action="store_true")
    parser.add_argument("--no-sigma-eff", action="store_true")
    parser.add_argument("--allow-missing-sigma-eff", action="store_true")

    args = parser.parse_args()

    manifest = pd.read_parquet(args.manifest)
    control_sample = select_control_sample(
        manifest,
        n_sample=args.control_sample_size,
        min_points=args.min_points,
        seed=args.seed,
    )

    amp_grid = _build_grid_linear(args.amp_min, args.amp_max, args.amp_steps)
    dur_grid = _build_grid_log(args.dur_min, args.dur_max, args.dur_steps)

    detection_kwargs = _build_detection_kwargs(args)

    run_injection_recovery(
        control_sample,
        detection_kwargs=detection_kwargs,
        amplitude_grid=amp_grid,
        duration_grid=dur_grid,
        n_injections_per_grid=args.n_injections_per_grid,
        skewness_range=(args.skew_min, args.skew_max),
        mag_err_order=args.mag_err_order,
        mag_err_sample=args.mag_err_sample,
        seed=args.seed,
        workers=max(1, args.workers),
        task_size=max(1, args.task_size),
        checkpoint_interval=max(1, args.checkpoint_interval),
        chunk_size=max(1, args.chunk_size),
        output_path=args.out,
        checkpoint_path=None,
        resume=not args.no_resume,
        overwrite=args.overwrite,
        max_trials=args.max_trials,
        show_progress=True,
    )


if __name__ == "__main__":
    main()
