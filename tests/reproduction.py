from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as pl

# NOTE: The 'naive' and 'biweight' methods are legacy implementations
# kept for backward compatibility. New code should use method='bayes'.
from malca.events import run_bayesian_significance
from malca.utils import read_lc_dat2
from malca.baseline import per_camera_gp_baseline
from malca.pre_filter import apply_pre_filters
from malca.filter import filter_signal_amplitude



CANDIDATE_USECOLS = {
    "path",
    "source_id",
    "Source_ID",
    "Source ID",
    "asas_sn_id",
    "mag_bin",
}


brayden_candidates: list[dict[str, object]] = [
    {"source": "J042214+152530", "source_id": "377957522430", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J202402+383938", "source_id": "42950993887", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J174328+343315", "source_id": "223339338105", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J080327-261620", "source_id": "601296043597", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": False},
    {"source": "J184916-473251", "source_id": "472447294641", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Known", "expected_detected": True},
    {"source": "J183153-284827", "source_id": "455267102087", "category": "Dippers", "mag_bin": "13.5_14", "search_method": "Known", "expected_detected": False},
    {"source": "J070519+061219", "source_id": "266288137752", "category": "Dippers", "mag_bin": "13.5_14", "search_method": "Known", "expected_detected": False},
    {"source": "J081523-385923", "source_id": "532576686103", "category": "Dippers", "mag_bin": "13.5_14", "search_method": "Known", "expected_detected": False},
    {"source": "J085816-430955", "source_id": "352187470767", "category": "Dippers", "mag_bin": "12_12.5", "search_method": "Known", "expected_detected": False},
    {"source": "J114712-621037", "source_id": "609886184506", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Known", "expected_detected": False},
    {"source": "J005437+644347", "source_id": "68720274411", "category": "Multiple Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Known", "expected_detected": True},
    {"source": "J062510-075341", "source_id": "377958261591", "category": "Multiple Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J124745-622756", "source_id": "515397118400", "category": "Multiple Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J175912-120956", "source_id": "326417831663", "category": "Multiple Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J181752-580749", "source_id": "644245387906", "category": "Multiple Eclipse Binaries", "mag_bin": "12_12.5", "search_method": "Known", "expected_detected": True},
    {"source": "J160757-574540", "source_id": "661425129485", "category": "Multiple Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": False},
    {"source": "J073924-272916", "source_id": "438086977939", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J074007-161608", "source_id": "360777377116", "category": "Single Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J094848-545959", "source_id": "635655234580", "category": "Single Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J162209-444247", "source_id": "412317159120", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J183606-314826", "source_id": "438086901547", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J205245-713514", "source_id": "463856535113", "category": "Single Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J212132+480140", "source_id": "120259184943", "category": "Single Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J225702+562312", "source_id": "25770019815", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J190316-195739", "source_id": "515396514761", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J175602+013135", "source_id": "231929175915", "category": "Single Eclipse Binaries", "mag_bin": "14_14.5", "search_method": "Known", "expected_detected": True},
    {"source": "J073234-200049", "source_id": "335007754417", "category": "Single Eclipse Binaries", "mag_bin": "14.5_15", "search_method": "Known", "expected_detected": True},
    {"source": "J223332+565552", "source_id": "60130040391", "category": "Single Eclipse Binaries", "mag_bin": "12.5_13", "search_method": "Known", "expected_detected": True},
    {"source": "J183210-173432", "source_id": "317827964025", "category": "Single Eclipse Binaries", "mag_bin": "12.5_13", "search_method": "Pipeline", "expected_detected": False},
]


def load_manifest_df(manifest_path: Path | str) -> pd.DataFrame:
    path = Path(manifest_path).expanduser()
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "source_id" not in df.columns and "asas_sn_id" in df.columns:
        df = df.rename(columns={"asas_sn_id": "source_id"})
    df["source_id"] = df["source_id"].astype(str)
    return df


def _candidate_usecols_from_header(path: Path, sep: str) -> list[str]:
    try:
        with path.open("r", newline="") as handle:
            reader = csv.reader(handle, delimiter=sep)
            header = next(reader, [])
    except Exception:
        return []

    return [col for col in header if col in CANDIDATE_USECOLS]


def load_candidates_df(cand_path: Path) -> pd.DataFrame:
    suffix = cand_path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        usecols = _candidate_usecols_from_header(cand_path, sep)
        if usecols:
            return pd.read_csv(cand_path, usecols=usecols)
        return pd.read_csv(cand_path)

    if suffix in {".parquet", ".pq"}:
        usecols: list[str] | None = None
        try:
            import pyarrow.parquet as pq
        except Exception:
            usecols = None
        else:
            try:
                schema_cols = pq.ParquetFile(cand_path).schema.names
                usecols = [col for col in schema_cols if col in CANDIDATE_USECOLS]
            except Exception:
                usecols = None

        if usecols:
            return pd.read_parquet(cand_path, columns=usecols)
        try:
            return pd.read_parquet(cand_path, columns=list(CANDIDATE_USECOLS))
        except Exception:
            return pd.read_parquet(cand_path)

    raise SystemExit(f"Unsupported candidates file format: {cand_path}")


def dataframe_from_candidates(data: Sequence[Mapping[str, object]] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(data or brayden_candidates).copy()
    df.rename(columns={"Source": "source", "Source ID": "source_id"}, inplace=True)
    df["source_id"] = df["source_id"].astype(str)
    return df


def target_map(df: pd.DataFrame) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for mag_bin, chunk in df.groupby("mag_bin"):
        grouped[str(mag_bin)] = set(chunk["source_id"].astype(str))
    return grouped


def records_from_manifest(df: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    records: dict[str, list[dict[str, object]]] = {}
    for rec in df.to_dict("records"):
        source_id = str(rec.get("source_id"))
        mag_bin = str(rec.get("mag_bin"))
        lc_dir = str(rec.get("lc_dir"))
        dat_path = rec.get("dat_path") or str(Path(lc_dir) / f"{source_id}.dat")
        record = {
            "mag_bin": mag_bin,
            "index_num": rec.get("index_num"),
            "index_csv": rec.get("index_csv"),
            "lc_dir": lc_dir,
            "asas_sn_id": source_id,
            "dat_path": dat_path,
            "found": bool(rec.get("dat_exists", True)),
        }
        records.setdefault(mag_bin, []).append(record)
    return records


def records_from_skypatrol_dir(df_targets: pd.DataFrame, skypatrol_dir: Path) -> dict[str, list[dict[str, object]]]:
    base = Path(skypatrol_dir)
    if not base.exists():
        return {}

    records: dict[str, list[dict[str, object]]] = {}
    for _, row in df_targets.iterrows():
        source_id = str(row.get("source_id"))
        mag_bin = str(row.get("mag_bin"))
        csv_path = base / f"{source_id}-light-curves.csv"
        if not csv_path.exists():
            continue
        rec = {
            "mag_bin": mag_bin,
            "index_num": None,
            "index_csv": None,
            "lc_dir": str(base),
            "asas_sn_id": source_id,
            "dat_path": str(csv_path),
            "found": True,
        }
        records.setdefault(mag_bin, []).append(rec)
    return records


def records_from_candidates_with_paths(
    df_targets: pd.DataFrame,
    *,
    path_prefix: Path | str | None = None,
    path_root: Path | str | None = None,
) -> dict[str, list[dict[str, object]]]:
    """
    Build records_map from candidates that have a 'path' column.
    Used when events.py output is passed directly to reproduction.py.
    """
    if "path" not in df_targets.columns:
        return {}

    records: dict[str, list[dict[str, object]]] = {}
    prefix = Path(path_prefix).expanduser() if path_prefix else None
    root = Path(path_root).expanduser() if path_root else None

    for _, row in df_targets.iterrows():
        source_id = str(row.get("source_id"))
        mag_bin = str(row.get("mag_bin", ""))
        path_str = str(row.get("path", ""))

        if not path_str or path_str == "nan":
            continue

        path = Path(path_str)
        if prefix and root:
            try:
                path = root / path.relative_to(prefix)
            except ValueError:
                pass
        lc_dir = str(path.parent)

        rec = {
            "mag_bin": mag_bin,
            "index_num": None,
            "index_csv": None,
            "lc_dir": lc_dir,
            "asas_sn_id": source_id,
            "dat_path": str(path),
            "found": path.exists(),
        }
        records.setdefault(mag_bin, []).append(rec)
    return records


def coerce_candidate_records(data) -> list[dict[str, object]]:
    if data is None:
        return list(brayden_candidates)

    if isinstance(data, pd.DataFrame):
        records = data.to_dict("records")
    else:
        records = list(data)

    if not records:
        return []

    first = records[0]
    if isinstance(first, Mapping):
        import re
        coerced: list[dict[str, object]] = []
        for rec in records:
            if not isinstance(rec, Mapping):
                continue
            new = dict(rec)
            source_id = str(new.get("source_id", new.get("Source_ID", ""))).strip()

            if not source_id and "path" in new:
                path_str = str(new["path"])
                source_id = Path(path_str).stem

            if not source_id:
                continue

            new["source_id"] = source_id
            new.setdefault("source", new.get("source", source_id))

            if "mag_bin" not in new or not new["mag_bin"]:
                if "path" in new:
                    path_str = str(new["path"])
                    mag_bin_match = re.search(r'(\d+(?:\.\d+)?_\d+(?:\.\d+)?)', path_str)
                    if mag_bin_match:
                        new["mag_bin"] = mag_bin_match.group(1)

            coerced.append(new)
        return coerced

    ids = []
    for entry in records:
        entry_str = str(entry)
        entry_path = Path(entry_str)
        stem = entry_path.stem
        source_id = stem.split("-")[0] if stem else entry_str
        ids.append(source_id)

    lookup = {c["source_id"]: c for c in brayden_candidates}
    coerced = []
    seen = set()
    for source_id in ids:
        if source_id in seen:
            continue
        seen.add(source_id)
        if source_id in lookup:
            coerced.append(lookup[source_id])
        else:
            coerced.append({"source": source_id, "source_id": source_id, "mag_bin": None})
    return coerced


def resolve_candidates(spec: str | None):
    if spec is None:
        return list(brayden_candidates)

    env = globals()
    if spec in env:
        return coerce_candidate_records(env[spec])

    cand_path = Path(spec)
    if cand_path.exists():
        df = load_candidates_df(cand_path)
        return coerce_candidate_records(df)

    raise SystemExit(f"Unknown candidates spec '{spec}'. Provide a built-in name or a valid file path.")


def clean_for_bayes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep this consistent with the Bayesian module's cleaning so event_indices align with plotting."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    mask = np.ones(len(out), dtype=bool)

    if "saturated" in out.columns:
        mask &= (out["saturated"] == 0)

    mask &= out["JD"].notna() & out["mag"].notna()

    if "error" in out.columns:
        mask &= out["error"].notna() & (out["error"] > 0.0) & (out["error"] < 1.0)

    out = out.loc[mask].sort_values("JD").reset_index(drop=True)
    return out


def plot_light_curve_with_dips(
    dfg: pd.DataFrame,
    dfv: pd.DataFrame,
    res_g: dict,
    res_v: dict,
    source_id: str,
    plot_path: Path,
):
    # Use 1x2 layout (side-by-side) with separate x-axes for each band
    # V-band and g-band have different JD ranges and should not share x-axis
    fig, axes = pl.subplots(1, 2, figsize=(16, 6))

    # Baseline parameters (match the SHO-ish defaults; note baseline function takes q not Q)
    baseline_kwargs = {
        "S0": 0.0005,
        "w0": 0.0031415926535897933,
        "q": 0.7,
        "jitter": 0.006,
        "sigma_floor": None,
        "add_sigma_eff_col": True,
    }

    camera_colors = pl.cm.tab10(np.linspace(0, 1, 10))

    def plot_band(ax, df_band: pd.DataFrame, res: dict, band_label: str):
        if df_band is None or df_band.empty or "JD" not in df_band.columns or "mag" not in df_band.columns:
            ax.text(0.5, 0.5, f"No {band_label}-band data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{source_id} - {band_label}-band (no data)", fontsize=12)
            return

        plot = df_band.copy()
        if "error" in plot.columns:
            plot = plot[plot["error"] <= 1.0]

        df_baseline = None
        try:
            df_baseline = per_camera_gp_baseline(plot, **baseline_kwargs)
        except Exception:
            df_baseline = None

        cam_col = "camera#" if "camera#" in plot.columns else None
        if cam_col and cam_col in plot.columns:
            cameras = sorted(plot[cam_col].dropna().unique())
            for i, cam in enumerate(cameras):
                cam_mask = plot[cam_col] == cam
                cam_data = plot[cam_mask]
                if cam_data.empty:
                    continue

                color = camera_colors[i % len(camera_colors)]
                label = f"Cam {cam}"

                if "error" in cam_data.columns:
                    ax.errorbar(
                        cam_data["JD"], cam_data["mag"], yerr=cam_data["error"],
                        fmt="o", markersize=3, alpha=0.6, color=color,
                        label=label, elinewidth=1.0, capsize=1.5
                    )
                else:
                    ax.scatter(cam_data["JD"], cam_data["mag"], s=10, alpha=0.6, color=color, label=label)

                if df_baseline is not None and "baseline" in df_baseline.columns:
                    cam_base = df_baseline[df_baseline[cam_col] == cam].sort_values("JD")
                    if not cam_base.empty:
                        ax.plot(
                            cam_base["JD"], cam_base["baseline"],
                            color=color, linewidth=2.0, alpha=0.9, linestyle="--",
                            label=f"Baseline {label}"
                        )
        else:
            if "error" in plot.columns:
                ax.errorbar(
                    plot["JD"], plot["mag"], yerr=plot["error"],
                    fmt="o", markersize=3, alpha=0.6,
                    label=f"{band_label}-band", elinewidth=1.0, capsize=1.5
                )
            else:
                ax.scatter(plot["JD"], plot["mag"], s=10, alpha=0.6, label=f"{band_label}-band")

            if df_baseline is not None and "baseline" in df_baseline.columns:
                df_sorted = df_baseline.sort_values("JD")
                ax.plot(
                    df_sorted["JD"], df_sorted["baseline"],
                    linewidth=2.0, alpha=0.9, linestyle="--", label="baseline"
                )

        # -------------------------------------------------------------
        # MODIFIED LOGIC: Plot lines ONLY for BIC-confirmed events
        # -------------------------------------------------------------
        
        # 1. Attempt to plot based on run_summaries (BIC aware)
        run_summaries = res.get("dip", {}).get("run_summaries", [])
        confirmed_count = 0
        
        if run_summaries:
            for summary in run_summaries:
                # Filter: skip if morphology is 'noise' or 'none'
                morph = summary.get("morphology", "none")
                if morph not in ("noise", "none"):
                    confirmed_count += 1
                    
                    # Try to get t0 from params, otherwise use range center
                    t0 = summary.get("params", {}).get("t0")
                    if t0 is None:
                        start_jd = summary.get("start_jd")
                        end_jd = summary.get("end_jd")
                        if start_jd is not None and end_jd is not None:
                            t0 = (start_jd + end_jd) / 2.0
                    
                    if t0 is not None and np.isfinite(t0):
                        # Plot distinct solid red line for confirmed event
                        ax.axvline(t0, color='red', alpha=0.6, linestyle="-", linewidth=1.5)

        else:
            # 2. Fallback for legacy results (no summaries): Plot all triggered indices
            dip_indices = res.get("dip", {}).get("event_indices", np.array([], dtype=int))
            if isinstance(dip_indices, (list, tuple)):
                dip_indices = np.asarray(dip_indices, dtype=int)
            
            if isinstance(dip_indices, np.ndarray) and dip_indices.size > 0:
                valid = dip_indices[(dip_indices >= 0) & (dip_indices < len(plot))]
                if valid.size:
                    dip_jds = plot.iloc[valid]["JD"].to_numpy()
                    for jd in dip_jds:
                        ax.axvline(jd, alpha=0.3, linestyle="--", linewidth=1)
                    confirmed_count = len(dip_jds) # approximations

        n_triggers = int(res.get("dip", {}).get("n_dips", 0))
        
        ax.set_ylabel(f"Magnitude ({band_label})", fontsize=12)
        ax.set_title(f"{source_id} - {band_label}-band (triggers={n_triggers}, confirmed={confirmed_count})", fontsize=12)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best", ncol=2)

    # V-band on left (earlier JD, typically ends around JD 2458500)
    # g-band on right (later JD, typically starts around JD 2458500)
    plot_band(axes[0], dfv, res_v, "V")
    plot_band(axes[1], dfg, res_g, "g")
    axes[0].set_xlabel("JD", fontsize=12)
    axes[1].set_xlabel("JD", fontsize=12)

    pl.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    pl.savefig(plot_path, dpi=150, bbox_inches="tight")
    pl.close()


def build_reproduction_report(
    candidates: Sequence[Mapping[str, object]] | None = None,
    *,
    out_dir: Path | str = "./peak_results_repro",
    out_format: str = "csv",
    n_workers: int | None = None,
    chunk_size: int = 250000,
    metrics_baseline_func=None,
    metrics_dip_threshold: float = 0.3,
    # posterior-prob thresholding (legacy)
    bayes_significance_threshold: float | None = 99.99997,
    bayes_p_points: int = 80,
    # NEW: log BF triggering
    bayes_trigger_mode: str = "logbf",          # "logbf" or "posterior_prob"
    bayes_logbf_threshold_dip: float = 5.0,     # trigger if max log BF >= this
    bayes_logbf_threshold_jump: float = 5.0,
    skypatrol_dir: Path | str | None = None,
    path_prefix: Path | str | None = None,
    path_root: Path | str | None = None,
    extra_columns: Iterable[str] | None = None,
    manifest_path: Path | str | None = None,
    method: str = "naive",
    verbose: bool = False,
    # Filter options
    skip_pre_filters: bool = False,
    min_time_span: float = 100.0,
    min_points_per_day: float = 0.05,
    min_cameras: int = 2,
    skip_vsx: bool = False,
    vsx_catalog: Path | str = Path("input/vsx/vsxcat.090525.csv"),
    vsx_max_sep: float = 3.0,
    min_mag_offset: float = 0.1,
    skip_post_filters: bool = False,
    **baseline_kwargs,
) -> pd.DataFrame:
    manifest_df = load_manifest_df(manifest_path) if manifest_path is not None else None

    baseline_candidates = candidates if candidates is not None else brayden_candidates
    df_targets = dataframe_from_candidates(baseline_candidates)

    manifest_subset = None
    if manifest_df is not None:
        manifest_subset = manifest_df[manifest_df["source_id"].isin(df_targets["source_id"])].copy()
        if not manifest_subset.empty:
            cols = [
                col
                for col in ["source_id", "mag_bin", "lc_dir", "index_num", "index_csv", "dat_path", "dat_exists"]
                if col in manifest_subset.columns
            ]
            df_targets = df_targets.merge(manifest_subset[cols], on="source_id", how="left")
            if "mag_bin_x" in df_targets.columns and "mag_bin_y" in df_targets.columns:
                df_targets["mag_bin"] = df_targets["mag_bin_y"].fillna(df_targets["mag_bin_x"])
                df_targets = df_targets.drop(columns=["mag_bin_x", "mag_bin_y"])

    target_map_dict = target_map(df_targets)
    records_map = records_from_manifest(manifest_subset) if manifest_subset is not None else None

    if (records_map is None or not records_map) and skypatrol_dir is not None:
        records_map = records_from_skypatrol_dir(df_targets, Path(skypatrol_dir))
        if verbose:
            n_found = sum(len(v) for v in records_map.values())
            print(f"[DEBUG] Built records_map from skypatrol_dir: {n_found} light curves found")

    if (records_map is None or not records_map) and "path" in df_targets.columns:
        records_map = records_from_candidates_with_paths(
            df_targets,
            path_prefix=path_prefix,
            path_root=path_root,
        )
        if verbose:
            n_found = sum(len(v) for v in records_map.values())
            print(f"[DEBUG] Built records_map from candidates 'path' column: {n_found} light curves found")

    # Apply pre-filters if manifest is provided and not skipped
    if manifest_subset is not None and not skip_pre_filters and records_map:
        if verbose:
            total_before = sum(len(v) for v in records_map.values())
            print(f"\n[PRE-FILTER] Applying pre-filters to {total_before} candidates...")
        
        # Prepare dataframe for pre_filter (needs 'path' column pointing to lc_dir)
        df_pre = manifest_subset.rename(columns={"lc_dir": "path"}).copy()
        
        try:
            df_filtered = apply_pre_filters(
                df_pre,
                apply_sparse=True,
                min_time_span=min_time_span,
                min_points_per_day=min_points_per_day,
                apply_vsx=not skip_vsx,
                vsx_max_sep_arcsec=vsx_max_sep,
                vsx_catalog_csv=vsx_catalog,
                apply_multi_camera=True,
                min_cameras=min_cameras,
                n_workers=n_workers or 10,
                show_tqdm=verbose,
            )
            
            # Update records_map to only include filtered sources
            filtered_ids = set(df_filtered["source_id"].astype(str))
            for mag_bin in list(records_map.keys()):
                records_map[mag_bin] = [
                    rec for rec in records_map[mag_bin]
                    if str(rec.get("asas_sn_id")) in filtered_ids
                ]
                # Remove empty mag_bins
                if not records_map[mag_bin]:
                    del records_map[mag_bin]
            
            total_after = sum(len(v) for v in records_map.values())
            if verbose:
                print(f"[PRE-FILTER] Kept {total_after}/{total_before} candidates after pre-filtering")
                print(f"[PRE-FILTER] Rejected {total_before - total_after} candidates")
        
        except Exception as e:
            if verbose:
                print(f"[PRE-FILTER] Warning: pre-filter failed: {e}")
                print(f"[PRE-FILTER] Continuing without pre-filtering...")

    if method == "naive":
        raise DeprecationWarning(
            "The 'naive' method is deprecated and has been removed. "
            "Please use method='bayes' instead, which is the current implementation. "
            "The naive method used legacy code from malca.old.lc_events_naive."
        )

    elif method == "biweight":
        raise DeprecationWarning(
            "The 'biweight' method is deprecated and has been removed. "
            "Please use method='bayes' instead, which is the current implementation. "
            "The biweight method used legacy code from malca.old.lc_events."
        )

    elif method == "bayes":
        if records_map is None or not records_map:
            raise SystemExit("Bayesian method requires light-curve paths. Provide --manifest or --skypatrol-dir.")

        rows = []

        for mag_bin in sorted(records_map):
            for rec in records_map[mag_bin]:
                asn = rec.get("asas_sn_id")
                lc_dir = rec.get("lc_dir")
                dat_path = rec.get("dat_path")
                has_path = bool(dat_path) and Path(str(dat_path)).exists()

                if verbose and not has_path:
                    print(f"[DEBUG] {asn}: dat_path missing: {dat_path}")

                try:
                    if str(dat_path).endswith('.csv') and has_path:
                        from malca.plot import read_skypatrol_csv
                        df_all = read_skypatrol_csv(str(dat_path))
                        # read_skypatrol_csv standardizes v_g_band to 0=g, 1=V
                        if not df_all.empty and "v_g_band" in df_all.columns:
                            dfg = df_all[df_all["v_g_band"] == 0].reset_index(drop=True)
                            dfv = df_all[df_all["v_g_band"] == 1].reset_index(drop=True)
                        else:
                            dfg, dfv = pd.DataFrame(), pd.DataFrame()
                    else:
                        dfg, dfv = (
                            read_lc_dat2(asn, lc_dir)
                            if asn and lc_dir and has_path
                            else (pd.DataFrame(), pd.DataFrame())
                        )
                except Exception as e:
                    if verbose:
                        print(f"[DEBUG] {asn}: data load failed: {e}")
                    dfg, dfv = pd.DataFrame(), pd.DataFrame()

                if verbose:
                    print(f"[DEBUG] {asn}: loaded g={len(dfg)} rows, v={len(dfv)} rows from {dat_path}")

                def apply_triggering(result: dict, band_name: str) -> dict:
                    """
                    Force triggering to be based on either:
                      - log BF local (preferred)
                      - posterior probability threshold (legacy)
                    """
                    for kind in ("dip", "jump"):
                        block = result.get(kind, {})
                        if not isinstance(block, dict):
                            result[kind] = {"significant": False}
                            continue

                        if bayes_trigger_mode == "logbf":
                            thr = bayes_logbf_threshold_dip if kind == "dip" else bayes_logbf_threshold_jump
                            log_bf_local = block.get("log_bf_local", None)
                            if log_bf_local is None:
                                # if the bayes module wasn't updated, this stays off
                                block["event_indices"] = np.array([], dtype=int)
                                block["significant"] = False
                                block["max_log_bf_local"] = np.nan
                            else:
                                lb = np.asarray(log_bf_local, float)
                                finite = np.isfinite(lb)
                                max_lb = float(np.nanmax(lb)) if finite.any() else np.nan
                                idx = np.nonzero(finite & (lb >= float(thr)))[0].astype(int)
                                block["event_indices"] = idx
                                block["significant"] = bool(np.isfinite(max_lb) and (max_lb >= float(thr)))
                                block["max_log_bf_local"] = max_lb

                            # counts
                            block["max_event_prob"] = np.nan
                            block["n_dips"] = int(len(block.get("event_indices", []))) if kind == "dip" else block.get("n_dips", 0)
                            block["n_jumps"] = int(len(block.get("event_indices", []))) if kind == "jump" else block.get("n_jumps", 0)

                            if verbose:
                                mx = block.get("max_log_bf_local", np.nan)
                                bf = block.get("bayes_factor", np.nan)
                                ct = len(block.get("event_indices", []))
                                print(f"[DEBUG] {asn} {band_name}: {kind} max_logBF={mx:.2f} thr={thr:.2f} "
                                      f"count={ct} globalBF={bf:.2f}")

                        else:
                            # posterior_prob mode
                            event_prob = np.asarray(block.get("event_probability", np.array([])), float)
                            max_prob = float(np.nanmax(event_prob)) if event_prob.size else np.nan
                            block["max_event_prob"] = max_prob

                            event_indices = block.get("event_indices", np.array([], dtype=int))
                            if isinstance(event_indices, (list, tuple)):
                                event_indices = np.asarray(event_indices, dtype=int)
                            if not isinstance(event_indices, np.ndarray):
                                event_indices = np.array([], dtype=int)

                            count_key = "n_dips" if kind == "dip" else "n_jumps"
                            block[count_key] = int(event_indices.size)

                            if verbose:
                                dip_bf = block.get("bayes_factor", np.nan)
                                print(f"[DEBUG] {asn} {band_name}: {kind} max_prob={max_prob:.6f} "
                                      f"count={int(event_indices.size)} globalBF={dip_bf:.2f}")

                        result[kind] = block

                    return result

                def bayes(df: pd.DataFrame, band_name: str):
                    dfc = clean_for_bayes(df)
                    if dfc is None or dfc.empty:
                        return {
                            "dip": {"significant": False, "bayes_factor": np.nan, "max_event_prob": np.nan, "n_dips": 0, "max_log_bf_local": np.nan},
                            "jump": {"significant": False, "bayes_factor": np.nan, "max_event_prob": np.nan, "n_jumps": 0, "max_log_bf_local": np.nan},
                        }

                    try:
                        # physics convention note: if converting sigma->prob, we use one-tailed Gaussian CDF (norm.cdf)
                        result = run_bayesian_significance(
                            dfc,
                            significance_threshold=float(bayes_significance_threshold) if bayes_significance_threshold is not None else 99.99997,
                            p_points=int(bayes_p_points),
                        )
                        result = apply_triggering(result, band_name)
                        return result
                    except Exception as e:
                        if verbose:
                            print(f"[DEBUG] {asn} {band_name}: run_bayesian_significance failed: {e}")
                        return {
                            "dip": {"significant": False, "bayes_factor": np.nan, "max_event_prob": np.nan, "n_dips": 0, "max_log_bf_local": np.nan},
                            "jump": {"significant": False, "bayes_factor": np.nan, "max_event_prob": np.nan, "n_jumps": 0, "max_log_bf_local": np.nan},
                        }

                res_g = bayes(dfg, "g")
                res_v = bayes(dfv, "V")

                if out_dir:
                    plot_path = Path(out_dir) / f"{asn}_dips.png"
                    plot_light_curve_with_dips(
                        clean_for_bayes(dfg),
                        clean_for_bayes(dfv),
                        res_g,
                        res_v,
                        str(asn),
                        plot_path,
                    )

                rows.append(
                    {
                        "mag_bin": str(rec.get("mag_bin")),
                        "asas_sn_id": asn,
                        "index_num": rec.get("index_num"),
                        "index_csv": rec.get("index_csv"),
                        "lc_dir": lc_dir,
                        "dat_path": dat_path,

                        "g_bayes_dip_significant": bool(res_g["dip"].get("significant", False)),
                        "v_bayes_dip_significant": bool(res_v["dip"].get("significant", False)),
                        "g_bayes_jump_significant": bool(res_g["jump"].get("significant", False)),
                        "v_bayes_jump_significant": bool(res_v["jump"].get("significant", False)),

                        "g_bayes_dip_max_prob": float(res_g["dip"].get("max_event_prob", np.nan)),
                        "v_bayes_dip_max_prob": float(res_v["dip"].get("max_event_prob", np.nan)),

                        "g_bayes_dip_bayes_factor": float(res_g["dip"].get("bayes_factor", np.nan)),
                        "v_bayes_dip_bayes_factor": float(res_v["dip"].get("bayes_factor", np.nan)),
                        "g_bayes_jump_bayes_factor": float(res_g["jump"].get("bayes_factor", np.nan)),
                        "v_bayes_jump_bayes_factor": float(res_v["jump"].get("bayes_factor", np.nan)),

                        "g_bayes_dip_max_logbf": float(res_g["dip"].get("max_log_bf_local", np.nan)),
                        "v_bayes_dip_max_logbf": float(res_v["dip"].get("max_log_bf_local", np.nan)),
                        "g_bayes_jump_max_logbf": float(res_g["jump"].get("max_log_bf_local", np.nan)),
                        "v_bayes_jump_max_logbf": float(res_v["jump"].get("max_log_bf_local", np.nan)),

                        "g_bayes_n_dips": int(res_g["dip"].get("n_dips", 0)),
                        "v_bayes_n_dips": int(res_v["dip"].get("n_dips", 0)),
                        "g_bayes_n_jumps": int(res_g["jump"].get("n_jumps", 0)),
                        "v_bayes_n_jumps": int(res_v["jump"].get("n_jumps", 0)),
                    }
                )
    else:
        rows = None

    if rows is None:
        rows_df = pd.DataFrame(columns=["source_id", "mag_bin"])
    elif isinstance(rows, list):
        rows_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["source_id", "mag_bin"])
    elif isinstance(rows, pd.DataFrame):
        rows_df = rows.copy() if not rows.empty else pd.DataFrame(columns=["source_id", "mag_bin"])
    else:
        rows_df = pd.DataFrame(columns=["source_id", "mag_bin"])

    # Apply signal amplitude filter if enabled
    if not rows_df.empty and min_mag_offset > 0:
        if verbose:
            print(f"\n[SIGNAL-FILTER] Applying signal amplitude filter (min_mag_offset={min_mag_offset})...")
            n_before = len(rows_df)
        
        try:
            rows_df = filter_signal_amplitude(
                rows_df,
                min_mag_offset=min_mag_offset,
                show_tqdm=verbose,
            )
            
            if verbose:
                n_after = len(rows_df)
                print(f"[SIGNAL-FILTER] Kept {n_after}/{n_before} detections after signal amplitude filter")
                print(f"[SIGNAL-FILTER] Rejected {n_before - n_after} low-amplitude detections")
        
        except Exception as e:
            if verbose:
                print(f"[SIGNAL-FILTER] Warning: signal amplitude filter failed: {e}")
                print(f"[SIGNAL-FILTER] Continuing without signal amplitude filtering...")

    if "source_id" not in rows_df.columns:
        if "asas_sn_id" in rows_df.columns:
            rows_df["source_id"] = rows_df["asas_sn_id"].astype(str)
        else:
            rows_df["source_id"] = ""
    rows_df = rows_df.drop(columns=["asas_sn_id"], errors="ignore")
    if "mag_bin" not in rows_df.columns:
        rows_df["mag_bin"] = ""

    merged = df_targets.merge(rows_df, on=["source_id", "mag_bin"], how="left", suffixes=("", "_det"))

    g_peaks = merged["g_n_peaks"] if "g_n_peaks" in merged.columns else pd.Series(np.nan, index=merged.index)
    v_peaks = merged["v_n_peaks"] if "v_n_peaks" in merged.columns else pd.Series(np.nan, index=merged.index)
    g_bayes = (
        merged["g_bayes_dip_significant"].fillna(False).astype(bool)
        if "g_bayes_dip_significant" in merged.columns
        else pd.Series(False, index=merged.index)
    )
    v_bayes = (
        merged["v_bayes_dip_significant"].fillna(False).astype(bool)
        if "v_bayes_dip_significant" in merged.columns
        else pd.Series(False, index=merged.index)
    )

    merged["detected"] = (
        (g_peaks.fillna(0).astype(float) > 0)
        | (v_peaks.fillna(0).astype(float) > 0)
        | g_bayes
        | v_bayes
    )

    def format_detection(row: pd.Series) -> str:
        if not row.get("detected", False):
            return "—"
        parts = [f"mag_bin={row.get('mag_bin', '')}"]

        if "g_n_peaks" in row and pd.notna(row["g_n_peaks"]):
            parts.append(f"g_peaks={int(row['g_n_peaks'])}")
        if "v_n_peaks" in row and pd.notna(row["v_n_peaks"]):
            parts.append(f"v_peaks={int(row['v_n_peaks'])}")

        def bayes_part(prefix: str) -> str | None:
            sig = bool(row.get(f"{prefix}_bayes_dip_significant", False))
            if not sig:
                return None
            bf = row.get(f"{prefix}_bayes_dip_bayes_factor")
            bf_str = f"{float(bf):.3f}" if pd.notna(bf) else "nan"
            n_dips = int(row.get(f"{prefix}_bayes_n_dips", 0))
            mxlog = row.get(f"{prefix}_bayes_dip_max_logbf")
            if pd.notna(mxlog):
                return f"{prefix}_bayes_dip (maxlogBF={float(mxlog):.2f}, BF={bf_str}, n={n_dips})"
            return f"{prefix}_bayes_dip (BF={bf_str}, n={n_dips})"

        for pref in ("g", "v"):
            part = bayes_part(pref)
            if part:
                parts.append(part)

        return "; ".join(parts)

    merged["detection_details"] = merged.apply(format_detection, axis=1)

    if extra_columns:
        cols = [c for c in extra_columns if c in merged.columns]
    else:
        cols = []

    ordered_cols = [
        "source",
        "source_id",
        "category",
        "mag_bin",
        "detected",
        "detection_details",
    ]
    for col in [
        "g_n_peaks",
        "v_n_peaks",
        "g_bayes_dip_significant",
        "v_bayes_dip_significant",
        "g_bayes_dip_max_prob",
        "v_bayes_dip_max_prob",
        "g_bayes_dip_max_logbf",
        "v_bayes_dip_max_logbf",
        "g_bayes_jump_significant",
        "v_bayes_jump_significant",
        "g_bayes_dip_bayes_factor",
        "v_bayes_dip_bayes_factor",
        "g_bayes_jump_bayes_factor",
        "v_bayes_jump_bayes_factor",
        "g_bayes_n_dips",
        "v_bayes_n_dips",
        "g_bayes_n_jumps",
        "v_bayes_n_jumps",
        "g_max_depth",
        "v_max_depth",
        "jd_first",
        "jd_last",
    ]:
        if col in merged.columns:
            ordered_cols.append(col)
    ordered_cols.extend([c for c in cols if c not in ordered_cols])

    remaining = [c for c in merged.columns if c not in ordered_cols]
    ordered_cols.extend(remaining)

    return merged[ordered_cols]


__all__ = [
    "brayden_candidates",
    "build_reproduction_report",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run targeted reproduction search on events.py candidates and summarize results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on events.py output CSV (uses 'path' column directly)
  python -m malca.reproduction --input output/strong_candidates_12_12.5.csv --method bayes

  # With manifest for legacy data
  python -m malca.reproduction --candidates candidates.csv --manifest manifest.csv --method bayes

  # With SkyPatrol CSV files
  python -m malca.reproduction --candidates candidates.csv --skypatrol-dir input/skypatrol2 --method bayes
"""
    )
    parser.add_argument("--out-dir", default="./results_test", help="Directory for peak_results output")
    parser.add_argument("--out-format", choices=("csv", "parquet"), default="csv")
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="ProcessPool worker count for dip_finder_naive.",
    )
    parser.add_argument("--chunk-size", type=int, default=250000, help="Rows per chunk flush for CSV output")
    parser.add_argument("--metrics-dip-threshold", type=float, default=0.3, help="Dip threshold for run_metrics")

    # posterior-prob thresholding (legacy)
    parser.add_argument(
        "--bayes-significance-threshold",
        type=float,
        default=None,
        help="Per-point posterior threshold (percentage, e.g. 99.99997). Only used if --bayes-trigger-mode=posterior_prob.",
    )
    parser.add_argument(
        "--bayes-sigma",
        type=float,
        default=None,
        help="Sigma threshold converted to a one-tailed Gaussian CDF (%). Only used if --bayes-trigger-mode=posterior_prob.",
    )
    parser.add_argument("--bayes-p-points", type=int, default=80, help="Number of logit-spaced probability grid points")

    # NEW: log BF triggering
    parser.add_argument(
        "--bayes-trigger-mode",
        choices=("logbf", "posterior_prob"),
        default="logbf",
        help="Trigger Bayesian detections using log BF (recommended) or posterior probability (legacy).",
    )
    parser.add_argument(
        "--bayes-logbf-threshold-dip",
        type=float,
        default=5.0,
        help="Dip triggers when max per-point log BF >= this (only if --bayes-trigger-mode=logbf).",
    )
    parser.add_argument(
        "--bayes-logbf-threshold-jump",
        type=float,
        default=5.0,
        help="Jump triggers when max per-point log BF >= this (only if --bayes-trigger-mode=logbf).",
    )

    parser.add_argument("--skypatrol-dir", default=None, help="Directory with SkyPatrol CSV files (<source_id>-light-curves.csv)")
    parser.add_argument("--manifest", default=None, help="Path to lc_manifest CSV/Parquet for targeted reproduction")
    parser.add_argument(
        "--path-prefix",
        default=None,
        help="Path prefix to rewrite for candidates with a 'path' column (e.g. /data/poohbah/1/assassin/rowan.90/lcsv2).",
    )
    parser.add_argument(
        "--path-root",
        default=None,
        help="Local root that replaces --path-prefix for candidates with a 'path' column.",
    )
    parser.add_argument(
        "--baseline-func",
        default=None,
        help="Naive-method baseline function import path (e.g. module:func). This does not affect Bayesian baseline.",
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help="Candidate spec (built-in list name or path to CSV/Parquet file from events.py).",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Alias for --candidates (path to events.py output CSV/Parquet)."
    )
    parser.add_argument(
        "--method",
        choices=("bayes",),
        default="bayes",
        help="Detection method. Only 'bayes' (Bayesian) is currently supported. "
             "Legacy methods 'naive' and 'biweight' have been deprecated.",
    )
    
    # Pre-filter options
    parser.add_argument("--skip-pre-filters", action="store_true", 
                        help="Skip pre-filtering step (sparse LC, multi-camera, VSX)")
    parser.add_argument("--min-time-span", type=float, default=100.0,
                        help="Min time span in days for sparse LC filter (default: 100)")
    parser.add_argument("--min-points-per-day", type=float, default=0.05,
                        help="Min cadence for sparse LC filter (default: 0.05)")
    parser.add_argument("--min-cameras", type=int, default=2,
                        help="Min cameras required for multi-camera filter (default: 2)")
    parser.add_argument("--skip-vsx", action="store_true",
                        help="Skip VSX known variable filter")
    parser.add_argument("--vsx-catalog", type=Path, default=Path("input/vsx/vsxcat.090525.csv"),
                        help="Path to VSX catalog CSV")
    parser.add_argument("--vsx-max-sep", type=float, default=3.0,
                        help="Max separation for VSX match in arcsec (default: 3.0)")
    
    # Signal amplitude filter
    parser.add_argument("--min-mag-offset", type=float, default=0.1,
                        help="Min magnitude offset for signal amplitude filter (0 to disable, default: 0.1)")
    
    # Post-filter options (placeholder for future)
    parser.add_argument("--skip-post-filters", action="store_true",
                        help="Skip post-filtering step (currently no post-filters implemented)")
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Print debug info")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    kwargs = {}
    if args.baseline_func:
        if ":" in args.baseline_func:
            mod_name, func_name = args.baseline_func.split(":", 1)
        else:
            mod_name, func_name = "baseline", args.baseline_func
        mod = __import__(mod_name, fromlist=[func_name])
        kwargs["metrics_baseline_func"] = getattr(mod, func_name)

    candidates_spec = args.input or args.candidates
    candidate_data = resolve_candidates(candidates_spec)

    # posterior-prob threshold (only if requested)
    bayes_significance_threshold = args.bayes_significance_threshold
    if args.bayes_trigger_mode == "posterior_prob":
        # physics convention note: one-tailed Gaussian threshold uses CDF(sigma)
        if args.bayes_sigma is not None:
            prob = stats.norm.cdf(args.bayes_sigma)
            bayes_significance_threshold = prob * 100.0
            if args.verbose:
                print(f"[DEBUG] Converting {args.bayes_sigma}-sigma to significance threshold: {bayes_significance_threshold:.8f}%")
        elif bayes_significance_threshold is None:
            prob = stats.norm.cdf(5.0)
            bayes_significance_threshold = prob * 100.0
            if args.verbose:
                print(f"[DEBUG] Using default 5-sigma significance threshold: {bayes_significance_threshold:.8f}%")
    else:
        bayes_significance_threshold = None
        if args.verbose:
            print(f"[DEBUG] Using logBF triggering: dip_thr={args.bayes_logbf_threshold_dip}, jump_thr={args.bayes_logbf_threshold_jump}")

    report = build_reproduction_report(
        candidates=candidate_data,
        out_dir=args.out_dir,
        out_format=args.out_format,
        n_workers=args.workers,
        chunk_size=args.chunk_size,
        metrics_dip_threshold=args.metrics_dip_threshold,
        bayes_significance_threshold=bayes_significance_threshold,
        bayes_p_points=args.bayes_p_points,
        bayes_trigger_mode=args.bayes_trigger_mode,
        bayes_logbf_threshold_dip=args.bayes_logbf_threshold_dip,
        bayes_logbf_threshold_jump=args.bayes_logbf_threshold_jump,
        skypatrol_dir=args.skypatrol_dir,
        path_prefix=args.path_prefix,
        path_root=args.path_root,
        manifest_path=args.manifest,
        method=args.method,
        verbose=args.verbose,
        # Filter parameters
        skip_pre_filters=args.skip_pre_filters,
        min_time_span=args.min_time_span,
        min_points_per_day=args.min_points_per_day,
        min_cameras=args.min_cameras,
        skip_vsx=args.skip_vsx,
        vsx_catalog=args.vsx_catalog,
        vsx_max_sep=args.vsx_max_sep,
        min_mag_offset=args.min_mag_offset,
        skip_post_filters=args.skip_post_filters,
        **kwargs,
    )

    # Print filtering summary
    print("\n" + "=" * 60)
    print("FILTERING SUMMARY")
    print("=" * 60)
    print(f"Pre-filters:          {'APPLIED' if not args.skip_pre_filters else 'SKIPPED'}")
    if not args.skip_pre_filters:
        print(f"  - Sparse LC filter:   min_time_span={args.min_time_span}d, min_cadence={args.min_points_per_day}/d")
        print(f"  - Multi-camera:       min_cameras={args.min_cameras}")
        print(f"  - VSX filter:         {'APPLIED' if not args.skip_vsx else 'SKIPPED'}")
    print(f"Signal amplitude:     {'APPLIED (min_mag_offset=' + str(args.min_mag_offset) + ')' if args.min_mag_offset > 0 else 'DISABLED'}")
    print(f"Post-filters:         {'APPLIED' if not args.skip_post_filters else 'SKIPPED (none implemented)'}")
    print("=" * 60 + "\n")

    columns = [
        "source",
        "source_id",
        "mag_bin",
        "detected",
        "detection_details",
        "g_n_peaks",
        "v_n_peaks",
        "g_bayes_dip_significant",
        "v_bayes_dip_significant",
        "g_bayes_n_dips",
        "v_bayes_n_dips",
        "g_bayes_dip_max_prob",
        "v_bayes_dip_max_prob",
        "g_bayes_dip_max_logbf",
        "v_bayes_dip_max_logbf",
        "g_bayes_dip_bayes_factor",
        "v_bayes_dip_bayes_factor",
    ]
    existing = [c for c in columns if c in report.columns]
    print(report[existing].to_string(index=False))


if __name__ == "__main__":
    main()
