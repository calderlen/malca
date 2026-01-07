"""
Skypatrol light-curve exploration utilities.

Loads `*-light-curves.csv` files from a directory, computes summary metrics,
and writes a set of informative plots. Designed to be notebook-friendly:

Example:
    from malca.skypatrol_explore import run_exploration
    run_exploration("input/skypatrol2")

Outputs (default): PNGs under output/skypatrol_explore/
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix


def load_lightcurve_csv(path: Path) -> pd.DataFrame:
    """Read one skypatrol CSV and return a cleaned DataFrame."""
    df = pd.read_csv(path, comment="#", skip_blank_lines=True)
    rename_map = {"Mag": "mag", "Mag Error": "error", "JD": "JD", "Filter": "filter", "Camera": "camera"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    for col in ["JD", "mag", "error"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    mask = (
        df["JD"].notna()
        & df["mag"].notna()
        & df["error"].notna()
        & (df["error"] > 0)
        & (df["error"] < 10)
    )
    return df.loc[mask].sort_values("JD").reset_index(drop=True)


def collect_metrics(paths: Sequence[Path]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load all light curves and compute summary metrics."""
    metrics: list[dict[str, object]] = []
    lcs: dict[str, pd.DataFrame] = {}
    for path in paths:
        df = load_lightcurve_csv(path)
        if df.empty:
            continue
        name = path.name
        lcs[name] = df
        metrics.append({
            "lc_name": name,
            "n_points": len(df),
            "mag_mean": df["mag"].mean(),
            "mag_std": df["mag"].std(),
            "mag_min": df["mag"].min(),
            "mag_max": df["mag"].max(),
            "mag_range": df["mag"].max() - df["mag"].min(),
            "error_med": df["error"].median(),
            "jd_span": df["JD"].max() - df["JD"].min(),
        })
    return pd.DataFrame.from_records(metrics), lcs


def plot_scatter_with_labels(df: pd.DataFrame, x: str, y: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df[x], df[y], s=30, alpha=0.7)
    for _, row in df.iterrows():
        ax.annotate(row["lc_name"].split("-")[0], (row[x], row[y]), fontsize=7, alpha=0.6)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hist(df: pd.DataFrame, col: str, out_path: Path, bins: int = 40) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[col].dropna(), bins=bins, alpha=0.8, edgecolor="k")
    ax.set_xlabel(col)
    ax.set_ylabel("count")
    ax.set_title(f"Histogram of {col}")
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lightcurve(df: pd.DataFrame, name: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    yerr = df["error"] if "error" in df.columns else None
    ax.errorbar(df["JD"], df["mag"], yerr=yerr, fmt=".", ms=3, alpha=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("JD")
    ax.set_ylabel("Mag")
    ax.set_title(name)
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_corr_heatmap(df: pd.DataFrame, cols: Sequence[str], out_path: Path) -> None:
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    ax.set_title("Correlation matrix")
    fig.colorbar(cax, ax=ax, shrink=0.8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scatter_matrix(df: pd.DataFrame, cols: Sequence[str], out_path: Path) -> None:
    axarr = scatter_matrix(df[cols], figsize=(8, 8), diagonal="hist", color="steelblue", alpha=0.7)
    # Rotate labels for readability
    for ax in axarr.flatten():
        ax.set_xlabel(ax.get_xlabel(), rotation=45, ha="right")
        ax.set_ylabel(ax.get_ylabel(), rotation=45, ha="right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_exploration(
    lc_root: str | Path,
    *,
    output_dir: str | Path = "output/skypatrol_explore",
    max_lightcurves: int | None = None,
    sample_plot_count: int = 5,
) -> dict[str, object]:
    """
    Explore skypatrol light curves and write summary plots.

    Parameters
    ----------
    lc_root : str | Path
        Directory containing `*-light-curves.csv` files.
    output_dir : str | Path
        Where to save plots (default: output/skypatrol_explore).
    max_lightcurves : int | None
        Limit how many files to load (None = all).
    sample_plot_count : int
        Number of individual light-curve plots to save.

    Returns
    -------
    dict with keys:
        metrics_df: DataFrame of summary metrics
        lcs: dict of light-curve DataFrames keyed by filename
        output_dir: Path to plots
    """
    lc_root = Path(lc_root)
    if not lc_root.is_absolute():
        lc_root = Path(__file__).resolve().parents[1] / lc_root
    output_dir = Path(output_dir)
    paths = sorted(lc_root.glob("*-light-curves.csv"))
    if max_lightcurves is not None:
        paths = paths[:max_lightcurves]
    print(f"[skypatrol_explore] found {len(paths)} files under {lc_root}")

    metrics_df, lcs = collect_metrics(paths)
    if metrics_df.empty:
        print("[skypatrol_explore] no data to plot")
        return {"metrics_df": metrics_df, "lcs": lcs, "output_dir": output_dir}

    # Scatter plots
    plot_scatter_with_labels(metrics_df, "n_points", "mag_std", output_dir / "mag_std_vs_n_points.png")
    plot_scatter_with_labels(metrics_df, "n_points", "mag_mean", output_dir / "mag_mean_vs_n_points.png")
    plot_scatter_with_labels(metrics_df, "jd_span", "mag_std", output_dir / "mag_std_vs_jd_span.png")
    plot_scatter_with_labels(metrics_df, "mag_std", "mag_range", output_dir / "mag_range_vs_mag_std.png")
    # Histograms
    plot_hist(metrics_df, "mag_std", output_dir / "hist_mag_std.png")
    plot_hist(metrics_df, "mag_mean", output_dir / "hist_mag_mean.png")
    plot_hist(metrics_df, "error_med", output_dir / "hist_error_med.png")
    plot_hist(metrics_df, "n_points", output_dir / "hist_n_points.png")
    # Correlation heatmap and scatter matrix across key metrics
    metric_cols = ["n_points", "mag_mean", "mag_std", "mag_range", "error_med", "jd_span"]
    plot_corr_heatmap(metrics_df, metric_cols, output_dir / "corr_heatmap.png")
    plot_scatter_matrix(metrics_df, metric_cols, output_dir / "scatter_matrix.png")
    # Sample individual light-curve plots
    for name in list(lcs.keys())[:sample_plot_count]:
        plot_lightcurve(lcs[name], name, output_dir / f"{name.replace('.csv', '')}_jd_mag.png")

    print(f"[skypatrol_explore] plots written to {output_dir.resolve()}")
    return {"metrics_df": metrics_df, "lcs": lcs, "output_dir": output_dir}


if __name__ == "__main__":
    run_exploration("input/skypatrol2")
