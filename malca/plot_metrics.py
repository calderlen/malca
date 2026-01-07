"""
Plot helper functions for inspecting light-curve metrics across a few sources.

Designed for quick use from scripts or Jupyter. No heavy dependencies beyond matplotlib.

Functions:
    plot_metrics_for_ids(...)  -> save 2D metric plots per light curve
    plot_3d_surface(...)       -> quick 3D scatter for exploring parameter space
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)


def plot_metrics_for_ids(
    df: pd.DataFrame,
    *,
    id_col: str = "asas_sn_id",
    ids: Sequence[str] | None = None,
    x_cols: Sequence[str] = ("p_points", "mag_points"),
    metrics: Sequence[str] = ("dip_bd", "jump_bd", "dip_best_p", "jump_best_p", "elapsed_s"),
    output_dir: Path | str = "/output/plots_metrics",
) -> list[Path]:
    """
    For each requested ID, make 2D plots of several metrics versus one or more x-axes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the metrics to plot.
    id_col : str
        Column that identifies the light curve (default: 'asas_sn_id').
    ids : sequence of str | None
        IDs to plot. If None, plots the first 5 unique IDs in the dataframe.
    x_cols : sequence of str
        Columns to use as x-axes (e.g., 'p_points', 'mag_points').
    metrics : sequence of str
        Metric columns to plot on the y-axis (multiple series per subplot).
    output_dir : Path | str
        Directory to write PNGs. Defaults to /output/plots_metrics.

    Returns
    -------
    list[Path]
        Paths to the generated plot files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if ids is None:
        ids = df[id_col].dropna().astype(str).unique()[:5]

    saved_paths: list[Path] = []

    for id_val in ids:
        sub = df[df[id_col].astype(str) == str(id_val)]
        if sub.empty:
            continue

        fig, axes = plt.subplots(1, len(x_cols), figsize=(6 * len(x_cols), 4), squeeze=False)
        for ax, x_col in zip(axes[0], x_cols):
            if x_col not in sub.columns:
                ax.set_visible(False)
                continue
            x_vals = sub[x_col].to_numpy()
            has_series = False
            for metric in metrics:
                if metric not in sub.columns:
                    continue
                y_vals = sub[metric].to_numpy()
                ax.plot(x_vals, y_vals, marker="o", label=metric)
                has_series = True
            ax.set_xlabel(x_col)
            ax.set_ylabel("value")
            ax.set_title(f"{id_val} vs {x_col}")
            if has_series:
                ax.grid(True, alpha=0.3)
        # Add a single legend if any metric was plotted
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(metrics), 4))
        fig.tight_layout()
        out_path = output_dir / f"{id_val}_metrics.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def plot_3d_surface(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    z_col: str,
    color_col: str | None = None,
    id_col: str | None = None,
    id_value: str | int | None = None,
    output_path: Path | str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Quick 3D scatter for exploring a parameter surface (usable in notebooks).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to plot.
    x_col, y_col, z_col : str
        Columns to use for the 3D axes.
    color_col : str | None
        Optional column to color points by (uses a colormap).
    id_col : str | None
        Optional ID column to filter on.
    id_value : str | int | None
        If provided with id_col, filter to this ID before plotting.
    output_path : Path | str | None
        If given, save the plot to this path.

    Returns
    -------
    (fig, ax)
        Matplotlib figure and axes (3D) for further tweaking or interactive use.
    """
    df_plot = df
    if id_col is not None and id_value is not None:
        df_plot = df_plot[df_plot[id_col].astype(str) == str(id_value)]
    if df_plot.empty:
        raise ValueError("No data to plot after filtering.")

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    xs = df_plot[x_col].to_numpy()
    ys = df_plot[y_col].to_numpy()
    zs = df_plot[z_col].to_numpy()

    if color_col and color_col in df_plot.columns:
        colors = df_plot[color_col].to_numpy()
        sc = ax.scatter(xs, ys, zs, c=colors, cmap="viridis", s=20, alpha=0.8)
        fig.colorbar(sc, ax=ax, shrink=0.6, label=color_col)
    else:
        ax.scatter(xs, ys, zs, s=20, alpha=0.8)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(f"3D view: {z_col} vs {x_col}, {y_col}")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)

    return fig, ax


if __name__ == "__main__":
    print("This module provides plotting helpers. Import and call plot_metrics_for_ids / plot_3d_surface from scripts or notebooks.")
