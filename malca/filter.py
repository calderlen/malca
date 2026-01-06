"""
Filters that run DURING events.py processing.
These filters use both baseline data and event detection results.

Note: This module is currently for signal amplitude filtering only.
Other filters have been separated into:
    - pre_filter.py: filters before events.py (based on raw LC data)
    - post_filter.py: filters after events.py (based on events.py output)

Filters:
11-12. filter_signal_amplitude - enforce minimum magnitude offset between event and baseline

Required input columns:
    baseline_mag (from baseline fitting during events.py),
    dip_best_mag_event, jump_best_mag_event (from events.py)
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm


def log_rejections(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    filter_name: str,
    log_csv: str | Path | None,
) -> None:
    """
    Log rejected candidates to a CSV file.
    """
    if log_csv is None:
        return

    # Try to find an ID column
    id_col = None
    for candidate in ["path", "asas_sn_id", "id", "source_id"]:
        if candidate in df_before.columns:
            id_col = candidate
            break

    if id_col is None:
        return

    before_ids = set(df_before[id_col].astype(str))
    after_ids = set(df_after[id_col].astype(str))
    rejected = sorted(before_ids - after_ids)
    if not rejected:
        return

    log_path = Path(log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df_log = pd.DataFrame({id_col: rejected, "filter": filter_name})
    df_log.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)


def filter_signal_amplitude(
    df: pd.DataFrame,
    *,
    min_mag_offset: float = 0.05,
    show_tqdm: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Enforce |best_mag_event - baseline_mag| > threshold (e.g., 0.05 - 0.1 mag).

    This filter ensures events have sufficient magnitude deviation from baseline.
    Requires baseline_mag to be computed during events.py baseline fitting.
    """
    n0 = len(df)
    pbar = tqdm(total=2, desc="filter_signal_amplitude", leave=False) if show_tqdm else None

    dip_diff = np.abs(df["dip_best_mag_event"] - df["baseline_mag"])
    jump_diff = np.abs(df["jump_best_mag_event"] - df["baseline_mag"])
    mask = (dip_diff > min_mag_offset) | (jump_diff > min_mag_offset)

    out = df.loc[mask].reset_index(drop=True)

    if pbar:
        pbar.update(1)

    if show_tqdm:
        tqdm.write(f"[filter_signal_amplitude] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_signal_amplitude", rejected_log_csv)

    if pbar:
        pbar.update(1)
        pbar.close()

    return out
