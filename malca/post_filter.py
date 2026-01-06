"""
Post-filters that run AFTER events.py.
These filters depend only on the output columns from events.py.

Filters:
7. filter_posterior_strength - require strong Bayes factors
8. filter_event_probability - require high event probabilities
9. filter_run_robustness - require sufficient run count and points
10. filter_morphology - require specific morphology with good BIC

Required input columns (from events.py):
    dip_bayes_factor, jump_bayes_factor,
    dip_max_log_bf_local, jump_max_log_bf_local,
    dip_max_event_prob, jump_max_event_prob,
    dip_run_count, jump_run_count,
    dip_max_run_points, jump_max_run_points,
    dip_best_morph, jump_best_morph,
    dip_best_delta_bic, jump_best_delta_bic
"""

from __future__ import annotations
from pathlib import Path
from time import perf_counter
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
    Assumes 'path' column exists (from events.py).
    """
    if log_csv is None:
        return

    before_ids = set(df_before["path"].astype(str))
    after_ids = set(df_after["path"].astype(str))
    rejected = sorted(before_ids - after_ids)
    if not rejected:
        return

    log_path = Path(log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df_log = pd.DataFrame({"path": rejected, "filter": filter_name})
    df_log.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)


def filter_posterior_strength(
    df: pd.DataFrame,
    *,
    min_bayes_factor: float = 10.0,
    require_finite_local_bf: bool = True,
    show_tqdm: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Require dip_bayes_factor or jump_bayes_factor > threshold.
    Optionally require dip_max_log_bf_local or jump_max_log_bf_local to be finite.
    """
    n0 = len(df)
    pbar = tqdm(total=2, desc="filter_posterior_strength", leave=False) if show_tqdm else None

    # At least one of dip or jump BF must exceed threshold
    mask = (df["dip_bayes_factor"].fillna(0) > min_bayes_factor) | \
           (df["jump_bayes_factor"].fillna(0) > min_bayes_factor)

    # Require finite local BF if requested
    if require_finite_local_bf:
        is_finite_dip = df["dip_max_log_bf_local"].notna() & np.isfinite(df["dip_max_log_bf_local"])
        is_finite_jump = df["jump_max_log_bf_local"].notna() & np.isfinite(df["jump_max_log_bf_local"])
        mask &= (is_finite_dip | is_finite_jump)

    out = df.loc[mask].reset_index(drop=True)

    if pbar:
        pbar.update(1)

    if show_tqdm:
        tqdm.write(f"[filter_posterior_strength] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_posterior_strength", rejected_log_csv)

    if pbar:
        pbar.update(1)
        pbar.close()

    return out


# =============================================================================
# Filter 8: Event probability
# ============================================================================

def filter_event_probability(
    df: pd.DataFrame,
    *,
    min_event_prob: float = 0.5,
    show_tqdm: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Keep light curves with dip_max_event_prob or jump_max_event_prob > threshold.
    """
    n0 = len(df)
    pbar = tqdm(total=2, desc="filter_event_probability", leave=False) if show_tqdm else None

    mask = (df["dip_max_event_prob"].fillna(0) > min_event_prob) | \
           (df["jump_max_event_prob"].fillna(0) > min_event_prob)

    out = df.loc[mask].reset_index(drop=True)

    if pbar:
        pbar.update(1)

    if show_tqdm:
        tqdm.write(f"[filter_event_probability] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_event_probability", rejected_log_csv)

    if pbar:
        pbar.update(1)
        pbar.close()

    return out


# =============================================================================
# Filter 9: Run robustness
# =============================================================================

def filter_run_robustness(
    df: pd.DataFrame,
    *,
    min_run_count: int = 1,
    min_run_points: int = 3,
    show_tqdm: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Require dip_run_count or jump_run_count >= min_run_count.
    Require dip_max_run_points or jump_max_run_points >= min_run_points.
    """
    n0 = len(df)
    pbar = tqdm(total=2, desc="filter_run_robustness", leave=False) if show_tqdm else None

    # Check run counts
    dip_count_ok = df["dip_run_count"].fillna(0) >= min_run_count
    jump_count_ok = df["jump_run_count"].fillna(0) >= min_run_count
    has_runs = dip_count_ok | jump_count_ok

    # Check run points
    dip_points_ok = (df["dip_max_run_points"].fillna(0) >= min_run_points) | ~dip_count_ok
    jump_points_ok = (df["jump_max_run_points"].fillna(0) >= min_run_points) | ~jump_count_ok

    mask = has_runs & (dip_points_ok | jump_points_ok)
    out = df.loc[mask].reset_index(drop=True)

    if pbar:
        pbar.update(1)

    if show_tqdm:
        tqdm.write(f"[filter_run_robustness] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_run_robustness", rejected_log_csv)

    if pbar:
        pbar.update(1)
        pbar.close()

    return out


# =============================================================================
# Filter 10: Morphology
# =============================================================================

def filter_morphology(
    df: pd.DataFrame,
    *,
    dip_morphology: str = "gaussian",
    jump_morphology: str = "paczynski",
    min_delta_bic: float = 10.0,
    show_tqdm: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Keep runs whose best morphology is 'gaussian' for dips or 'paczynski' for jumps,
    with dip_best_delta_bic/jump_best_delta_bic >= threshold to reject noise-like runs.
    """
    n0 = len(df)
    pbar = tqdm(total=2, desc="filter_morphology", leave=False) if show_tqdm else None

    # Check morphology for dips
    dip_morph_ok = (df["dip_best_morph"].fillna("").str.lower() == dip_morphology.lower()) & \
                   (df["dip_best_delta_bic"].fillna(0) >= min_delta_bic)

    # Check morphology for jumps
    jump_morph_ok = (df["jump_best_morph"].fillna("").str.lower() == jump_morphology.lower()) & \
                    (df["jump_best_delta_bic"].fillna(0) >= min_delta_bic)

    mask = dip_morph_ok | jump_morph_ok
    out = df.loc[mask].reset_index(drop=True)

    if pbar:
        pbar.update(1)

    if show_tqdm:
        tqdm.write(f"[filter_morphology] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_morphology", rejected_log_csv)

    if pbar:
        pbar.update(1)
        pbar.close()

    return out


# =============================================================================
# Main orchestration
# =============================================================================

def apply_post_filters(
    df: pd.DataFrame,
    *,
    # Filter 7: posterior strength
    apply_posterior_strength: bool = True,
    min_bayes_factor: float = 10.0,
    require_finite_local_bf: bool = True,
    # Filter 8: event probability
    apply_event_probability: bool = True,
    min_event_prob: float = 0.5,
    # Filter 9: run robustness
    apply_run_robustness: bool = True,
    min_run_count: int = 1,
    min_run_points: int = 3,
    # Filter 10: morphology
    apply_morphology: bool = False,
    dip_morphology: str = "gaussian",
    jump_morphology: str = "paczynski",
    min_delta_bic: float = 10.0,
    # General
    show_tqdm: bool = True,
    rejected_log_csv: str | Path | None = "rejected_post_filter.csv",
) -> pd.DataFrame:
    """
    Apply post-filters after running events.py.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe from events.py
    apply_* : bool
        Whether to apply each filter
    show_tqdm : bool
        Show progress bars
    rejected_log_csv : str | Path | None
        Path to log rejected candidates

    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    df_filtered = df.copy()
    n_start = len(df_filtered)

    filters = []

    if apply_posterior_strength:
        filters.append(("posterior_strength", filter_posterior_strength, {
            "min_bayes_factor": min_bayes_factor,
            "require_finite_local_bf": require_finite_local_bf,
            "show_tqdm": show_tqdm,
            "rejected_log_csv": rejected_log_csv,
        }))

    if apply_event_probability:
        filters.append(("event_probability", filter_event_probability, {
            "min_event_prob": min_event_prob,
            "show_tqdm": show_tqdm,
            "rejected_log_csv": rejected_log_csv,
        }))

    if apply_run_robustness:
        filters.append(("run_robustness", filter_run_robustness, {
            "min_run_count": min_run_count,
            "min_run_points": min_run_points,
            "show_tqdm": show_tqdm,
            "rejected_log_csv": rejected_log_csv,
        }))

    if apply_morphology:
        filters.append(("morphology", filter_morphology, {
            "dip_morphology": dip_morphology,
            "jump_morphology": jump_morphology,
            "min_delta_bic": min_delta_bic,
            "show_tqdm": show_tqdm,
            "rejected_log_csv": rejected_log_csv,
        }))

    # Apply filters sequentially
    total_steps = len(filters)
    if total_steps > 0:
        with tqdm(total=total_steps, desc="apply_post_filters", leave=True, disable=not show_tqdm) as pbar:
            for label, func, kwargs in filters:
                n_before = len(df_filtered)
                start = perf_counter()
                df_filtered = func(df_filtered, **kwargs)
                elapsed = perf_counter() - start
                n_after = len(df_filtered)

                pbar.set_postfix_str(f"{label}: {n_before} → {n_after} ({elapsed:.2f}s)")
                pbar.update(1)

    n_end = len(df_filtered)
    if show_tqdm:
        tqdm.write(f"\n[apply_post_filters] Total: {n_start} → {n_end} ({n_end/n_start*100:.1f}% kept)")

    return df_filtered.reset_index(drop=True)
