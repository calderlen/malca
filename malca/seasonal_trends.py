"""
Refactor of the LTvar-style seasonal-trend code.

Key behavior preserved from the brute-force version:
- Read ASAS-SN .dat light curves, keep only good points and g-band (good/bad==1, v/g?==0)
- Convert times from (jd ~ JD-2450000) to full JD via JD = jd + 2450000
- Special hard-coded Target filter: 17181160895 drops JD < 2.458e6
- Compute “season gap midpoints” from RA and dspring, then keep only midpoints inside [min(JD), max(JD)]
- Require at least 2 midpoints (otherwise skip) like the original (it `continue`s on mid_length==1)
- Cap at 12 seasons by using at most the *earliest 11* midpoints (same net effect as using mid[-1]..mid[-11])
- Define seasons with strict inequalities (points exactly on a midpoint are excluded)
- Compute per-season medians for *non-empty* seasons, keep their original season numbers as x-values
  (this is what the giant if/elif ladder was trying to do with e.g. indexes=[1,5,6,...])
- Fit linear and quadratic polynomials to (season_index, season_median)
- Compute “max diff” using the same (buggy but preserved) algebra as in the snippet.

Notes:
- The original uses ID['ra_deg'] but treats it like hours in the formula.
  Default below preserves that behavior. If your RA is truly degrees, pass --ra-is-deg to convert deg->hours.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.stats import mad_std
from astropy.table import Table
from tqdm import tqdm


LC_COLUMNS = ["jd", "mag", "error", "good/bad", "camera", "v/g?", "saturated/unsaturated", "camera,field"]


@dataclass(frozen=True)
class Config:
    root: Path
    mag_bin: str
    output: Path
    dir_start: int
    dir_end: int
    dspring: float
    ra_is_deg: bool
    max_seasons: int
    n_midpoints: int
    min_points_per_season: int
    min_seasons_for_quadratic: int
    write_per_dir: bool


def parse_args() -> Config:
    p = argparse.ArgumentParser(prog="LTvar_refactor.py")

    p.add_argument("--root", default="/data/poohbah/1/assassin/rowan.90/lcsv2/", type=str)
    p.add_argument("--mag-bin", default="13_13.5", type=str)
    p.add_argument("--output", default=None, type=str, help="Combined output CSV (default: LTvar<MAG>.csv)")
    p.add_argument("--dir-start", type=int, default=0)
    p.add_argument("--dir-end", type=int, default=30)

    # Preserve constants/behavior
    p.add_argument("--dspring", type=float, default=2460023.5)
    p.add_argument(
        "--ra-is-deg",
        action="store_true",
        help="Convert ID['ra_deg'] from degrees to hours before the dspring formula.",
    )
    p.add_argument("--max-seasons", type=int, default=12)
    p.add_argument(
        "--n-midpoints",
        type=int,
        default=None,
        help="How many yearly midpoints to generate before filtering to data range (default: dir_end+1, like the snippet).",
    )
    p.add_argument(
        "--min-points-per-season",
        type=int,
        default=1,
        help="Treat seasons with < this many points as empty. (The snippet mostly uses 0, sometimes <=1; default 1 is safest.)",
    )
    p.add_argument(
        "--min-seasons-for-quadratic",
        type=int,
        default=3,
        help="Need at least this many non-empty seasons to do degree-2 polyfit (default 3).",
    )
    p.add_argument(
        "--write-per-dir",
        action="store_true",
        help="Also write per-directory CSVs to <MAG_BIN>/new/<x>.csv (similar spirit to the snippet).",
    )

    a = p.parse_args()

    root = Path(a.root)
    mag_bin = a.mag_bin
    out = a.output
    if out is None:
        out = f"LTvar{mag_bin.replace('_','-')}.csv"
    output = Path(out)

    n_midpoints = a.n_midpoints if a.n_midpoints is not None else (a.dir_end + 1)

    return Config(
        root=root,
        mag_bin=mag_bin,
        output=output,
        dir_start=a.dir_start,
        dir_end=a.dir_end,
        dspring=float(a.dspring),
        ra_is_deg=bool(a.ra_is_deg),
        max_seasons=int(a.max_seasons),
        n_midpoints=int(n_midpoints),
        min_points_per_season=int(a.min_points_per_season),
        min_seasons_for_quadratic=int(a.min_seasons_for_quadratic),
        write_per_dir=bool(a.write_per_dir),
    )


def read_index_csv(path: Path) -> pd.DataFrame:
    # Matches the snippet's weird sep of comma OR tab.
    return pd.read_table(path, sep=r"\,|\t", engine="python")


def read_lc_dat(path: Path) -> pd.DataFrame:
    df = pd.read_table(path, sep=r"\s+", names=LC_COLUMNS, engine="python")
    # Preserve intended time convention: input 'jd' looks like JD-2450000 in many ASAS-SN products.
    df["JD"] = df["jd"].astype(float) + 2450000.0
    return df


def filter_lc(df: pd.DataFrame, target_id: int) -> pd.DataFrame:
    # Preserve the snippet's cuts
    df = df[df["good/bad"] == 1]
    df = df[df["v/g?"] == 0]  # g-band

    # Special-case hard-coded target behavior
    if target_id == 17181160895:
        df = df[df["JD"] >= 2.458e6]

    return df


def seasonal_midpoints_from_ra(
    ra_val: float, *,
    ra_is_deg: bool,
    dspring: float,
    n_midpoints: int,
) -> np.ndarray:
    """
    Replicates:
      date1 = dspring + 365.25*(RA-12.0)/24.0
      date2 = date1 + 365.25/2.0 + 365.25
      mid(n) = date2 - n*365.25
    """
    ra_hours = ra_val / 15.0 if ra_is_deg else ra_val
    n = np.arange(n_midpoints, dtype=float)

    date1 = dspring + 365.25 * (ra_hours - 12.0) / 24.0
    date2 = date1 + 365.25 / 2.0 + 365.25
    mid = date2 - n * 365.25

    return np.asarray(mid, dtype=float)


def choose_midpoints_in_range(mid: np.ndarray, tmin: float, tmax: float, max_seasons: int) -> np.ndarray:
    """
    Keep only midpoints within (tmin, tmax) like the snippet, then cap to max_seasons.

    The brute-force code effectively uses at most 11 midpoints (for 12 seasons) by indexing from the end.
    Since mid is generated in decreasing order, mid[-1]..mid[-11] are the 11 *smallest* midpoints.
    Equivalent, once sorted ascending: keep the earliest 11 midpoints (smallest times).

    Returns sorted ascending midpoints.
    """
    mid_in = mid[(mid > tmin) & (mid < tmax)]
    if mid_in.size == 0:
        return np.array([], dtype=float)

    mid_sorted = np.sort(mid_in)

    max_midpoints = max_seasons - 1
    if mid_sorted.size > max_midpoints:
        mid_sorted = mid_sorted[:max_midpoints]

    return mid_sorted


def assign_seasons_strict(JD: np.ndarray, mids_sorted: np.ndarray) -> np.ndarray:
    """
    Seasons are defined by strict inequalities:

      S=1: JD < mid0
      S=2: mid0 < JD < mid1
      ...
      S=k: mid_{k-2} < JD < mid_{k-1}
      S=k+1: JD > mid_{k-1}

    Points exactly equal to any midpoint are excluded (strict).
    """
    if mids_sorted.size == 0:
        return np.full(JD.shape, -1, dtype=int)

    # Exclude exact-boundary points (strict inequalities)
    mask = np.ones(JD.shape, dtype=bool)
    for m in mids_sorted:
        mask &= ~np.isclose(JD, m, rtol=0.0, atol=0.0)

    JD2 = JD[mask]
    # np.digitize: returns 0..len(mids), where 0 means < mids[0], len(mids) means > mids[-1]
    season2 = np.digitize(JD2, mids_sorted, right=False) + 1  # seasons numbered starting at 1

    season = np.full(JD.shape, -1, dtype=int)
    season[mask] = season2
    return season


def season_medians_with_gap_indices(
    mags: np.ndarray,
    season_idx: np.ndarray,
    *,
    min_points_per_season: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute median and MAD per season for seasons that have enough points.
    Return (indexes, meds, meds_err) where indexes are the *season numbers* (gap-preserving).
    """
    good = season_idx > 0
    if not np.any(good):
        return np.array([]), np.array([]), np.array([])

    idxs = []
    meds = []
    errs = []

    for s in np.unique(season_idx[good]):
        sel = (season_idx == s)
        m = mags[sel]
        if m.size >= min_points_per_season:
            idxs.append(int(s))
            meds.append(float(np.median(m)))
            errs.append(float(mad_std(m)))

    if len(idxs) == 0:
        return np.array([]), np.array([]), np.array([])

    # Ensure sorted by season number
    order = np.argsort(idxs)
    return np.asarray(idxs)[order], np.asarray(meds)[order], np.asarray(errs)[order]


def compute_trend_metrics(indexes: np.ndarray, meds: np.ndarray) -> tuple[float, float, float, float, float]:
    """
    Return:
      (lin_slope, quad_slope, coeff1, coeff2, max_diff)

    Preserves the snippet's naming/buggy algebra:
      coeffs = polyfit(x, y, 2) gives [a,b,c]
      quadratic_slope = a
      c1 = b
      c2 = c
      te = -c2/(2*a)   (bug preserved)
      me = c1-(c2^2)/(4*a) (bug preserved)
      m(x) = c1 + c2*x + a*x^2 (bug preserved)
    """
    # Linear slope
    lin = np.polyfit(indexes, meds, 1)
    lin_slope = float(lin[0])

    # Quadratic
    quad = np.polyfit(indexes, meds, 2)
    a = float(quad[0])
    c1 = float(quad[1])
    c2 = float(quad[2])

    # Fitted endpoints (preserved)
    x0 = float(indexes[0])
    x1 = float(indexes[-1])

    m0 = c1 + c2 * x0 + a * x0 * x0
    m1 = c1 + c2 * x1 + a * x1 * x1

    # Handle near-linear case safely
    if np.isclose(a, 0.0):
        diff = abs(m1 - m0)
        return lin_slope, a, c1, c2, float(diff)

    te = -c2 / (2.0 * a)
    me = c1 - (c2 * c2) / (4.0 * a)

    if (te > x0) and (te < x1):
        m1m0 = abs(m1 - m0)
        m1me = abs(m1 - me)
        m0me = abs(m0 - me)
        diff = max(m1m0, m1me, m0me)
    else:
        diff = abs(m1 - m0)

    return lin_slope, a, c1, c2, float(diff)


def process_one_file(
    lc_path: Path,
    id_df: pd.DataFrame,
    cfg: Config,
) -> dict | None:
    """
    Return one output row dict, or None if skipped.
    """
    target_str = lc_path.stem
    try:
        target = int(target_str)
    except ValueError:
        return None

    # Lookup row in index
    rows = id_df.loc[id_df["asas_sn_id"] == target]
    if rows.empty:
        return None
    row = rows.iloc[0]

    ra_val = float(row["ra_deg"])
    p_mag = float(row["pstarrs_g_mag"])

    df = read_lc_dat(lc_path)
    df = filter_lc(df, target)

    if df.empty:
        return None

    JD = df["JD"].to_numpy(dtype=float)
    mag = df["mag"].to_numpy(dtype=float)

    # Whole-LC stats
    lc_median = float(np.median(mag))
    lc_mad = float(mad_std(mag))
    lc_dispersion = float(np.ptp(mag))

    # Midpoints + seasons
    mid_all = seasonal_midpoints_from_ra(
        ra_val,
        ra_is_deg=cfg.ra_is_deg,
        dspring=cfg.dspring,
        n_midpoints=cfg.n_midpoints,
    )
    mids = choose_midpoints_in_range(mid_all, float(JD.min()), float(JD.max()), cfg.max_seasons)

    # Match snippet behavior: if only one midpoint survives, skip
    if mids.size == 1:
        return None
    # Also skip if <2 midpoints (more conservative, prevents degenerate seasonalization)
    if mids.size < 2:
        return None

    season_idx = assign_seasons_strict(JD, mids)
    indexes, meds, _meds_err = season_medians_with_gap_indices(
        mag, season_idx, min_points_per_season=cfg.min_points_per_season
    )

    # Need enough seasons to fit quadratic
    if meds.size < cfg.min_seasons_for_quadratic:
        return None

    lin_slope, quad_slope, c1, c2, diff = compute_trend_metrics(indexes.astype(float), meds.astype(float))

    return {
        "ASAS-SN ID": target,
        "Pstarss gmag": p_mag,
        "Median": lc_median,
        "Median_err": lc_mad,
        "Dispersion": lc_dispersion,
        "Slope": lin_slope,
        "Quad Slope": quad_slope,
        "coeff1": c1,
        "coeff2": c2,
        "max diff": diff,
    }


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    cfg = parse_args()

    all_rows: list[dict] = []

    # Directory indices
    dirs = list(range(cfg.dir_start, cfg.dir_end + 1))

    for x in dirs:
        t0 = time.time()
        print(f"Starting {cfg.mag_bin} directory {x}")

        index_path = cfg.root / cfg.mag_bin / f"index{x}.csv"
        lc_dir = cfg.root / cfg.mag_bin / f"lc{x}_cal"

        if not index_path.exists() or not lc_dir.exists():
            print(f"  Skipping {x}: missing {index_path} or {lc_dir}")
            continue

        id_df = read_index_csv(index_path)

        files = sorted([p for p in lc_dir.iterdir() if p.suffix == ".dat"])
        dir_rows: list[dict] = []

        for lc_path in tqdm(files, desc=f"Processing lc{x}_cal", unit="file"):
            row = process_one_file(lc_path, id_df, cfg)
            if row is None:
                continue
            dir_rows.append(row)

        all_rows.extend(dir_rows)

        if cfg.write_per_dir:
            per_dir_path = Path(cfg.mag_bin) / "new" / f"{x}.csv"
            write_csv_rows(per_dir_path, dir_rows)

        dt = time.time() - t0
        print(f"Ending {x} | Execution time (s): {dt:.2f}")

    # Combined output
    write_csv_rows(cfg.output, all_rows)

    # Also provide an Astropy Table version in-memory (mirrors the snippet's table idea)
    if all_rows:
        tbl = Table(rows=all_rows)
        # You can uncomment this if you want astropy's CSV writer instead:
        # tbl.write(cfg.output, format="csv", overwrite=True)


if __name__ == "__main__":
    main()
