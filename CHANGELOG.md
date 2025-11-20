# Changelog

## 21-11-2025
- Added SkyPatrol CSV defaults to `df_plot` and taught the plotting/residual helpers to auto-detect and load the SkyPatrol format so `SKYPATROL_CSV_PATHS` can be plotted directly.
- Extended `reproduce_candidates.py` so the `--candidates` flag can point at built-in lists (e.g., `skypatrol_lightcurve_files`) or files while keeping detection summaries resilient when no peaks are returned.
- Cleaned up the `README.md` dependency list to match the current pure-Python baseline implementations.

## 19-11-2025
-  switched from Boolean masks to binary search w/ `np.searchsorted` in `lc_baseline.rolling_time_median` and `lc_baseline.rolling_time_mad`

## 17-11-2025
- Added a CLI entry point directly to `df_filter`, eliminating the need for the separate `script_filter_naive`.
- Removed `script_filter_naive` to avoid duplicated logic.
- Added `build_manifest` to search for asas-sn ID's in `lc_cal` subdirs BEFORE doing reproduction check, so `reproduce_candidates` only needs to check against a csv file instead of searching through all `mag_bins` each time.
- Edited `reproduce_candidates` to match the `brayden_candidates` with `lc_manifest.csv` instead of iterating through all `mag_bins` each time. 
