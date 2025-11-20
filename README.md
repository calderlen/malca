# ASAS-SN Variability Tools

Collection of utilities, scripts, and notebooks I built to automate the ASAS-SN dipper/variable search pipeline.  Modules under `calder/` implement the light-curve peak search, candidate filtering, manifest generation, and reproduction harnesses used to validate survey data products.

Follow changes in the [`CHANGELOG.md`](CHANGELOG.md) 

## Modules

- `calder/`
  - `lc_dips.py` – peak-search engine (`naive_dip_finder`) that walks the ASAS-SN light-curve archive and writes `peaks_<mag_bin>_<timestamp>.{csv,parquet}`.
  - `df_filter.py` – end-to-end filtering pipeline (`filter_csv`) plus individual filter helpers (dip fraction, multi-camera, VSX class enrichment, periodicity checks, etc.).
  - `reproduce_candidates.py` – targeted reproduction harness that reads `lc_manifest` outputs and verifies historical candidates.
  - `build_manifest.py` – utility that maps `source_id → (mag_bin, index_csv, lc_dir, dat_path)` so reproduction runs can skip directory scans.
  - `script_search_naive.py` – thin CLI wrapper around `naive_dip_finder` for full-bin searches.
  - Other helpers (`df_utils.py`, `lc_baseline.py`, `lc_metrics.py`, `vsx_crossmatch.py`, etc.) that provide baselines, metrics, plotting, and catalog crossmatching

## Dependencies
- numpy, pandas, scipy, astropy
- tqdm
- pyarrow
