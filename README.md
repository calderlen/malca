# MALCA: Multi-timescale ASAS-SN Light Curve Analysis

Pipeline to search for peaks and dips in ASAS-SN light curves

## how to use
- `malca/manifest.py`  
  Build a CSV/Parquet mapping ASAS-SN IDs to light-curve paths:  
  `python malca/manifest.py --index-root <path_to_index_root> --lc-root <path_to_lc_root> --out ./lc_manifest.parquet [--mag-bin 12_12.5 ...]`

- `malca/pre_filter.py`  
  Run pre-filters on candidates before event detection (sparse, periodic, VSX, bright-nearby, multi-camera):  
  `python -m malca.pre_filter --help` (expects an input CSV with `asas_sn_id`/`path`; configure filters via flags).

- `malca/events.py`  
  Bayesian event detection over light curves:  
  `python malca/events.py --input <paths_or_glob> --output results.parquet --workers 8 [--mag-bins ...] [--trigger-mode logbf|posterior_prob]`

- `malca/post_filter.py`  
  Post-process detected events (rule-based pruning/aggregation):  
  `python -m malca.post_filter --input <events_csv_or_parquet> --output <filtered_csv>`

- `malca/reproduce_candidates.py`  
  Targeted reproduction over candidate lists (naive/biweight/Bayesian):  
  `python malca/reproduce_candidates.py --method bayes --manifest ./lc_manifest.parquet --out-dir ./results_repro --out-format csv [--candidates <file_or_builtin>] [--n-workers 8]`

- `malca/filter.py`  
  Apply legacy/aggregate filtering to peak/dip tables:  
  `python malca/filter.py <peaks_csv_or_dir> [--biweight] [--band g|v|both|either] [--latest-per-bin] [--output <csv>] [--output-dir <dir>]`

- `malca/fp_analysis.py`  
  Compare pre/post filter retention:  
  `python malca/fp_analysis.py --pre <pre_csv_or_dir> --post <post_csv_or_dir> [--id-col asas_sn_id]`

- VSX utilities  
  - `malca/vsx_crossmatch.py`: build/propagate VSX matches; see `python malca/vsx_crossmatch.py --help`.  
  - `malca/vsx_filter.py`: filter VSX classes; see `python malca/vsx_filter.py --help`.  
  - `malca/vsx_reproducibility.py`: crossmatch reproducibility checks; see `python malca/vsx_reproducibility.py --help`.

- Plotting  
  - `malca/plot_results_bayes.py`: batch plotting of Bayesian results; see `python malca/plot_results_bayes.py --help`.  
  - `malca/plot.py`: helpers for plotting individual light curves (import or run `python -m malca.plot` for ad hoc use).

## dependencies
- numpy
- pandas
- scipy
- numba
- astropy
- tqdm
- matplotlib
- celerite2
- pyarrow (required for parquet outputs)
- duckdb (optional; required for `--output-format duckdb`)
- jupyter / ipykernel (optional; for notebooks)
