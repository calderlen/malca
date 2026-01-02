# MALCA: Multi-timescale ASAS-SN Light Curve Analysis

Pipeline to search for peaks and dips in ASAS-SN light curves

## how to use
- `scripts/lc_manifest.py`  
  Build a csv/parquet file mapping ASAS-SN IDs to LC paths:  
  `python scripts/lc_manifest.py --index-root <path_to_index_root> --lc-root <path_to_lc_root> --out ./lc_manifest.csv [--mag-bin 12_12.5 ...]`

- `malca/events.py`  
  Event detection orchestration (CLI) wrapping the Bayesian scoring.
- `malca/events_bayes.py`  
  Bayesian dip/peak search (writes CSV/Parquet), with optional baseline override from `malca/baseline.py`:  
  `python malca/events_bayes.py --mode dips|peaks --out-dir ./results_bayes --chunk-size 5000 [--mag-bin 12_12.5 ...]`  
  Example with a baseline: `--baseline-func baseline:per_camera_median_baseline`

- `scripts/reproduce_candidates.py`  
  Search for peaks/dips from a list of candidates (default: Brayden list):  
  `python scripts/reproduce_candidates.py --method biweight|naive --manifest ./lc_manifest.csv --out-dir ./results_repro --out-format csv [--candidates <file_or_builtin>] [--n-workers 8]`

- `malca/filter.py`  
  Filter the list of peaks/dips:  
  `python malca/filter.py <peaks_csv_or_dir> [--biweight] [--band g|v|both|either] [--latest-per-bin] [--output <csv>] [--output-dir <dir>]`

- `scripts/fp_analysis.py`  
  Summarize false-positive reduction (pre vs post filter retention):  
  `python scripts/fp_analysis.py --pre <pre_csv_or_dir> --post <post_csv_or_dir> [--id-col asas_sn_id]`

## dependencies
- numpy
- pandas
- scipy
- astropy
- tqdm
- celerite2
- pyarrow (optional; required for Parquet outputs)

## layout
```

├── docs
│   ├── ARCHITECTURE.md
│   ├── NOTES.md
│   └── TODO.md
├── environment.yml
├── .gitignore
├── pyproject.toml
├── README.md
├── output/                
├── malca
│   ├── baseline.py
│   ├── df_utils.py
│   ├── events.py
│   ├── events_bayes.py
│   ├── filter.py
│   ├── lc_utils.py
│   ├── julia
│   │   ├── baseline.jl
│   │   ├── df_utils.jl
│   │   ├── events.jl
│   │   ├── events_bayes.jl
│   │   └── lc_utils.jl
│   ├── old
│   │   ├── __init__.py
│   │   ├── lc_events_naive.py
│   │   ├── lc_events.py
│   │   └── lc_metrics.py
│   ├── plot.py
│   ├── stats.py
│   ├── test
│   │   ├── test.py
│   │   └── test_skypatrol.py
│   └── vsx
│       ├── crossmatch.py
│       ├── filter.py
│       └── reproducibility.py
├── scripts
│   ├── fp_analysis.py
│   ├── lc_manifest.py
│   ├── plot_results_bayes.py
│   ├── plot_results.py
│   └── reproduce_candidates.py
```
