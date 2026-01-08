# MALCA: Multi-timescale ASAS-SN Light Curve Analysis

## How to use
#### What files are expected
- Per-mag-bin directories: `/data/poohbah/1/assassin/rowan.90/lcsv2/<mag_bin>/`
  - Index CSVs: `index*.csv` with columns like `asas_sn_id, ra_deg, dec_deg, pm_ra, pm_dec, ...`
  - Light curves: `lc<num>_cal/` folders containing `<asas_sn_id>.dat2` (preferred) or `<asas_sn_id>.csv`
- Optional catalogs (only needed if you turn on those filters):
  - VSX: `input/vsx/vsx_cleaned.csv`
  - ASAS-SN bright-star index: `input/vsx/asassn_catalog.csv`

#### Typical run (large batches)
1) Build a manifest (map IDs -> light-curve directories):
   ```bash
   python malca/manifest.py --index-root /data/poohbah/1/assassin/rowan.90/lcsv2 --lc-root /data/poohbah/1/assassin/rowan.90/lcsv2 --mag-bin 13_13.5 --out /output/lc_manifest_13_13.5.parquet --workers 8
   ```
2) Pre-filter and run events in batches with resume support:
   ```bash
   python -m malca.filtered_events --mag-bin 13_13.5 --n-workers 20 --min-time-span 100 --min-points-per-day 0.05 --max-power 0.5 --min-cameras 2 --batch-size 2000 --lc-root /data/poohbah/1/assassin/rowan.90/lcsv2 --index-root /data/poohbah/1/assassin/rowan.90/lcsv2 -- --output /output/lc_events_results_13_13.5.csv --workers 20
   ```
   - The wrapper builds/loads the manifest, runs pre-filters from `malca/pre_filter.py`, then calls `malca/events.py` in batches.
   - Resume: if interrupted, it skips already processed paths using the `*_PROCESSED.txt` checkpoint next to the output.

3)  Post-filter events:
   ```bash
   python -m malca.post_filter --input /output/lc_events_results_13_13.5.csv --output /output/lc_events_results_13_13.5_filtered.csv
   ```

### Running pieces manually
- Build manifest only:
  `python malca/manifest.py --index-root <index_dir> --lc-root <lc_dir> --mag-bin 12_12.5 --out /output/lc_manifest.parquet`
- Pre-filter only (expects columns `asas_sn_id` and `path` pointing to lc_dir):
  `python -m malca.pre_filter --help`
- Events only:
  `python -m malca.events --input /path/to/lc*_cal/*.dat2 --output /output/results.parquet --workers 16`
- Post-filter only:
  `python -m malca.post_filter --input /output/results.parquet --output /output/results_filtered.parquet`
- Targeted reproduction of specific candidates (Bayesian):
  `python malca/reproduce_candidates.py --method bayes --manifest /output/lc_manifest.parquet --candidates my_targets.csv --out-dir /output/results_repro --out-format csv --n-workers 8`
- Batch reproduction helper (same idea, alternate entry point):
  `python malca/reproduction.py --method bayes --manifest /output/lc_manifest.parquet --candidates my_targets.csv --out-dir /output/results_repro --out-format csv --n-workers 8`
- Plot a single light curve with baseline/residuals:
  `python -m malca.plot --input /path/to/lc123.dat2 --output /output/plot.png`
- Batch plot Bayesian results:
  `python malca/plot_results_bayes.py --input /output/lc_events_results_13_13.5.csv --out-dir /output/plots`
- Plot metrics (multiple y vs p_points/mag_points; optional 3D scatter via matplotlib 3D):
  ```python
  import pandas as pd
  from malca.plot_metrics import plot_metrics_for_ids, plot_3d_surface

  df = pd.read_csv("/output/lc_events_results_13_13.5.csv")
  plot_metrics_for_ids(df, id_col="asas_sn_id", ids=["123", "456"], x_cols=("p_points", "mag_points"))
  plot_3d_surface(df, x_col="p_points", y_col="mag_points", z_col="dip_bd", color_col="elapsed_s")
  ```

### Notes
- Always generate the manifest first; everything else depends on knowing where each `<asas_sn_id>.dat2` lives.
- If you don’t have the VSX or ASAS-SN catalog files, leave those filters off (`apply_vsx=False`, `apply_bns=False`).


### Dependencies
- numpy, pandas, scipy, numba, astropy, tqdm, matplotlib
- celerite2
- pyarrow (Parquet support)
- duckdb (optional, only if using `--output-format duckdb`)
