# MALCA: Multi-timescale ASAS-SN Light Curve Analysis

## How to use
#### What files are expected
- Per-mag-bin directories: `/data/poohbah/1/assassin/rowan.90/lcsv2/<mag_bin>/`
  - Index CSVs: `index*.csv` with columns like `asas_sn_id, ra_deg, dec_deg, pm_ra, pm_dec, ...`
- Light curves: `lc<num>_cal/` folders containing `<asas_sn_id>.dat2`
- Optional catalogs:
  - VSX: `input/vsx/vsx_cleaned.csv` (pre-filtered to exclude known periodic/eclipsing/transient variables)
  - Note: Bright nearby star (BNS) filtering is handled upstream by ASAS-SN during LC generation

#### Typical run (large batches)
1) Build a manifest (map IDs -> light-curve directories):
   ```bash
   python malca/manifest.py --index-root /data/poohbah/1/assassin/rowan.90/lcsv2 --lc-root /data/poohbah/1/assassin/rowan.90/lcsv2 --mag-bin 13_13.5 --out /output/lc_manifest_13_13.5.parquet --workers 8
   ```
2) Pre-filter and run events in batches with resume support:
   ```bash
   python -m malca.events_filtered --mag-bin 13_13.5 --n-workers 20 --min-time-span 100 --min-points-per-day 0.05 --min-cameras 2 --vsx-catalog input/vsx/vsx_cleaned.csv --batch-size 2000 --lc-root /data/poohbah/1/assassin/rowan.90/lcsv2 --index-root /data/poohbah/1/assassin/rowan.90/lcsv2 -- --output /output/lc_events_results_13_13.5.csv --workers 20
   ```
   - The wrapper builds/loads the manifest, runs pre-filters from `malca/pre_filter.py`, then calls `malca/events.py` in batches.
   - Pre-filters: sparse LC removal, multi-camera requirement, VSX known variable rejection
   - Resume: if interrupted, it skips already processed paths using the `*_PROCESSED.txt` checkpoint next to the output.
   - To disable VSX filter: add `--skip-vsx`

3) Post-filter events (expensive validations on candidates only):
   ```bash
   python -m malca.post_filter --input /output/lc_events_results_13_13.5.csv --output /output/lc_events_results_13_13.5_filtered.csv
   ```
   - Post-filters: periodicity (LSP), Gaia RUWE, periodic catalog crossmatch

## Pipeline Architecture


1. **Pre-filtering**:
   - Sparse LC removal
   - Multi-camera requirement
   - VSX crossmatchg

2. **Event detection**:
   - Run `events.py`
   - Fit dip models, compute Bayes factors, extract metrics

3. **Post-filtering**:
   - Periodicity: Bootstrap Lomb-Scargle periodogram
   - Gaia RUWE: Query Gaia catalog for binary contamination
   - Periodic catalog: Crossmatch to known periodic variables

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
- Injection-recovery testing (validate pipeline completeness/contamination):
  ```python
  from malca.injection_recovery import (
      select_control_sample,
      run_injection_recovery,
      compute_detection_efficiency,
      plot_detection_efficiency
  )
  import pandas as pd
  import numpy as np

  # Load manifest and select 10,000 clean control LCs
  manifest = pd.read_parquet("/output/lc_manifest_13_13.5.parquet")
  control_sample = select_control_sample(manifest, n_sample=10000)

  # Define detection function (your pipeline)
  def my_detection_func(df_lc):
      # Run events.py logic on single LC
      # Return True if dip detected, False otherwise
      pass

  # Run injection-recovery on amplitude × duration grid
  results = run_injection_recovery(
      control_sample,
      my_detection_func,
      amplitude_grid=np.linspace(0.1, 2.0, 100),  # 100 amplitude values
      duration_grid=np.logspace(0.5, 2.5, 100),   # 100 duration values (log scale)
      n_injections_per_grid=100                    # 100 trials per grid point
  )

  # Compute and plot detection efficiency
  amp_centers, dur_centers, efficiency_grid = compute_detection_efficiency(results)
  plot_detection_efficiency(amp_centers, dur_centers, efficiency_grid)
  ```
  - Injects synthetic dips with skew-normal profiles and realistic noise
  - Measures completeness as function of dip amplitude and duration
  - Generates heatmap like ZTF dipper paper Figure 5

### Notes
- Always generate the manifest first; everything else depends on knowing where each `<asas_sn_id>.dat2` lives.
- VSX filter is enabled by default in `events_filtered.py` - requires `input/vsx/vsx_cleaned.csv` catalog
  - To disable: add `--skip-vsx` flag
  - The catalog is pre-filtered to exclude known periodic/eclipsing/transient variables
- Bright nearby star (BNS) filtering is NOT needed - ASAS-SN already filtered this during LC generation
- Pre-filters (run on all sources before event detection):
  - Sparse LC removal: reject LCs with insufficient time span or cadence
  - Multi-camera requirement: require observations from ≥2 cameras
  - VSX crossmatch: reject known variables within 3 arcsec
- Post-filters (run on detected candidates only):
  - Periodicity validation: bootstrap Lomb-Scargle periodogram with significance testing
  - Gaia RUWE check: flag potential binary contamination (RUWE > 1.4)
  - Periodic catalog crossmatch: check against known periodic variable catalogs


### Dependencies
- numpy, pandas, scipy, numba, astropy, tqdm, matplotlib
- celerite2
- pyarrow (Parquet support)
- duckdb (optional, only if using `--output-format duckdb`)
