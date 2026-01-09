# MALCA: Multi-timescale ASAS-SN Light Curve Analysis

## How to use
#### What files are expected
- Per-mag-bin directories: `/data/poohbah/1/assassin/rowan.90/lcsv2/<mag_bin>/`
  - Index CSVs: `index*.csv` with columns like `asas_sn_id, ra_deg, dec_deg, pm_ra, pm_dec, ...`
- Light curves: `lc<num>_cal/` folders containing `<asas_sn_id>.dat2`
- Optional catalogs:
  - VSX: `input/vsx/vsx_cleaned.csv` (pre-filtered to exclude known periodic/eclipsing/transient variables)
  - Note: Bright nearby star (BNS) filtering is handled upstream by ASAS-SN during LC generation

#### Dependencies
- Core pipeline: numpy, pandas, scipy, numba, astropy, celerite2, matplotlib, tqdm
- Optional outputs: pyarrow (parquet), duckdb
- Notebooks/EDA: jupyterlab, ipykernel, seaborn, scikit-learn, joblib

#### Typical run (large batches)
1) Build a manifest (map IDs -> light-curve directories):
   ```bash
   python malca/manifest.py --index-root /data/poohbah/1/assassin/rowan.90/lcsv2 --lc-root /data/poohbah/1/assassin/rowan.90/lcsv2 --mag-bin 13_13.5 --out /output/lc_manifest_13_13.5.parquet --workers 10
   ```
2) Pre-filter and run events in batches with resume support:
   ```bash
   python -m malca.events_filtered --mag-bin 13_13.5 --workers 10 --min-time-span 100 --min-points-per-day 0.05 --min-cameras 2 --vsx-catalog input/vsx/vsx_cleaned.csv --batch-size 2000 --lc-root /data/poohbah/1/assassin/rowan.90/lcsv2 --index-root /data/poohbah/1/assassin/rowan.90/lcsv2 -- --output /output/lc_events_results_13_13.5.csv --workers 10
   ```
   - The wrapper builds/loads the manifest, runs pre-filters from `malca/pre_filter.py`, then calls `malca/events.py` in batches.
   - Pre-filters: sparse LC removal, multi-camera requirement, VSX known variable rejection
   - Resume: if interrupted, it skips already processed paths using the `*_PROCESSED.txt` checkpoint next to the output.
   - To disable VSX filter: add `--skip-vsx`

3) Post-filter events (strict quality cuts on candidates only):
   ```bash
   python -m malca.post_filter --input /output/lc_events_results_13_13.5.csv --output /output/lc_events_results_13_13.5_filtered.csv

   # With custom thresholds
   python -m malca.post_filter --input results.csv --output filtered.csv --min-bayes-factor 20 --min-event-prob 0.7 --apply-morphology
   ```
   - **Implemented filters**: posterior strength (Bayes factors), event probability, run robustness, morphology
   - **Placeholder filters** (not yet implemented): periodicity (LSP), Gaia RUWE, periodic catalog crossmatch

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
  `python -m malca.events --input /path/to/lc*_cal/*.dat2 --output /output/results.parquet --workers 10`
  - Default Bayesian grid is 12x12 (12 p-grid points × 12 mag-grid points). Change p-grid with `--p-points`.
- Post-filter only:
  `python -m malca.post_filter --input /output/results.parquet --output /output/results_filtered.parquet`
- Targeted reproduction of specific candidates (Bayesian):
  `python malca/reproduce_candidates.py --method bayes --manifest /output/lc_manifest.parquet --candidates my_targets.csv --out-dir /output/results_repro --out-format csv --workers 10`
- Batch reproduction helper (same idea, alternate entry point):
  `python malca/reproduction.py --method bayes --manifest /output/lc_manifest.parquet --candidates my_targets.csv --out-dir /output/results_repro --out-format csv --workers 10`
- Plot light curves with baseline/residuals:
  ```bash
  # Single file
  python -m malca.plot --input /path/to/lc123.dat2 --out-dir /output/plots --format png

  # Multiple files (glob patterns supported)
  python -m malca.plot --input input/skypatrol2/*.csv --out-dir /output/plots --skip-events

  # All files from events.py results
  python -m malca.plot --events /output/lc_events_results_13_13.5_filtered.csv --out-dir /output/plots
  ```
- Batch plot Bayesian results (SkyPatrol CSVs, filtered by events output):
  `python -m malca.plot_results_bayes /path/to/*-light-curves.csv --results-csv /output/lc_events_results_13_13.5.csv --out-dir /output/plots`
- Event scoring (dips or microlensing):
  ```bash
  # Score dip events (default)
  python -m malca.score --events /output/lc_events_results_13_13.5.csv --output /output/dipper_scores.csv --event-type dip

  # Score microlensing events (Paczyński curves)
  python -m malca.score --events /output/lc_events_results_13_13.5.csv --output /output/microlens_scores.csv --event-type microlensing
  ```
- Plot metrics (multiple y vs p_points/mag_points; optional 3D scatter via matplotlib 3D):
  ```python
  import pandas as pd
  from malca.plot_metrics import plot_metrics_for_ids, plot_3d_surface

  df = pd.read_csv("/output/lc_events_results_13_13.5.csv")
  plot_metrics_for_ids(df, id_col="asas_sn_id", ids=["123", "456"], x_cols=("p_points", "mag_points"))
  plot_3d_surface(df, x_col="p_points", y_col="mag_points", z_col="dip_bd", color_col="elapsed_s")
  ```
- Injection-recovery testing (validate pipeline completeness/contamination):
  `python -m malca.injection_recovery --manifest /output/lc_manifest_13_13.5.parquet --out /output/injection_recovery_13_13.5.csv --workers 10`
  ```python
  from malca.injection_recovery import (
      select_control_sample,
      run_injection_recovery,
      compute_detection_efficiency,
      plot_detection_efficiency,
  )
  import pandas as pd
  import numpy as np

  manifest = pd.read_parquet("/output/lc_manifest_13_13.5.parquet")
  control_sample = select_control_sample(manifest, n_sample=10000)

  results = run_injection_recovery(
      control_sample,
      detection_kwargs={},  # uses events.py defaults
      amplitude_grid=np.linspace(0.1, 2.0, 100),
      duration_grid=np.logspace(0.5, 2.5, 100),
      n_injections_per_grid=100,
      workers=10,
  )

  # Compute and plot detection efficiency
  amp_centers, dur_centers, efficiency_grid = compute_detection_efficiency(results)
  plot_detection_efficiency(amp_centers, dur_centers, efficiency_grid)
  ```
  - Injects synthetic dips with skew-normal profiles and realistic noise
  - Measures completeness as function of dip amplitude and duration
  - Generates heatmap like ZTF dipper paper Figure 5
- Seasonal trend summary (LTVar-style):
  `python -m malca.ltv --mag-bin 13_13.5 --output /output/ltv_13_13.5.csv --workers 10`
- Quick stats for a single LC file:
  `python -m malca.stats /path/to/lc123.dat2`
- False-positive reduction summary (pre vs post filter):
  `python -m malca.fp_analysis --pre /output/pre.csv --post /output/post.csv`

### CLI modules
- `malca.manifest`: `python -m malca.manifest --index-root /data/poohbah/1/assassin/rowan.90/lcsv2 --lc-root /data/poohbah/1/assassin/rowan.90/lcsv2 --mag-bin 13_13.5 --out /output/lc_manifest_13_13.5.parquet --workers 10`
- `malca.events_filtered`: `python -m malca.events_filtered --mag-bin 13_13.5 --workers 10 --min-time-span 100 --min-points-per-day 0.05 --min-cameras 2 --vsx-catalog input/vsx/vsx_cleaned.csv --batch-size 2000 --lc-root /data/poohbah/1/assassin/rowan.90/lcsv2 --index-root /data/poohbah/1/assassin/rowan.90/lcsv2 -- --output /output/lc_events_results_13_13.5.csv --workers 10`
- `malca.events`: `python -m malca.events --input /path/to/lc*_cal/*.dat2 --output /output/results.csv --workers 10`
- `malca.post_filter`: `python -m malca.post_filter --input /output/results.csv --output /output/results_filtered.csv`
- `malca.plot`: `python -m malca.plot --input /path/to/lc123.dat2 --out-dir /output/plots --format png`
- `malca.score`: `python -m malca.score --events /output/results.csv --output /output/dipper_scores.csv --event-type dip`
- `malca.injection_recovery`: `python -m malca.injection_recovery --manifest /output/lc_manifest_13_13.5.parquet --out /output/injection_recovery_13_13.5.csv --workers 10`
- `malca.ltv`: `python -m malca.ltv --mag-bin 13_13.5 --output /output/ltv_13_13.5.csv --workers 10`
- `malca.stats`: `python -m malca.stats /path/to/lc123.dat2`
- `malca.fp_analysis`: `python -m malca.fp_analysis --pre /output/pre.csv --post /output/post.csv`
- `malca.reproduction`: `python -m malca.reproduction --method bayes --manifest /output/lc_manifest.parquet --candidates /output/targets.csv --out-dir /output/results_repro --out-format csv --workers 10`
- `malca.vsx_filter`: `python -m malca.vsx_filter`
- `malca.vsx_crossmatch`: `python -m malca.vsx_crossmatch`
- `malca.vsx_reproducibility`: `python -m malca.vsx_reproducibility`
- `malca.skypatrol_explore`: `python -m malca.skypatrol_explore`

#### Legacy/old scripts
- `malca.old.plot_results_bayes`: `python -m malca.old.plot_results_bayes /path/to/*-light-curves.csv --results-csv /output/results.csv --out-dir /output/plots`
- `malca.old.plot_results`: `python -m malca.old.plot_results /output/peaks.csv --csv-dir input/skypatrol2 --out-dir /output/plots`
- `malca.old.lc_events`: `python -m malca.old.lc_events --mode dips --mag-bin 13_13.5 --out-dir /output/lc_events_old --format csv --workers 10`
- `malca.old.lc_filter`: `python -m malca.old.lc_filter /output/peaks.csv --output /output/peaks_filtered.csv`
- `malca.old.falsepositives`: `python -m malca.old.falsepositives --pre /output/pre.csv --post /output/post.csv`
- `malca.old.stats`: `python -m malca.old.stats /path/to/lc123.dat2`
- `malca.old.test_skypatrol`: `python -m malca.old.test_skypatrol input/skypatrol2/*.csv --out /output/test_results.csv`

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
