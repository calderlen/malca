# MALCA: ASAS-SN light-curve filtering and event finding

## How to use
#### What files are expected
- Per-mag-bin directories: `/data/poohbah/1/assassin/rowan.90/lcsv2/<mag_bin>/`
  - Index CSVs: `index*.csv` with columns like `asas_sn_id, ra_deg, dec_deg, pm_ra, pm_dec, ...`
  - Light curves: `lc<num>_cal/` folders containing `<asas_sn_id>.dat2` (preferred) or `<asas_sn_id>.csv`
- Optional catalogs (only needed if you turn on those filters):
  - VSX: `results_crossmatch/vsx_cleaned_*.csv`
  - ASAS-SN bright-star index: `results_crossmatch/asassn_index_masked_concat_cleaned_*.csv`

#### Typical run (large batches)
1) Build a manifest (map IDs -> light-curve directories):
   ```bash
   python malca/manifest.py --index-root /data/poohbah/1/assassin/rowan.90/lcsv2 \
     --lc-root /data/poohbah/1/assassin/rowan.90/lcsv2 \
     --mag-bin 13_13.5 \
     --out lc_manifest_13_13.5.parquet
   ```
2) Pre-filter and run events in batches with resume support:
   ```bash
   python filtered_events.py --mag-bin 13_13.5 --n-workers 16 \
     --min-time-span 100 --min-points-per-day 0.05 --max-power 0.5 --min-cameras 2 \
     --batch-size 2000 --lc-root /data/poohbah/1/assassin/rowan.90/lcsv2 \
     --index-root /data/poohbah/1/assassin/rowan.90/lcsv2 \
     -- --output ./output/lc_events_results_13_13.5.csv --workers 32
   ```
   - The wrapper builds/loads the manifest, runs pre-filters from `malca/pre_filter.py`, then calls `malca/events.py` in batches.
   - Resume: if interrupted, it skips already processed paths using the `*_PROCESSED.txt` checkpoint next to the output.

3)  Post-filter events:
   ```bash
   python -m malca.post_filter --input ./output/lc_events_results_13_13.5.csv \
     --output ./output/lc_events_results_13_13.5_filtered.csv
   ```

### Running pieces manually
- Build manifest only:
  `python malca/manifest.py --index-root <index_dir> --lc-root <lc_dir> --mag-bin 12_12.5 --out lc_manifest.parquet`
- Pre-filter only (expects columns `asas_sn_id` and `path` pointing to lc_dir):
  `python -m malca.pre_filter --help`
- Events only:
  `python -m malca.events --input /path/to/lc*_cal/*.dat2 --output results.parquet --workers 16`
- Post-filter only:
  `python -m malca.post_filter --input results.parquet --output results_filtered.parquet`
- Targeted reproduction of specific candidates (Bayesian):
  `python malca/reproduce_candidates.py --method bayes --manifest lc_manifest.parquet --candidates my_targets.csv --out-dir results_repro --out-format csv --n-workers 8`
- Batch reproduction helper (same idea, alternate entry point):
  `python malca/reproduction.py --method bayes --manifest lc_manifest.parquet --candidates my_targets.csv --out-dir results_repro --out-format csv --n-workers 8`
- Plot a single light curve with baseline/residuals:
  `python -m malca.plot --input /path/to/lc123.dat2 --output plot.png`
- Batch plot Bayesian results:
  `python malca/plot_results_bayes.py --input ./output/lc_events_results_13_13.5.csv --out-dir ./plots`

### Notes
- Always generate the manifest first; everything else depends on knowing where each `<asas_sn_id>.dat2` lives.
- If you don’t have the VSX or ASAS-SN catalog files, leave those filters off (`apply_vsx=False`, `apply_bns=False`).


### Dependencies
- numpy, pandas, scipy, numba, astropy, tqdm, matplotlib
- celerite2
- pyarrow (Parquet support)
- duckdb (optional, only if using `--output-format duckdb`)
