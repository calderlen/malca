
## MALCA (Multi-timescale ASAS-SN Light Curve Analysis)

## Modules (Python)
- `malca/events.py`
  - orchestration and CLI for Bayesian event (dip/jump) scoring
  - Bayesian event scoring internals (formerly excursions_bayes); uses `malca/baseline.py`, `malca/utils.py`
- `malca/baseline.py`
  - baseline computation: global/rolling mean/median, per-camera variants, GP baseline
- `malca/utils.py`
  - light-curve I/O and parsing helpers, basic cleaning, JD/year conversions
- `malca/plot.py`
  - plotting/light-curve helpers
- `malca/stats.py`
  - statistical helpers
- `malca/filter.py`
  - post-processing/filtering of detected events
- `malca/pre_filter.py`
- `malca/post_filter.py`
- `malca/manifest.py`
- `malca/ltv.py`
- `malca/fp_analysis.py`
- `malca/plot_results_bayes.py`
- `malca/reproduce_candidates.py`
- `malca/reproduction.py`
- `malca/vsx_crossmatch.py`
- `malca/vsx_filter.py`
- `malca/vsx_reproducibility.py`
- `malca/filtered_events.py`
  - wrapper script: build/load manifest, run pre-filters, and call `malca.events` in batches with resume support
- `malca/plot_metrics.py`
  - plotting helpers for light-curve metrics (2D multi-metric plots and simple 3D scatter)
- `malca/dipper_score.py`
  - dipper scoring metric based on dip depth, width, detections, and Gaussian fit chi2

## Legacy Python modules (`malca/old/`)
- `df_utils.py`
  - legacy peak search utilities and metrics helpers
- `falsepositives.py`
- `lc_events.py`
- `lc_events_naive.py`
- `lc_filter.py`
- `lc_metrics.py`
- `plot_results.py`
- `stats.py`
- `test_skypatrol.py`

## Modules (Julia)
- `malca/julia/baseline.jl`
- `malca/julia/df_utils.jl`
- `malca/julia/events.jl`
- `malca/julia/lc_utils.jl`
    - since the ASAS-SN light curves are in two bands, V and G which have offsets, we analyze them separately. this function looks at only one of the two bands of the light curve at a time, converts the mag to delta, and finds dips/peaks a minimum distance of 50 data points apart (~1 data point per day), scores the light curve band and computes other metrics
  - `prefix_metrics`
  - `lc_proc`
    - intakes a single light curve's g and v bands
    - finds the first and last jd
    - outputs a dict summarizing information about the light curve
  - `event_finder`
    - orchestrates parallel processing of light curves (record prep, worker submission, peak/dip analysis via `lc_proc`, and buffered writing of results) test
