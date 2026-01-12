
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
- `malca/score.py`
  - event scoring metric for both dips (Gaussian fits) and microlensing (Paczyński curves)
  - computes scores based on event amplitude, width, detections, and fit chi2






