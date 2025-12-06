# ASAS-SN Variability Tools

Pipeline to search for peaks and dips in ASAS-SN light curves


## Modules
- `calder/`
  - `lc_dips` - searches for dips 
    - `clean_lc`
      - removes saturated light curves
      - mask any NaN's in JD and mag, may be unnecessary?
      - sort light curves by JD, may be unnecessary ?
    - `empty_metrics`
      - creates empty dict for dip metrics for a single band
    - `process_record_naive`
      - optionally computes baseline of a single light curve
      - finds peaks in baseline residuals for a single light curve
      - computes dip metrics for a single light curve
    - `_biweight_gaussian_metrics`
    - `_compute_biweight_delta` 
    - `_compute_biweight_delta_peaks`
    - `process_record_biweight`
    - `naive_dip_finder`
    - `biweight_dip_finder`

  - `lc_peaks`
    - `fit_paczynski_peaks`
    - `_compute_biweight_delta_peaks`
    - `process_record_microlensing`
    - `microlensing_peak_finder`


## Dependencies
  - numpy
  - pandas
  - scipy
  - astropy
  - tqdm
  - pyarrow (optional; required for Parquet outputs)


