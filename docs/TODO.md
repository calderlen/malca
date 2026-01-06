todo short-term
- ✅ filter_sparse_lightcurves assumes dat2 columns that are not present (COMPLETED - now computes from dat2)
- ✅ filter_periodic_lightcurves assumes dat2 columns that are not present (COMPLETED - now computes Lomb-Scargle from dat2)
- ✅ filter_multi_camera assumes columns in dat2 that are not actually there (COMPLETED - now counts cameras from dat2)
- ✅ filter_bright_nearby_stars needs to do implicit filter by crossmatching LCs with index (COMPLETED - now does catalog crossmatch)
- ✅ pre_filter refers to "vsx_match_sep_arcsec" and "vsx_class" that aren't output anywhere (COMPLETED - now does inline VSX crossmatch)
- ✅ parallelize pre_filter (COMPLETED - added n_workers parameter with ProcessPoolExecutor)
- See docs/PRE_FILTER_UPDATES.md for details on all changes

todo long-term
- obtain "recently calibrated" light curves for 15-15.5, 15.5-16 mag. dom's newly calibrated light curves only go down to 15 mag
- need to collect camera# for each peak detected
- crossmatch w/ VSX earlier in the pipeline so you can filter out known types earlier -- can also grab data such as period (when available) to reduce LSP calls; make sure code is aware when period is detected and when it isn't
- adapt camfilter and camfilter2 from more_utils?


