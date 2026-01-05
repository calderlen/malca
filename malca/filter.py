"""
filter list


existing filters in old/lc_filter.py:
1. are there any dips/peaks
    - prob need to make it so that only V band is considered going forward to speed things up
2. multi camera filter
     - removes cadndiates that only detected on one camera
3. filter sparse lightcurves
    - remove candidates with less than 100 days of observation
    - remove candidates with less than 1 data point per 20 days on average
4. filter periodic candidates
    - remove candidates with strong periodicity
5. vsx crossmatch filter
6. filter bright nearby stars -- is filter.py being applied BEFORE or AFTER events.py? because filter.py is applied after events.py, then events.py needs to remove BNS. if filter.py is applied before events.py, then filter.py needs to remove BNS. need to confirm which it is.


new filters
7. posterior strength
    - require dip_bayes_factor or jump_bayes_factor > threshold
    - require max_log_bf_local finite
8. event probability
    - keep light curves with dip_max_event_prob or jump_max_event_prob > threshold
9. run robustness
    - require dip_run_count or jump_run_count >= 1
    - require dip_max_run_points/.jump_max_run_points >= run_min_points
10. morphology?
    - keep runs whose best morphology is 'gaussian' for dips or 'pacyznski' for jumps, with delta_bic_null >= threshold to reject noise-like runs
11. signal ampltiude
    - enforce a minimum best_mag_event offset relative to baseline_mag
12. enforce |best_mag_event - baseline_mag| > threshold, e.g. 0.05 - 0.1 mag
13. 
"""
