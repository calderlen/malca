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

    
"""
