import os
import re
from glob import glob

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import mad_std, sigma_clip
from tqdm import tqdm

colors = ["#6b8bcd", "#b3b540", "#8f62ca", "#5eb550", "#c75d9c", "#4bb092", "#c5562f", "#6c7f39",
              "#ce5761", "#c68c45", '#b5b246', '#d77fcc', '#7362cf', '#ce443f', '#3fc1bf', '#cda735',
              '#a1b055']


def gaussian(t, amp, t0, sigma, baseline):
    """
    Gaussian kernel + baseline term (shared between events and dipper scoring).
    """
    return baseline + amp * np.exp(-0.5 * ((t - t0) / sigma) ** 2)


def get_id_col(df: pd.DataFrame) -> str:
    """Find the ID column in a dataframe."""
    for candidate in ["asas_sn_id", "id", "source_id", "path"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("No ID column found. Expected one of: asas_sn_id, id, source_id, path")


def clean_lc(df, max_error_absolute=1.0, max_error_sigma=5.0):
    base_mask = np.ones(len(df), dtype=bool)
    base_mask &= (df["saturated"] == 0)
    base_mask &= df["JD"].notna() & df["mag"].notna()
    base_mask &= df["error"].notna() & (df["error"] > 0.0)

    mask = base_mask.copy()
    if base_mask.sum() > 0:
        errors = df.loc[base_mask, "error"].values
        mask &= (df["error"] < max_error_absolute)

        clipped = sigma_clip(
            errors,
            sigma=max_error_sigma,
            cenfunc="median",
            stdfunc=mad_std,
        )
        clipped_mask = np.asarray(clipped.mask)
        if clipped_mask.shape == errors.shape:
            # Create a full-length mask for clipped errors
            clipped_full = np.zeros(len(df), dtype=bool)
            clipped_full[base_mask] = clipped_mask
            mask &= ~clipped_full

    df = df.loc[mask]
    df = df.sort_values("JD").reset_index(drop=True)
    return df


def compute_time_stats(df_lc: pd.DataFrame) -> dict:
    """Compute time span and cadence stats from a light curve DataFrame."""
    if df_lc.empty:
        return {"time_span_days": 0.0, "points_per_day": 0.0}

    jd = df_lc["JD"].values
    jd = jd[np.isfinite(jd)]
    if len(jd) < 2:
        return {"time_span_days": 0.0, "points_per_day": 0.0}

    time_span_days = float(jd.max() - jd.min())
    points_per_day = len(jd) / time_span_days if time_span_days > 0 else 0.0

    return {
        "time_span_days": time_span_days,
        "points_per_day": points_per_day,
    }


def compute_n_cameras(df_lc: pd.DataFrame) -> int:
    """Count unique cameras from a light curve DataFrame."""
    if df_lc.empty:
        return 0
    cameras = df_lc["camera#"].dropna().unique()
    return int(len(cameras))


def year_to_jd(year):
    jd_epoch = 2449718.5
    year_epoch = 1995
    days_in_year = 365.25

    return (year - year_epoch) * days_in_year + (jd_epoch - 2450000.0)


def jd_to_year(jd):
    jd_epoch = 2449718.5
    year_epoch = 1995
    days_in_year = 365.25

    return year_epoch + ((jd + 2450000.0) - jd_epoch) / days_in_year


def read_lc_dat2(asassn_id, path):

    dat2_path = os.path.join(path, f"{asassn_id}.dat2")
    if os.path.exists(dat2_path):
        file = dat2_path
                      
        columns = ["JD",
                   "mag",
                   "error",
                   "good_bad",
                   "camera#",
                   "v_g_band",
                   "saturated",
                   "cam_field"]

        # Use whitespace delimiter instead of fixed-width to handle
        # variable-width JD values (4-digit vs 5-digit integer parts).
        # read_fwf infers column widths from early rows, which fails when
        # JD transitions from 9999 to 10000+ (leading digit gets truncated).
        df = pd.read_csv(
            file,
            header=None,
            names=columns,
            sep=r'\s+',
        )
    
                                               
        df[["camera_name", "field"]] = df["cam_field"].str.split("/", expand=True)

                                      
        df = df.drop(columns=["cam_field"])

                        
        df = df.astype({
            "JD": "float64",
            "mag": "float64",
            "error": "float64",
            "good_bad": "int64",
            "camera#": "int64",
            "v_g_band": "int64",
            "saturated": "int64",
            "camera_name": "string",
            "field": "string"
        })

        df_g = df.loc[df["v_g_band"] == 0].reset_index(drop=True)    
        df_v = df.loc[df["v_g_band"] == 1].reset_index(drop=True)

        return df_g, df_v

    raise FileNotFoundError(
        f"Light curve file not found: {dat2_path}"
    )

def read_lc_csv(asassn_id, path):
    csv_path = os.path.join(path, f"{asassn_id}.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(csv_path)

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df["JD"] = df["jd"] + 2450000.0

    df_g = df[df["phot_filter"] == "g"].copy().reset_index(drop=True)
    df_v = df[df["phot_filter"] == "V"].copy().reset_index(drop=True)

    return df_g, df_v


def read_lc_raw(asassn_id, path):
    raw_path = os.path.join(path, f"{asassn_id}.raw")
    if not os.path.exists(raw_path):
        return pd.DataFrame()
    columns = [
        "camera#",
        "median",
        "sig1_low",
        "sig1_high",
        "p90_low",
        "p90_high",
    ]
    df = pd.read_csv(
        raw_path,
        delim_whitespace=True,
        header=None,
        names=columns,
        dtype={
            "camera#": "int64",
            "median": "float64",
            "sig1_low": "float64",
            "sig1_high": "float64",
            "p90_low": "float64",
            "p90_high": "float64",
        },
    )
    return df


def match_index_to_lc(
    index_path: str = "/data/poohbah/1/assassin/lenhart/code/calder/lcsv2_masked/",
    lc_path:    str = "/data/poohbah/1/assassin/rowan.90/lcsv2",
    mag_bins:   list = ['12_12.5','12.5_13','13_13.5','13.5_14','14_14.5','14.5_15'],
    id_column:  str = "asas_sn_id",
):
    """
    Generator function that iterates over index*_masked.csv files in lcsv2_masked/<mag_bin>/, find corresponding lc<num>_cal/ directories in lcsv2/<mag_bin>/, and yield one record per asas_sn_id with whether its .dat file exists. Outputs a dict
    """

    idx_pattern = re.compile(r"index(\d+)_masked\.csv$", re.IGNORECASE)

    for mag_bin in tqdm(mag_bins, desc="Bins", unit="bin"):
        idx_paths = sorted(glob(os.path.join(index_path, mag_bin, "index*_masked.csv")))
        for idx_csv in tqdm(idx_paths, desc=f"{mag_bin} index CSVs", leave=False):

                                                        
            idx_num = int(idx_pattern.search(os.path.basename(idx_csv)).group(1))

            lc_dir = os.path.join(lc_path, mag_bin, f"lc{idx_num}_cal")

            ids = (
                pd.read_csv(idx_csv, dtype={id_column: "string"})[id_column]
                .dropna()
                .astype(str)
                .unique()
            )

            for asn in ids:
                dat_path = os.path.join(lc_dir, f"{asn}.dat")
                found = os.path.exists(dat_path)
                yield {
                    "mag_bin":      mag_bin,
                    "index_num":    idx_num,
                    "index_csv":    idx_csv,
                    "lc_dir":       lc_dir,
                    "asas_sn_id":   asn,
                    "dat_path":     dat_path if found else None,
                    "found":        found,
                }


def custom_id(ra_val,dec_val):
    """
    ADOPTED FROM BRAYDEN JOHANTGEN'S CODE: https://github.com/johantgen13/Dippers_Project.git
    """
    c = SkyCoord(ra=ra_val*u.degree, dec=dec_val*u.degree, frame='icrs')
    ra_num = c.ra.hms
    dec_num = c.dec.dms

    if int(dec_num[0]) < 0:
        cust_id = 'J'+str(int(c.ra.hms[0])).rjust(2,'0')+str(int(c.ra.hms[1])).rjust(2,'0')+str(int(round(c.ra.hms[2]))).rjust(2,'0')+'$-$'+str(int(c.dec.dms[0])*(-1)).rjust(2,'0')+str(int(c.dec.dms[1])*(-1)).rjust(2,'0')+str(int(round(c.dec.dms[2])*(-1))).rjust(2,'0')
    else:
        cust_id = 'J'+str(int(c.ra.hms[0])).rjust(2,'0')+str(int(c.ra.hms[1])).rjust(2,'0')+str(int(round(c.ra.hms[2]))).rjust(2,'0')+'$+$'+str(int(c.dec.dms[0])).rjust(2,'0')+str(int(c.dec.dms[1])).rjust(2,'0')+str(int(round(c.dec.dms[2]))).rjust(2,'0')

    return cust_id


def plotparams(ax, labelsize=15):
    """
    ADAPTED FROM BRAYDEN JOHANTGEN'S CODE: https://github.com/johantgen13/Dippers_Project.git
    """

    ax.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(direction='in', which='both', labelsize=labelsize)
    ax.tick_params('both', length=8, width=1.8, which='major')
    ax.tick_params('both', length=4, width=1, which='minor')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    return ax



def divide_cameras():
    """
    ADAPTED FROM BRAYDEN JOHANTGEN'S CODE: https://github.com/johantgen13/Dippers_Project.git
    """
    pass
