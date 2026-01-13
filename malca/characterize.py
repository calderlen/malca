"""
Multi-wavelength characterization for ASAS-SN dipper candidates.

This module consolidates:
- Gaia DR3 querying (astrometry, astrophysics, 2MASS/WISE photometry)
- StarHorse local catalog join (stellar ages, masses)
- 3D dust extinction via dustmaps3d (Wang et al. 2025)
- YSO classification (Koenig & Leisawitz 2014)
- Galactic population classification

Usage:
    python -m malca.characterize --input output/events.csv --output output/characterized.csv --dust
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress astropy warnings
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)


# =============================================================================
# GAIA DR3 QUERYING
# =============================================================================

def query_gaia_by_ids(source_ids: list[str | int], chunk_size: int = 1000, cache_file: str | None = None) -> pd.DataFrame:
    """
    Query Gaia DR3 for a list of Source IDs.
    
    Retrieves astrometry, astrophysics, and 2MASS/WISE photometry via ADQL joins.
    """
    from astroquery.gaia import Gaia
    
    cached_df = pd.DataFrame()
    ids_to_query = [str(x) for x in source_ids if str(x).isdigit()]
    
    if cache_file and Path(cache_file).exists():
        print(f"Loading Gaia cache from {cache_file}...")
        cached_df = pd.read_parquet(cache_file)
        if "source_id" in cached_df.columns:
            cached_df["source_id"] = cached_df["source_id"].astype(str)
            processed_ids = set(cached_df["source_id"])
            ids_to_query = [x for x in ids_to_query if x not in processed_ids]
            if ids_to_query:
                print(f"Use {len(processed_ids)} cached sources. Querying {len(ids_to_query)} new sources.")
            else:
                return cached_df
    
    if not ids_to_query:
        return cached_df

    results = []
    
    for i in tqdm(range(0, len(ids_to_query), chunk_size), desc="Querying Gaia DR3"):
        chunk_ids = ids_to_query[i : i + chunk_size]
        ids_str = ",".join(chunk_ids)
        
        query = f"""
        SELECT
            g.source_id,
            g.ra, g.dec,
            g.parallax, g.parallax_error, g.ruwe,
            g.pmra, g.pmdec,
            g.phot_g_mean_mag, g.bp_rp,
            g.teff_gspphot, g.logg_gspphot, g.mh_gspphot,
            g.distance_gspphot, g.ag_gspphot,
            
            xm_tm.original_ext_source_id AS tmass_id,
            tm.j_m AS tmass_j, tm.h_m AS tmass_h, tm.ks_m AS tmass_k,
            tm.j_msigcom AS tmass_j_err, tm.h_msigcom AS tmass_h_err, tm.ks_msigcom AS tmass_k_err,
            
            xm_aw.original_ext_source_id AS allwise_id,
            aw.w1mpro AS unwise_w1, aw.w2mpro AS unwise_w2,
            aw.w1sigmpro AS unwise_w1_err, aw.w2sigmpro AS unwise_w2_err
            
        FROM gaiadr3.gaia_source AS g
        
        LEFT JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xm_tm
            ON g.source_id = xm_tm.source_id
        LEFT JOIN external.tmass_psc AS tm
            ON xm_tm.original_ext_source_id = tm.designation
            
        LEFT JOIN gaiadr3.allwise_best_neighbour AS xm_aw
            ON g.source_id = xm_aw.source_id
        LEFT JOIN external.allwise AS aw
            ON xm_aw.original_ext_source_id = aw.designation
            
        WHERE g.source_id IN ({ids_str})
        """
        
        try:
            job = Gaia.launch_job_async(query)
            chunk_df = job.get_results().to_pandas()
            results.append(chunk_df)
        except Exception as e:
            print(f"Error querying Gaia chunk {i}: {e}")
            continue
            
    new_results = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    if not new_results.empty and "source_id" in new_results.columns:
        new_results["source_id"] = new_results["source_id"].astype(str)
        
    full_df = pd.concat([cached_df, new_results], ignore_index=True) if not new_results.empty else cached_df
    
    if cache_file and not new_results.empty:
        full_df.to_parquet(cache_file, index=False)
        
    return full_df


# =============================================================================
# STARHORSE LOCAL CATALOG
# =============================================================================

def query_starhorse_by_ids(source_ids: list[str | int], starhorse_file: str | Path | None = None, use_tap: bool = True) -> pd.DataFrame:
    """
    Retrieve StarHorse 2021 stellar parameters (Anders et al.).
    
    **Recommended**: TAP queries (default, use_tap=True)
    - Queries gaia.aip.de TAP service remotely
    - No large download required
    - Returns age, mass, distance, extinction
    
    **Alternative**: Local catalog join (use_tap=False)
    - Requires downloading ~100GB catalog from https://cdsarc.cds.unistra.fr/viz-bin/cat/I/354
    - Faster for repeated queries on same dataset
    """
    if use_tap:
        # TAP query via pyvo
        try:
            import pyvo
        except ImportError:
            print("Error: 'pyvo' package required for TAP queries. Install with: pip install pyvo")
            return pd.DataFrame()
        
        # Convert IDs to strings
        valid_ids = [str(x) for x in source_ids if str(x).isdigit()]
        if not valid_ids:
            return pd.DataFrame()
            
        print(f"Querying StarHorse via TAP for {len(valid_ids)} sources...")
        
        # Query in chunks (TAP has query length limits)
        chunk_size = 1000
        results = []
        
        for i in tqdm(range(0, len(valid_ids), chunk_size), desc="StarHorse TAP"):
            chunk_ids = valid_ids[i:i+chunk_size]
            ids_str = ",".join(chunk_ids)
            
            query = f"""
            SELECT 
                source_id,
                teff50, logg50, met50,
                dist50, dist16, dist84,
                av50, av16, av84,
                mass50, mass16, mass84,
                age50, age16, age84
            FROM gaiaedr3_contrib.starhorse
            WHERE source_id IN ({ids_str})
            """
            
            try:
                tap_service = pyvo.dal.TAPService("https://gaia.aip.de/tap")
                result = tap_service.search(query)
                chunk_df = result.to_table().to_pandas()
                results.append(chunk_df)
            except Exception as e:
                print(f"TAP query error for chunk {i}: {e}")
                continue
        
        if not results:
            print("Warning: No StarHorse results from TAP queries.")
            return pd.DataFrame()
            
        sh_df = pd.concat(results, ignore_index=True)
        sh_df['source_id'] = sh_df['source_id'].astype(str)
        
        print(f"Retrieved {len(sh_df)} StarHorse entries via TAP.")
        return sh_df
        
    else:
        # Local catalog join (original implementation)
        if starhorse_file is None:
            starhorse_file = os.environ.get('STARHORSE_PATH', 'input/starhorse/starhorse2021.parquet')
            
        starhorse_path = Path(starhorse_file)
        
        if not starhorse_path.exists():
            print(f"Warning: StarHorse catalog not found at {starhorse_path}")
            print("Tip: Use use_tap=True to query remotely instead of downloading 100GB catalog.")
            return pd.DataFrame()
            
        print(f"Loading StarHorse catalog from {starhorse_path}...")
        
        try:
            if str(starhorse_path).endswith('.parquet'):
                sh_df = pd.read_parquet(starhorse_path)
            elif str(starhorse_path).endswith('.fits') or str(starhorse_path).endswith('.fits.gz'):
                from astropy.table import Table
                sh_df = Table.read(starhorse_path).to_pandas()
            else:
                sh_df = pd.read_csv(starhorse_path)
        except Exception as e:
            print(f"Error loading StarHorse: {e}")
            return pd.DataFrame()
            
        # Standardize column name
        if 'Source' in sh_df.columns:
            sh_df = sh_df.rename(columns={'Source': 'source_id'})
        elif 'EDR3Name' in sh_df.columns:
            sh_df = sh_df.rename(columns={'EDR3Name': 'source_id'})
            
        if 'source_id' not in sh_df.columns:
            print("Warning: Could not find source_id column in StarHorse catalog.")
            return pd.DataFrame()
            
        sh_df['source_id'] = sh_df['source_id'].astype(str)
        
        # Filter to requested IDs
        valid_ids = set(str(x) for x in source_ids if str(x).isdigit())
        sh_filtered = sh_df[sh_df['source_id'].isin(valid_ids)]
        
        print(f"Found {len(sh_filtered)}/{len(valid_ids)} sources in StarHorse catalog.")
        
        return sh_filtered


# =============================================================================
# 3D DUST EXTINCTION (dustmaps3d - Wang et al. 2025)
# =============================================================================

def get_dust_extinction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Query 3D dust extinction using dustmaps3d (Wang et al. 2025).
    All-sky coverage, ~350MB data, fast queries.
    """
    if df.empty:
        return df
        
    df = df.copy()
    df['A_v_3d'] = 0.0
    df['ebv_3d'] = np.nan
    
    try:
        from dustmaps3d import dustmaps3d
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except ImportError:
        print("Error: 'dustmaps3d' package not installed. Run: pip install dustmaps3d")
        return df

    if 'ra' not in df.columns or 'dec' not in df.columns:
        print("Warning: Missing ra/dec columns for dust query.")
        return df
        
    # Distance (in pc from Gaia, need kpc for dustmaps3d)
    if 'distance_gspphot' in df.columns:
        dist_pc = df['distance_gspphot'].values
    elif 'parallax' in df.columns:
        plx = df['parallax'].values
        valid_plx = (np.isfinite(plx)) & (plx > 0)
        dist_pc = np.full(len(df), np.nan)
        dist_pc[valid_plx] = 1000.0 / plx[valid_plx]
    else:
        print("Warning: No distance info for dust query.")
        return df
    
    dist_kpc = dist_pc / 1000.0
    valid_mask = (np.isfinite(df['ra'])) & (np.isfinite(df['dec'])) & (np.isfinite(dist_kpc)) & (dist_kpc > 0)
    
    if not valid_mask.any():
        return df
    
    # Convert RA/Dec to Galactic l, b
    coords = SkyCoord(ra=df.loc[valid_mask, 'ra'].values * u.deg, 
                      dec=df.loc[valid_mask, 'dec'].values * u.deg, 
                      frame='icrs')
    galactic = coords.galactic
    
    l = galactic.l.deg
    b = galactic.b.deg
    d = dist_kpc[valid_mask]
    
    try:
        print(f"Querying dustmaps3d for {valid_mask.sum()} sources...")
        ebv, dust_density, sigma, max_dist = dustmaps3d(l, b, d)
        
        A_V = 3.1 * ebv
        
        df.loc[valid_mask, 'ebv_3d'] = ebv
        df.loc[valid_mask, 'A_v_3d'] = A_V
        df.loc[valid_mask, 'dust_sigma'] = sigma
        df.loc[valid_mask, 'dust_max_dist_kpc'] = max_dist
        
        print(f"Dust query complete. Mean A_V = {A_V[np.isfinite(A_V)].mean():.3f}")
        
    except Exception as e:
        print(f"Error querying dustmaps3d: {e}")
        
    df['A_v_3d'] = df['A_v_3d'].fillna(0.0)
    
    return df


# =============================================================================
# YSO CLASSIFICATION (Koenig & Leisawitz 2014)
# =============================================================================

def classify_yso(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify YSO candidates using 2MASS-WISE color-color diagram.
    Supports dust-corrected colors if A_v_3d is present.
    """
    df = df.copy()
    
    # Map columns (support multiple naming conventions)
    col_map = {
        'H': ['Hmag', 'tmass_h'], 
        'K': ['Kmag', 'tmass_k'], 
        'W1': ['W1mag', 'unwise_w1', 'w1mpro'], 
        'W2': ['W2mag', 'unwise_w2', 'w2mpro']
    }
    
    vals = {}
    for bands, candidates in col_map.items():
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
        vals[bands] = found
        
    if not all(vals.values()):
        df['yso_class'] = 'unknown'
        return df
        
    H = df[vals['H']]
    K = df[vals['K']]
    W1 = df[vals['W1']]
    W2 = df[vals['W2']]
        
    hk_color = H - K
    w1w2_color = W1 - W2
    
    # Dust Correction
    if 'A_v_3d' in df.columns and df['A_v_3d'].sum() > 0:
        av = df['A_v_3d'].fillna(0.0)
        hk_color = hk_color - (0.18 * av)
        w1w2_color = w1w2_color - (0.05 * av)
        df['H_K_dered'] = hk_color
        df['W1_W2_dered'] = w1w2_color
    
    df['H_K'] = hk_color 
    df['W1_W2'] = w1w2_color
    
    # Classification criteria
    class_i = df['W1_W2'] > 0.8
    class_ii = ((df['W1_W2'] > 0.25) & (df['W1_W2'] < 0.8) & (df['H_K'] > 0.3))
    trans = ((df['W1_W2'] > 0.25) & (df['W1_W2'] < 0.8) & (df['H_K'] < 0.3))
    ms = df['W1_W2'] < 0.25
    
    df['yso_class'] = 'unknown'
    df.loc[class_i, 'yso_class'] = 'Class I'
    df.loc[class_ii, 'yso_class'] = 'Class II'
    df.loc[trans, 'yso_class'] = 'Transition Disk'
    df.loc[ms, 'yso_class'] = 'Main Sequence'
    
    return df


# =============================================================================
# GALACTIC POPULATION CLASSIFICATION
# =============================================================================

def classify_galactic_population(df: pd.DataFrame) -> pd.DataFrame:
    """Classify stars into thin/thick disk based on age (StarHorse) or metallicity (Gaia)."""
    if df.empty:
        return df
    df = df.copy()
    
    # Use StarHorse age if available
    if 'age50' in df.columns and 'met50' in df.columns:
        low_alpha = (df['age50'] < 8) & (df['met50'] > -0.5)
        high_alpha = (df['age50'] > 8) | (df['met50'] < -0.5)
        df['population'] = 'unknown'
        df.loc[low_alpha, 'population'] = 'thin_disk'
        df.loc[high_alpha, 'population'] = 'thick_disk'
    elif 'mh_gspphot' in df.columns:
        # Fallback to Gaia metallicity
        thin = df['mh_gspphot'] > -0.4
        thick = df['mh_gspphot'] <= -0.4
        df['population'] = 'unknown'
        df.loc[thin, 'population'] = 'thin_disk_candidate'
        df.loc[thick, 'population'] = 'thick_disk_candidate'
        
    return df


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-wavelength characterization for dipper candidates")
    parser.add_argument("--input", type=Path, required=True, help="Input events CSV/Parquet (must have asas_sn_id)")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV/Parquet")
    parser.add_argument("--crossmatch", type=Path, 
                        default=Path("input/vsx/asassn_x_vsx_matches_20250919_2252.csv"),
                        help="Path to ASAS-SN x VSX crossmatch CSV (must contain asas_sn_id and gaia_id)")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Gaia query chunk size")
    parser.add_argument("--cache", type=Path, default=Path("output/gaia_cache.parquet"), help="Cache file for Gaia queries")
    parser.add_argument("--dust", action="store_true", help="Enable dustmaps3d 3D extinction query")
    parser.add_argument("--starhorse", type=str, default=None, help="StarHorse stellar ages/masses: 'tap' for remote TAP query (recommended), or path to local catalog file")
    
    args = parser.parse_args()
    
    # Load input
    print(f"Loading {args.input}...")
    if str(args.input).endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
        
    if "asas_sn_id" not in df.columns:
        print(f"Error: 'asas_sn_id' column not found in input {args.input}")
        return

    # Load Crossmatch
    xmatch_path = args.crossmatch.expanduser()
    if not xmatch_path.exists():
        print(f"Error: Crossmatch file {xmatch_path} not found.")
        if "gaia_id" in df.columns:
            print("Warning: Crossmatch not found, but 'gaia_id' is present in input. Proceeding with input IDs.")
            df_merged = df
        else:
            return
    else:
        print(f"Loading crossmatch file {xmatch_path}...")
        xmatch_cols = ["asas_sn_id", "gaia_id", "tmass_id", "allwise_id"]
        try:
            header = pd.read_csv(xmatch_path, nrows=0).columns
            use_cols = ["asas_sn_id"] + [c for c in xmatch_cols if c in header and c != "asas_sn_id"]
            df_xmatch = pd.read_csv(xmatch_path, usecols=use_cols, dtype=str)
            
            print("Merging events with crossmatch table...")
            df["asas_sn_id"] = df["asas_sn_id"].astype(str)
            df_xmatch["asas_sn_id"] = df_xmatch["asas_sn_id"].astype(str)
            
            df_merged = df.merge(df_xmatch, on="asas_sn_id", how="left")
            print(f"Merged {len(df_merged)} rows.")
            
        except Exception as e:
            print(f"Error reading crossmatch file: {e}")
            return

    # Extract Gaia IDs
    if "gaia_id" not in df_merged.columns:
        print("Error: 'gaia_id' not found after merge.")
        df_char = df_merged
    else:
        if args.cache:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            
        missing_gaia = df_merged["gaia_id"].isna().sum()
        print(f"Found Gaia IDs for {len(df_merged) - missing_gaia}/{len(df_merged)} sources.")
        
        gaia_ids = df_merged["gaia_id"].dropna().unique().tolist()
        
        if not gaia_ids:
            print("No Gaia IDs found to query.")
            df_char = df_merged
        else:
            # Query Gaia
            print(f"Querying Gaia DR3 for {len(gaia_ids)} sources...")
            gaia_df = query_gaia_by_ids(
                gaia_ids, 
                chunk_size=args.chunk_size,
                cache_file=str(args.cache) if args.cache else None
            )
            
            if gaia_df.empty:
                print("Warning: No results from Gaia query.")
                df_char = df_merged
            else:
                df_merged["gaia_id"] = df_merged["gaia_id"].astype(str)
                gaia_df["source_id"] = gaia_df["source_id"].astype(str)
                
                print("Merging Gaia results...")
                df_char = df_merged.merge(gaia_df, left_on="gaia_id", right_on="source_id", how="left", suffixes=("", "_gaia"))
                
                # Galactic Population
                print("Classifying Galactic populations...")
                df_char = classify_galactic_population(df_char)
                
                # StarHorse join (TAP queries by default)
                if args.starhorse:
                    print("Loading StarHorse catalog for ages...")
                    # If user provides a file path, use local catalog; otherwise use TAP
                    use_tap_query = not Path(args.starhorse).exists() if args.starhorse != "tap" else True
                    sh_df = query_starhorse_by_ids(gaia_ids, starhorse_file=args.starhorse if not use_tap_query else None, use_tap=use_tap_query)
                    if not sh_df.empty:
                        df_char = df_char.merge(sh_df, on='source_id', how='left', suffixes=('', '_sh'))
                        if 'age50' in df_char.columns:
                            df_char = classify_galactic_population(df_char)
                
                # Dust correction
                if args.dust:
                    print("Computing 3D dust extinction (dustmaps3d)...")
                    df_char = get_dust_extinction(df_char)
                
                # YSO Classification
                if "tmass_j" in df_char.columns:
                    print("Classifying YSOs...")
                    df_char = classify_yso(df_char)
                else:
                    print("Warning: IR photometry columns not found for YSO classification.")
    
    # Save results
    print("Saving results...")
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if str(output_path).endswith(".parquet"):
        df_char.to_parquet(output_path, index=False)
    else:
        df_char.to_csv(output_path, index=False)
        
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
