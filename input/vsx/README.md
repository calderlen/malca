# VSX Crossmatch Data

**NOTE**: The default catalog locations for pre_filter.py are in `results_crossmatch/`:
- `results_crossmatch/asassn_index_masked_concat_cleaned_20250926_1557.csv`
- `results_crossmatch/vsx_cleaned_20250926_1557.csv`

This directory can be used for alternative VSX/ASAS-SN catalog files.

## Required File Formats

### ASAS-SN Index File
The ASAS-SN index file should contain:
- `asas_sn_id`, `ra_deg`, `dec_deg`, `pm_ra`, `pm_dec`
- Photometric columns: `gaia_mag`, `pstarrs_g_mag`, `pstarrs_r_mag`, etc.
- Catalog cross-references: `gaia_id`, `hip_id`, `tyc_id`, `tmass_id`, etc.

Example structure:
```
asas_sn_id,ra_deg,dec_deg,pm_ra,pm_dec,gaia_mag,pstarrs_g_mag,...
395137802047,166.54080613,-12.80643723,-23.85,-1.21,11.51,12.014,...
```

### VSX Catalog File
The VSX catalog should contain:
- `ra`, `dec` - Coordinates in degrees
- `class` - Variable star classification

## Usage

### Pre-computed Crossmatches
If you already have index files with coordinates and proper motion (default), the filters will use them directly.

### On-the-fly Crossmatching
`malca/vsx_crossmatch.py` can generate crossmatches between ASAS-SN and VSX catalogs.

`malca/pre_filter.py` can perform crossmatching on-the-fly if needed (see filter_vsx_match and filter_bright_nearby_stars).
