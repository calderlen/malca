import matplotlib.pyplot as pl
import matplotlib.ticker as tick
import pandas as pd
import numpy as np
import scipy.signal
from astropy.time import Time
from df_utils import jd_to_year, year_to_jd

# these are all of the non-derived columns we have to work with -- consider joining them together here as necessary


asassn_columns=["JD",
                "mag",
                'error', 
                'good_bad', #1=good, 0 =bad
                'camera#', 
                'v_g_band', #1=V, 0=g
                'saturated',
                'cam_field']
  
asassn_raw_columns = [
                'cam#',
                'median',
                '1siglow', 
               '1sighigh', 
               '90percentlow',
               '90percenthigh']

asassn_index_columns = ['asassn_id',
                        'ra_deg',
                        'dec_deg',
                        'refcat_id',
                        'gaia_id', 
                        'hip_id',
                        'tyc_id',
                        'tmass_id',
                        'sdss_id',
                        'allwise_id',
                        'tic_id',
                        'plx',
                        'plx_d',
                        'pm_ra',
                        'pm_ra_d',
                        'pm_dec',
                        'pm_dec_d',
                        'gaia_mag',
                        'gaia_mag_d',
                        'gaia_b_mag',
                        'gaia_b_mag_d',
                        'gaia_r_mag',
                        'gaia_r_mag_d',
                        'gaia_eff_temp',
                        'gaia_g_extinc',
                        'gaia_var',
                        'sfd_g_extinc',
                        'rp_00_1',
                        'rp_01',
                        'rp_10',
                        'pstarrs_g_mag',
                        'pstarrs_g_mag_d',
                        'pstarrs_g_mag_chi',
                        'pstarrs_g_mag_contrib',
                        'pstarrs_r_mag',
                        'pstarrs_r_mag_d',
                        'pstarrs_r_mag_chi',
                        'pstarrs_r_mag_contrib',
                        'pstarrs_i_mag',
                        'pstarrs_i_mag_d',
                        'pstarrs_i_mag_chi',
                        'pstarrs_i_mag_contrib',
                        'pstarrs_z_mag',
                        'pstarrs_z_mag_d',
                        'pstarrs_z_mag_chi',
                        'pstarrs_z_mag_contrib',
                        'nstat']


# stats you have to work with, this is everything you've derived from the above data and the file structure


# in lc_dips.process_record_naive
    #   mag_bin
    #   asas_sn_id
    #   index_num
    #   index_csv
    #   lc_dir
    #   dat_path
    #   raw_path
    #   g_n_peaks
    #   g_mean_mag
    #   g_peaks_idx
    #   g_peaks_jd
    #   v_n_peaks
    #   v_mean_mag
    #   v_peaks_idx
    #   v_peaks_jd
    #   jd_first
    #   jd_last
    #   n_rows_g
    #   n_rows_v
    
# in lc_dips.naive_dip_finder
    #    n_dip_runs,
    #    n_jump_runs,
    #    n_dip_points,
    #    n_jump_points,
    #    most_recent_dip,
    #    most_recent_jump,
    #    max_depth,
    #    max_height,
    #    max_dip_duration,
    #    max_jump_duration,
    #    dip_fraction
    #    jump_fraction

def process_and_plot_from_csv(csv_file_path, lc_data_dir='Updated_LC_data/'):
    """
    Reads a CSV of candidates, fetches their light curves, processes peaks, 
    and generates the plots.
    
    """
    
    # 1. Read the CSV
    df = pd.read_csv(csv_file_path)
    
    # 2. Build the dictionary structure required by the plotter
    light_curve_dict = {}
    
    print(f"Processing {len(df)} sources from {csv_file_path}...")
    
    for index, row in df.iterrows():
        asassn_id = str(row['Source_ID'])
        
        # Generate the ID string (e.g., J073234-200049)
        # If 'Source' column exists, use it, otherwise calculate from RA/Dec
        if 'Source' in row:
            source_label = row['Source']
        else:
            # Fallback if Source name isn't in CSV, requires custom_id function
            try:
                source_label = custom_id(row['RA'], row['Dec'])
            except NameError:
                source_label = f"ID:{asassn_id}"

        # 3. Read Light Curve Data
        # Attempting to use your existing read function pattern
        try:
            # Assuming read_lightcurve_dat returns (v_data, g_data)
            # You might need to adjust this if your function name is different
            v_df, g_df = read_lightcurve_dat(asassn_id, lc_data_dir) 
        except Exception as e:
            print(f"Skipping {source_label} ({asassn_id}): Could not read LC data. {e}")
            continue

        # 4. Process g-band for peaks (Logic from your snippets)
        peaks = []
        if g_df is not None and not g_df.empty:
            # Basic filtering (optional, based on your snippet)
            # g_df = g_df.loc[g_df.Mag < 15].reset_index(drop=True) 
            
            mean_mag = np.mean(g_df['Mag'])
            mag_avg = [i - mean_mag for i in g_df['Mag']]
            
            # Peak finding parameters from your code
            found_peaks, _ = scipy.signal.find_peaks(
                mag_avg, 
                prominence=0.17, 
                distance=25, 
                height=0.3, 
                width=2
            )
            peaks = [int(p) for p in found_peaks]

        # 5. Determine Plot Limits (Optional auto-scaling)
        # You can make this smarter, currently just None to let matplotlib autosale
        y_lims = None
        # If you want to strictly follow your snippet logic:
        if g_df is not None and not g_df.empty:
            y_min = 11 # Default or calculated from data
            y_max = g_df['Mag'].max() + 0.5
            # y_lims = (y_min, y_max) # Uncomment to enforce limits

        # 6. Add to dictionary
        light_curve_dict[source_label] = {
            'g_data': g_df,
            'v_data': v_df,
            'peaks': peaks,
            'title': f"{source_label} (ID: {asassn_id})",
            # 'ylim': y_lims, # Add this if you calculated limits
            # 'xlim': (-1700, 2950) # Add this if you want fixed x-axis like your snippets
        }

    # 7. Call the plotting function
    if len(light_curve_dict) > 0:
        plot_light_curves(light_curve_dict)
    else:
        print("No valid data found to plot.")




def plot_light_curves(light_curve_dict, output_dir='plots'):
    """
    Generates light curve plots from the processed dictionary.
    """

    # Plotting Parameters
    colors = ["#6b8bcd", "#b3b540", "#8f62ca", "#5eb550", "#c75d9c", "#4bb092", 
              "#c5562f", "#6c7f39", "#ce5761", "#c68c45", '#b5b246', '#d77fcc', 
              '#7362cf', '#ce443f', '#3fc1bf', '#cda735', '#a1b055']
    
    num_plots = len(light_curve_dict)
    cols = 3
    rows = (num_plots + cols - 1) // cols  # Ceiling division to determine rows
    
    # Adjust figure size dynamically based on rows
    fig = pl.figure(figsize=(16, 4 * rows))
    gs = fig.add_gridspec(rows, cols, hspace=0, wspace=0)
    axs = gs.subplots(sharex=False, sharey=False) 
    
    # Handle single plot case (axs is not an array) vs multiple (axs is array)
    if num_plots > 1:
        axs_flat = axs.flatten()
    else:
        axs_flat = [axs]

    # Iterate through the dictionary and plot
    for i, (source_name, data) in enumerate(light_curve_dict.items()):
        ax = axs_flat[i]
        
        g_df = data.get('g_data')
        v_df = data.get('v_data')
        peaks = data.get('peaks', [])
        title = data.get('title', source_name)
        
        # Plot g-band
        if g_df is not None and not g_df.empty:
            cams = g_df["Camera"].unique()
            for j, cam in enumerate(cams):
                cam_data = g_df[g_df["Camera"] == cam]
                cam_jd = cam_data["JD"].astype(float) - 2.458 * 10**6
                cam_mag = cam_data["Mag"].astype(float)
                color = colors[j % len(colors)]
                ax.scatter(cam_jd, cam_mag, color=color, alpha=0.6, marker='.', label=f'g-{cam}')

        # Plot V-band
        if v_df is not None and not v_df.empty:
             cams = v_df["Camera"].unique()
             for j, cam in enumerate(cams):
                cam_data = v_df[v_df["Camera"] == cam]
                cam_jd = cam_data["JD"].astype(float) - 2.458 * 10**6
                cam_mag = cam_data["Mag"].astype(float)
                color = colors[j % len(colors)] 
                ax.scatter(cam_jd, cam_mag, color=color, alpha=0.6, marker='.', label=f'V-{cam}')

        # Vertical Lines for Peaks
        if len(peaks) > 0 and g_df is not None:
            # Calculate limits for lines if not provided
            y_min = data.get('ylim', (g_df['Mag'].min(), g_df['Mag'].max()))[0]
            y_max = data.get('ylim', (g_df['Mag'].min(), g_df['Mag'].max()))[1] + 0.2
            
            for peak_idx in peaks:
                 if peak_idx < len(g_df):
                    peak_jd = g_df.iloc[peak_idx]['JD'] - 2.458 * 10**6
                    ax.vlines(peak_jd, y_min, y_max, "k", alpha=0.4)

        # Formatting
        ax.invert_yaxis()
        ax.minorticks_on()
        ax.tick_params(axis='x', direction='in', top=False, labelbottom=True, bottom=True, pad=-15, labelsize=12)
        ax.tick_params(axis='y', direction='in', right=False, pad=-35, labelsize=12)
        ax.tick_params('both', length=6, width=1.5, which='major')
        ax.tick_params('both', direction='in', length=4, width=1, which='minor')
        ax.yaxis.set_minor_locator(tick.MultipleLocator(0.1))
        
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)

        ax.set_title(title, y=1.0, pad=-40, size=15)

        if 'xlim' in data:
            ax.set_xlim(data['xlim'])
        if 'ylim' in data:
            ax.set_ylim(data['ylim'])
        
        # Attempt Secondary Axis (Years)
        try:
            secax = ax.secondary_xaxis('top', functions=(lambda x: jd_to_year(x + 2.458*10**6), 
                                                         lambda x: year_to_jd(x) - 2.458*10**6))
            secax.xaxis.set_tick_params(direction='in', labelsize=12, pad=-18, length=6, width=1.5)
        except:
            pass

    # Hide unused subplots
    for j in range(i + 1, len(axs_flat)):
        axs_flat[j].set_visible(False)

    fig.supxlabel('Julian Date $- 2458000$ [d]', fontsize=30, y=0.05)
    fig.supylabel('Mag', fontsize=30, x=0.08)

    pl.show()