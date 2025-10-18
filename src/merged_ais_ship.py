import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats


def create_merged_ais_ship(filtered_ais_path, filtered_ship_path, merged_path):

    # --- Load filtered AIS data ---
    df_ais = pd.read_parquet(filtered_ais_path).dropna()

    # --- Merge with ship filtered data ---
    df_ship = pd.read_parquet(filtered_ship_path)
    df_merged = df_ais.merge(
            df_ship[["MMSI", "VesselType", "Length", "Width", "Draft", "Cargo", "TransceiverClass"]],
            on="MMSI",
            how="left"
        ).dropna()

    df_merged.to_parquet(merged_path, engine="pyarrow", index=False)


def plot_merged_ais_ship(
    dataset_path,
    mercator_path,
    lon_c,
    lat_c,
    a_lon,
    b_lat,
    tboat,
    LON_MIN,
    LON_MAX,
    LAT_MIN,
    LAT_MAX
):
    """
    Plot filtered AIS points with Mercator velocity field as background.
    
    Parameters
    ----------
    filtered_ais_path : str
        Path to AIS parquet file
    filtered_ship_path : str
        Path to ship parquet file
    mercator_path : str
        Path to Mercator NetCDF file (containing uo/vo)
    lon_c, lat_c : float
        Ellipse center coordinates
    a_lon, b_lat : float
        Semi-axes of the ellipse (longitude, latitude)
    tboat : datetime-like
        Reference time for AIS and Mercator data
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX : float
        Geographic boundaries of the map
    """

    # --- Load MERGED data ---
    df_filtered = pd.read_parquet(dataset_path).dropna()

    # --- Load Mercator data ---
    loaded_mercator = xr.open_dataset(mercator_path)
    LON_m = loaded_mercator['longitude'].values
    LAT_m = loaded_mercator['latitude'].values
    LON2D, LAT2D = np.meshgrid(LON_m, LAT_m, indexing='xy')
    ellipse = ((LON2D - lon_c)**2 / a_lon**2) + ((LAT2D - lat_c)**2 / b_lat**2)

    # --- Create figure with Cartopy ---
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(9, 6), subplot_kw={'projection': proj})

    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='0.92', zorder=0)
    ax.coastlines(resolution='50m', linewidth=0.8)
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)

    # Select Mercator field at given time
    hour_mercator = loaded_mercator.sel(time=tboat)
    hour_velocity = np.sqrt(hour_mercator['uo'].values[0]**2 + hour_mercator['vo'].values[0]**2)

    # Select AIS points within one hour window
    hour_ais_df = df_filtered[
        (df_filtered['BaseDateTime'] >= tboat) & 
        (df_filtered['BaseDateTime'] < tboat + pd.Timedelta(hours=1))
    ]

    # Compute angular difference between Heading and COG
    hour_diff_rad = np.arctan2(
        np.sin(np.deg2rad(hour_ais_df['Heading'] - hour_ais_df['COG'])),
        np.cos(np.deg2rad(hour_ais_df['Heading'] - hour_ais_df['COG']))
    )

    # --- Background velocity field ---
    im = ax.pcolormesh(
        LON_m, LAT_m, hour_velocity,
        shading='auto', cmap='YlOrRd', vmin=0, vmax=5
    )

    # --- Filtered AIS points ---
    sc = ax.scatter(
        hour_ais_df['LON'], hour_ais_df['LAT'],
        c=np.abs(hour_diff_rad), cmap='binary', s=1.5, alpha=0.6,
        vmin=0, vmax=np.pi/8, transform=proj
    )

    # --- Ellipse contour ---
    ax.contour(
        LON2D, LAT2D, ellipse,
        levels=[1], colors='limegreen',
        linestyles=':', linewidths=2,
        transform=proj
    )

    # --- Colorbars ---
    cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label('Boat Drift [rad]', fontsize=10)

    cbar_velocity = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar_velocity.set_label('Current Velocity [m.s-1]', fontsize=10)

    # --- Gridlines ---
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, x_inline=False, y_inline=False)
    gl.right_labels = False
    gl.top_labels = False

    # --- Legend ---
    boat_proxy = ax.scatter([], [], c='gray', s=10, label='Boats', transform=proj)
    ellipse_proxy = plt.Line2D([0], [0], color='limegreen',
                               linestyle=':', linewidth=2, label='Study Area')
    ax.legend(handles=[boat_proxy, ellipse_proxy], loc='upper right')

    # --- Final plot ---
    plt.title(f'AIS points (filtered) with Velocity background\n{tboat}', fontsize=12)
    plt.tight_layout()
    plt.show()

def prefilter_outliers(x, method="mad", k=3.5):
    """
    x: array-like (pd.Series or np.array) of degree difference
    method: "mad" (par défaut) ou "iqr"
    k: threshold (3.5 pour MAD, 1.5 à 3 pour IQR)
    return: (x_filtre (pd.Series), mask_bool)
    """
    x = pd.Series(x, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna() # To be sure

    if method == "mad":
        med = x.median()
        mad = stats.median_abs_deviation(x, scale="normal")
        if mad == 0:
            mask = pd.Series(True, index=x.index)
        else:
            z = (x - med).abs() / mad
            mask = z <= k
    elif method == "iqr":
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - k*iqr, q3 + k*iqr
        mask = (x >= low) & (x <= high)
    else:
        raise ValueError("method must be 'mad' or 'iqr'")

    return x[mask]

def winsorize_mad(x, k=4):
    x = pd.Series(x, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    from scipy import stats
    med = x.median()
    mad = stats.median_abs_deviation(x, scale="normal")
    if mad == 0:
        return x
    low, high = med - k*mad, med + k*mad
    return x.clip(lower=low, upper=high)

def normality_test(x):
    x = pd.Series(x).dropna().astype(float)
    n = len(x)
    if n == 0:
        raise ValueError("No data post filter.")
    if n > 5000:
        k2, p = stats.normaltest(x)
        return {"test": "D’Agostino K²", "stat": k2, "pvalue": p, "n": n}
    else:
        W, p = stats.shapiro(x)
        return {"test": "Shapiro-Wilk", "stat": W, "pvalue": p, "n": n}


def polar_histogram_with_stats(filtered_df, bins=360, figsize=(14, 10)):
    """
    Create a polar histogram with descriptive statistics.

    Parameters:
    -----------
    filtered_df : pandas.DataFrame
        DataFrame containing 'Heading' and 'COG' columns
    bins : int
        Number of bins for the histogram (default: 360 = 0.5° / bin)
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    fig : matplotlib.figure.Figure
    stats_dict : dict
        Dictionary with descriptive statistics
    """

    # Compute Angular Difference in Radians
    diff_rad = np.arctan2(
        np.sin(np.deg2rad(filtered_df['Heading'] - filtered_df['COG'])),
        np.cos(np.deg2rad(filtered_df['Heading'] - filtered_df['COG']))
    )
    abs_diff_degree = np.abs(np.rad2deg(diff_rad))

    # Descriptive statistics
    stats_dict = {
        'Number of observations': len(abs_diff_degree),
        'Mean (deg)': np.mean(abs_diff_degree),
        'Median (deg)': np.median(abs_diff_degree),
        'Standard deviation (deg)': np.std(abs_diff_degree),
        'Min (deg)': np.min(abs_diff_degree),
        'Max (deg)': np.max(abs_diff_degree),
        'Q1 (deg)': np.percentile(abs_diff_degree, 25),
        'Q3 (deg)': np.percentile(abs_diff_degree, 75),
        'IQR (deg)': np.percentile(abs_diff_degree, 75) - np.percentile(abs_diff_degree, 25),
    }

    fig = plt.figure(figsize=figsize)

    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], width_ratios=[2, 2, 1],
                         hspace=0.3, wspace=0.3)

    # 1. Polar Histogram (quarter circle)
    ax_polar = fig.add_subplot(gs[0, :2], projection='polar')

    theta_bins = np.linspace(0, np.pi/8, bins + 1)  # From 0 to 22.5°
    hist_values, bin_edges = np.histogram(np.deg2rad(abs_diff_degree), bins=theta_bins)

    theta_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = bin_edges[1] - bin_edges[0]

    bars = ax_polar.bar(theta_centers, hist_values, width=width,
                       alpha=0.7, color='steelblue', edgecolor='navy', linewidth=1)

    ax_polar.set_theta_zero_location('N')  # 0° at the top
    ax_polar.set_theta_direction(-1)  # clockwise
    ax_polar.set_thetalim(0, np.pi/8)  # Restrict to [0, π/8]

    theta_ticks = np.linspace(0, np.pi/8, 7)
    theta_labels = [f'{int(np.rad2deg(t))}°' for t in theta_ticks]
    ax_polar.set_thetagrids(np.rad2deg(theta_ticks), theta_labels)

    ax_polar.set_title('Polar Histogram of |Heading - COG|',
                      pad=20, fontsize=14, fontweight='bold')
    ax_polar.set_ylabel('Frequency', labelpad=30)

    # 2. Classic Histogram
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_hist.hist(winsorize_mad(np.rad2deg(diff_rad)), bins=50, alpha=0.7, color='lightcoral',
                edgecolor='darkred', density=True)
    ax_hist.set_xlabel('Relative Difference (degrees)')
    ax_hist.set_ylabel('Density')
    ax_hist.set_title('Distribution of Relative Differences', fontsize=12, fontweight='bold')
    ax_hist.grid(True, alpha=0.3)
    x_density = np.linspace(np.min(np.rad2deg(diff_rad)), np.max(np.rad2deg(diff_rad)), 100)
    try:
        kde = stats.gaussian_kde(np.rad2deg(diff_rad))
        ax_hist.plot(x_density, kde(x_density), 'r-', linewidth=2, label='KDE')
        ax_hist.legend()
    except:
        pass

    # 3. Box plot
    ax_box = fig.add_subplot(gs[1, 1])
    bp = ax_box.boxplot([abs_diff_degree], patch_artist=True,
                       labels=['Differences (°)'])
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][0].set_alpha(0.7)
    ax_box.set_title('Boxplot', fontsize=12, fontweight='bold')
    ax_box.grid(True, alpha=0.3)

    # 4. Stats Table
    ax_stats = fig.add_subplot(gs[:, 2])
    ax_stats.axis('off')

    stats_text = "DESCRIPTIVE STATISTICS\n" + "="*25 + "\n\n"

    main_stats = [
        f"N observations: {stats_dict['Number of observations']}",
        f"",
        f"Mean: {stats_dict['Mean (deg)']:.2f}°",
        f"",
        f"Median: {stats_dict['Median (deg)']:.2f}°",
        f"",
        f"Std deviation: {stats_dict['Standard deviation (deg)']:.2f}°",
        f"",
        f"Min: {stats_dict['Min (deg)']:.2f}°",
        f"",
        f"Max: {stats_dict['Max (deg)']:.2f}°",
        f"",
        f"Q1: {stats_dict['Q1 (deg)']:.2f}°",
        f"Q3: {stats_dict['Q3 (deg)']:.2f}°",
        f"IQR: {stats_dict['IQR (deg)']:.2f}°"
    ]

    stats_text += "\n".join(main_stats)

    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    fig.suptitle('Angular Difference Analysis between Heading and COG',
                fontsize=16, fontweight='bold', y=0.95)

    plt.tight_layout()

    return fig, stats_dict


def advanced_polar_analysis(filtered_df, bins=36):
    """
    Advanced polar analysis with statistical tests
    """

    diff_rad = np.arctan2(
        np.sin(np.deg2rad(filtered_df['Heading'] - filtered_df['COG'])),
        np.cos(np.deg2rad(filtered_df['Heading'] - filtered_df['COG']))
    )
    diff_degree = np.rad2deg(diff_rad)
    abs_diff_rad = np.abs(diff_rad)

    print("ADVANCED STATISTICAL ANALYSIS")
    print("=" * 40)

    print(f"\nNormality test (Shapiro-Wilk if n<5000, D'Agostino otherwise):")
    res = normality_test(winsorize_mad(diff_degree))
    print(f"N before: {len(pd.Series(diff_degree).dropna())} | N after: {res['n']}")
    print(f"Test: {res['test']} | stat={res['stat']:.3f} | p-value={res['pvalue']:.3g}")

    mean_direction = np.mean(abs_diff_rad)
    circular_var = np.var(abs_diff_rad)

    print(f"\nCircular statistics:")
    print(f"Mean direction: {np.rad2deg(mean_direction):.2f}°")
    print(f"Circular variance: {circular_var:.4f}")
    print(f"Concentration: {1 - circular_var:.4f}")

    percentiles = [5, 10, 25, 50, 75, 90, 95]
    print(f"\nPercentiles (degrees):")
    for p in percentiles:
        val = np.percentile(np.rad2deg(abs_diff_rad), p)
        print(f"P{p}: {val:.2f}°")
