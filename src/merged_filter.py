import pandas as pd
import numpy as np
import xarray as xr

def detect_coord_names(ds):
    """Guess the coordinate names (lon, lat, time, depth)."""
    def _detect(cands):
        for n in cands:
            if n in ds.coords or n in ds.variables:
                return n
        return None

    lon = _detect(("lon", "longitude", "x"))
    lat = _detect(("lat", "latitude", "y"))
    time = _detect(("time",)) or "time"
    depth = _detect(("depth", "z", "lev", "level"))
    return {"lon": lon, "lat": lat, "time": time, "depth": depth}


def select_surface_if_any(ds, depth_name):
    """If a vertical dimension exists, take the first level (surface)."""
    if depth_name and depth_name in ds.dims:
        return ds.isel({depth_name: 0})
    return ds

def ensure_datetime64_utc(series):
    """Converts a pandas Series to a UTC-based numpy datetime64[ns] object."""
    return pd.to_datetime(series, errors="coerce", utc=True).to_numpy(dtype="datetime64[ns]")


def harmonize_longitudes(lon_values, ds_lon):
    """Aligns DataFrame longitudes to match the dataset convention (either [-180, 180] or [0, 360])."""
    lon = np.asarray(lon_values, dtype=float)
    lon_min = float(np.nanmin(ds_lon.values))
    lon_max = float(np.nanmax(ds_lon.values))
    if lon_min >= 0.0 and lon_max > 180.0:      # dataset in 0..360
        lon = np.where(lon < 0.0, lon + 360.0, lon)
    elif lon_max <= 180.0:                       # dataset in -180..180
        lon = np.where(lon > 180.0, lon - 360.0, lon)
    return lon

def compute_phi_deg_from_uv(u, v):
    """Compute the sae current bearing in degrees, clockwise from North: PHI = atan2(u, v), mapped to [0, 360)."""
    phi = np.degrees(np.arctan2(u, v))
    return (phi + 360.0) % 360.0

def add_uv_phi_to_df(
    df_merged_dataset,
    mercator_path,
    lon_col="LON",
    lat_col="LAT",
    time_col="BaseDateTime",
):
    """
    Sample 'uo' and 'vo' (nearest in space/time) from the xarray Dataset at AIS points,
    compute PHI, and return a new DataFrame with columns U, V, and PHI.
    Also saves to Parquet if `outfile_parquet` is provided.
    """
    # --- Load merged and mercator data ---
    loaded_mercator = xr.open_dataset(mercator_path)

    # Contact details & surface area
    names = detect_coord_names(loaded_mercator)
    ds = select_surface_if_any(loaded_mercator, names["depth"])

    # Force the dataset time to datetime64 if possible (if it fails, it's okay in the notebook)
    if names["time"] in ds.coords and not np.issubdtype(ds[names["time"]].dtype, np.datetime64):
        ds = ds.convert_calendar("proleptic_gregorian", use_cftime=False)

    # Targets to interpolate
    lons_df = df_merged_dataset[lon_col].to_numpy(dtype=float)
    lats_df = df_merged_dataset[lat_col].to_numpy(dtype=float)
    times_df = ensure_datetime64_utc(df_merged_dataset[time_col])

    # Harmonizes longitudes relative to the dataset
    lons_df = harmonize_longitudes(lons_df, ds[names["lon"]])

    # Prepare 1D DataArray “obs” for vectorized interpolation
    obs_dim = "obs"
    t_da = xr.DataArray(times_df, dims=(obs_dim,))
    lon_da = xr.DataArray(lons_df, dims=(obs_dim,))
    lat_da = xr.DataArray(lats_df, dims=(obs_dim,))

    # Useful subset and nearest neighbor interpolation
    ds_needed = ds[["uo", "vo"]]
    sampled = ds_needed.interp({names["time"]: t_da, names["lon"]: lon_da, names["lat"]: lat_da},
                               method="nearest")

    # Retrieves U,V aligned with obs, then calculates PHI
    u = np.asarray(sampled["uo"].values)
    v = np.asarray(sampled["vo"].values)
    phi = compute_phi_deg_from_uv(u, v)

    out = df_merged_dataset.drop(columns=['Status']).dropna().copy()
    out["U"] = u
    out["V"] = v
    out["PHI"] = phi

    out = out.dropna()

    return out

def create_merged_dataset(
    merged_path,
    mercator_path,
    dataset_path,
    lon_c,
    lat_c,
    a_lon,
    b_lat):

    # --- Load merged data ---
    df_merged_dataset = pd.read_parquet(merged_path).dropna()

    # --- Stufy area filter ---
    df_merged_dataset = df_merged_dataset[ ((df_merged_dataset['LON'] - lon_c)**2 / a_lon**2 + (df_merged_dataset['LAT'] - lat_c)**2 / b_lat**2) <= 1 ]
    
    # --- Drift angle filter ---
    diff_rad = np.arctan2(
        np.sin(np.deg2rad(df_merged_dataset['Heading'] - df_merged_dataset['COG'])),
        np.cos(np.deg2rad(df_merged_dataset['Heading'] - df_merged_dataset['COG']))
    )
    df_merged_dataset['Drift'] = np.rad2deg(diff_rad)
    df_merged_dataset = df_merged_dataset.loc[df_merged_dataset['Drift'].abs() <= 8]

    # --- Merged with Mercator (u,v,phi) ---
    out = add_uv_phi_to_df(df_merged_dataset,
                            mercator_path,
                            lon_col="LON",
                            lat_col="LAT",
                            time_col="BaseDateTime"
                            )

    out.to_parquet(dataset_path, engine="pyarrow", index=False)

