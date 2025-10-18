import pyarrow.parquet as pq
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def sampled_maps(dst_path, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, SOG_MIN, SOG_MAX, sampling_norm=False):
  # Definition of the area of interest for Cartopy
  lon_bounds = [LON_MIN, LON_MAX]
  lat_bounds = [LAT_MIN, LAT_MAX]

  # Optimized reading by chunks with specific columns only
  columns_needed = ['BaseDateTime', 'LON', 'LAT', 'SOG']
  chunk_size = 50000  # Process by chunks of 50k rows

  print("Reading the file by chunks...")

  # Use pyarrow to read the parquet file
  pf = pq.ParquetFile(dst_path)

  # Determine the time period without loading everything
  first_chunk = pf.read_row_group(0).to_pandas()
  start_date = first_chunk['BaseDateTime'].min()
  end_date = pf.read_row_group(pf.num_row_groups - 1).to_pandas()['BaseDateTime'].max() # Get max date from the last row group
  del first_chunk  # Free memory

  print(f"Data period: {start_date} to {end_date}")

  # Definition of target periods
  period_1h_end = start_date + timedelta(hours=1)
  period_1d_end = start_date + timedelta(days=1)
  period_1m_end = start_date + timedelta(days=30)

  # Lists to store sampled data
  data_1h_sample = []
  data_1d_sample = []
  data_1m_sample = []

  # Sampling parameters to avoid memory overload
  if sampling_norm:
    max_points_1m = pf.metadata.num_rows
    max_points_1d = max_points_1m//31
    max_points_1h = max_points_1d//24
  else:
    max_points_1m = pf.metadata.num_rows
    max_points_1d = max_points_1m
    max_points_1h = max_points_1m


  print("Processing data by chunks...")

  # Process chunk by chunk using pyarrow
  for rg in range(pf.num_row_groups):
      tbl = pf.read_row_group(rg, columns=columns_needed)
      chunk = tbl.to_pandas()

      # Time filtering for 1 hour
      mask_1h = (chunk['BaseDateTime'] >= start_date) & (chunk['BaseDateTime'] <= period_1h_end)
      if mask_1h.any() and len(pd.concat(data_1h_sample, ignore_index=True) if data_1h_sample else pd.DataFrame()) < max_points_1h:
          chunk_1h = chunk[mask_1h]
          # Sampling if too many points
          if len(chunk_1h) > (max_points_1h - (len(pd.concat(data_1h_sample, ignore_index=True) if data_1h_sample else pd.DataFrame()))):
              chunk_1h = chunk_1h.sample(n=max_points_1h - (len(pd.concat(data_1h_sample, ignore_index=True) if data_1h_sample else pd.DataFrame())))
          data_1h_sample.append(chunk_1h)

      # Time filtering for 1 day
      mask_1d = (chunk['BaseDateTime'] >= start_date) & (chunk['BaseDateTime'] <= period_1d_end)
      if mask_1d.any() and len(pd.concat(data_1d_sample, ignore_index=True) if data_1d_sample else pd.DataFrame()) < max_points_1d:
          chunk_1d = chunk[mask_1d]
          # Sampling if too many points
          if len(chunk_1d) > (max_points_1d - (len(pd.concat(data_1d_sample, ignore_index=True) if data_1d_sample else pd.DataFrame()))):
              chunk_1d = chunk_1d.sample(n=max_points_1d - (len(pd.concat(data_1d_sample, ignore_index=True) if data_1d_sample else pd.DataFrame())))
          data_1d_sample.append(chunk_1d)

      # Time filtering for 1 month
      mask_1m = (chunk['BaseDateTime'] >= start_date) & (chunk['BaseDateTime'] <= period_1m_end)
      if mask_1m.any() and len(pd.concat(data_1m_sample, ignore_index=True) if data_1m_sample else pd.DataFrame()) < max_points_1m:
          chunk_1m = chunk[mask_1m]
          # Sampling if too many points
          if len(chunk_1m) > (max_points_1m - (len(pd.concat(data_1m_sample, ignore_index=True) if data_1m_sample else pd.DataFrame()))):
              chunk_1m = chunk_1m.sample(n=max_points_1m - (len(pd.concat(data_1m_sample, ignore_index=True) if data_1m_sample else pd.DataFrame())))
          data_1m_sample.append(chunk_1m)

      # If enough data is collected for all periods, break the loop
      if (len(pd.concat(data_1h_sample, ignore_index=True) if data_1h_sample else pd.DataFrame()) >= max_points_1h and
          len(pd.concat(data_1d_sample, ignore_index=True) if data_1d_sample else pd.DataFrame()) >= max_points_1d and
          len(pd.concat(data_1m_sample, ignore_index=True) if data_1m_sample else pd.DataFrame()) >= max_points_1m):
          break


  # Concatenation of sampled chunks
  print("Concatenating sampled data...")

  data_1h = pd.concat(data_1h_sample, ignore_index=True) if data_1h_sample else pd.DataFrame()
  data_1d = pd.concat(data_1d_sample, ignore_index=True) if data_1d_sample else pd.DataFrame()
  data_1m = pd.concat(data_1m_sample, ignore_index=True) if data_1m_sample else pd.DataFrame()

  # Free memory of intermediate lists
  del data_1h_sample, data_1d_sample, data_1m_sample


  # Create the figure with subplots
  fig = plt.figure(figsize=(18, 6))

  # Map projection configuration
  projection = ccrs.PlateCarree()

  datasets = [data_1h, data_1d, data_1m]
  titles = ['Maritime traffic - 1 hour', 'Maritime traffic - 1 day', 'Maritime traffic - 1 month']
  periods = ['1hour', '1day', '1month']

  for i, (data, title, period) in enumerate(zip(datasets, titles, periods)):
      ax = fig.add_subplot(1, 3, i+1, projection=projection)

      # Geographic extent configuration
      ax.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]],
                    crs=ccrs.PlateCarree())

      # Add geographic features
      ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
      ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='gray')
      ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.7)
      ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)

      # Add gridlines
      gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
      gl.top_labels = False
      gl.right_labels = False

      if len(data) > 0:
          # Scatter plot with plasma colormap according to speed
          scatter = ax.scatter(data['LON'], data['LAT'],
                            c=data['SOG'],
                            cmap='plasma',
                            s=1.5,
                            alpha=0.6,
                            vmin=SOG_MIN, vmax=SOG_MAX,
                            transform=ccrs.PlateCarree())

          # Add colorbar
          cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
          cbar.set_label('Speed (SOG) in knots', fontsize=10)

          # Data info
          ax.text(0.02, 0.98, f'{len(data)} points\n{period}',
                  transform=ax.transAxes,
                  verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                  fontsize=9)
      else:
          ax.text(0.5, 0.5, f'No data\nfor {period}',
                  transform=ax.transAxes,
                  horizontalalignment='center',
                  verticalalignment='center',
                  fontsize=12,
                  bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))


      ax.set_title(title, fontsize=12, pad=10)

  fig.suptitle('AIS Trajectories - Vessel passages (sampled)\n'
              f'Area: {lon_bounds[0]}°/{lon_bounds[1]}° E, {lat_bounds[0]}°/{lat_bounds[1]}° N',
              fontsize=14, y=0.95)

  plt.tight_layout()
  plt.subplots_adjust(top=0.85)

  plt.savefig("/content/drive/MyDrive/GithubProject/AIS-Deep-Learning-Project/results/ais_trajectories_sampled.png", dpi=300, bbox_inches='tight')  # PNG haute résolution

  plt.show()

  print(f"\n{'='*60}")
  print("DETAILED STATISTICS (sampled data)")
  print(f"{'='*60}")

  for period, data in zip(periods, datasets):
      print(f"\n{period.upper()}:")
      if len(data) > 0:
          print(f"  - Number of points: {len(data):,}")
          print(f"  - Mean speed: {data['SOG'].mean():.1f} knots")
          print(f"  - Median speed: {data['SOG'].median():.1f} knots")
          print(f"  - Time range: {data['BaseDateTime'].min()} to {data['BaseDateTime'].max()}")
      else:
          print(f"  - No data available")

  print(f"\nMemory optimization:")
  print(f"- Reading by chunks of {chunk_size:,} lines (using pyarrow row groups)")
  print(f"- Sampling: max {max_points_1h:,}/1hour, {max_points_1d:,}/1day, {max_points_1m:,}/1month")
  print(f"- Loaded columns: {columns_needed}")
