
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import plotly.graph_objects as go
from tqdm import tqdm  # Import tqdm

# Interpolation mode:
# 0 - Uninterpolated data
# 1 - Cubic interpolation on all data points based on vessel ID and sorted by time
# 2 - New interpolation by segments
interpolation_mode = 2  # Set the desired interpolation mode here

# Read ais_train.csv
ais_train = pd.read_csv("ais_train.csv", sep='|')

# Temporal features
ais_train['time'] = pd.to_datetime(ais_train['time'])
ais_train['elapsed_time'] = (ais_train['time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# Filter out unrealistic speeds
ais_train = ais_train[ais_train['sog'] < 25]

# Map 'navstat' values
ais_train['navstat'] = ais_train['navstat'].replace(8, 0)  # Under way sailing -> Under way using engine
ais_train = ais_train[~((ais_train['navstat'].isin([1, 5])) & (ais_train['sog'] > 0))]
ais_train = ais_train[~((ais_train['navstat'] == 2) & (ais_train['sog'] > 5))]

# Create 'status' column: 'moving' or 'moored'
ais_train['status'] = 'moving'
ais_train.loc[ais_train['navstat'].isin([1, 5]), 'status'] = 'moored'

# Assign Trajectory IDs based on status changes (only needed for interpolation_mode == 2)
if interpolation_mode == 2:
    def assign_trajectory_ids(df):
        df = df.sort_values('elapsed_time')
        df['status_shift'] = df['status'] != df['status'].shift()
        df['trajectory_id'] = df['status_shift'].cumsum()
        df = df.drop('status_shift', axis=1)
        return df

    ais_train = ais_train.groupby('vesselId', group_keys=False).apply(assign_trajectory_ids).reset_index(drop=True)

# Define helper functions
def haversine_distance(lat1, lon1, lat2, lon2):
    # Earth radius in kilometers
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon / 2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
    dlon_rad = np.radians(lon2 - lon1)
    x = np.sin(dlon_rad) * np.cos(lat2_rad)
    y = np.cos(lat1_rad)*np.sin(lat2_rad) - np.sin(lat1_rad)*np.cos(lat2_rad)*np.cos(dlon_rad)
    initial_bearing = np.arctan2(x, y)
    return (np.degrees(initial_bearing) + 360) % 360

# Initialize list to collect processed data frames
processed_trajectories = []

# Loop over vessels with tqdm progress bar
vessel_ids = ais_train['vesselId'].unique()
print("Processing vessels...")
for vessel_id in tqdm(vessel_ids, desc='Vessels'):
    vessel_data = ais_train[ais_train['vesselId'] == vessel_id].sort_values('elapsed_time').reset_index(drop=True)

    if interpolation_mode == 0:
        # Uninterpolated data
        processed_trajectories.append(vessel_data)
    elif interpolation_mode == 1:
        # Cubic interpolation on all data points based on vessel ID and sorted by time
        # Ensure at least two data points
        if len(vessel_data) < 2:
            continue

        # Prepare data for interpolation
        times = vessel_data['elapsed_time'].values
        latitudes = vessel_data['latitude'].values
        longitudes = vessel_data['longitude'].values

        # Remove duplicates in times
        times, unique_indices = np.unique(times, return_index=True)
        latitudes = latitudes[unique_indices]
        longitudes = longitudes[unique_indices]

        if len(times) < 2:
            continue

        # Convert lat/lon to radians
        lat_rad = np.radians(latitudes)
        lon_rad = np.radians(longitudes)

        # Convert to 3D Cartesian coordinates
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)

        # Create new time points for interpolation (every hour)
        start_time = times.min()
        end_time = times.max()
        new_times = np.arange(start_time, end_time + 1, 3600)  # Every hour in seconds
        # Ensure new_times are within the original times
        new_times = new_times[(new_times >= times.min()) & (new_times <= times.max())]

        # Interpolate x, y, z using cubic splines
        cs_x = CubicSpline(times, x)
        cs_y = CubicSpline(times, y)
        cs_z = CubicSpline(times, z)

        x_interp = cs_x(new_times)
        y_interp = cs_y(new_times)
        z_interp = cs_z(new_times)

        # Normalize to unit sphere
        norm = np.sqrt(x_interp**2 + y_interp**2 + z_interp**2)
        x_interp /= norm
        y_interp /= norm
        z_interp /= norm

        # Convert back to lat/lon
        lat_interp = np.degrees(np.arcsin(z_interp))
        lon_interp = np.degrees(np.arctan2(y_interp, x_interp))

        # Handle longitude wrap-around
        lon_interp = (lon_interp + 360) % 360
        # Adjust longitudes > 180 to negative values (from -180 to 180)
        lon_interp[lon_interp > 180] -= 360

        # Create interpolated DataFrame
        interp_df = pd.DataFrame({
            'vesselId': vessel_id,
            'elapsed_time': new_times,
            'latitude': lat_interp,
            'longitude': lon_interp,
            'status': 'moving'  # Assume moving for interpolation
        })

        # Recalculate 'sog' and 'cog'
        lat_prev = np.roll(lat_interp, 1)
        lon_prev = np.roll(lon_interp, 1)
        time_prev = np.roll(new_times, 1)
        distances = haversine_distance(lat_prev, lon_prev, lat_interp, lon_interp)
        time_diffs = (new_times - time_prev) / 3600  # Convert time difference to hours
        time_diffs[0] = np.nan  # First element has no previous point
        sog = distances / time_diffs  # Speed in km/h
        cog = calculate_bearing(lat_prev, lon_prev, lat_interp, lon_interp)
        cog[0] = np.nan  # First element has no previous point

        # Assign 'sog' and 'cog' to interp_df
        interp_df['sog'] = sog
        interp_df['cog'] = cog

        # Drop the first row as it has NaN values
        interp_df = interp_df.iloc[1:].reset_index(drop=True)

        # Convert 'elapsed_time' back to 'time'
        interp_df['time'] = pd.to_datetime(interp_df['elapsed_time'], unit='s')

        # Append to processed_trajectories
        processed_trajectories.append(interp_df)
    elif interpolation_mode == 2:
        # Interpolation by segments
        # Loop over trajectory IDs
        trajectory_ids = vessel_data['trajectory_id'].unique()
        for trajectory_id in trajectory_ids:
            trajectory_data = vessel_data[vessel_data['trajectory_id'] == trajectory_id]
            status = trajectory_data['status'].iloc[0]
            if status == 'moving':
                # Perform cubic spline interpolation in 3D Cartesian coordinates
                # Ensure at least two data points
                if len(trajectory_data) < 2:
                    continue
                # Prepare data for interpolation
                times = trajectory_data['elapsed_time'].values
                latitudes = trajectory_data['latitude'].values
                longitudes = trajectory_data['longitude'].values

                # Remove duplicates in times
                times, unique_indices = np.unique(times, return_index=True)
                latitudes = latitudes[unique_indices]
                longitudes = longitudes[unique_indices]

                if len(times) < 2:
                    continue

                # Convert lat/lon to radians
                lat_rad = np.radians(latitudes)
                lon_rad = np.radians(longitudes)

                # Convert to 3D Cartesian coordinates
                x = np.cos(lat_rad) * np.cos(lon_rad)
                y = np.cos(lat_rad) * np.sin(lon_rad)
                z = np.sin(lat_rad)

                # Create new time points for interpolation (every hour)
                start_time = times.min()
                end_time = times.max()
                new_times = np.arange(start_time, end_time + 1, 3600)  # Every hour in seconds
                # Ensure new_times are within the original times
                new_times = new_times[(new_times >= times.min()) & (new_times <= times.max())]

                # Interpolate x, y, z using cubic splines
                cs_x = CubicSpline(times, x)
                cs_y = CubicSpline(times, y)
                cs_z = CubicSpline(times, z)

                x_interp = cs_x(new_times)
                y_interp = cs_y(new_times)
                z_interp = cs_z(new_times)

                # Normalize to unit sphere
                norm = np.sqrt(x_interp**2 + y_interp**2 + z_interp**2)
                x_interp /= norm
                y_interp /= norm
                z_interp /= norm

                # Convert back to lat/lon
                lat_interp = np.degrees(np.arcsin(z_interp))
                lon_interp = np.degrees(np.arctan2(y_interp, x_interp))

                # Handle longitude wrap-around
                lon_interp = (lon_interp + 360) % 360
                # Adjust longitudes > 180 to negative values (from -180 to 180)
                lon_interp[lon_interp > 180] -= 360

                # Create interpolated DataFrame
                interp_df = pd.DataFrame({
                    'vesselId': vessel_id,
                    'trajectory_id': trajectory_id,
                    'elapsed_time': new_times,
                    'latitude': lat_interp,
                    'longitude': lon_interp,
                    'status': status
                })

                # Recalculate 'sog' and 'cog'
                lat_prev = np.roll(lat_interp, 1)
                lon_prev = np.roll(lon_interp, 1)
                time_prev = np.roll(new_times, 1)
                distances = haversine_distance(lat_prev, lon_prev, lat_interp, lon_interp)
                time_diffs = (new_times - time_prev) / 3600  # Convert time difference to hours
                time_diffs[0] = np.nan  # First element has no previous point
                sog = distances / time_diffs  # Speed in km/h
                cog = calculate_bearing(lat_prev, lon_prev, lat_interp, lon_interp)
                cog[0] = np.nan  # First element has no previous point

                # Assign 'sog' and 'cog' to interp_df
                interp_df['sog'] = sog
                interp_df['cog'] = cog

                # Drop the first row as it has NaN values
                interp_df = interp_df.iloc[1:].reset_index(drop=True)

                # Convert 'elapsed_time' back to 'time'
                interp_df['time'] = pd.to_datetime(interp_df['elapsed_time'], unit='s')

                # Append to processed_trajectories
                processed_trajectories.append(interp_df)
            else:
                # Handle 'moored' segments
                # Resample or keep as is (e.g., resample to daily data)
                trajectory_data = trajectory_data.set_index('time')
                moored_daily = trajectory_data.resample('1D').first().reset_index()
                if moored_daily.empty:
                    continue
                moored_daily['vesselId'] = vessel_id
                moored_daily['trajectory_id'] = trajectory_id
                # Set 'sog' to zero
                moored_daily['sog'] = 0.0
                moored_daily['cog'] = np.nan
                # Append to processed_trajectories
                processed_trajectories.append(moored_daily[['vesselId', 'trajectory_id', 'elapsed_time', 'latitude', 'longitude', 'sog', 'cog', 'status', 'time']])
    else:
        print(f"Invalid interpolation_mode: {interpolation_mode}")
        break

# Combine all processed data
ais_train_processed = pd.concat(processed_trajectories, ignore_index=True)

# Now, proceed with further processing or analysis
# For example, we can visualize the data

# Let's pick a vessel to plot
# Find a vessel that crosses the International Date Line
def find_vessel_crossing_idl(df):
    for vessel_id in df['vesselId'].unique():
        vessel_data = df[df['vesselId'] == vessel_id].sort_values('elapsed_time')
        lon_diff = np.abs(np.diff(vessel_data['longitude'].values))
        if np.any(lon_diff > 180):
            return vessel_id
    return None

#vessel_id_to_plot = find_vessel_crossing_idl(ais_train_processed)
vessel_id_to_plot = "61e9f3bfb937134a3c4bfe9f"

if vessel_id_to_plot is None:
    print("No vessel crossing the International Date Line found.")
    # Choose a vessel to plot
    vessel_id_to_plot = ais_train_processed['vesselId'].unique()[0]
    print(f"Plotting vessel: {vessel_id_to_plot}")
else:
    print(f"Vessel crossing the IDL found: {vessel_id_to_plot}")

vessel_data = ais_train_processed[ais_train_processed['vesselId'] == vessel_id_to_plot]

# Plotting code
import plotly.express as px

fig = px.scatter_mapbox(
    vessel_data,
    lat="latitude",
    lon="longitude",
    hover_name="trajectory_id" if interpolation_mode == 2 else 'vesselId',
    hover_data=["status", "sog", "cog"],
    color="trajectory_id" if interpolation_mode == 2 else None,
    zoom=3,
)

fig.update_layout(mapbox_style="open-street-map")
fig.update_traces(mode='lines+markers')
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

