
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline

# Read ais_train.csv
ais_train = pd.read_csv("ais_train.csv", sep='|')

# Temporal features
ais_train['time'] = pd.to_datetime(ais_train['time'])
ais_train['elapsed_time'] = (ais_train['time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
ais_train['day_of_week'] = ais_train['time'].dt.dayofweek  # Monday=0, Sunday=6
ais_train['hour_of_day'] = ais_train['time'].dt.hour
ais_train = pd.get_dummies(ais_train, columns=['day_of_week', 'hour_of_day'], drop_first=True)

# Filter out unrealistic speeds
ais_train = ais_train[ais_train['sog'] < 25]

# Map 'navstat' values
ais_train['navstat'] = ais_train['navstat'].replace(8, 0)  # Under way sailing -> Under way using engine
ais_train = ais_train[~((ais_train['navstat'].isin([1, 5])) & (ais_train['sog'] > 0))]
ais_train = ais_train[~((ais_train['navstat'] == 2) & (ais_train['sog'] > 5))]

# One-hot encode 'navstat'
ais_train = pd.get_dummies(ais_train, columns=['navstat'])

# Split cyclic values into x and y
ais_train['cog_sin'] = np.sin(np.radians(ais_train['cog']))
ais_train['cog_cos'] = np.cos(np.radians(ais_train['cog']))
ais_train['heading_sin'] = np.sin(np.radians(ais_train['heading']))
ais_train['heading_cos'] = np.cos(np.radians(ais_train['heading']))

# Merge with vessel data
vessels = pd.read_csv("vessels.csv", sep='|')[['shippingLineId', 'vesselId']]
vessels['new_id'] = range(len(vessels))
vessel_id_to_new_id = dict(zip(vessels['vesselId'], vessels['new_id']))
ais_train = pd.merge(ais_train, vessels, on='vesselId', how='left')

# Create a variable called vesselId (choose a vessel ID from the dataset)
vesselId = ais_train['vesselId'].unique()[7]

# Filter the data for the selected vesselId and sort by 'time'
vessel_data = ais_train[ais_train['vesselId'] == vesselId].sort_values('time').reset_index(drop=True)

# Clean vessel_data
vessel_data = vessel_data.dropna(subset=['latitude', 'longitude'])

# Ensure that 'time' is datetime and sorted
vessel_data = vessel_data.sort_values('time')

# Convert 'time' to numeric format for interpolation (seconds since epoch)
vessel_data['time_numeric'] = vessel_data['time'].astype(np.int64) // 10**9

# Ensure that 'time_numeric' is strictly increasing
vessel_data = vessel_data.drop_duplicates(subset='time_numeric')
vessel_data = vessel_data.sort_values('time_numeric').reset_index(drop=True)

# Interpolation flag
interpolate = False  # Set to False to disable interpolation

if interpolate:
    # Create a new time index with daily frequency
    start_time = vessel_data['time'].min().normalize()
    end_time = vessel_data['time'].max().normalize()
    new_time_index = pd.date_range(start=start_time, end=end_time, freq='1D')

    # Convert 'new_time_index' to numeric format
    new_time_numeric = new_time_index.astype(np.int64) // 10**9

    # Prepare original data for interpolation
    original_times = vessel_data['time_numeric'].values
    original_latitudes = vessel_data['latitude'].values
    original_longitudes = vessel_data['longitude'].values

    # Convert latitude and longitude to radians
    lat_rad = np.radians(original_latitudes)
    lon_rad = np.radians(original_longitudes)

    # Convert to 3D Cartesian coordinates on a unit sphere
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    # Create cubic spline interpolators for x, y, z
    cs_x = CubicSpline(original_times, x)
    cs_y = CubicSpline(original_times, y)
    cs_z = CubicSpline(original_times, z)

    # Interpolate at new time points within the range of original_times
    t_min = original_times.min()
    t_max = original_times.max()
    valid_mask = (new_time_numeric >= t_min) & (new_time_numeric <= t_max)
    interp_times_valid = new_time_numeric[valid_mask]

    if len(interp_times_valid) == 0:
        print("No valid interpolation times within the range of original data.")
        plotting_data = vessel_data
    else:
        x_interp = cs_x(interp_times_valid)
        y_interp = cs_y(interp_times_valid)
        z_interp = cs_z(interp_times_valid)

        # Normalize the interpolated coordinates to lie on the unit sphere
        norm = np.sqrt(x_interp**2 + y_interp**2 + z_interp**2)
        x_interp /= norm
        y_interp /= norm
        z_interp /= norm

        # Convert back to latitude and longitude
        lat_interp = np.degrees(np.arcsin(z_interp))
        lon_interp = np.degrees(np.arctan2(y_interp, x_interp))

        # Create interpolated DataFrame
        vessel_data_interp = pd.DataFrame({
            'time_numeric': interp_times_valid,
            'latitude': lat_interp,
            'longitude': lon_interp
        })

        # Convert 'time_numeric' back to datetime
        vessel_data_interp['time'] = pd.to_datetime(vessel_data_interp['time_numeric'], unit='s')

        # Use the interpolated data for plotting
        plotting_data = vessel_data_interp.sort_values('time').reset_index(drop=True)
else:
    # If not interpolating, use the original data
    plotting_data = vessel_data.sort_values('time').reset_index(drop=True)

# Calculate time differences between consecutive points
plotting_data['time_diff'] = plotting_data['time'].shift(-1) - plotting_data['time']
plotting_data['time_diff_hours'] = plotting_data['time_diff'].dt.total_seconds() / 3600

# Calculate midpoints between consecutive points
plotting_data['mid_lat'] = (plotting_data['latitude'] + plotting_data['latitude'].shift(-1)) / 2
plotting_data['mid_lon'] = (plotting_data['longitude'] + plotting_data['longitude'].shift(-1)) / 2

# Create text labels for time differences
plotting_data['time_diff_label'] = plotting_data['time_diff_hours'].round(2).astype(str) + ' hrs'

# Remove the last row (as it has NaN values due to shifting)
plotting_data = plotting_data[:-1]

# Create the map figure
fig = go.Figure()

# Add the trajectory line
fig.add_trace(go.Scattermapbox(
    mode='lines+markers',
    lon=plotting_data['longitude'],
    lat=plotting_data['latitude'],
    marker=dict(size=6, color='blue'),
    line=dict(width=2, color='blue'),
    name='Trajectory'
))

# Add time difference annotations at midpoints
fig.add_trace(go.Scattermapbox(
    mode='text',
    lon=plotting_data['mid_lon'],
    lat=plotting_data['mid_lat'],
    text=plotting_data['time_diff_label'],
    textfont=dict(size=10, color='black'),
    textposition='top center',
    name='Time Difference'
))

fig.update_layout(
    mapbox={
        'style': "open-street-map",
        'zoom': 1 if interpolate else 4,
        'center': {'lon': plotting_data['longitude'].mean(), 'lat': plotting_data['latitude'].mean()}
    },
    title=f"Trajectory of vessel {vesselId} with Time Differences {'(Interpolated Daily)' if interpolate else ''}"
)

fig.show()

