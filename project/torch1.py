import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy
from scipy.interpolate import CubicSpline
from tqdm import tqdm

print("PyTorch version:", torch.__version__)

# Set device preference: MPS > CUDA > CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

"""
    Load and Preprocess Data
"""

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

# One-hot encode 'navstat'
ais_train = pd.get_dummies(ais_train, columns=['navstat'])

# Merge with vessel data
vessels = pd.read_csv("vessels.csv", sep='|')[['shippingLineId', 'vesselId']]
vessels['new_id'] = range(len(vessels))
vessel_id_to_new_id = dict(zip(vessels['vesselId'], vessels['new_id']))
ais_train = pd.merge(ais_train, vessels, on='vesselId', how='left')

# Merge port data
ports = pd.read_csv("ports.csv", sep='|')[['portId', 'latitude', 'longitude']]
ports = ports.rename(columns={'latitude': 'port_latitude', 'longitude': 'port_longitude'})
ais_train = pd.merge(ais_train, ports, on='portId', how='left')
ais_train = ais_train[~ais_train['portId'].isnull()]  # Remove rows with null ports

def haversine_distance(lat1, lon1, lat2, lon2):
    # Earth radius in nautical miles
    R = 3440.065
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
    diff_long = np.radians(lon2 - lon1)
    x = np.sin(diff_long) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - (np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(diff_long))
    initial_bearing = np.arctan2(x, y)
    return (np.degrees(initial_bearing) + 360) % 360

"""
    Cubic Spline Interpolation for Each Vessel
"""

# List to store processed trajectories
processed_trajectories = []

# Group data by vesselId
vessel_ids = ais_train['vesselId'].unique()
for vessel_id in tqdm(vessel_ids, desc="Interpolating Vessels"):

    vessel_data = ais_train[ais_train['vesselId'] == vessel_id].sort_values('elapsed_time')

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
    })

    # Assign 'portId' using merge_asof
    vessel_port_data = vessel_data[['elapsed_time', 'portId']].drop_duplicates().sort_values('elapsed_time')
    interp_df = pd.merge_asof(
        interp_df.sort_values('elapsed_time'),
        vessel_port_data,
        on='elapsed_time',
        direction='nearest'
    )

    # Recalculate 'sog' and 'cog'
    lat_prev = np.roll(lat_interp, 1)
    lon_prev = np.roll(lon_interp, 1)
    time_prev = np.roll(new_times, 1)
    distances = haversine_distance(lat_prev, lon_prev, lat_interp, lon_interp)
    time_diffs = (new_times - time_prev) / 3600  # Convert time difference to hours
    time_diffs[0] = np.nan  # First element has no previous point
    sog = distances / time_diffs  # Speed in knots
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

# Combine all interpolated data
ais_train_interpolated = pd.concat(processed_trajectories, ignore_index=True)

print("ais_train", len(ais_train))
print("ais_train_interpolated", len(ais_train_interpolated))

"""
    Continue Preprocessing with Interpolated Data
"""

# Temporal features
ais_train_interpolated['day_of_week'] = ais_train_interpolated['time'].dt.dayofweek
ais_train_interpolated['hour_of_day'] = ais_train_interpolated['time'].dt.hour
ais_train_interpolated = pd.get_dummies(ais_train_interpolated, columns=['day_of_week', 'hour_of_day'], drop_first=True)

# Handle cyclic features for 'cog'
ais_train_interpolated['cog_sin'] = np.sin(np.radians(ais_train_interpolated['cog']))
ais_train_interpolated['cog_cos'] = np.cos(np.radians(ais_train_interpolated['cog']))

# Merge with vessels and ports data
ais_train_interpolated = pd.merge(ais_train_interpolated, vessels, on='vesselId', how='left')
ais_train_interpolated = pd.merge(ais_train_interpolated, ports, on='portId', how='left')

# Calculate 'distance_to_port' and 'bearing_to_port'
ais_train_interpolated['distance_to_port'] = haversine_distance(
    ais_train_interpolated['latitude'], ais_train_interpolated['longitude'],
    ais_train_interpolated['port_latitude'], ais_train_interpolated['port_longitude']
)
ais_train_interpolated['bearing_to_port'] = calculate_bearing(
    ais_train_interpolated['latitude'], ais_train_interpolated['longitude'],
    ais_train_interpolated['port_latitude'], ais_train_interpolated['port_longitude']
)

# Define input and target features
input_features = [
    'latitude', 'longitude', 'sog', 'cog_sin', 'cog_cos', 'elapsed_time',
    'distance_to_port', 'bearing_to_port'
]
input_features.extend([col for col in ais_train_interpolated.columns if 'day_of_week_' in col])
input_features.extend([col for col in ais_train_interpolated.columns if 'hour_of_day_' in col])

target_columns = ['latitude', 'longitude']

# Initialize scalers
scaler_input = MinMaxScaler()
scaler_output = MinMaxScaler()

# Drop rows with NaN values in input features
ais_train_interpolated = ais_train_interpolated.dropna(subset=input_features + target_columns)

# Scale input and output features
input_data = scaler_input.fit_transform(ais_train_interpolated[input_features])
output_data = scaler_output.fit_transform(ais_train_interpolated[target_columns])

# Add scaled features back to DataFrame
ais_train_interpolated[input_features] = input_data
ais_train_interpolated[target_columns] = output_data

"""
    Create Sequences for Model Training
"""

# Function to create sequences per vessel
def create_sequences_per_vessel(df, time_steps):
    X, y = [], []
    vessel_ids = df['vesselId'].unique()
    for vessel_id in vessel_ids:
        vessel_data = df[df['vesselId'] == vessel_id].sort_values('elapsed_time')
        inputs = vessel_data[input_features].values
        targets = vessel_data[target_columns].values
        if len(inputs) < time_steps:
            continue  # Skip sequences shorter than time_steps
        for i in range(len(inputs) - time_steps):
            X.append(inputs[i:i + time_steps])
            y.append(targets[i + time_steps])
    return np.array(X), np.array(y)

# Create sequences
time_step = 10
X, y = create_sequences_per_vessel(ais_train_interpolated, time_step)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_val = torch.from_numpy(X_val).float()
y_val = torch.from_numpy(y_val).float()

# Create TensorDatasets and DataLoaders
batch_size = 128
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

"""
    Haversine Loss Function
"""

def haversine_loss(y_true, y_pred):
    R = 6371.0  # Earth radius in kilometers

    # Ensure constants are tensors of the same dtype and device as y_true
    pi_over_180 = torch.tensor(np.pi / 180.0, dtype=y_true.dtype, device=y_true.device)

    lat_true = y_true[:, 0] * pi_over_180
    lon_true = y_true[:, 1] * pi_over_180
    lat_pred = y_pred[:, 0] * pi_over_180
    lon_pred = y_pred[:, 1] * pi_over_180

    dlat = lat_pred - lat_true
    dlon = lon_pred - lon_true

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat_true) * torch.cos(lat_pred) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    distance = R * c

    # Return mean distance over the batch
    return torch.mean(distance)

"""
    Define and Train the Model
"""

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(LSTMModel, self).__init__()
        # Set bidirectional=True and adjust hidden sizes accordingly
        self.lstm1 = nn.LSTM(
            input_size, hidden_size1, batch_first=True, bidirectional=True
        )
        self.lstm2 = nn.LSTM(
            hidden_size1 * 2, hidden_size2, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(0.2)
        # Adjust the input size of the fully connected layer
        self.fc = nn.Linear(hidden_size2 * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.dropout(out[:, -1, :])  # Get the last time step
        out = self.fc(out)
        return out

# Initialize the model
input_size = X_train.shape[2]
hidden_size1 = 512
hidden_size2 = 256
output_size = y_train.shape[1]
model = LSTMModel(input_size, hidden_size1, hidden_size2, output_size).to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters())

# Training loop
num_epochs = 100
best_val_loss = float('inf')
patience = 5
counter = 0
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(num_epochs):
    # Training
    model.train()
    train_losses = []

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = haversine_loss(batch_y, outputs)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)
            loss = haversine_loss(batch_y, outputs)
            val_losses.append(loss.item())

    avg_val_loss = np.mean(val_losses)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

# Load best model weights
model.load_state_dict(best_model_wts)

"""
    Prepare Test Data and Make Predictions
"""

# Load test data
ais_test = pd.read_csv("ais_test.csv")
ais_test['time'] = pd.to_datetime(ais_test['time'])
ais_test['elapsed_time'] = (ais_test['time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
ais_test['new_id'] = ais_test['vesselId'].map(vessel_id_to_new_id)

ais_test['day_of_week'] = ais_test['time'].dt.dayofweek
ais_test['hour_of_day'] = ais_test['time'].dt.hour

# One-hot encode
ais_test = pd.get_dummies(ais_test, columns=['day_of_week', 'hour_of_day'], drop_first=True)

# Merge with vessels and ports data
ais_test = pd.merge(ais_test, vessels, on='vesselId', how='left')

# Ensure all columns in ais_test match those in input_features
for col in input_features:
    if col not in ais_test.columns:
        ais_test[col] = 0

# Scale the test data using the same scaler
input_data_test = scaler_input.transform(ais_test[input_features])
ais_test_scaled = ais_test.copy()
ais_test_scaled[input_features] = input_data_test

# Prepare sequences for each vessel in the test set
def create_sequences_for_test(df_train, df_test, time_steps):
    X_test = []
    test_ids = []
    for idx, row in df_test.iterrows():
        vessel_id = row['vesselId']
        current_time = row['elapsed_time']

        # Get the historical data for this vessel up to the current_time
        vessel_train_data = df_train[df_train['vesselId'] == vessel_id]
        vessel_test_data = df_test[df_test['vesselId'] == vessel_id]

        # Combine and sort
        vessel_data = pd.concat([vessel_train_data, vessel_test_data], ignore_index=True)
        vessel_data = vessel_data.sort_values('elapsed_time')

        # Select data up to the current_time (excluding the current row)
        historical_data = vessel_data[vessel_data['elapsed_time'] < current_time]

        # Get the last 'time_steps' entries
        historical_sequence = historical_data.tail(time_steps)[input_features].values

        if len(historical_sequence) < time_steps:
            # Pad with zeros if not enough historical data
            padding = np.zeros((time_steps - len(historical_sequence), len(input_features)))
            historical_sequence = np.vstack([padding, historical_sequence])

        X_test.append(historical_sequence)
        test_ids.append(row['ID'])  # Assuming 'ID' is unique per test row

    return np.array(X_test), test_ids


# Create sequences for each test row
X_test, test_ids = create_sequences_for_test(ais_train_interpolated, ais_test_scaled, time_step)

# Convert to PyTorch tensor
X_test = torch.from_numpy(X_test).float()

# Create a DataLoader for test data
test_dataset = TensorDataset(X_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Make predictions in batches
predictions = []

model.eval()
with torch.no_grad():
    for batch_X in test_loader:
        batch_X = batch_X[0].to(device)  # batch_X is a tuple
        outputs = model(batch_X)
        predictions.append(outputs.cpu().numpy())

# Concatenate all batch predictions
y_pred = np.concatenate(predictions, axis=0)

# Inverse transform predictions
y_pred_inverse = scaler_output.inverse_transform(y_pred)


# Prepare submission
submission_df = pd.DataFrame({
    'ID': test_ids,
    'longitude_predicted': y_pred_inverse[:, target_columns.index('longitude')],
    'latitude_predicted': y_pred_inverse[:, target_columns.index('latitude')]
})

# Ensure the submission file has the required columns
submission_df = submission_df[['ID', 'longitude_predicted', 'latitude_predicted']]

# Save submission file
submission_df.to_csv("submission.csv", index=False)

# Display submission
print(submission_df.head())
print(f"Submission DataFrame shape: {submission_df.shape}")

print(f"Number of predictions: {len(y_pred_inverse)}")
print(f"Number of test IDs: {len(test_ids)}")
assert len(y_pred_inverse) == len(test_ids), "Mismatch between predictions and test IDs"
