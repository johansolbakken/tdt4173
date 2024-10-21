import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU Available:", tf.config.list_physical_devices('GPU'))

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Enable memory growth for GPU
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

"""
    Load and Preprocess Data
"""

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

# Merge with vessel data
vessels = pd.read_csv("vessels.csv", sep='|')[['shippingLineId', 'vesselId']]
vessels['new_id'] = range(len(vessels))
vessel_id_to_new_id = dict(zip(vessels['vesselId'], vessels['new_id']))
ais_train = pd.merge(ais_train, vessels, on='vesselId', how='left')

# Merge port data
ports = pd.read_csv("ports.csv", sep='|')[['portId', 'latitude', 'longitude']]
ports = ports.rename(columns={'latitude': 'port_latitude', 'longitude': 'port_longitude'})
ais_train = pd.merge(ais_train, ports, on='portId', how='left')
ais_train = ais_train[~ais_train['portId'].isnull()] # remove with null ports

def haversine_distance(lat1, lon1, lat2, lon2):
    # Earth radius in nautical miles
    R = 3440.065
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
    diff_long = np.radians(lon2 - lon1)
    x = np.sin(diff_long) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - (np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(diff_long))
    initial_bearing = np.arctan2(x, y)
    return (np.degrees(initial_bearing) + 360) % 360

# Calculate features
print("distance to port calculation")
ais_train['distance_to_port'] = haversine_distance(
    ais_train['latitude'], ais_train['longitude'],
    ais_train['port_latitude'], ais_train['port_longitude']
)
print("bearing to port calculation")
ais_train['bearing_to_port'] = calculate_bearing(
    ais_train['latitude'], ais_train['longitude'],
    ais_train['port_latitude'], ais_train['port_longitude']
)

print("done calculating")

# Define input and target features
input_features = ['latitude', 'longitude', 'sog', 'cog', 'heading', 'elapsed_time', 'vesselId']
input_features.extend([col for col in ais_train.columns if 'day_of_week_' in col])
input_features.extend([col for col in ais_train.columns if 'hour_of_day_' in col])
navstat_columns = [col for col in ais_train.columns if col.startswith('navstat_')]
input_features.extend(navstat_columns)
input_features.extend(['distance_to_port', 'bearing_to_port'])
target_columns = ['latitude', 'longitude']

# Initialize scalers
scaler_input = MinMaxScaler()
scaler_output = MinMaxScaler()

# Scale input and output features
input_data = scaler_input.fit_transform(ais_train[input_features])
output_data = scaler_output.fit_transform(ais_train[target_columns])

# Add scaled features back to DataFrame
ais_train_scaled = ais_train.copy()
ais_train_scaled[input_features] = input_data
ais_train_scaled[target_columns] = output_data

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
X, y = create_sequences_per_vessel(ais_train_scaled, time_step)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)


"""
    Haversine loss
"""
def haversine_loss(y_true, y_pred):
    R = 6371.0  # Earth radius in kilometers
    pi_over_180 = K.constant(np.pi / 180.0, dtype='float64')

    # Ensure the tensors are of type float64 for precision
    lat_true = K.cast(y_true[:, 0], 'float64') * pi_over_180
    lon_true = K.cast(y_true[:, 1], 'float64') * pi_over_180
    lat_pred = K.cast(y_pred[:, 0], 'float64') * pi_over_180
    lon_pred = K.cast(y_pred[:, 1], 'float64') * pi_over_180

    # Compute differences
    dlat = lat_pred - lat_true
    dlon = lon_pred - lon_true

    # Haversine formula
    a = K.sin(dlat / 2)**2 + K.cos(lat_true) * K.cos(lat_pred) * K.sin(dlon / 2)**2
    c = 2 * tf.atan2(K.sqrt(a), K.sqrt(1 - a))
    distance = R * c

    # Return mean distance over the batch
    return K.mean(distance)

"""
    Define and Train the Model
"""

# Define the LSTM Model
model = Sequential()
model.add(LSTM(1024, return_sequences=True, input_shape=(time_step, X_train.shape[2])))
model.add(LSTM(512))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1]))

# Compile the model
model.compile(optimizer='adam', loss=haversine_loss)

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

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

# Ensure all columns in ais_test match those in ais_train
for col in input_features:
    if col not in ais_test.columns:
        ais_test[col] = 0

# One-hot encode 'navstat' in test data (if available)
# If 'navstat' is not available in test data, you may need to handle this accordingly

# Merge with last known positions from training data
# Get the last 'time_step' records for each vessel from training data
last_positions = ais_train_scaled.groupby('vesselId').apply(lambda x: x.sort_values('elapsed_time').tail(time_step))
last_positions = last_positions.reset_index(drop=True)

# Prepare sequences for each vessel in the test set
vessel_sequences = {}
for vessel_id in ais_test['vesselId'].unique():
    if vessel_id in last_positions['vesselId'].values:
        vessel_data = last_positions[last_positions['vesselId'] == vessel_id]
        seq = vessel_data[input_features].values
        if len(seq) < time_step:
            # Pad sequences if necessary
            seq = np.pad(seq, ((time_step - len(seq), 0), (0, 0)), mode='constant')
        vessel_sequences[vessel_id] = seq
    else:
        # If no data available, create a default sequence (e.g., zeros or mean values)
        seq = np.zeros((time_step, len(input_features)))
        vessel_sequences[vessel_id] = seq

# Create test sequences
X_test = []
for idx, row in ais_test.iterrows():
    vessel_id = row['vesselId']
    seq = vessel_sequences[vessel_id]
    X_test.append(seq)
X_test = np.array(X_test)

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform predictions
y_pred_inverse = scaler_output.inverse_transform(y_pred)

"""
    Prepare Submission File
"""

# Prepare submission
submission_df = pd.DataFrame({
    'ID': ais_test['ID'].values,
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
