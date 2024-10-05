import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from geopy.distance import geodesic
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, concatenate, Flatten, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

def haversine_loss(y_true, y_pred):
    """
    Custom loss function to calculate the mean Haversine distance between predicted and true coordinates.
    """
    # Convert degrees to radians manually
    y_true_rad = y_true * (np.pi / 180.0)
    y_pred_rad = y_pred * (np.pi / 180.0)

    lat_true = y_true_rad[:, 0]
    lon_true = y_true_rad[:, 1]
    lat_pred = y_pred_rad[:, 0]
    lon_pred = y_pred_rad[:, 1]

    # Haversine formula components
    dlat = lat_pred - lat_true
    dlon = lon_pred - lon_true

    sin_dlat = tf.math.sin(dlat / 2)
    sin_dlon = tf.math.sin(dlon / 2)

    a = sin_dlat**2 + tf.math.cos(lat_true) * tf.math.cos(lat_pred) * sin_dlon**2

    # Clip 'a' to prevent invalid values due to floating-point errors
    a = tf.clip_by_value(a, 0.0, 1.0)

    c = 2 * tf.math.atan2(tf.math.sqrt(a), tf.math.sqrt(1.0 - a))

    radius = 6371.0  # Earth's radius in kilometers
    distance = radius * c

    # Return the mean distance as the loss
    return K.mean(distance)

def prepare_sequences(data, feature_cols, target_cols, sequence_length):
    X, y, vessel_types = [], [], []
    data_values = data[feature_cols + target_cols + ['vessel_type_encoded']].values
    for i in range(sequence_length, len(data_values) + 1):
        X.append(data_values[i - sequence_length:i, :len(feature_cols)])
        y.append(data_values[i - 1, len(feature_cols):len(feature_cols)+len(target_cols)])
        vessel_types.append(data_values[i - 1, -1])  # Vessel type code at the last time step
    return np.array(X), np.array(y), np.array(vessel_types)

def prepare_test_sequences(ais_test, ais_train, features, sequence_length, scaler):
    X_test, vessel_test = [], []
    for idx, row in ais_test.iterrows():
        vessel_id = row['vesselId']
        test_time = pd.to_datetime(row['time'])
        
        # Get historical data for this vessel
        vessel_history = ais_train[
            (ais_train['vesselId'] == vessel_id) &
            (ais_train['time'] < test_time)
        ].sort_values(by='time')
        
        # Select the last `sequence_length` entries
        vessel_history = vessel_history.iloc[-sequence_length:].copy()
        
        # If not enough history, pad with zeros or appropriate values
        if len(vessel_history) < sequence_length:
            # Create a padding DataFrame with the necessary columns
            required_columns = vessel_history.columns
            padding = pd.DataFrame(0, index=range(sequence_length - len(vessel_history)), columns=required_columns)
            vessel_history = pd.concat([padding, vessel_history], ignore_index=True)
        
        # Apply same transformations as in training
        # Extract time-related features
        vessel_history['time'] = pd.to_datetime(vessel_history['time'])
        vessel_history['hour'] = vessel_history['time'].dt.hour
        vessel_history['day_of_week'] = vessel_history['time'].dt.dayofweek

        # Cyclical encoding for hour and day_of_week
        vessel_history['hour_sin'] = np.sin(2 * np.pi * vessel_history['hour'] / 24)
        vessel_history['hour_cos'] = np.cos(2 * np.pi * vessel_history['hour'] / 24)
        vessel_history['day_sin'] = np.sin(2 * np.pi * vessel_history['day_of_week'] / 7)
        vessel_history['day_cos'] = np.cos(2 * np.pi * vessel_history['day_of_week'] / 7)
    
        # Handle missing values
        vessel_history[features] = vessel_history[features].ffill().fillna(0)

        # Normalize features using the same scaler
        vessel_history[features] = scaler.transform(vessel_history[features])
        
        X_test.append(vessel_history[features].values)
        vessel_test.append(row['vessel_type_encoded'])  # Ensure vessel type is encoded appropriately
    return np.array(X_test), np.array(vessel_test)

def prepare_test_data(ais_test, vessels, vessel_type_categories):
    merged_test = ais_test.copy()

    # Merge AIS test data with vessel data
    merged_test = pd.merge(
        merged_test,
        vessels[['vesselId', 'vesselType']],
        on='vesselId',
        how='left'
    )

    # Handle missing vesselType
    merged_test['vesselType'].fillna('-1', inplace=True)

    # Encode vessel type as numeric categories (ensure consistency with training data)
    merged_test['vesselType'] = pd.Categorical(merged_test['vesselType'], categories=vessel_type_categories)
    merged_test['vesselType'] = merged_test['vesselType'].fillna('-1')
    merged_test['vessel_type_encoded'] = merged_test['vesselType'].cat.codes

    return merged_test

def main() -> None:
    # Load AIS and optional datasets
    ais_train = pd.read_csv('ais_train.csv', sep='|')
    ais_test = pd.read_csv('ais_test.csv')  # Assuming comma-separated
    vessels = pd.read_csv('vessels.csv', sep='|')
    ports = pd.read_csv('ports.csv', sep='|')
    schedules = pd.read_csv('schedules_to_may_2024.csv', sep='|')

    # Verify column names
    print("AIS Train Columns:", ais_train.columns)
    print("AIS Test Columns:", ais_test.columns)
    print("Vessels Columns:", vessels.columns)
    print("Ports Columns:", ports.columns)

    # Convert 'time' columns to datetime in ais_train and ais_test
    ais_train['time'] = pd.to_datetime(ais_train['time'])
    ais_test['time'] = pd.to_datetime(ais_test['time'])

    # Merge AIS data with vessel data to get vessel types and other info
    merged_data = pd.merge(
        ais_train,
        vessels[['vesselId', 'vesselType']],
        on='vesselId',
        how='left'
    )

    # Handle missing vesselType
    merged_data['vesselType'] = merged_data['vesselType'].astype('object')
    merged_data['vesselType'] = merged_data['vesselType'].fillna('-1')
    
    # Extract time-related features
    merged_data['hour'] = merged_data['time'].dt.hour
    merged_data['day_of_week'] = merged_data['time'].dt.dayofweek

    # Cyclical encoding for hour and day_of_week
    merged_data['hour_sin'] = np.sin(2 * np.pi * merged_data['hour']/24)
    merged_data['hour_cos'] = np.cos(2 * np.pi * merged_data['hour']/24)
    merged_data['day_sin'] = np.sin(2 * np.pi * merged_data['day_of_week']/7)
    merged_data['day_cos'] = np.cos(2 * np.pi * merged_data['day_of_week']/7)

    # Create future latitude and longitude columns by shifting the original values
    merged_data = merged_data.sort_values(by=['vesselId', 'time'])
    merged_data['future_latitude'] = merged_data.groupby('vesselId')['latitude'].shift(-1)
    merged_data['future_longitude'] = merged_data.groupby('vesselId')['longitude'].shift(-1)

    # Drop rows with missing future positions
    merged_data.dropna(subset=['future_latitude', 'future_longitude'], inplace=True)

    # Encode vessel type as numeric categories
    merged_data['vesselType'] = merged_data['vesselType'].astype('category')
    vessel_type_categories = merged_data['vesselType'].cat.categories.tolist()
    if 'Unknown' not in vessel_type_categories:
        vessel_type_categories.append('Unknown')
    merged_data['vesselType'].cat.set_categories(vessel_type_categories)
    merged_data['vessel_type_encoded'] = merged_data['vesselType'].cat.codes

    # Include current latitude and longitude in features
    features = ['latitude', 'longitude', 'speed', 'heading',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos']

    # Check for 'speed' and 'heading' in columns, fill if missing
    if 'speed' not in merged_data.columns:
        merged_data['speed'] = 0
    if 'heading' not in merged_data.columns:
        merged_data['heading'] = 0

    # Handle missing values
    merged_data[features] = merged_data[features].ffill().fillna(0)

    # Normalize numerical features
    scaler = MinMaxScaler()
    merged_data[features] = scaler.fit_transform(merged_data[features])

    # Define the target columns
    target = ['future_latitude', 'future_longitude']

    # Use early data as training, later data as validation
    train_data, val_data = train_test_split(merged_data, test_size=0.2, shuffle=False)

    sequence_length = 5
    X_train, y_train, vessel_train = prepare_sequences(train_data, features, target, sequence_length)
    X_val, y_val, vessel_val = prepare_sequences(val_data, features, target, sequence_length)
 
    # Prepare vessel type for embedding
    vessel_type_max = merged_data['vessel_type_encoded'].max() + 1  # For input_dim in Embedding layer

    # Build the model with embedding and additional layers
    # Inputs
    feature_input = Input(shape=(sequence_length, len(features)), name='feature_input')
    vessel_input = Input(shape=(1,), name='vessel_input')

    # Vessel type embedding
    vessel_embedding = Embedding(input_dim=vessel_type_max, output_dim=10, input_length=1)(vessel_input)
    vessel_embedding = Flatten()(vessel_embedding)
    vessel_embedding = RepeatVector(sequence_length)(vessel_embedding)

    # Concatenate feature inputs and vessel embedding
    lstm_input = concatenate([feature_input, vessel_embedding], axis=-1)

    # LSTM layers
    x = LSTM(128, return_sequences=True)(lstm_input)
    x = Dropout(0.2)(x)
    x = LSTM(64)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(2)(x)  # Predict latitude and longitude

    # Define the model
    model = Model(inputs=[feature_input, vessel_input], outputs=output)

    # Compile the model with custom loss function
    # model.compile(optimizer='adam', loss=haversine_loss)
    model.compile(optimizer='adam', loss='mean_absolute_error')

    model.summary()

    # Train the model with callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)

    model.fit(
        [X_train, vessel_train],
        y_train,
        epochs=1,
        batch_size=64,
        validation_data=([X_val, vessel_val], y_val),
        callbacks=[early_stopping, reduce_lr],
    )

    # Evaluate on validation set
    predictions_val = model.predict([X_val, vessel_val])
    val_data = val_data.iloc[sequence_length - 1:].copy()  # Align with predictions
    val_data['pred_latitude'] = predictions_val[:, 0]
    val_data['pred_longitude'] = predictions_val[:, 1]

    val_data['error_distance'] = val_data.apply(lambda row: calculate_distance(
        row['future_latitude'], row['future_longitude'], row['pred_latitude'], row['pred_longitude']), axis=1)

    mean_error_distance = val_data['error_distance'].mean()
    print(f'Mean Geodetic Error on Validation Set: {mean_error_distance:.2f} km')

    # Prepare test data
    merged_test = prepare_test_data(ais_test, vessels, vessel_type_categories)

    # Apply same transformations to test data
    merged_test['hour'] = merged_test['time'].dt.hour
    merged_test['day_of_week'] = merged_test['time'].dt.dayofweek

    # Cyclical encoding for hour and day_of_week
    merged_test['hour_sin'] = np.sin(2 * np.pi * merged_test['hour']/24)
    merged_test['hour_cos'] = np.cos(2 * np.pi * merged_test['hour']/24)
    merged_test['day_sin'] = np.sin(2 * np.pi * merged_test['day_of_week']/7)
    merged_test['day_cos'] = np.cos(2 * np.pi * merged_test['day_of_week']/7)

    # Include current latitude and longitude in features
    features = ['latitude', 'longitude', 'speed', 'heading',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos']

    # Prepare test sequences using historical data
    X_test, vessel_test = prepare_test_sequences(merged_test, merged_data, features, sequence_length, scaler)

    # Make predictions on test set
    predictions_test = model.predict([X_test, vessel_test])

    # Prepare submission file
    submission_df = ais_test.copy()
    submission_df['latitude'] = predictions_test[:, 0]
    submission_df['longitude'] = predictions_test[:, 1]
    submission_df = submission_df[['ID', 'longitude', 'latitude']]
    submission_df.to_csv('submission.csv', index=False)
    print("Submission file 'submission.csv' has been created.")

if __name__ == "__main__":
    main()
