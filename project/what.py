import pandas as pd
import logging
import sys
from typing import Tuple
from colorlog import ColoredFormatter
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Bidirectional, GRU, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
import numpy as np
import tensorflow as tf
from rtree import index
from shapely.geometry import Point
import time
from tqdm import tqdm


# Configure the colorful logger
def setup_logger() -> logging.Logger:
    """Set up a colorful logger for the pipeline.
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("ML_Pipeline")
    logger.setLevel(logging.INFO)
    
    # Define log colors for different levels
    formatter = ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s",
        datefmt=None,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'white',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        }
    )
    
    # Stream handler for console output
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger

# Initialize the logger
logger = setup_logger()

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the AIS and auxiliary datasets.

    Returns:
        A tuple containing the loaded DataFrames:
        (ais_train, ais_test, vessels, ports, schedules).
    """
    logger.info("Loading datasets.")
    ais_train = pd.read_csv('ais_train.csv', sep='|')
    ais_test = pd.read_csv('ais_test.csv')
    vessels = pd.read_csv('vessels.csv', sep='|')
    ports = pd.read_csv('ports.csv', sep='|')
    schedules = pd.read_csv('schedules_to_may_2024.csv', sep='|')
    logger.info("Datasets loaded successfully.")
    return ais_train, ais_test, vessels, ports, schedules



def prepare_data(
    ais_train: pd.DataFrame,
    vessels: pd.DataFrame, 
    ports: pd.DataFrame, 
    schedules: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """Prepare the data for the LSTM model, including additional time-based, vessel-specific, and port proximity features.
    
    Args:
        ais_train: DataFrame containing AIS training data.
        
    Returns:
        Tuple containing the feature array (X), target array (y), and the fitted scaler.
    """
    logger.info("Preparing data for the model.")
    
    # Convert the 'time' column to datetime format for feature extraction
    ais_train['time'] = pd.to_datetime(ais_train['time'])
    
    # Extract hour of the day and day of the week as new features
    ais_train['hour_of_day'] = ais_train['time'].dt.hour
    ais_train['day_of_week'] = ais_train['time'].dt.dayofweek

    # Calculate the time elapsed since the first recorded entry for each vessel
    ais_train['time_elapsed'] = (ais_train['time'] - ais_train['time'].min()).dt.total_seconds()
    
    # Compute the sine and cosine of the course over ground (cog) to represent direction
    ais_train['cog_sin'] = np.sin(np.deg2rad(ais_train['cog']))
    ais_train['cog_cos'] = np.cos(np.deg2rad(ais_train['cog']))

    # Categorize the speed of the vessel
    ais_train['speed_category'] = pd.cut(ais_train['sog'], bins=[-1, 5, 15, np.inf], labels=[0, 1, 2])

    # Merge vessel-specific information into ais_train data
    ais_train = ais_train.merge(vessels[['vesselId', 'maxSpeed', 'length', 'yearBuilt']], on='vesselId', how='left')

    # Fill missing values in vessel-specific data with appropriate defaults
    ais_train['maxSpeed'] = ais_train['maxSpeed'].fillna(ais_train['maxSpeed'].mean())

    # Measure time to create the R-tree index for ports
    start_time = time.time()
    port_idx = index.Index()
    for idx, row in ports.iterrows():
        port_idx.insert(idx, (row['longitude'], row['latitude'], row['longitude'], row['latitude']))
    end_time = time.time()
    logger.info(f"Time to build R-tree index for ports: {end_time - start_time:.2f} seconds")

    # Measure time to calculate the nearest port for each vessel with progress and ETA
    total_vessels = len(ais_train)
    start_time = time.time()
    closest_ports = []

    for _, vessel in tqdm(ais_train.iterrows(), total=total_vessels, desc="Calculating nearest ports", unit="vessel"):
        point = Point(vessel['longitude'], vessel['latitude'])
        nearest_port_idx = list(port_idx.nearest((vessel['longitude'], vessel['latitude'], vessel['longitude'], vessel['latitude']), 1))[0]
        closest_port = ports.iloc[nearest_port_idx]
        distance_to_port = point.distance(Point(closest_port['longitude'], closest_port['latitude']))
        closest_ports.append(distance_to_port)

    end_time = time.time()
    logger.info(f"Time to calculate closest port for all vessels: {end_time - start_time:.2f} seconds for {total_vessels} vessels")

    # Add the calculated distances to the AIS data
    ais_train['distance_to_nearest_port'] = closest_ports

    speed_threshold = 5.0  # knots
    distance_threshold = 1.0  # kilometers

    # Identify if the vessel is anchored
    ais_train['anchored'] = (ais_train['sog'] < speed_threshold) & (ais_train['distance_to_nearest_port'] < distance_threshold)


    # Extract the relevant features, including the new ones
    features = ais_train[['latitude', 'longitude', 'sog', 'cog_sin', 'cog_cos', 'hour_of_day', 'day_of_week', 
                      'time_elapsed', 'speed_category', 'maxSpeed', 'distance_to_nearest_port', 'anchored']].values
    target = ais_train[['latitude', 'longitude']].shift(-1).ffill().values

    # Normalize features
    feature_scaler = MinMaxScaler()
    features_scaled = feature_scaler.fit_transform(features)
    
    # Reshape for LSTM input: (samples, timesteps, features)
    X = features_scaled.reshape((features_scaled.shape[0], 1, features_scaled.shape[1]))
    
    # Normalize target data (latitude and longitude)
    target_scaler = MinMaxScaler()
    y = target_scaler.fit_transform(target)    
    
    logger.info("Data preparation complete.")
    return X, y, feature_scaler, target_scaler

def geodesic_loss(y_true, y_pred):
    """Calculate the Haversine distance between true and predicted coordinates.
    
    Args:
        y_true: Tensor of true coordinates (latitude, longitude).
        y_pred: Tensor of predicted coordinates (latitude, longitude).
    
    Returns:
        Tensor representing the geodesic distance (Haversine distance) between the true and predicted points.
    """
    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    
    # Split the latitude and longitude into separate tensors
    lat_true, lon_true = tf.split(y_true, num_or_size_splits=2, axis=1)
    lat_pred, lon_pred = tf.split(y_pred, num_or_size_splits=2, axis=1)
    
    # Convert degrees to radians manually
    lat_true = lat_true * tf.constant(np.pi / 180.0)
    lon_true = lon_true * tf.constant(np.pi / 180.0)
    lat_pred = lat_pred * tf.constant(np.pi / 180.0)
    lon_pred = lon_pred * tf.constant(np.pi / 180.0)
    
    # Compute the differences between true and predicted coordinates
    dlat = lat_pred - lat_true
    dlon = lon_pred - lon_true
    
    # Haversine formula
    a = tf.square(tf.sin(dlat / 2)) + tf.cos(lat_true) * tf.cos(lat_pred) * tf.square(tf.sin(dlon / 2))
    c = 2 * tf.atan2(tf.sqrt(a), tf.sqrt(1 - a))
    distance = R * c
    
    return tf.reduce_mean(distance)

def build_model(input_shape: Tuple[int, int]) -> Sequential:
    """Build the LSTM model.
    
    Args:
        input_shape: Shape of the input data (timesteps, features).
        
    Returns:
        Compiled LSTM model.
    """
    logger.info("Building the LSTM model.")
 
    model = Sequential()
    model.add(Input(shape=input_shape))  # Use Input layer to specify the shape
    lstm = True
    if lstm:
        model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
        model.add(BatchNormalization())  # Batch normalization after the first LSTM layer
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(units=100)))
        model.add(BatchNormalization())  # Batch normalization after the first LSTM layer
        model.add(Dropout(0.2))
    else:
        model.add(GRU(units=100, return_sequences=True))
        model.add(BatchNormalization())  # Batch normalization after the first LSTM layer
        model.add(Dropout(0.2))
        model.add(GRU(units=100))
        model.add(BatchNormalization())  # Batch normalization after the first LSTM layer
    model.add(Dense(units=2))  # Output: latitude and longitude
    model.compile(optimizer='adam', loss=geodesic_loss)
    logger.info("Model built successfully.")
    return model

def train_model(model: Sequential, X_train: np.ndarray, y_train: np.ndarray) -> None:
    """Train the LSTM model.
    
    Args:
        model: The LSTM model to train.
        X_train: Training features.
        y_train: Training targets.
    """
    logger.info("Starting model training.")
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2,callbacks=[early_stopping, reduce_lr])
    logger.info("Model training complete.")

def generate_submission(model: Sequential, ais_test: pd.DataFrame, feature_scaler: MinMaxScaler, target_scaler: MinMaxScaler) -> None:
    """Generate a submission file with the predicted vessel positions.
    
    Args:
        model: Trained LSTM model.
        ais_test: DataFrame containing AIS test data.
        feature_scaler: Scaler used to normalize the features.
        target_scaler: Scaler used to normalize the target coordinates.
    """
    logger.info("Generating predictions for the test set.")
    
    # Use the scaling_factor as the main feature from the test data
    test_features = ais_test[['scaling_factor']].values
    
    # Since the test data only has one feature, we need to adjust the input shape to match the model's expectation
    num_train_features = feature_scaler.n_features_in_
    test_features_padded = np.zeros((test_features.shape[0], num_train_features))
    test_features_padded[:, :test_features.shape[1]] = test_features

    # Normalize the padded test features to match the training data scale
    test_features_scaled = feature_scaler.transform(test_features_padded)
    X_test = test_features_scaled.reshape((test_features_scaled.shape[0], 1, test_features_scaled.shape[1]))
    
    # Make predictions using the model
    predictions = model.predict(X_test)
    
    # Inverse transform the predictions using the target scaler
    predictions = target_scaler.inverse_transform(predictions)
    
    # Create the submission DataFrame in the required format
    submission = pd.DataFrame({
        'ID': ais_test['ID'],
        'longitude_predicted': predictions[:, 1],
        'latitude_predicted': predictions[:, 0]
    })
    
    # Ensure that the submission file has exactly 51739 rows as required
    assert submission.shape[0] == 51739, "The submission file must have exactly 51739 rows."
    
    # Save the predictions to submission.csv
    submission.to_csv('submission.csv', index=False)
    logger.info("Submission file saved as submission.csv.")

def main() -> None:
    """Main function to run the machine learning pipeline."""
    logger.info("Starting the machine learning pipeline.")
    ais_train, ais_test, vessels, ports, schedules = load_data()

    # Verify column names
    print(f"AIS Train Columns:\n{ais_train.head()}")
    print(f"AIS Test Columns:\n{ais_test.columns}")
    print(f"Vessels Columns:\n{vessels.head()}")
    print(f"Ports Columns:\n{ports.head()}")
    print(f"Schedules:\n{schedules.head()}")

    # Prepare the data
    X_train, y_train, feature_scaler, target_scaler = prepare_data(ais_train, vessels, ports, schedules)

    # Build and train the LSTM model
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    train_model(model, X_train, y_train)

    # Generate the submission file
    generate_submission(model, ais_test, feature_scaler, target_scaler)
    
    logger.info("Machine learning pipeline finished successfully.")

if __name__ == "__main__":
    main()
