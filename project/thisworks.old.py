import pandas as pd
import numpy as np
from geopy.distance import geodesic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from typing import List, Tuple
import logging
import sys
from colorlog import ColoredFormatter

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
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        }
    )
    
    # Stream handler for console output
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger

# Initialize the logger
logger = setup_logger()

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the geodesic distance between two latitude-longitude points.

    Args:
        lat1: Latitude of the first point.
        lon1: Longitude of the first point.
        lat2: Latitude of the second point.
        lon2: Longitude of the second point.

    Returns:
        The distance in kilometers between the two points.
    """
    distance = geodesic((lat1, lon1), (lat2, lon2)).kilometers
    return distance

def prepare_sequences(
    data: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare input and target sequences for training the LSTM model.

    Args:
        data: DataFrame containing the data.
        feature_cols: List of feature column names.
        target_cols: List of target column names.
        sequence_length: The length of each sequence.

    Returns:
        A tuple containing the input sequences (X) and target values (y).
    """
    logger.info("Preparing sequences for training/validation.")
    X, y = [], []
    data_values = data[feature_cols + target_cols].values
    for i in range(sequence_length, len(data_values) + 1):
        X.append(data_values[i - sequence_length:i, :len(feature_cols)])
        y.append(data_values[i - 1, len(feature_cols):])
    logger.debug(f"Prepared {len(X)} sequences.")
    return np.array(X), np.array(y)

def prepare_test_sequences(
    data: pd.DataFrame,
    feature_cols: List[str],
    sequence_length: int
) -> np.ndarray:
    """Prepare input sequences for the test data.

    Args:
        data: DataFrame containing the test data.
        feature_cols: List of feature column names.
        sequence_length: The length of each sequence.

    Returns:
        An array of input sequences for the test data.
    """
    logger.info("Preparing sequences for testing.")
    X = []
    data_values = data[feature_cols].values
    num_samples = len(data_values)
    for i in range(num_samples):
        start_idx = max(0, i - sequence_length + 1)
        seq = data_values[start_idx:i+1]
        # Pad sequences that are shorter than the required length
        if len(seq) < sequence_length:
            padding = np.zeros((sequence_length - len(seq), len(feature_cols)))
            seq = np.vstack((padding, seq))
            logger.debug(f"Padded sequence at index {i} with zeros.")
        X.append(seq)
    logger.debug(f"Prepared {len(X)} test sequences.")
    return np.array(X)

def prepare_test_data(
    ais_test: pd.DataFrame,
    vessels: pd.DataFrame,
    vessel_type_categories: pd.Index,
    ports: pd.DataFrame,  # Added ports dataset
    features: List[str],
    scaler: StandardScaler,
    low_speed_threshold: float = 5.0,  # Default threshold for low speed in knots
    port_radius_km: float = 10.0  # Radius to consider proximity to port
) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare and preprocess the test data.

    Args:
        ais_test: DataFrame containing the AIS test data.
        vessels: DataFrame containing vessel information.
        vessel_type_categories: Categories of vessel types from training data.
        ports: DataFrame containing ports information.
        features: List of feature column names to include.
        scaler: Fitted StandardScaler instance for feature scaling.
        low_speed_threshold: Speed below which vessel is considered to have low speed.
        port_radius_km: Radius in kilometers to consider a vessel near a port.

    Returns:
        A tuple containing the merged test DataFrame and the list of feature columns.
    """
    logger.info("Preparing test data.")
    merged_test = ais_test.copy()

    # Merge AIS test data with vessel data
    merged_test = pd.merge(
        merged_test,
        vessels[['vesselId', 'vesselType']],
        on='vesselId',
        how='left'
    )
    logger.debug("Merged AIS test data with vessel data.")

    # Ensure 'vesselType' is of object (string) type and handle missing values
    merged_test['vesselType'] = merged_test['vesselType'].astype('object').fillna('Unknown')
    logger.debug("Handled missing vesselType values.")

    # Extract time-related features
    merged_test['time'] = pd.to_datetime(merged_test['time'])
    merged_test['hour'] = merged_test['time'].dt.hour
    merged_test['day_of_week'] = merged_test['time'].dt.dayofweek
    logger.debug("Extracted time-related features (hour, day_of_week).")

    # Encode vessel type using categories from training data
    merged_test['vesselType'] = pd.Categorical(
        merged_test['vesselType'],
        categories=vessel_type_categories
    )
    merged_test['vessel_type_encoded'] = merged_test['vesselType'].cat.codes
    logger.debug("Encoded vesselType as numeric categories.")

    # **Feature Engineering: Low Speed Indicator**
    merged_test['low_speed'] = (merged_test['sog'] < low_speed_threshold).astype(int)
    features.append('low_speed')  # Add to feature list
    logger.debug("Added low_speed indicator feature.")

    # # **Feature Engineering: Proximity to Port**
    # # Calculate distance to each port and take the minimum distance
    # def compute_min_distance(row):
    #     vessel_coord = (row['latitude'], row['longitude'])
    #     distances = ports.apply(
    #         lambda port: calculate_distance(
    #             vessel_coord[0], vessel_coord[1],
    #             port['latitude'], port['longitude']
    #         ),
    #         axis=1
    #     )
    #     min_distance = distances.min()
    #     return min_distance
    #
    # logger.info("Calculating minimum distance to ports for each vessel.")
    # merged_test['min_distance_to_port'] = merged_test.apply(compute_min_distance, axis=1)
    # features.append('min_distance_to_port')  # Add to feature list
    # logger.debug("Added min_distance_to_port feature.")
    #
    # # **Feature Engineering: Within Port Radius**
    # merged_test['within_port_radius'] = (merged_test['min_distance_to_port'] <= port_radius_km).astype(int)
    # features.append('within_port_radius')  # Add to feature list
    # logger.debug("Added within_port_radius feature.")

    # **Handling Missing Features**
    # Ensure all required features are present
    for feature in features:
        if feature not in merged_test.columns:
            merged_test[feature] = 0  # Assign a default value or handle appropriately
            logger.warning(f"Feature '{feature}' missing in test data. Filled with default values.")

    # Handle missing feature values
    merged_test[features] = merged_test[features].fillna(0)
    logger.debug("Handled missing feature values in test data.")

    # **Feature Scaling**
    logger.info("Applying feature scaling to test data.")
    merged_test[features] = scaler.transform(merged_test[features])
    logger.debug("Applied StandardScaler to test data features.")

    logger.info("Test data preparation completed.")
    return merged_test, features

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
    logger.debug("Datasets loaded successfully.")
    return ais_train, ais_test, vessels, ports, schedules

def preprocess_data(
    ais_train: pd.DataFrame,
    vessels: pd.DataFrame,
    ports: pd.DataFrame  # Added ports dataset for consistency
) -> Tuple[pd.DataFrame, pd.Index]:
    """Preprocess the training data and merge with vessel information.

    Args:
        ais_train: DataFrame containing the AIS training data.
        vessels: DataFrame containing vessel information.
        ports: DataFrame containing ports information.

    Returns:
        A tuple containing the merged training DataFrame and vessel type categories.
    """
    logger.info("Preprocessing training data.")
    # Merge AIS data with vessel data to get vessel types and other info
    merged_data = pd.merge(
        ais_train,
        vessels[['vesselId', 'vesselType']],
        on='vesselId',
        how='left'
    )
    logger.debug("Merged AIS training data with vessel data.")

    # Ensure 'vesselType' is of object type and handle missing values
    merged_data['vesselType'] = merged_data['vesselType'].astype('object').fillna('Unknown')
    logger.debug("Handled missing vesselType values in training data.")

    # Extract time-related features
    merged_data['time'] = pd.to_datetime(merged_data['time'])
    merged_data['hour'] = merged_data['time'].dt.hour
    merged_data['day_of_week'] = merged_data['time'].dt.dayofweek
    logger.debug("Extracted time-related features (hour, day_of_week) in training data.")

    # Sort data and create future position columns
    merged_data = merged_data.sort_values(by=['vesselId', 'time'])
    merged_data['future_latitude'] = merged_data.groupby('vesselId')['latitude'].shift(-1)
    merged_data['future_longitude'] = merged_data.groupby('vesselId')['longitude'].shift(-1)
    logger.debug("Created future latitude and longitude columns.")

    # Drop rows with missing future positions
    initial_length = len(merged_data)
    merged_data.dropna(subset=['future_latitude', 'future_longitude'], inplace=True)
    logger.debug(f"Dropped {initial_length - len(merged_data)} rows with missing future positions.")

    # Encode vessel type as numeric categories
    merged_data['vesselType'] = merged_data['vesselType'].astype('category')
    vessel_type_categories = merged_data['vesselType'].cat.categories
    merged_data['vessel_type_encoded'] = merged_data['vesselType'].cat.codes
    logger.debug("Encoded vesselType as numeric categories in training data.")

    # **Feature Engineering: Low Speed Indicator**
    low_speed_threshold = 5.0  # Example threshold in knots
    merged_data['low_speed'] = (merged_data['sog'] < low_speed_threshold).astype(int)
    logger.debug("Added low_speed indicator feature.")

    # # **Feature Engineering: Proximity to Port**
    # def compute_min_distance(row):
    #     vessel_coord = (row['latitude'], row['longitude'])
    #     distances = ports.apply(
    #         lambda port: calculate_distance(
    #             vessel_coord[0], vessel_coord[1],
    #             port['latitude'], port['longitude']
    #         ),
    #         axis=1
    #     )
    #     min_distance = distances.min()
    #     return min_distance
    #
    # logger.info("Calculating minimum distance to ports for each vessel in training data.")
    # merged_data['min_distance_to_port'] = merged_data.apply(compute_min_distance, axis=1)
    # logger.debug("Added min_distance_to_port feature.")
    #
    # # **Feature Engineering: Within Port Radius**
    # port_radius_km = 10.0  # Example radius in kilometers
    # merged_data['within_port_radius'] = (merged_data['min_distance_to_port'] <= port_radius_km).astype(int)
    # logger.debug("Added within_port_radius feature.")

    # **Handling Missing Features**
    # Handle missing 'cog', 'sog', 'latitude', 'longitude' if any
    for feature in ['cog', 'sog', 'latitude', 'longitude']:
        if feature not in merged_data.columns:
            merged_data[feature] = 0
            logger.warning(f"Feature '{feature}' missing in training data. Filled with default values.")
        else:
            merged_data[feature] = merged_data[feature].fillna(merged_data[feature].mean())
            logger.debug(f"Handled missing values in feature '{feature}' by imputing with mean.")

    # **Additional Features from Feature Engineering**
    # Ensure that the new features are included in the feature list
    additional_features = ['low_speed', 'min_distance_to_port', 'within_port_radius']
    for feat in additional_features:
        if feat not in merged_data.columns:
            merged_data[feat] = 0
            logger.warning(f"Additional feature '{feat}' missing in training data. Filled with default values.")

    logger.info("Training data preprocessing completed.")
    return merged_data, vessel_type_categories

def split_data(
    merged_data: pd.DataFrame,
    features: List[str],
    target: List[str],
    sequence_length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, StandardScaler]:
    """Split the data into training and validation sets, apply feature scaling, and prepare sequences.

    Args:
        merged_data: DataFrame containing the preprocessed data.
        features: List of feature column names.
        target: List of target column names.
        sequence_length: The length of each sequence.

    Returns:
        A tuple containing training and validation sequences and targets, validation data, and the scaler:
        (X_train, y_train, X_val, y_val, val_data, scaler).
    """
    logger.info("Splitting data into training and validation sets.")
    # Use early data as training, later data as validation
    train_data, val_data = train_test_split(merged_data, test_size=0.2, shuffle=False)
    logger.debug(f"Training data size: {len(train_data)}, Validation data size: {len(val_data)}")

    # Handle missing feature values
    train_data[features] = train_data[features].fillna(0)
    val_data[features] = val_data[features].fillna(0)
    logger.debug("Handled missing feature values in training and validation data.")

    # **Feature Scaling**
    scaler = StandardScaler()
    logger.info("Fitting StandardScaler on training data features.")
    scaler.fit(train_data[features])
    logger.debug("Fitted StandardScaler on training data features.")
    train_data[features] = scaler.transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])
    logger.debug("Applied StandardScaler to training and validation data features.")

    # Prepare sequences
    X_train, y_train = prepare_sequences(train_data, features, target, sequence_length)
    X_val, y_val = prepare_sequences(val_data, features, target, sequence_length)
    logger.info("Data splitting and sequence preparation completed.")
    return X_train, y_train, X_val, y_val, val_data, scaler

def build_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """Build and compile the LSTM model.

    Args:
        input_shape: A tuple representing the input shape (sequence_length, num_features).

    Returns:
        A compiled Keras model ready for training.
    """
    logger.info("Building the LSTM model.")
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(2))  # Predict latitude and longitude
    model.compile(optimizer='adam', loss='mean_absolute_error')
    logger.debug("LSTM model built and compiled.")
    return model

def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> tf.keras.callbacks.History:
    """Train the LSTM model.

    Args:
        model: The compiled Keras model.
        X_train: Training input sequences.
        y_train: Training target values.
        X_val: Validation input sequences.
        y_val: Validation target values.

    Returns:
        The history object containing training details.
    """
    logger.info("Starting model training.")
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=64)
    logger.info("Model training completed.")
    return history

def evaluate_model(
    model: tf.keras.Model,
    X_val: np.ndarray,
    val_data: pd.DataFrame,
    sequence_length: int
) -> float:
    """Evaluate the model on the validation set and calculate error distance.

    Args:
        model: The trained Keras model.
        X_val: Validation input sequences.
        val_data: DataFrame containing the validation data.
        sequence_length: The length of each sequence.

    Returns:
        The mean geodesic error distance on the validation set.
    """
    logger.info("Evaluating the model on the validation set.")
    predictions_val = model.predict(X_val)
    logger.debug(f"Number of predictions: {predictions_val.shape[0]}")
    logger.debug(f"Number of validation samples: {len(val_data) - sequence_length + 1}")

    # Align with predictions
    val_data_aligned = val_data.iloc[sequence_length - 1:].copy()
    logger.debug(f"Aligned validation data size: {len(val_data_aligned)}")

    # Assign predictions
    val_data_aligned['pred_latitude'] = predictions_val[:, 0]
    val_data_aligned['pred_longitude'] = predictions_val[:, 1]
    logger.debug("Predictions on validation set obtained.")

    # Calculate the error distance for each prediction
    val_data_aligned['error_distance'] = val_data_aligned.apply(
        lambda row: calculate_distance(
            row['future_latitude'],
            row['future_longitude'],
            row['pred_latitude'],
            row['pred_longitude']
        ),
        axis=1
    )
    mean_error_distance = val_data_aligned['error_distance'].mean()
    logger.info(f"Mean Geodetic Error on Validation Set: {mean_error_distance:.2f} km")
    return mean_error_distance

def prepare_submission(
    ais_test: pd.DataFrame,
    merged_test: pd.DataFrame,
    predictions_test: np.ndarray
) -> pd.DataFrame:
    """Prepare the submission DataFrame.

    Args:
        ais_test: DataFrame containing the AIS test data.
        merged_test: DataFrame containing the merged test data.
        predictions_test: Numpy array containing the predicted coordinates.

    Returns:
        A DataFrame ready to be saved as a submission file.
    """
    logger.info("Preparing submission DataFrame.")
    # Align predictions with test data
    submission_df = merged_test.copy()
    submission_df['longitude_predicted'] = predictions_test[:, 1]
    submission_df['latitude_predicted'] = predictions_test[:, 0]
    logger.debug("Added predicted coordinates to submission DataFrame.")

    # Prepare submission file
    if 'ID' not in ais_test.columns:
        ais_test.reset_index(inplace=True)
        ais_test.rename(columns={'index': 'ID'}, inplace=True)
        logger.warning("'ID' column not found in ais_test. Assigned index as 'ID'.")

    submission_df['ID'] = ais_test['ID'].values.astype(int)
    submission_df = submission_df[['ID', 'longitude_predicted', 'latitude_predicted']]
    logger.debug("Reordered submission DataFrame columns.")

    logger.info("Submission DataFrame preparation completed.")
    return submission_df

def main() -> None:
    """Main function to run the machine learning pipeline."""
    logger.info("Starting the Machine Learning Pipeline.")

    # Print TensorFlow and Keras versions
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"Keras version: {tf.keras.__version__}")

    # List physical devices
    physical_devices = tf.config.list_physical_devices()
    logger.debug(f"Physical devices: {physical_devices}")

    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info("TensorFlow is using the following GPU(s):")
        for gpu in gpus:
            logger.info(f"\t{gpu}")
    else:
        logger.warning("No GPU found. TensorFlow will use the CPU.")

    # Load datasets
    ais_train, ais_test, vessels, ports, schedules = load_data()

    # Verify column names
    logger.debug(f"AIS Train Columns: {list(ais_train.columns)}")
    logger.debug(f"AIS Test Columns: {list(ais_test.columns)}")
    logger.debug(f"Vessels Columns: {list(vessels.columns)}")
    logger.debug(f"Ports Columns: {list(ports.columns)}")

    # Preprocess training data
    merged_data, vessel_type_categories = preprocess_data(ais_train, vessels, ports)

    # Define features and target columns (Original + Additional Features)
    features = ['hour', 'day_of_week', 'vessel_type_encoded', 'cog', 'sog', 
               'low_speed']
    target = ['future_latitude', 'future_longitude']
    logger.debug(f"Features: {features}")
    logger.debug(f"Target: {target}")

    # Split data and prepare sequences with feature scaling
    sequence_length = 5
    X_train, y_train, X_val, y_val, val_data, scaler = split_data(merged_data, features, target, sequence_length)
    
    # Build the LSTM model
    input_shape = (sequence_length, len(features))
    model = build_model(input_shape)
    model.summary(print_fn=lambda x: logger.debug(x))

    # Train the model
    train_model(model, X_train, y_train, X_val, y_val)

    # Evaluate the model
    mean_error_distance = evaluate_model(model, X_val, val_data, sequence_length)

    # Prepare test data with the same features as training and apply scaling
    merged_test, test_features = prepare_test_data(
        ais_test, vessels, vessel_type_categories, ports, 
        features, scaler, low_speed_threshold=5.0, port_radius_km=10.0
    )

    # Prepare test sequences
    X_test = prepare_test_sequences(merged_test, test_features, sequence_length)

    # Verify the shape of test data
    logger.debug(f"Test sequences shape: {X_test.shape}")  # Should be (num_samples, 5, num_features)

    # Make predictions on test data
    logger.info("Making predictions on the test set.")
    predictions_test = model.predict(X_test)
    logger.debug("Predictions on test set obtained.")

    # Prepare submission DataFrame
    submission_df = prepare_submission(ais_test, merged_test, predictions_test)

    # Save submission file
    submission_df.to_csv('submission.csv', index=False)
    logger.info("Submission file 'submission.csv' has been created.")

    logger.info("Machine Learning Pipeline completed successfully.")

if __name__ == "__main__":
    main()
