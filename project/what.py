import pandas as pd
import logging
import sys
from typing import Tuple
from colorlog import ColoredFormatter
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import numpy as np

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
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Prepare the data for the LSTM model, including additional time-based features.
    
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
    
    # Extract the relevant features, including the new ones
    features = ais_train[['latitude', 'longitude', 'sog', 'cog', 'hour_of_day', 'day_of_week', 'time_elapsed']].values
    target = ais_train[['latitude', 'longitude']].shift(-1).ffill().values

    # Normalize features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Reshape for LSTM input: (samples, timesteps, features)
    X = features_scaled.reshape((features_scaled.shape[0], 1, features_scaled.shape[1]))
    y = target
    
    logger.info("Data preparation complete.")
    return X, y, scaler

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
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=2))  # Output: latitude and longitude
    model.compile(optimizer='adam', loss='mse')
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
    model.fit(X_train, y_train, epochs=2, batch_size=32, validation_split=0.2)
    logger.info("Model training complete.")

def generate_submission(model: Sequential, ais_test: pd.DataFrame, scaler: MinMaxScaler) -> None:
    """Generate a submission file with the predicted vessel positions.
    
    Args:
        model: Trained LSTM model.
        ais_test: DataFrame containing AIS test data.
        scaler: Scaler used to normalize the features.
    """
    logger.info("Generating predictions for the test set.")
    
    # Use the scaling_factor as the main feature from the test data
    test_features = ais_test[['scaling_factor']].values
    
    # Since the test data only has one feature, we need to adjust the input shape to match the model's expectation
    num_train_features = scaler.n_features_in_
    test_features_padded = np.zeros((test_features.shape[0], num_train_features))
    test_features_padded[:, :test_features.shape[1]] = test_features

    # Normalize the padded test features to match the training data scale
    test_features_scaled = scaler.transform(test_features_padded)
    X_test = test_features_scaled.reshape((test_features_scaled.shape[0], 1, test_features_scaled.shape[1]))
    
    # Make predictions using the model
    predictions = model.predict(X_test)
    
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
    X_train, y_train, scaler = prepare_data(ais_train, vessels, ports, schedules)

    # Build and train the LSTM model
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    train_model(model, X_train, y_train)

    # Generate the submission file
    generate_submission(model, ais_test, scaler)
    
    logger.info("Machine learning pipeline finished successfully.")

if __name__ == "__main__":
    main()
