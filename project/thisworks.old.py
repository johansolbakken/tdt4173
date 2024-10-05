import pandas as pd
import numpy as np
from geopy.distance import geodesic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.model_selection import train_test_split
import tensorflow as tf

def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

def prepare_sequences(data, feature_cols, target_cols, sequence_length):
    X, y = [], []
    data_values = data[feature_cols + target_cols].values
    for i in range(sequence_length, len(data_values) + 1):
        X.append(data_values[i - sequence_length:i, :len(feature_cols)])
        y.append(data_values[i - 1, len(feature_cols):])
    return np.array(X), np.array(y)

def prepare_test_sequences(data, feature_cols, sequence_length):
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
        X.append(seq)
    return np.array(X)

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
    # Ensure 'vesselType' is of object (string) type
    merged_test['vesselType'] = merged_test['vesselType'].astype('object')

    # Handle missing 'vesselType' values
    merged_test['vesselType'] = merged_test['vesselType'].fillna('Unknown')

    # Extract time-related features
    merged_test['time'] = pd.to_datetime(merged_test['time'])
    merged_test['hour'] = merged_test['time'].dt.hour
    merged_test['day_of_week'] = merged_test['time'].dt.dayofweek

    # Encode vessel type as numeric categories (ensure consistency with training data)
    merged_test['vesselType'] = pd.Categorical(merged_test['vesselType'], categories=vessel_type_categories)
    merged_test['vessel_type_encoded'] = merged_test['vesselType'].cat.codes

    # Define the feature columns
    features = ['hour', 'day_of_week', 'vessel_type_encoded']

    # Handle missing feature values
    merged_test[features] = merged_test[features].fillna(0)  # Or appropriate imputation

    return merged_test, features

def main() -> None:
    print("TensorFlow version:", tf.__version__)
    print("Keras version:", tf.keras.__version__)

    physical_devices = tf.config.list_physical_devices()
    print("Physical devices:", physical_devices)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("TensorFlow is using the following GPU(s):")
        for gpu in gpus:
            print("\t" + str(gpu))
    else:
        print("No GPU found. TensorFlow will use the CPU.")


    # Load AIS and optional datasets
    ais_train = pd.read_csv('ais_train.csv', sep='|')
    ais_test = pd.read_csv('ais_test.csv')  # Removed sep parameter
    vessels = pd.read_csv('vessels.csv', sep='|')
    ports = pd.read_csv('ports.csv', sep='|')
    schedules = pd.read_csv('schedules_to_may_2024.csv', sep='|')

    # Verify column names
    print("AIS Train Columns:", ais_train.columns)
    print("AIS Test Columns:", ais_test.columns)
    print("Vessels Columns:", vessels.columns)
    print("Ports Columns:", ports.columns)

    # Merge AIS data with vessel data to get vessel types and other info
    merged_data = pd.merge(
        ais_train,
        vessels[['vesselId', 'vesselType']],
        on='vesselId',
        how='left'
    )

    # Ensure 'vesselType' is of object (string) type
    merged_data['vesselType'] = merged_data['vesselType'].astype('object')

    # Handle missing 'vesselType' values
    merged_data['vesselType'] = merged_data['vesselType'].fillna('Unknown')

    # Rest of your code...
    # Extract time-related features
    merged_data['time'] = pd.to_datetime(merged_data['time'])
    merged_data['hour'] = merged_data['time'].dt.hour
    merged_data['day_of_week'] = merged_data['time'].dt.dayofweek

    # Create future latitude and longitude columns by shifting the original values
    merged_data = merged_data.sort_values(by=['vesselId', 'time'])
    merged_data['future_latitude'] = merged_data.groupby('vesselId')['latitude'].shift(-1)
    merged_data['future_longitude'] = merged_data.groupby('vesselId')['longitude'].shift(-1)

    # Drop rows with missing future positions
    merged_data.dropna(subset=['future_latitude', 'future_longitude'], inplace=True)

    # Encode vessel type as numeric categories
    merged_data['vesselType'] = merged_data['vesselType'].astype('category')
    vessel_type_categories = merged_data['vesselType'].cat.categories
    merged_data['vessel_type_encoded'] = merged_data['vesselType'].cat.codes

    # Define the feature columns
    features = ['hour', 'day_of_week', 'vessel_type_encoded']

    # Define the target columns
    target = ['future_latitude', 'future_longitude']

    # Handle missing feature values
    merged_data[features] = merged_data[features].fillna(0)

    # Use early data as training, later data as validation
    train_data, val_data = train_test_split(merged_data, test_size=0.2, shuffle=False)

    sequence_length = 5
    X_train, y_train = prepare_sequences(train_data, features, target, sequence_length)
    X_val, y_val = prepare_sequences(val_data, features, target, sequence_length)

    # Build the LSTM model
    model = Sequential()
    model.add(Input(shape=(sequence_length, len(features))))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(2))  # Predict latitude and longitude

    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.summary()

    # Train the model
    model.fit(X_train, y_train, epochs=1, validation_data=(X_val, y_val))

    # Evaluate on validation set
    predictions_val = model.predict(X_val)
    val_data = val_data.iloc[sequence_length - 1:]  # Align with predictions
    val_data['pred_latitude'] = predictions_val[:, 0]
    val_data['pred_longitude'] = predictions_val[:, 1]

    val_data['error_distance'] = val_data.apply(lambda row: calculate_distance(
        row['future_latitude'], row['future_longitude'], row['pred_latitude'], row['pred_longitude']), axis=1)

    mean_error_distance = val_data['error_distance'].mean()
    print(f'Mean Geodetic Error on Validation Set: {mean_error_distance} km')

    # Prepare test data
    merged_test, features = prepare_test_data(ais_test, vessels, vessel_type_categories)

    # Prepare test sequences
    X_test = prepare_test_sequences(merged_test, features, sequence_length)

    # Make predictions on test set
    predictions_test = model.predict(X_test)

    # Align predictions with test data
    submission_df = merged_test.copy()
    submission_df['longitude_predicted'] = predictions_test[:, 1]
    submission_df['latitude_predicted'] = predictions_test[:, 0]

    # Prepare submission file
    if 'ID' not in ais_test.columns:
        ais_test.reset_index(inplace=True)
        ais_test.rename(columns={'index': 'ID'}, inplace=True)

    submission_df['ID'] = ais_test['ID'].values
    submission_df['ID'] = submission_df['ID'].astype(int)

    submission_df = submission_df[['ID', 'longitude_predicted', 'latitude_predicted']]

    # Save submission file
    submission_df.to_csv('submission.csv', index=False)
    print("Submission file 'submission.csv' has been created.")

if __name__ == "__main__":
    main()
