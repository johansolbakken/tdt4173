import pandas as pd
from typing import List, Optional
from datetime import datetime
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class AisTrain:
    def __init__(self, time: datetime, cog: float, sog: float, rot: int, heading: int, navstat: int, etaRaw: str, latitude: float, longitude: float, vesselId: str, portId: str):
        self.time = time
        self.cog = cog
        self.sog = sog
        self.rot = rot
        self.heading = heading
        self.navstat = navstat
        self.etaRaw = etaRaw
        self.latitude = latitude
        self.longitude = longitude
        self.vesselId = vesselId
        self.portId = portId

class Vessel:
    def __init__(self, shippingLineId: str, vesselId: str, CEU: int, DWT: float, GT: float, NT: Optional[float], vesselType: float, homePort: str, length: float, maxHeight: Optional[float], maxSpeed: Optional[float], maxWidth: Optional[float], rampCapacity: Optional[float], yearBuilt: int):
        self.shippingLineId = shippingLineId
        self.vesselId = vesselId
        self.CEU = CEU
        self.DWT = DWT
        self.GT = GT
        self.NT = NT
        self.vesselType = vesselType
        self.homePort = homePort
        self.length = length
        self.maxHeight = maxHeight
        self.maxSpeed = maxSpeed
        self.maxWidth = maxWidth
        self.rampCapacity = rampCapacity
        self.yearBuilt = yearBuilt

class Port:
    def __init__(self, portId: str, name: str, portLocation: str, longitude: float, latitude: float, UN_LOCODE: str, countryName: str, ISO: str):
        self.portId = portId
        self.name = name
        self.portLocation = portLocation
        self.longitude = longitude
        self.latitude = latitude
        self.UN_LOCODE = UN_LOCODE
        self.countryName = countryName
        self.ISO = ISO

class Schedule:
    def __init__(self, vesselId: str, shippingLineId: str, shippingLineName: str, portId: str, portLatitude: float, portLongitude: float):
        self.vesselId = vesselId
        self.shippingLineId = shippingLineId
        self.shippingLineName = shippingLineName
        self.portId = portId
        self.portLatitude = portLatitude
        self.portLongitude = portLongitude

def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

def main() -> None:
    # Load AIS and optional datasets
    ais_train = pd.read_csv('ais_train.csv', sep='|')
    ais_test = pd.read_csv('ais_test.csv', sep='|')
    vessels = pd.read_csv('vessels.csv', sep='|')
    ports = pd.read_csv('ports.csv', sep='|')
    schedules = pd.read_csv('schedules_to_may_2024.csv', sep='|')

    # Merge AIS data with port data to get the port's latitude and longitude
    merged_data = pd.merge(
        ais_train, 
        ports[['portId', 'latitude', 'longitude']],  # Select relevant columns from ports
        left_on='portId', right_on='portId', 
        suffixes=('', '_port')  # To differentiate vessel and port lat/lon
    )

    # Calculate distance to port
    merged_data['distance_to_port'] = merged_data.apply(lambda row: calculate_distance(
        row['latitude'], row['longitude'], row['latitude_port'], row['longitude_port']), axis=1)

    # Extract time-related features
    ais_train['timestamp'] = pd.to_datetime(ais_train['timestamp'])
    ais_train['hour'] = ais_train['timestamp'].dt.hour
    ais_train['day_of_week'] = ais_train['timestamp'].dt.dayofweek
    # Create future latitude and longitude columns by shifting the original values
    ais_train['future_latitude'] = ais_train['latitude'].shift(-1)
    ais_train['future_longitude'] = ais_train['longitude'].shift(-1)

    # Encode vessel type as numeric categories
    ais_train['vessel_type_encoded'] = ais_train['vessel_type'].astype('category').cat.codes

    # Use early data as training, later data as validation
    train_data, val_data = train_test_split(ais_train, test_size=0.2, shuffle=False)
    # Prepare the data for LSTM
    def prepare_sequences(data, feature_cols, target_cols, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[feature_cols].iloc[i:i+sequence_length].values)
            y.append(data[target_cols].iloc[i+sequence_length].values)
        return np.array(X), np.array(y)

    # Define the feature columns
    features = ['cog', 'sog', 'rot', 'heading', 'navstat', 'distance_to_port', 'hour', 'day_of_week', 'vessel_type_encoded']

    # Define the target columns
    target = ['future_latitude', 'future_longitude']  # Assuming you have future latitude and longitude columns

    sequence_length = 5
    X_train, y_train = prepare_sequences(train_data, features, target, sequence_length)
    X_val, y_val = prepare_sequences(val_data, features, target, sequence_length)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=(sequence_length, len(features))))
    model.add(Dense(2))  # Predict latitude and longitude

    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.summary()

    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

    # Make predictions
    predictions = model.predict(X_val)

    # Calculate geodetic distance between predictions and actual positions
    val_data['pred_latitude'] = predictions[:, 0]
    val_data['pred_longitude'] = predictions[:, 1]

    val_data['error_distance'] = val_data.apply(lambda row: calculate_distance(
        row['future_latitude'], row['future_longitude'], row['pred_latitude'], row['pred_longitude']), axis=1)

    mean_error_distance = val_data['error_distance'].mean()
    print(f'Mean Geodetic Error: {mean_error_distance} km')



if __name__ == "__main__":
    main()

