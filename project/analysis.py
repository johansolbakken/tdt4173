#!/usr/bin/env python3

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from geopy.distance import geodesic
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load and preprocess data
logger.info("Loading ais_train.csv")
ais_train = pd.read_csv("ais_train.csv", sep="|")
ais_train['time'] = pd.to_datetime(ais_train['time']).dt.tz_localize(None)  # Make 'time' timezone-naive
ais_train = ais_train.rename(columns={'portId': 'maybePortId'})
logger.info("Finished loading ais_train.csv")

# Map vessel ids
logger.info("Mapping vessel ids")
vessel_mapping = {vessel: idx for idx, vessel in enumerate(ais_train['vesselId'].unique())}
ais_train['vesselId'] = ais_train['vesselId'].map(vessel_mapping)

# Pruning the data
logger.info("Pruning ais_train.csv data")
ais_train = ais_train[ais_train['cog'] != 360]  # cog=360 is not available
ais_train = ais_train[(ais_train['cog'] <= 360) | (ais_train['cog'] > 409.5)]  # this range should not be used
ais_train = ais_train[ais_train['heading'] != 511]  # unavailable
ais_train = ais_train[ais_train['sog'] < 25]
ais_train['navstat'] = ais_train['navstat'].replace(8, 0)  # Under way sailing -> Under way using engine
ais_train = ais_train[~((ais_train['navstat'].isin([1, 5])) & (ais_train['sog'] > 0))]
ais_train = ais_train[~((ais_train['navstat'] == 2) & (ais_train['sog'] > 5))]
logger.info("Finished pruning ais_train.csv")

logger.info("Loading vessels.csv")
vessels = pd.read_csv("vessels.csv", sep='|')
vessels['vesselId'] = vessels['vesselId'].map(vessel_mapping)
logger.info("Finished loading and processing vessels.csv")

logger.info("Loading schedules_to_may_2024.csv")
schedules = pd.read_csv("schedules_to_may_2024.csv", sep="|")
schedules['vesselId'] = schedules['vesselId'].map(vessel_mapping)
schedules['sailingDate'] = pd.to_datetime(schedules['sailingDate']).dt.tz_localize(None)
schedules['arrivalDate'] = pd.to_datetime(schedules['arrivalDate']).dt.tz_localize(None)
schedules = schedules.dropna(subset=['portLatitude'])  # drop nan values
schedules = schedules.drop_duplicates()  # many duplicate values
logger.info("Finished loading and processing schedules_to_may_2024.csv")

logger.info("Loading ports.csv")
ports = pd.read_csv('ports.csv', sep='|')
logger.info("Finished loading ports.csv")

logger.info("Loading ais_test.csv")
ais_test = pd.read_csv("ais_test.csv")  # sep=","
ais_test['time'] = pd.to_datetime(ais_test['time']).dt.tz_localize(None)  # Make 'time' timezone-naive
ais_test['vesselId'] = ais_test['vesselId'].map(vessel_mapping)
logger.info("Finished loading ais_test.csv")

def feature_engineering(df):
    logger.info("Performing feature engineering")
    df['elapsed_time'] = (df['time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

    # Merge schedules with the ais_train data
    df = pd.merge(df, schedules[['vesselId', 'arrivalDate', 'sailingDate']], on='vesselId', how='left')

    # Calculate time differences between the vessel's current time and arrival/sailing times
    df['time_to_arrival'] = (df['arrivalDate'] - df['time']).dt.total_seconds() / 3600  # Hours to arrival
    df['time_to_sailing'] = (df['sailingDate'] - df['time']).dt.total_seconds() / 3600  # Hours to sailing

    # Determine if the vessel is in port (i.e., between arrival and sailing)
    df['in_port'] = ((df['time'] >= df['arrivalDate']) & (df['time'] <= df['sailingDate'])).astype(int)

    logger.info("Finished feature engineering")
    return df

logger.info("Sampling ais_train data")
if False:
    ais_train = ais_train.sample(frac=0.1, random_state=42)

ais_train = feature_engineering(ais_train)

# Prepare features and targets
logger.info("Preparing features and targets")
target_lat = 'latitude'
target_lon = 'longitude'
features = ['vesselId', 'elapsed_time', 'time_to_arrival', 'time_to_sailing', 'in_port']

train_data = ais_train[ais_train['time'] < '2024-05-01']
val_data = ais_train[ais_train['time'] >= '2024-05-01']

X_train = train_data[features]
y_train_lat = train_data[target_lat]
y_train_lon = train_data[target_lon]
X_val = val_data[features]
y_val_lat = val_data[target_lat]
y_val_lon = val_data[target_lon]

# Train model for latitude
logger.info("Training latitude model")
model_lat = xgb.XGBRegressor()
model_lat.fit(X_train, y_train_lat, verbose=1)

# Train model for longitude
logger.info("Training longitude model")
model_lon = xgb.XGBRegressor()
model_lon.fit(X_train, y_train_lon, verbose=1)

if False:
    def calculate_geodesic_distance(true_coords, pred_coords):
        logger.info("Calculating geodesic distance")
        distances = [geodesic(true, pred).km for true, pred in zip(true_coords, pred_coords)]
        return distances

    # Predict on validation set
    logger.info("Predicting on validation set")
    val_pred_lat = model_lat.predict(X_val)
    val_pred_lon = model_lon.predict(X_val)

    true_coords = list(zip(y_val_lat, y_val_lon))
    pred_coords = list(zip(val_pred_lat, val_pred_lon))

    distances = calculate_geodesic_distance(true_coords, pred_coords)
    weighted_mean_geodetic_distance = sum(distances) / len(distances)
    logger.info(f"Weighted Mean Geodetic Distance: {weighted_mean_geodetic_distance}")

# Feature engineering and prediction on test set
logger.info("Performing feature engineering on ais_test and predicting")
ais_test = feature_engineering(ais_test)
X_test = ais_test[features]

test_pred_lat = model_lat.predict(X_test)
test_pred_lon = model_lon.predict(X_test)

# Generate submission
logger.info("Generating submission.csv")
submission = pd.DataFrame({
    'ID': ais_test['ID'],
    'longitude_predicted': test_pred_lon,
    'latitude_predicted': test_pred_lat
})

submission.to_csv('submission.csv', index=False)
logger.info("Finished generating submission.csv")
