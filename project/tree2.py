#!/usr/bin/env python3

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import radians, cos, sin, sqrt, atan2

# Function to calculate the Haversine distance between two points
def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Radius of Earth in kilometers (mean radius)
    r = 6371.0
    return r * c  # distance in kilometers
 
"""
ais_train.csv:
Index(['time', 'cog', 'sog', 'rot', 'heading', 'navstat', 'etaRaw', 'latitude',
       'longitude', 'vesselId', 'portId', 'elapsed_time'], dtype='object')
"""
ais_train = pd.read_csv("ais_train.csv", sep="|")

ais_train = ais_train.drop('portId', axis=1) # can be misleading

# map vessel ids
vessel_mapping = {vessel: idx for idx, vessel in enumerate(ais_train['vesselId'].unique())}

ais_train = ais_train[ais_train['cog']!=360] # cog=360 is not available
ais_train = ais_train[(ais_train['cog'] <= 360) | (ais_train['cog'] > 409.5)] # this range should not be used

# The 'turning' column (-1=left, 1=right, 0=no turning)
ais_train['turning'] = 0
ais_train.loc[ais_train['rot'] < 0, 'turning'] = -1
ais_train.loc[ais_train['rot'] >= 0, 'turning'] = 1
ais_train.loc[ais_train['rot'] == 128, 'turning'] = 0
ais_train.loc[ais_train['rot'] == -128, 'turning'] = 0

ais_train = ais_train[ais_train['heading'] != 511] # unavailable
ais_train = ais_train[ais_train['sog'] < 25]

# Map 'navstat' values
ais_train['navstat'] = ais_train['navstat'].replace(8, 0)  # Under way sailing -> Under way using engine
ais_train = ais_train[~((ais_train['navstat'].isin([1, 5])) & (ais_train['sog'] > 0))]
ais_train = ais_train[~((ais_train['navstat'] == 2) & (ais_train['sog'] > 5))]

"""
vessels.csv
Index(['shippingLineId', 'vesselId', 'CEU', 'DWT', 'GT', 'NT', 'vesselType',
       'breadth', 'depth', 'draft', 'enginePower', 'freshWater', 'fuel',
       'homePort', 'length', 'maxHeight', 'maxSpeed', 'maxWidth',
       'rampCapacity', 'yearBuilt'],
      dtype='object')
"""
vessels = pd.read_csv("vessels.csv", sep='|')
vessels['vesselId'] = vessels['vesselId'].map(vessel_mapping)

"""
schedules_to_may_2024.csv
Index(['vesselId', 'shippingLineId', 'shippingLineName', 'arrivalDate',
       'sailingDate', 'portName', 'portId', 'portLatitude', 'portLongitude'],
      dtype='object')
"""
schedules = pd.read_csv("schedules_to_may_2024.csv", sep="|")
schedules['vesselId'] = schedules['vesselId'].map(vessel_mapping)
schedules = schedules.dropna(subset=['portLatitude']) # drop nan values
schedules = schedules.drop_duplicates() # many duplicate values

"""
ports.csv
Index(['portId', 'name', 'portLocation', 'longitude', 'latitude', 'UN_LOCODE',
       'countryName', 'ISO'],
      dtype='object')
"""
ports = pd.read_csv('ports.csv', sep='|')
ports = ports.drop('portLocation', axis=1)
ports = ports.drop('UN_LOCODE', axis=1)
ports = ports.drop('countryName', axis=1)
ports = ports.drop('ISO', axis=1)
ports = ports.drop('name', axis=1)
ports = ports.rename(columns={'longitude': 'portLon', 'latitude': 'portLat'})

"""
ais_test.csv
Index(['ID', 'vesselId', 'time', 'scaling_factor'], dtype='object')
"""
ais_test = pd.read_csv("ais_test.csv") # sep=","

def create_features(df):
    df['vesselId'] = df['vesselId'].map(vessel_mapping)

    # Temporal features
    df['time'] = pd.to_datetime(df['time'])
    df['elapsed_time'] = (df['time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

    # Merge
    df = pd.merge(df, vessels, on='vesselId', how='left')
    df = pd.merge(df, ports, left_on='homePort', right_on='portId', how='left')

    df['distance_from_home'] = df.apply(
        lambda row: haversine(row['portLat'], row['portLon'], row['portLat'], row['portLon']), axis=1)

    # Define 'isHome' based on a threshold distance (e.g., within 10 km
    home_radius_km = 10
    df['isHome'] = np.where(df['distance_from_home'] <= home_radius_km, 1, 0)

    return df

ais_train = create_features(ais_train)
ais_test = create_features(ais_test)

features = ['elapsed_time', 'vesselId', 'isHome', 'distance_from_home']

"""
Need to train then
"""
X = ais_train[features]
y_latitude = ais_train['latitude']
y_longitude = ais_train['longitude']

# Split data into train and validation sets (70% training, 30% validation)
X_train, X_val, y_lat_train, y_lat_val, y_lon_train, y_lon_val = train_test_split(
    X, y_latitude, y_longitude, test_size=0.3, random_state=42)

# Convert data into DMatrix format (optimized for XGBoost)
dtrain_lat = xgb.DMatrix(X_train, label=y_lat_train)
dval_lat = xgb.DMatrix(X_val, label=y_lat_val)

dtrain_lon = xgb.DMatrix(X_train, label=y_lon_train)
dval_lon = xgb.DMatrix(X_val, label=y_lon_val)

# Define XGBoost parameters (tuning these might improve performance)
params = {
    'objective': 'reg:squarederror', # regression task
    'eval_metric': 'rmse',           # evaluation metric
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# Train the model for latitude prediction
bst_lat = xgb.train(params, dtrain_lat, num_boost_round=100,
                    evals=[(dval_lat, 'eval')], early_stopping_rounds=10)

# Train the model for longitude prediction
bst_lon = xgb.train(params, dtrain_lon, num_boost_round=100,
                    evals=[(dval_lon, 'eval')], early_stopping_rounds=10)

# Make predictions
lat_predictions = bst_lat.predict(xgb.DMatrix(ais_test[features]))
lon_predictions = bst_lon.predict(xgb.DMatrix(ais_test[features]))

# Prepare the submission file
submission = pd.DataFrame({
    'ID': ais_test['ID'],
    'latitude_predicted': lat_predictions,
    'longitude_predicted': lon_predictions
})

submission.to_csv("submission.csv", index=False)
print(submission)
print("Submission file saved as 'submission.csv'")
