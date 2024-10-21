#!/usr/bin/env python3

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

"""
ais_train.csv:
Index(['time', 'cog', 'sog', 'rot', 'heading', 'navstat', 'etaRaw', 'latitude',
       'longitude', 'vesselId', 'portId', 'elapsed_time'], dtype='object')
"""
ais_train = pd.read_csv("ais_train.csv", sep="|")

ais_train = ais_train.drop('portId', axis=1) # can be misleading

# map vessel ids
vessel_mapping = {vessel: idx for idx, vessel in enumerate(ais_train['vesselId'].unique())}
ais_train['vesselId'] = ais_train['vesselId'].map(vessel_mapping)

ais_train = ais_train[ais_train['cog']!=360] # cog=360 is not available
ais_train = ais_train[(ais_train['cog'] <= 360) | (ais_train['cog'] > 409.5)] # this range should not be used

# The 'turning' column (-1=left, 1=right, 0=no turning)
ais_train['turning'] = 0
ais_train.loc[ais_train['rot'] < 0, 'turning'] = -1
ais_train.loc[ais_train['rot'] >= 0, 'turning'] = 1
ais_train.loc[ais_train['rot'] == 128, 'turning'] = 0
ais_train.loc[ais_train['rot'] == -128, 'turning'] = 0

ais_train = ais_train[ais_train['heading'] != 511] # unavailable

# Temporal features
ais_train['time'] = pd.to_datetime(ais_train['time'])
ais_train['elapsed_time'] = (ais_train['time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# Filter out unrealistic speeds
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

"""
ais_test.csv
Index(['ID', 'vesselId', 'time', 'scaling_factor'], dtype='object')
"""
ais_test = pd.read_csv("ais_test.csv") # sep=","
ais_test['vesselId'] = ais_test['vesselId'].map(vessel_mapping)

# Temporal features
ais_test['time'] = pd.to_datetime(ais_test['time'])
ais_test['elapsed_time'] = (ais_test['time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

"""
Need to train then
"""
X = ais_train[['elapsed_time', 'vesselId']]
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
lat_predictions = bst_lat.predict(xgb.DMatrix(ais_test[['elapsed_time', 'vesselId']]))
lon_predictions = bst_lon.predict(xgb.DMatrix(ais_test[['elapsed_time', 'vesselId']]))

# Prepare the submission file
submission = pd.DataFrame({
    'ID': ais_test['ID'],
    'latitude_predicted': lat_predictions,
    'longitude_predicted': lon_predictions
})

submission.to_csv("submission.csv", index=False)
print(submission)
print("Submission file saved as 'submission.csv'")
