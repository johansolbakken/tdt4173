import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Load the training data
ais_train = pd.read_csv("ais_train.csv", sep='|')
ais_test = pd.read_csv("ais_test.csv")

# Feature Engineering: Time-related features
ais_train['time'] = pd.to_datetime(ais_train['time'])
ais_train['elapsed_time'] = (ais_train['time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
ais_train['day_of_week'] = ais_train['time'].dt.dayofweek
ais_train['hour_of_day'] = ais_train['time'].dt.hour
ais_train['month'] = ais_train['time'].dt.month

# Encode vesselId
vessel_mapping = {vessel: idx for idx, vessel in enumerate(ais_train['vesselId'].unique())}
ais_train['vesselId'] = ais_train['vesselId'].map(vessel_mapping)
ais_test['vesselId'] = ais_test['vesselId'].map(vessel_mapping)

# If any vesselId is missing in ais_test (not seen in ais_train), set it to a new category
ais_test['vesselId'].fillna(-1, inplace=True)

# Input features for training
input_features = ['elapsed_time', 'vesselId', 'day_of_week', 'hour_of_day', 'month']

# Separate target columns for latitude and longitude
X_train, X_val, y_train_lat, y_val_lat = train_test_split(ais_train[input_features], ais_train['latitude'], test_size=0.2, random_state=42)
X_train_lon, X_val_lon, y_train_lon, y_val_lon = train_test_split(ais_train[input_features], ais_train['longitude'], test_size=0.2, random_state=42)

# Convert vesselId to categorical for LightGBM
categorical_features = ['vesselId']

# LightGBM model
lgb_model = lgb.LGBMRegressor()

# Define the parameter grid to search
param_grid = {
    'num_leaves': [20, 31, 40],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [500, 1000, 2000],
    'max_depth': [7, 8, 9],
    'min_child_samples': [20, 30],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Set up the GridSearchCV for latitude model
grid_search_lat = GridSearchCV(estimator=lgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
grid_search_lat.fit(X_train, y_train_lat)

# Output the best parameters and best score for latitude
print(f"Best parameters for latitude: {grid_search_lat.best_params_}")
print(f"Best score for latitude (neg RMSE): {grid_search_lat.best_score_}")

# Train longitude model
grid_search_lon = GridSearchCV(estimator=lgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
grid_search_lon.fit(X_train_lon, y_train_lon)

# Output the best parameters and best score for longitude
print(f"Best parameters for longitude: {grid_search_lon.best_params_}")
print(f"Best score for longitude (neg RMSE): {grid_search_lon.best_score_}")

# Use the best estimators for final predictions on the validation set
best_model_lat = grid_search_lat.best_estimator_
best_model_lon = grid_search_lon.best_estimator_

y_pred_lat = best_model_lat.predict(X_val)
y_pred_lon = best_model_lon.predict(X_val_lon)

# Evaluate the models
rmse_lat = np.sqrt(mean_squared_error(y_val_lat, y_pred_lat))
rmse_lon = np.sqrt(mean_squared_error(y_val_lon, y_pred_lon))

print(f"Validation RMSE for latitude: {rmse_lat}")
print(f"Validation RMSE for longitude: {rmse_lon}")

# Preprocess the test data (time-based features)
ais_test['time'] = pd.to_datetime(ais_test['time'])
ais_test['elapsed_time'] = (ais_test['time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
ais_test['day_of_week'] = ais_test['time'].dt.dayofweek
ais_test['hour_of_day'] = ais_test['time'].dt.hour
ais_test['month'] = ais_test['time'].dt.month

# Make predictions using the best models
X_test = ais_test[input_features]
test_predictions_lat = best_model_lat.predict(X_test)
test_predictions_lon = best_model_lon.predict(X_test)

# Create the submission DataFrame (modify based on actual target columns)
submission_df = pd.DataFrame({
    'ID': ais_test['ID'].values,
    'latitude_predicted': test_predictions_lat,
    'longitude_predicted': test_predictions_lon
})

# Save the submission file to CSV
submission_df.to_csv("submission.csv", index=False)

# Print out the first few rows of the submission for verification
print(submission_df.head())
