#!/usr/bin/env python3

import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import LightGBMModel
import lightgbm as lgb

# Load ais_train.csv with separator '|'
train_df = pd.read_csv("ais_train.csv", sep="|")
train_df["time"] = pd.to_datetime(train_df["time"])

# Load ais_test.csv with separator ','
test_df = pd.read_csv("ais_test.csv", sep=",")
test_df["time"] = pd.to_datetime(test_df["time"])

# Use 'vesselId' instead of 'vessel_id'
# Select only vessel IDs that are in both train and test datasets
common_vessel_ids = set(train_df["vesselId"]).intersection(set(test_df["vesselId"]))
train_df = train_df[train_df["vesselId"].isin(common_vessel_ids)]

# Group the training data by vesselId
groups = train_df.groupby("vesselId")

# Initialize dictionaries to store TimeSeries objects, last training times, and feature importances
timeseries_dict = {}
last_train_time = {}
feature_importances = {}

# Process each vesselId group
for vessel_id, group_df in groups:
    # Sort on time
    group_df = group_df.sort_values("time")
    # Set index to time
    group_df = group_df.set_index("time")
    # Select features (latitude and longitude)
    features_df = group_df[["latitude", "longitude"]]
    # Resample data to hourly frequency with mean and linear interpolation
    features_df = features_df.resample("h").mean().interpolate(method="cubic")
    # Create Darts TimeSeries object
    ts = TimeSeries.from_dataframe(features_df, value_cols=["latitude", "longitude"])
    # Store the TimeSeries object and last training time
    timeseries_dict[vessel_id] = ts
    last_train_time[vessel_id] = features_df.index.max()

# Initialize a dictionary to store predictions
predictions = {}

# Fit LightGBM models, predict, and compute feature importance for each TimeSeries object
for vessel_id, ts in timeseries_dict.items():
    # Get the last training time
    last_time = last_train_time[vessel_id]
    # Get test times for this vessel
    vessel_test_df = test_df[test_df["vesselId"] == vessel_id]
    test_times = vessel_test_df["time"]
    # Compute the time differences in hours
    time_diffs = (test_times - last_time).dt.total_seconds() / 3600
    # Get the maximum forecast horizon needed
    max_n = int(np.ceil(time_diffs.max()))
    if max_n <= 0:
        continue  # Skip if no future times to predict
    # Initialize LightGBM model with lag parameters
    model = LightGBMModel(
        lags=48,  # the correct value is between 48 and 96
        learning_rate=0.1,
    )
    # Fit the model
    model.fit(ts)
    # Predict up to the maximum horizon needed
    forecast = model.predict(max_n)
    # Store the forecast and last time
    predictions[vessel_id] = (forecast, last_time)

    # Extract feature importance from the LightGBM model
    lgb_model = model.model  # Access the underlying LightGBM model
    importance = lgb_model.feature_importance(importance_type="split")
    feature_importances[vessel_id] = importance

# Display feature importances
for vessel_id, importance in feature_importances.items():
    print(f"Feature importance for vessel {vessel_id}: {importance}")

# Initialize a list to store submission rows
submission_rows = []

# Generate predictions for the submission file
for idx, row in test_df.iterrows():
    vessel_id = row["vesselId"]
    test_time = row["time"]
    test_id = row["ID"]  # Assuming 'ID' column exists in test_df
    # Check if predictions are available for this vessel_id
    if vessel_id in predictions:
        forecast_ts, last_time = predictions[vessel_id]
        time_diff = (test_time - last_time).total_seconds() / 3600
        index = (
            int(np.round(time_diff)) - 1
        )  # Adjust index since forecast starts from last_time + 1 hour
        # Convert forecast_ts to DataFrame
        forecast_df = forecast_ts.pd_dataframe()
        # Check if index is within forecast horizon
        if 0 <= index < len(forecast_df):
            predicted_lat = forecast_df["latitude"].iloc[index]
            predicted_lon = forecast_df["longitude"].iloc[index]
        else:
            predicted_lat = np.nan
            predicted_lon = np.nan
    else:
        predicted_lat = np.nan
        predicted_lon = np.nan
    # Append the prediction to the submission list
    submission_rows.append(
        {
            "ID": test_id,
            "longitude_predicted": predicted_lon,
            "latitude_predicted": predicted_lat,
        }
    )

# Create a submission DataFrame from the list
submission_df = pd.DataFrame(submission_rows)

# Save the submission file
submission_df.to_csv("submission.csv", index=False)

print(submission_df)
