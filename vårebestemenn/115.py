import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load training data
train = pd.read_csv('ais_train.csv', sep='|')

# Load test data
test = pd.read_csv('ais_test.csv', sep=',')

# Convert 'time' column to datetime
train['time'] = pd.to_datetime(train['time'])
test['time'] = pd.to_datetime(test['time'])

# Map 'vesselId' to unique integers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['vesselId'] = le.fit_transform(train['vesselId'])
test['vesselId'] = le.transform(test['vesselId'])

# Sort datasets by 'vesselId' and 'time'
train = train.sort_values(by=['vesselId', 'time'])
test = test.sort_values(by=['vesselId', 'time'])

# Create 'previous_lat', 'previous_lon', and 'delta_time' in the training set
train['previous_lat'] = train.groupby('vesselId')['latitude'].shift(1)
train['previous_lon'] = train.groupby('vesselId')['longitude'].shift(1)
train['delta_time'] = train.groupby('vesselId')['time'].diff().dt.total_seconds()

# Drop rows with missing values resulting from the shift operation
train = train.dropna(subset=['previous_lat', 'previous_lon', 'delta_time'])

# Prepare training features and targets
X_train = train[['vesselId', 'previous_lat', 'previous_lon', 'delta_time']]
y_train_lat = train['latitude']
y_train_lon = train['longitude']

# Initialize 'previous_lat', 'previous_lon', and 'delta_time' in the test set
test['previous_lat'] = np.nan
test['previous_lon'] = np.nan
test['delta_time'] = np.nan

# Retrieve last known positions from the training set
last_positions = train.groupby('vesselId').apply(lambda x: x.iloc[-1])[['vesselId', 'latitude', 'longitude', 'time']]
last_positions = last_positions.set_index('vesselId')

# Train separate Random Forest models for latitude and longitude
model_lat = RandomForestRegressor(n_estimators=50, random_state=42)
model_lat.fit(X_train, y_train_lat)

model_lon = RandomForestRegressor(n_estimators=50, random_state=42)
model_lon.fit(X_train, y_train_lon)

# Prepare a list to collect the prediction results
submission_rows = []

# Loop over each vessel in the test data
for vessel_id in test['vesselId'].unique():
    vessel_test_data = test[test['vesselId'] == vessel_id].copy()
    vessel_test_data = vessel_test_data.sort_values(by='time')
    
    # Check if the vessel_id exists in the last_positions
    if vessel_id in last_positions.index:
        prev_lat = last_positions.loc[vessel_id, 'latitude']
        prev_lon = last_positions.loc[vessel_id, 'longitude']
        last_time = last_positions.loc[vessel_id, 'time']
    else:
        # If vessel_id is not in the training data, skip prediction
        continue
    
    # Iterate over each record for the vessel
    for idx, row in vessel_test_data.iterrows():
        delta_time = (row['time'] - last_time).total_seconds()
        
        # Prepare the feature vector
        X_test_row = pd.DataFrame({
            'vesselId': [vessel_id],
            'previous_lat': [prev_lat],
            'previous_lon': [prev_lon],
            'delta_time': [delta_time]
        })
        
        # Predict latitude and longitude
        predicted_lat = model_lat.predict(X_test_row)[0]
        predicted_lon = model_lon.predict(X_test_row)[0]
        
        # Update previous values for the next iteration
        prev_lat = predicted_lat
        prev_lon = predicted_lon
        last_time = row['time']
        
        # Append the prediction to the submission list
        submission_rows.append({
            'ID': row['ID'],
            'longitude_predicted': predicted_lon,
            'latitude_predicted': predicted_lat
        })

# Create a submission DataFrame from the list
submission_df = pd.DataFrame(submission_rows)

# Merge the predictions with the test data based on 'ID'
final_submission = test[['ID']].merge(submission_df, on='ID', how='left')

# Save the submission file
final_submission.to_csv('submission.csv', index=False)
