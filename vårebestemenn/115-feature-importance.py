import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load training and test data
train = pd.read_csv("ais_train.csv", sep="|")
test = pd.read_csv("ais_test.csv", sep=",")

# Convert 'time' column to datetime
train["time"] = pd.to_datetime(train["time"])
test["time"] = pd.to_datetime(test["time"])

# Map 'vesselId' to unique integers
le = LabelEncoder()
train["vesselId"] = le.fit_transform(train["vesselId"])
test["vesselId"] = le.transform(test["vesselId"])

# Sort datasets by 'vesselId' and 'time'
train = train.sort_values(by=["vesselId", "time"])
test = test.sort_values(by=["vesselId", "time"])

# Create 'previous_lat', 'previous_lon', and 'delta_time' in the training set
train["previous_lat"] = train.groupby("vesselId")["latitude"].shift(1)
train["previous_lon"] = train.groupby("vesselId")["longitude"].shift(1)
train["delta_time"] = train.groupby("vesselId")["time"].diff().dt.total_seconds()

# Drop rows with missing values resulting from the shift operation
train = train.dropna(subset=["previous_lat", "previous_lon", "delta_time"])

# Prepare training features and targets
X_train = train[["vesselId", "previous_lat", "previous_lon", "delta_time"]]
y_train_lat = train["latitude"]
y_train_lon = train["longitude"]

# Train separate Random Forest models for latitude and longitude
model_lat = RandomForestRegressor(n_estimators=50, random_state=42)
model_lat.fit(X_train, y_train_lat)

model_lon = RandomForestRegressor(n_estimators=50, random_state=42)
model_lon.fit(X_train, y_train_lon)

# Feature Importance for Latitude Model
lat_feature_importance = pd.DataFrame(
    {"Feature": X_train.columns, "Importance": model_lat.feature_importances_}
).sort_values(by="Importance", ascending=False)

# Feature Importance for Longitude Model
lon_feature_importance = pd.DataFrame(
    {"Feature": X_train.columns, "Importance": model_lon.feature_importances_}
).sort_values(by="Importance", ascending=False)

# Display the feature importances
print("Feature Importance for Latitude Prediction Model:")
print(lat_feature_importance)
print("\nFeature Importance for Longitude Prediction Model:")
print(lon_feature_importance)

# Plot feature importance for both models
plt.figure(figsize=(12, 6))

# Plot for Latitude Model
plt.subplot(1, 2, 1)
plt.barh(lat_feature_importance["Feature"], lat_feature_importance["Importance"])
plt.xlabel("Importance Score")
plt.title("Feature Importance for Latitude Model")
plt.gca().invert_yaxis()  # Highest importance at the top

# Plot for Longitude Model
plt.subplot(1, 2, 2)
plt.barh(lon_feature_importance["Feature"], lon_feature_importance["Importance"])
plt.xlabel("Importance Score")
plt.title("Feature Importance for Longitude Model")
plt.gca().invert_yaxis()  # Highest importance at the top

plt.tight_layout()
plt.show()
