import pandas as pd
import numpy as np

print("=" * 60)
print("FEATURE EXTRACTION")
print("=" * 60)

# Load dataset
DATASET_PATH = "data/public/gesture_dataset/gesture_dataset.csv"

df = pd.read_csv(DATASET_PATH)

# Sensor columns
sensor_columns = [
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz"
]

feature_rows = []

# Process every gesture sample
for sample_id, sample in df.groupby("sample_id"):

    features = {}

    features["sample_id"] = sample_id

    # Label of the complete gesture
    features["label"] = sample["label"].iloc[0]

    for sensor in sensor_columns:

        values = sample[sensor]

        features[f"{sensor}_mean"] = values.mean()

        features[f"{sensor}_std"] = values.std()

        features[f"{sensor}_min"] = values.min()

        features[f"{sensor}_max"] = values.max()

        features[f"{sensor}_rms"] = np.sqrt(np.mean(values ** 2))

    feature_rows.append(features)

# Create DataFrame
features_df = pd.DataFrame(feature_rows)

print("\nFeature Dataset Shape:")
print(features_df.shape)

print("\nFirst Five Samples:")
print(features_df.head())

# Save feature dataset
OUTPUT_PATH = "data/processed/features_dataset.csv"

features_df.to_csv(OUTPUT_PATH, index=False)

print("\nFeature dataset saved successfully!")