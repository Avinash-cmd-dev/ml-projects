import pandas as pd
import numpy as np

print("=" * 60)
print("FEATURE EXTRACTION")
print("=" * 60)

# Load preprocessed dataset
DATASET_PATH = "data/public/gesture_dataset/gesture_dataset.csv"

df = pd.read_csv(DATASET_PATH)

feature_rows = []

# Group by BOTH label and sample_id
for (label, sample_id), sample in df.groupby(["label", "sample_id"]):

    features = {}

    features["sample_id"] = sample_id
    features["label"] = label

    # Accelerometer Features
    for axis in ["ax", "ay", "az"]:
        features[f"{axis}_mean"] = sample[axis].mean()
        features[f"{axis}_std"] = sample[axis].std()
        features[f"{axis}_min"] = sample[axis].min()
        features[f"{axis}_max"] = sample[axis].max()
        features[f"{axis}_rms"] = np.sqrt(np.mean(sample[axis] ** 2))

    # Gyroscope Features
    for axis in ["gx", "gy", "gz"]:
        features[f"{axis}_mean"] = sample[axis].mean()
        features[f"{axis}_std"] = sample[axis].std()
        features[f"{axis}_min"] = sample[axis].min()
        features[f"{axis}_max"] = sample[axis].max()
        features[f"{axis}_rms"] = np.sqrt(np.mean(sample[axis] ** 2))

    feature_rows.append(features)

# Create DataFrame
feature_df = pd.DataFrame(feature_rows)

print("\nFeature Dataset Shape:")
print(feature_df.shape)

print("\nGesture Distribution:")
print(feature_df["label"].value_counts())

print("\nFirst Five Samples:")
print(feature_df.head())

# Save feature dataset
feature_df.to_csv("data/processed/features_dataset.csv", index=False)

print("\nFeature dataset saved successfully!")