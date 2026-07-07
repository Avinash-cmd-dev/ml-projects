import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
DATASET_PATH = "data/public/gesture_dataset/gesture_dataset.csv"

df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("PREPROCESSING")
print("=" * 60)

# Features
X = df[["ax", "ay", "az", "gx", "gy", "gz"]]

# Labels
y = df["label"]

print("\nFeature Shape:")
print(X.shape)

print("\nLabel Shape:")
print(y.shape)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\nGesture Classes:")
for i, gesture in enumerate(encoder.classes_):
    print(f"{i} --> {gesture}")

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFirst Five Normalized Samples:")
print(X_scaled[:5])

# Save processed dataset
processed = pd.DataFrame(X_scaled, columns=X.columns)
processed["label"] = y_encoded

processed.to_csv("data/processed/processed_dataset.csv", index=False)

print("\nProcessed dataset saved successfully!")
