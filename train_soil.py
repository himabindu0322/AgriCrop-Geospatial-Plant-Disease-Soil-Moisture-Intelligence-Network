import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("dataset/Plant_Parameters.csv")

print(df.head())

# Remove text column
df = df.drop("Plant Type", axis=1)

# Features
X = df.drop("Moisture", axis=1)

# Target
y = df["Moisture"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Accuracy
score = model.score(X_test, y_test)

print(f"Soil Moisture Prediction Accuracy: {score:.4f}")

# Save model
joblib.dump(
    model,
    "models/soil_model.pkl"
)

print("Soil Model Saved Successfully!")
