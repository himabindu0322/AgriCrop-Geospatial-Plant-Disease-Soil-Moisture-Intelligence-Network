import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv(
    "dataset/Plant_Parameters.csv"
)

print(df.head())

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

score = model.score(
    X_test,
    y_test
)

print("Accuracy:", score)

joblib.dump(
    model,
    "models/soil_model.pkl"
)

print("Soil Model Saved")
