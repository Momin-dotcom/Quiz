import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load model
model = joblib.load("models/logistic_model.pkl")

# Load test data
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Predict
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

with open("results/metrics.txt", "w") as f:
    f.write(f"Final Accuracy: {accuracy:.4f}")

print(f"Final Accuracy: {accuracy:.4f}")