import pandas as pd
import joblib
#import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# # Create models folder if not exists
# os.makedirs("models", exist_ok=True)

# Load processed data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train.values.ravel())

# Save model
joblib.dump(model, "models/logistic_model.pkl")

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")