import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Create processed folder if not exists
os.makedirs("data/processed", exist_ok=True)

# Load dataset (you can use any CSV, example: titanic.csv)
df = pd.read_csv("D:/ML OPS/titanic.csv")

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# Drop non-numeric columns (simple approach)
df = df.select_dtypes(include=["number"])

# Separate features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save processed data
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Preprocessing completed successfully.")