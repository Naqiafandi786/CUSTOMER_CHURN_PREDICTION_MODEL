# -------------------------------
# TRAIN TELCO CUSTOMER CHURN MODEL
# -------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import os

# --- 1. Load dataset ---
data = pd.read_csv("DATA/churn_data.csv")

# --- 2. Clean and preprocess ---
# Remove spaces from column names
data.columns = data.columns.str.strip()

# Convert 'TotalCharges' to numeric
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors='coerce')

# Drop missing values
data.dropna(inplace=True)

# Drop 'customerID'
data.drop("customerID", axis=1, inplace=True)

# Encode target column
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

# --- 3. Feature Engineering ---
data["NumServices"] = (
    (data["PhoneService"] == "Yes").astype(int)
    + (data["MultipleLines"] == "Yes").astype(int)
    + (data["OnlineSecurity"] == "Yes").astype(int)
    + (data["OnlineBackup"] == "Yes").astype(int)
    + (data["DeviceProtection"] == "Yes").astype(int)
    + (data["TechSupport"] == "Yes").astype(int)
    + (data["StreamingTV"] == "Yes").astype(int)
    + (data["StreamingMovies"] == "Yes").astype(int)
)

# Group tenure
bins = [0, 12, 24, 48, 60, 100]
labels = ['0-12', '12-24', '24-48', '48-60', '60+']
data['tenure_group'] = pd.cut(data['tenure'], bins=bins, labels=labels, right=False)

# Average monthly cost
data['AvgMonthly'] = data['TotalCharges'] / (data['tenure'] + 1)

# --- 4. One-hot encoding ---
data_encoded = pd.get_dummies(data, drop_first=False)

# --- 5. Split into train/test ---
X = data_encoded.drop("Churn", axis=1)
y = data_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 6. Handle imbalance ---
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# --- 7. Train Random Forest model ---
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_res, y_train_res)

# --- 8. Evaluate model ---
y_pred = rf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- 9. Save model and feature columns ---
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/rf_churn_model.joblib")
joblib.dump(list(X.columns), "models/feature_columns.joblib")

print("\n✅ Model training complete and saved successfully!")