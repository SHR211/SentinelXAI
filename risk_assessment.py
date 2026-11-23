import pandas as pd
import numpy as np
import joblib

# Load saved model
model = joblib.load("sentinelxai_model.pkl")

# Load saved feature names
feature_names = joblib.load("sentinelxai_features.pkl")

# Load UNSW-NB15 testing dataset
df = pd.read_csv("UNSW_NB15_testing-set.csv")

# Keep the original label for checking
actual_label = df["label"].iloc[0]

# Remove columns that are not model input
X = df.drop(columns=["id", "label", "attack_cat"], errors="ignore")

# Convert categorical columns into numbers
X = pd.get_dummies(
    X,
    columns=["proto", "service", "state"],
    drop_first=True
)

# Make sure testing features have exactly the same columns as training
X = X.reindex(columns=feature_names, fill_value=0)

# Take one network activity for assessment
sample = X.iloc[[0]]

# Predict
prediction = model.predict(sample)[0]

# Get attack probability
attack_probability = model.predict_proba(sample)[0][1]

# Convert probability into risk score
risk_score = round(attack_probability * 100, 2)

# Determine risk level
if risk_score < 30:
    risk_level = "Low"
elif risk_score < 60:
    risk_level = "Medium"
elif risk_score < 80:
    risk_level = "High"
else:
    risk_level = "Critical"

# Display result
print("\n========== SentinelXAI Risk Assessment ==========")

if prediction == 0:
    print("Status       : NORMAL")
else:
    print("Status       : ATTACK")

print("Attack Probability :", round(attack_probability * 100, 2), "%")
print("Risk Score         :", risk_score, "/ 100")
print("Risk Level         :", risk_level)

print("Actual Dataset Label:", actual_label)

print("=================================================")