import pandas as pd
import joblib
import shap

# Load saved model and feature list
model = joblib.load("sentinelxai_model.pkl")
feature_names = joblib.load("sentinelxai_features.pkl")

# Load testing dataset
df = pd.read_csv("UNSW_NB15_testing-set.csv")

# Keep actual label
actual_label = df["label"].iloc[0]

# Prepare input data
X = df.drop(columns=["id", "label", "attack_cat"], errors="ignore")

# Convert categorical columns into numbers
X = pd.get_dummies(
    X,
    columns=["proto", "service", "state"],
    drop_first=True
)

# Match training features exactly
X = X.reindex(columns=feature_names, fill_value=0)

# Select one network activity
sample = X.iloc[[0]]

# Model prediction
prediction = model.predict(sample)[0]

# Attack probability
attack_probability = model.predict_proba(sample)[0][1]

# Risk score
risk_score = round(attack_probability * 100, 2)

# Risk level
if risk_score < 30:
    risk_level = "Low"
elif risk_score < 60:
    risk_level = "Medium"
elif risk_score < 80:
    risk_level = "High"
else:
    risk_level = "Critical"


# -------------------------------
# SHAP EXPLANATION
# -------------------------------

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

# Handle different SHAP output formats
if isinstance(shap_values, list):
    values = shap_values[1][0]
else:
    if len(shap_values.shape) == 3:
        values = shap_values[0, :, 1]
    else:
        values = shap_values[0]

# Create feature contribution table
explanation = pd.DataFrame({
    "Feature": feature_names,
    "SHAP_Value": values,
    "Feature_Value": sample.iloc[0].values
})

# Sort by absolute contribution
explanation["Absolute_Impact"] = explanation["SHAP_Value"].abs()
explanation = explanation.sort_values(
    by="Absolute_Impact",
    ascending=False
)

# Display result
print("\n========== SentinelXAI XAI Explanation ==========")

print("Prediction        :", "ATTACK" if prediction == 1 else "NORMAL")
print("Attack Probability:", round(attack_probability * 100, 2), "%")
print("Risk Score        :", risk_score, "/ 100")
print("Risk Level        :", risk_level)
print("Actual Label      :", actual_label)

print("\nTop 10 Important Features:")

for i, row in explanation.head(10).iterrows():

    direction = "INCREASED RISK" if row["SHAP_Value"] > 0 else "DECREASED RISK"

    print(
        f"{row['Feature']} : "
        f"{direction} "
        f"(SHAP = {row['SHAP_Value']:.4f})"
    )

print("\n=================================================")