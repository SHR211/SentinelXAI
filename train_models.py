import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# =========================================================
# SETTINGS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_FILE = os.path.join(
    BASE_DIR,
    "UNSW_NB15_training-set.csv"
)


# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading training dataset...")

df = pd.read_csv(TRAIN_FILE)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# =========================================================
# PREPROCESSING
# =========================================================

def preprocess_training_data(df):

    data = df.copy()

    # Remove target-related / identifier columns
    for column in ["id", "attack_cat", "label"]:
        if column in data.columns:
            data = data.drop(columns=[column])

    # Categorical columns
    categorical_columns = [
        "proto",
        "service",
        "state"
    ]

    categorical_columns = [
        col
        for col in categorical_columns
        if col in data.columns
    ]

    # One-hot encoding
    if categorical_columns:

        data = pd.get_dummies(
            data,
            columns=categorical_columns,
            drop_first=True
        )

    # Convert all values to numeric
    for column in data.columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Replace infinite values
    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing values
    data = data.fillna(0)

    return data


# =========================================================
# PREPARE X AND Y
# =========================================================

print("\nPreparing features...")

X = preprocess_training_data(df)

y = df["label"].astype(int)

print("Processed feature shape:", X.shape)

print("\nClass distribution:")
print(y.value_counts())


# =========================================================
# SAVE FEATURE NAMES
# =========================================================

feature_names = X.columns.tolist()

joblib.dump(
    feature_names,
    os.path.join(
        BASE_DIR,
        "sentinelxai_features.pkl"
    )
)

print("\nFeature names saved.")


# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))


# =========================================================
# 1. RANDOM FOREST
# =========================================================

print("\n==============================")
print("Training Random Forest...")
print("==============================")

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(X_val)

rf_accuracy = accuracy_score(
    y_val,
    rf_pred
)

print(
    f"Random Forest Accuracy: "
    f"{rf_accuracy * 100:.2f}%"
)


# =========================================================
# 2. XGBOOST
# =========================================================

print("\n==============================")
print("Training XGBoost...")
print("==============================")

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)

xgb_model.fit(
    X_train,
    y_train
)

xgb_pred = xgb_model.predict(X_val)

xgb_accuracy = accuracy_score(
    y_val,
    xgb_pred
)

print(
    f"XGBoost Accuracy: "
    f"{xgb_accuracy * 100:.2f}%"
)


# =========================================================
# 3. LOGISTIC REGRESSION
# =========================================================

print("\n==============================")
print("Training Logistic Regression...")
print("==============================")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_val_scaled = scaler.transform(
    X_val
)

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

lr_model.fit(
    X_train_scaled,
    y_train
)

lr_pred = lr_model.predict(
    X_val_scaled
)

lr_accuracy = accuracy_score(
    y_val,
    lr_pred
)

print(
    f"Logistic Regression Accuracy: "
    f"{lr_accuracy * 100:.2f}%"
)


# =========================================================
# SAVE MODELS
# =========================================================

print("\nSaving trained models...")

joblib.dump(
    rf_model,
    os.path.join(
        BASE_DIR,
        "sentinelxai_rf.pkl"
    )
)

joblib.dump(
    xgb_model,
    os.path.join(
        BASE_DIR,
        "sentinelxai_xgb.pkl"
    )
)

joblib.dump(
    lr_model,
    os.path.join(
        BASE_DIR,
        "sentinelxai_lr.pkl"
    )
)

joblib.dump(
    scaler,
    os.path.join(
        BASE_DIR,
        "sentinelxai_scaler.pkl"
    )
)

# =========================================================
# ENSEMBLE VALIDATION
# =========================================================

print("\n==============================")
print("ENSEMBLE VALIDATION")
print("==============================")

rf_probability = rf_model.predict_proba(
    X_val
)[:, 1]

xgb_probability = xgb_model.predict_proba(
    X_val
)[:, 1]

lr_probability = lr_model.predict_proba(
    X_val_scaled
)[:, 1]


# Weighted ensemble
final_probability = (
    0.40 * rf_probability
    + 0.40 * xgb_probability
    + 0.20 * lr_probability
)


final_prediction = (
    final_probability >= 0.50
).astype(int)


ensemble_accuracy = accuracy_score(
    y_val,
    final_prediction
)

print(
    f"Ensemble Accuracy: "
    f"{ensemble_accuracy * 100:.2f}%"
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        final_prediction
    )
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_val,
        final_prediction
    )
)


# =========================================================
# FINISHED
# =========================================================

print("\n===================================")
print("MODEL TRAINING COMPLETED")
print("===================================")

print("\nGenerated files:")

print("sentinelxai_rf.pkl")
print("sentinelxai_xgb.pkl")
print("sentinelxai_lr.pkl")
print("sentinelxai_features.pkl")
print("sentinelxai_scaler.pkl")
