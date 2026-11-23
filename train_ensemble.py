import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


# =========================================================
# FILE PATHS
# =========================================================

TRAIN_FILE = "UNSW_NB15_training-set.csv"
TEST_FILE = "UNSW_NB15_testing-set.csv"

RF_FILE = "sentinelxai_rf.pkl"
XGB_FILE = "sentinelxai_xgb.pkl"
LR_FILE = "sentinelxai_lr.pkl"

FEATURE_FILE = "sentinelxai_features.pkl"
SCALER_FILE = "sentinelxai_scaler.pkl"


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading training dataset...")

train_df = pd.read_csv(TRAIN_FILE)

print("Training shape:", train_df.shape)

print("\nLoading testing dataset...")

test_df = pd.read_csv(TEST_FILE)

print("Testing shape:", test_df.shape)


# =========================================================
# PREPROCESSING
# =========================================================

def preprocess_dataframe(df):

    data = df.copy()

    # Remove columns that should not be ML inputs
    columns_to_drop = []

    if "id" in data.columns:
        columns_to_drop.append("id")

    if "attack_cat" in data.columns:
        columns_to_drop.append("attack_cat")

    if columns_to_drop:
        data = data.drop(
            columns=columns_to_drop
        )

    # Convert categorical columns
    categorical_columns = [
        "proto",
        "service",
        "state"
    ]

    existing_columns = [
        col
        for col in categorical_columns
        if col in data.columns
    ]

    if existing_columns:

        data = pd.get_dummies(
            data,
            columns=existing_columns,
            drop_first=True
        )

    return data


# =========================================================
# PREPARE TRAINING DATA
# =========================================================

train_processed = preprocess_dataframe(
    train_df
)

test_processed = preprocess_dataframe(
    test_df
)


# =========================================================
# SEPARATE X AND Y
# =========================================================

X_train = train_processed.drop(
    columns=["label"]
)

y_train = train_processed["label"]


X_test = test_processed.drop(
    columns=["label"]
)

y_test = test_processed["label"]


# =========================================================
# MATCH TEST FEATURES WITH TRAIN FEATURES
# =========================================================

feature_names = X_train.columns.tolist()

X_test = X_test.reindex(
    columns=feature_names,
    fill_value=0
)


# Make sure numeric
X_train = X_train.apply(
    pd.to_numeric,
    errors="coerce"
)

X_test = X_test.apply(
    pd.to_numeric,
    errors="coerce"
)

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)


print("\nFinal feature count:", len(feature_names))

print("X_train:", X_train.shape)

print("X_test :", X_test.shape)


# =========================================================
# SCALE DATA FOR LOGISTIC REGRESSION
# =========================================================

print("\nPreparing StandardScaler...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# =========================================================
# MODEL 1 — RANDOM FOREST
# =========================================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

rf_model = RandomForestClassifier(

    n_estimators=150,

    random_state=42,

    n_jobs=-1,

    class_weight="balanced"

)

rf_model.fit(
    X_train,
    y_train
)


rf_pred = rf_model.predict(
    X_test
)

rf_probability = rf_model.predict_proba(
    X_test
)[:, 1]


rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)


print(
    f"Random Forest Accuracy: "
    f"{rf_accuracy * 100:.2f}%"
)


# =========================================================
# MODEL 2 — XGBOOST
# =========================================================

print("\n" + "=" * 60)
print("TRAINING XGBOOST")
print("=" * 60)

xgb_model = XGBClassifier(

    n_estimators=150,

    max_depth=6,

    learning_rate=0.1,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=42,

    n_jobs=-1

)

xgb_model.fit(
    X_train,
    y_train
)


xgb_pred = xgb_model.predict(
    X_test
)

xgb_probability = xgb_model.predict_proba(
    X_test
)[:, 1]


xgb_accuracy = accuracy_score(
    y_test,
    xgb_pred
)


print(
    f"XGBoost Accuracy: "
    f"{xgb_accuracy * 100:.2f}%"
)


# =========================================================
# MODEL 3 — LOGISTIC REGRESSION
# =========================================================

print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)

lr_model = LogisticRegression(

    max_iter=1000,

    solver="liblinear",

    random_state=42

)

lr_model.fit(
    X_train_scaled,
    y_train
)


lr_pred = lr_model.predict(
    X_test_scaled
)

lr_probability = lr_model.predict_proba(
    X_test_scaled
)[:, 1]


lr_accuracy = accuracy_score(
    y_test,
    lr_pred
)


print(
    f"Logistic Regression Accuracy: "
    f"{lr_accuracy * 100:.2f}%"
)


# =========================================================
# 3-MODEL ENSEMBLE
# =========================================================

print("\n" + "=" * 60)
print("CREATING 3-MODEL ENSEMBLE")
print("=" * 60)

# Equal weighting
ensemble_probability = (

    0.40 * rf_probability

    + 0.40 * xgb_probability

    + 0.20 * lr_probability

)


ensemble_prediction = (
    ensemble_probability >= 0.50
).astype(int)


ensemble_accuracy = accuracy_score(
    y_test,
    ensemble_prediction
)


print(
    f"\nEnsemble Accuracy: "
    f"{ensemble_accuracy * 100:.2f}%"
)


# =========================================================
# ENSEMBLE REPORT
# =========================================================

print("\n" + "=" * 60)
print("ENSEMBLE CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        ensemble_prediction
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        ensemble_prediction
    )
)


# =========================================================
# SAVE MODELS
# =========================================================

print("\n" + "=" * 60)
print("SAVING MODELS")
print("=" * 60)


joblib.dump(
    rf_model,
    RF_FILE
)

print(
    "Saved:",
    RF_FILE
)


joblib.dump(
    xgb_model,
    XGB_FILE
)

print(
    "Saved:",
    XGB_FILE
)


joblib.dump(
    lr_model,
    LR_FILE
)

print(
    "Saved:",
    LR_FILE
)


joblib.dump(
    feature_names,
    FEATURE_FILE
)

print(
    "Saved:",
    FEATURE_FILE
)


joblib.dump(
    scaler,
    SCALER_FILE
)

print(
    "Saved:",
    SCALER_FILE
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("SENTINELXAI TRAINING COMPLETED")
print("=" * 60)

print(
    f"Random Forest Accuracy     : "
    f"{rf_accuracy * 100:.2f}%"
)

print(
    f"XGBoost Accuracy           : "
    f"{xgb_accuracy * 100:.2f}%"
)

print(
    f"Logistic Regression       : "
    f"{lr_accuracy * 100:.2f}%"
)

print(
    f"3-Model Ensemble Accuracy : "
    f"{ensemble_accuracy * 100:.2f}%"
)

print("\nModels saved successfully.")