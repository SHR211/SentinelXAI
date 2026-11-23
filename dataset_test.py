import pandas as pd

df = pd.read_csv("UNSW_NB15_training-set.csv")

print(df.head())
print(df.shape)
print(df.columns)
print("\nLabel Distribution:")
print(df["label"].value_counts())

print("\nAttack Categories:")
print(df["attack_cat"].value_counts())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nData Types:")
print(df.dtypes)

print("\nCategorical Columns:")
print(df.select_dtypes(include="object").columns)
# Convert categorical columns into numbers
df = pd.get_dummies(
    df,
    columns=['proto', 'service', 'state'],
    drop_first=True
)

print("After Encoding:")
print(df.shape)
# Separate features and target
X_train = df.drop(columns=['label', 'attack_cat', 'id'])
y_train = df['label']

print("\nFeatures:")
print(X_train.columns)

print("\nTarget:")
print(y_train.value_counts())
# Load testing dataset
test_df = pd.read_csv("UNSW_NB15_testing-set.csv")

# Encode categorical columns in testing data
test_df = pd.get_dummies(
    test_df,
    columns=['proto', 'service', 'state'],
    drop_first=True
)

# Separate testing features and target
X_test = test_df.drop(columns=['label', 'attack_cat', 'id'])
y_test = test_df['label']

# Make testing features match training features
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print("\nTraining Data:")
print(X_train.shape)

print("\nTesting Data:")
print(X_test.shape)

print("\nTesting Target:")
print(y_test.value_counts())
# Step 4: Train Random Forest Model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed!")

# Make predictions on testing data
y_pred = model.predict(X_test)

# Model accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import joblib

# Save the trained model
joblib.dump(model, "sentinelxai_model.pkl")

# Save the feature names
joblib.dump(X_train.columns.tolist(), "sentinelxai_features.pkl")

print("\nModel saved successfully!")
print("Feature list saved successfully!")