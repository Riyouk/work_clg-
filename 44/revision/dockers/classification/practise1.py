print("\nChurn Prediction ML Pipeline Started...\n")

# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv("question1.csv")   # CSV must be in same folder

print("Data Loaded Successfully!")
print(df.head())
print("\nShape:", df.shape)

# ---------------------------------------------------------
# OUTLIER HANDLING (IQR CLIPPING)
# ---------------------------------------------------------
numeric_cols = [
    "tenure_months", "monthly_charges", "total_charges",
    "num_complaints", "avg_daily_usage_minutes",
    "days_since_last_login", "age"
]

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)

print("\nOutliers handled using IQR method.")

# ---------------------------------------------------------
# ENCODING CATEGORICAL VARIABLES
# ---------------------------------------------------------
df["tech_support"] = df["tech_support"].map({"Yes":1, "No":0})
df["gender"] = df["gender"].map({"Male":1, "Female":0})

df["is_autopay"] = df["is_autopay"].astype(int)
df["is_paperless"] = df["is_paperless"].astype(int)
df["is_premium_streaming"] = df["is_premium_streaming"].astype(int)

df = pd.get_dummies(df, columns=[
    "contract_type", "payment_method",
    "internet_service", "location"
], drop_first=True)

print("\nEncoding Completed.")

# ---------------------------------------------------------
# FEATURE SCALING
# ---------------------------------------------------------
X = df.drop(columns=["customer_id", "is_churn"])
y = df["is_churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature Scaling Done.")

# ---------------------------------------------------------
# TRAIN-TEST SPLIT (STRATIFIED)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

print("\nTrain-Test Split Completed.")

# ---------------------------------------------------------
# BASE MODELS
# ---------------------------------------------------------

# Logistic Regression
log_model = LogisticRegression(max_iter=500, class_weight="balanced")
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Random Forest (Baseline)
rf_base = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
rf_base.fit(X_train, y_train)
y_pred_rf_base = rf_base.predict(X_test)

print("\nBaseline Models Trained.")

# ---------------------------------------------------------
# BASE MODEL EVALUATION
# ---------------------------------------------------------
def evaluate(name, y_test, y_pred):
    print(f"\n--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

evaluate("Logistic Regression", y_test, y_pred_log)
evaluate("Random Forest (Baseline)", y_test, y_pred_rf_base)

print("\nConfusion Matrix (Baseline RF):")
print(confusion_matrix(y_test, y_pred_rf_base))

# ---------------------------------------------------------
# RANDOM FOREST HYPERPARAMETER TUNING (GRIDSEARCHCV)
# ---------------------------------------------------------
print("\nStarting Random Forest Hyperparameter Tuning...\n")

param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)

grid = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring="recall",       # Optimize for CHURN recall
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

best_rf = grid.best_estimator_

print("\nBest Parameters Found:")
print(grid.best_params_)

# Evaluate best model
y_pred_best = best_rf.predict(X_test)

print("\n----- BEST RANDOM FOREST PERFORMANCE -----")
evaluate("Best Random Forest", y_test, y_pred_best)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))

# ---------------------------------------------------------
# K-FOLD CROSS VALIDATION ON BEST MODEL
# ---------------------------------------------------------
scores = cross_val_score(best_rf, X_scaled, y, cv=cv, scoring="recall")
print("\nStratified K-Fold Recall Scores:", scores)
print("Mean Recall:", np.mean(scores))

# ---------------------------------------------------------
# DONE
# ---------------------------------------------------------
print("\nChurn Prediction Pipeline Finished Successfully!\n")
