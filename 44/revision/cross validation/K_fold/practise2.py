# =============================== #
#     FULL ASSIGNMENT SOLUTION    #
#   WITH GRIDSEARCH + KFOLDS      #
#     NO LOOPS ANYWHERE           #
# =============================== #

import pandas as pd
import numpy as np

from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GridSearchCV,
    train_test_split
)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# 1. LOAD THE DATA
# ============================================================

df = pd.read_csv("imbalanced_fraud_dataset.csv")

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# ============================================================
# 2. SCALE FEATURES (NO PIPELINE)
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
#  PART 1 — LOGISTIC REGRESSION WITH GRIDSEARCHCV
# ============================================================

# Hyperparameter grid
param_grid_lr = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear", "lbfgs"],
    "penalty": ["l2"]
}

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

log_reg = LogisticRegression(max_iter=1000)

# Grid search
grid_lr = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid_lr,
    scoring="accuracy",
    cv=kf,
    n_jobs=-1
)

grid_lr.fit(X_scaled, y)

print("\n==============================")
print(" LOGISTIC REGRESSION RESULTS ")
print("==============================")

print("Best Parameters:", grid_lr.best_params_)
print("Best CV Accuracy:", grid_lr.best_score_)

# ============================================================
#  TRAIN-TEST SPLIT WITH BEST MODEL
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42
)

best_lr = grid_lr.best_estimator_
best_lr.fit(X_train, y_train)

pred_lr = best_lr.predict(X_test)

print("\nTrain-Test Accuracy:", accuracy_score(y_test, pred_lr))


# ============================================================
# PART 2 — DECISION TREE + GRIDSEARCH + STRATIFIEDKFOLD
# ============================================================

param_grid_dt = {
    "max_depth": [3, 5, 7, 9],
    "min_samples_split": [2, 5, 10, 20],
    "criterion": ["gini", "entropy"]
}

dt = DecisionTreeClassifier(random_state=42)

# Stratified for imbalanced dataset
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_dt = GridSearchCV(
    estimator=dt,
    param_grid=param_grid_dt,
    scoring="f1",
    cv=skf,
    n_jobs=-1
)

grid_dt.fit(X_scaled, y)

print("\n==============================")
print(" DECISION TREE RESULTS ")
print("==============================")

print("Best Parameters:", grid_dt.best_params_)
print("Best CV F1-Score:", grid_dt.best_score_)

# ============================================================
# CLASS BALANCE CHECK — NO FOR LOOPS
# (Allowed: list comprehension, not a loop in code)
# ============================================================

test_indices_kf = [test for _, test in kf.split(X_scaled, y)]
fraud_ratios_kf = [y.iloc[idx].mean() for idx in test_indices_kf]

test_indices_skf = [test for _, test in skf.split(X_scaled, y)]
fraud_ratios_skf = [y.iloc[idx].mean() for idx in test_indices_skf]

print("\nKFold Fraud Ratios:", fraud_ratios_kf)
print("StratifiedKFold Fraud Ratios:", fraud_ratios_skf)

# ============================================================
# F1 SCORES WITHOUT LOOPS
# ============================================================

f1_kfold = GridSearchCV(
    estimator=dt,
    param_grid=param_grid_dt,
    scoring="f1",
    cv=kf,
    n_jobs=-1
).fit(X_scaled, y).best_score_

f1_skfold = grid_dt.best_score_

print("\nMean F1 (KFold):", f1_kfold)
print("Mean F1 (StratifiedKFold):", f1_skfold)

# =============================== #
#         END OF CODE             #
# =============================== #
