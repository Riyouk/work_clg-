# ===============================
# Support Vector Machine (SVM)
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    RocCurveDisplay
)

# 1. Load Dataset
data = load_breast_cancer()
X, y = data.data, data.target

print("Features:", data.feature_names)
print("Target classes:", data.target_names)

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Feature Scaling (VERY IMPORTANT for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Baseline SVM Model
svm_clf = SVC(kernel="rbf", probability=True, random_state=42)
svm_clf.fit(X_train_scaled, y_train)

# Predictions
y_pred = svm_clf.predict(X_test_scaled)

# Evaluation
print("\n===== Baseline SVM =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

# ROC Curve
RocCurveDisplay.from_estimator(svm_clf, X_test_scaled, y_test)
plt.title("ROC Curve (Baseline SVM)")
plt.show()

# 5. Hyperparameter Tuning for SVM
param_grid = {
    'C': [0.1, 1, 10, 100],       # Regularization strength
    'gamma': [1, 0.1, 0.01, 0.001], # Kernel coefficient for RBF
    'kernel': ['rbf', 'poly', 'sigmoid'] # Types of kernels
}

grid_svm = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_svm.fit(X_train_scaled, y_train)

print("\n===== Hyperparameter Tuning Results =====")
print("Best Parameters:", grid_svm.best_params_)
print("Best CV Accuracy:", grid_svm.best_score_)

# 6. Best Model
best_svm = grid_svm.best_estimator_
y_pred_best = best_svm.predict(X_test_scaled)

print("\n===== Best SVM Model =====")
print("Test Accuracy:", accuracy_score(y_test, y_pred_best))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
print("\nClassification Report:\n", classification_report(y_test, y_pred_best, target_names=data.target_names))

# ROC Curve for Best SVM
RocCurveDisplay.from_estimator(best_svm, X_test_scaled, y_test)
plt.title("ROC Curve (Best SVM)")
plt.show()