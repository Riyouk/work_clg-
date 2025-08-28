# ===============================
# Decision Tree Classifier
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import pickle 
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

# ⚠ Note: Decision Trees don’t require scaling (they are scale-invariant)
# But to keep consistent pipeline, we can still scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Baseline Decision Tree
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)  # not using scaled version (tree doesn’t need scaling)

# Predictions
y_pred = dt_clf.predict(X_test)

# Evaluation
print("\n===== Baseline Decision Tree =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

# ROC Curve
RocCurveDisplay.from_estimator(dt_clf, X_test, y_test)
plt.title("ROC Curve (Baseline Decision Tree)")
plt.show()

# 4. Hyperparameter Tuning
param_grid = {
    "criterion": ["gini", "entropy", "log_loss"],  # impurity measures
    "max_depth": [3, 5, 7, 10, None],  # depth of tree
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42),
                       param_grid,
                       cv=5,
                       scoring="accuracy",
                       n_jobs=-1)

grid_dt.fit(X_train, y_train)

print("\n===== Hyperparameter Tuning Results =====")
print("Best Parameters:", grid_dt.best_params_)
print("Best CV Accuracy:", grid_dt.best_score_)

# 5. Best Model
best_dt = grid_dt.best_estimator_
y_pred_best = best_dt.predict(X_test)

print("\n===== Best Decision Tree Model =====")
print("Test Accuracy:", accuracy_score(y_test, y_pred_best))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
print("\nClassification Report:\n", classification_report(y_test, y_pred_best, target_names=data.target_names))

# ROC Curve for Best Decision Tree
RocCurveDisplay.from_estimator(best_dt, X_test, y_test)
plt.title("ROC Curve (Best Decision Tree)")
plt.show()

# 6. Visualize the Tree
plt.figure(figsize=(18, 10))
plot_tree(best_dt, feature_names=data.feature_names,
          class_names=data.target_names,
          filled=True, fontsize=8)
plt.show()

# Print text-based tree
tree_rules = export_text(best_dt, feature_names=list(data.feature_names))
print("\n===== Decision Tree Rules =====")
print(tree_rules)


pickle.dump()