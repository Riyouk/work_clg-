import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, RocCurveDisplay
from sklearn.model_selection import GridSearchCV

# 1. Load Dataset
data = load_breast_cancer()
X, y = data.data, data.target

print("Features:", data.feature_names)
print("Target classes:", data.target_names)
# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Feature Scaling (important for LR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Logistic Regression Model
log_reg = LogisticRegression(max_iter=5000, solver="lbfgs")
log_reg.fit(X_train_scaled, y_train)

# 5. Predictions
y_pred = log_reg.predict(X_test_scaled)
y_proba = log_reg.predict_proba(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)

# 6. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))


# Define hyperparameters
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],  # solvers that support l1 and l2
    'max_iter': [6110]
}

log_reg = LogisticRegression()
grid_lr = GridSearchCV(log_reg, param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
grid_lr.fit(X_train_scaled, y_train)

print("Best Logistic Regression Params:", grid_lr.best_params_)
print("Best CV Accuracy:", grid_lr.best_score_)

#best estimator objects 
best_lr = grid_lr.best_estimator_
y_predict = best_lr.predict(X_test_scaled)

cm = confusion_matrix(y_test,y_predict)

#6. evaluation 
print("accuracy : ",accuracy_score(y_test,y_predict))
print("\nconfusion matrix : ",cm)
print("\nclssification report : ",classification_report(y_test,y_predict,target_names=data.target_names))