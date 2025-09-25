# ===============================================
# Phishing Website Detection - ML Classification with PCA
# Dataset: Phishing_Legitimate_full.csv
# Objective: Classify websites as Legitimate (1) or Phishing (-1)
# ===============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("C:/Users/User/mini_project/MiniProject/classification/Phishing_Legitimate_full.csv")

print("First 5 rows:\n", df.head())
print("Dataset shape:", df.shape)

# -------------------------------
# 2. Features and Target
# -------------------------------
X = df.drop("CLASS_LABEL", axis=1)
y = df["CLASS_LABEL"]

# -------------------------------
# 3. Train-Test Split (80-20)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 4. Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 5. Apply PCA (Dimensionality Reduction)
# -------------------------------
# Reduce to 95% variance or set n_components to a specific number
pca = PCA(n_components=0.95, random_state=42)  
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Original number of features:", X_train_scaled.shape[1])
print("Reduced number of features after PCA:", X_train_pca.shape[1])

# -------------------------------
# 6. Random Forest Classifier
# -------------------------------
rf_clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# -------------------------------
# 7. Hyperparameter Grid for GridSearchCV
# -------------------------------
param_grid = {
    "n_estimators": [100],
    "max_depth": [10,None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"]
}

grid = GridSearchCV(
    estimator=rf_clf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring="accuracy"
)

# Fit GridSearchCV on PCA-transformed data
grid.fit(X_train_pca, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

# -------------------------------
# 8. Evaluate Best Model on Test Set
# -------------------------------
best_rf = grid.best_estimator_
y_pred_best_rf = best_rf.predict(X_test_pca)

print("\nRandom Forest Results (After Tuning + PCA):")
print("Accuracy:", accuracy_score(y_test, y_pred_best_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_best_rf))

# -------------------------------
# 9. Confusion Matrix Visualization
# -------------------------------
cm = confusion_matrix(y_test, y_pred_best_rf)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legitimate", "Phishing"],
            yticklabels=["Legitimate", "Phishing"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest with PCA")
plt.show()
