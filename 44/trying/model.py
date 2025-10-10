# ===============================
# Smart Packing Assistant (Trip Dataset)
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------
# Load CSV
# ------------------------------
df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/trying/smart_packing_assistant_dataset.csv")  # Replace with your CSV path

# ------------------------------
# Preview Data
# ------------------------------
print(df.head())
print(df.info())

# ------------------------------
# Convert Essential_Item_Categories to list
# ------------------------------
df['Essential_Item_Categories'] = df['Essential_Item_Categories'].apply(lambda x: [i.strip() for i in x.split(",")])

# ------------------------------
# Encode categorical features
# ------------------------------
categorical_cols = ['Trip_Type', 'Destination_Climate', 'Gender', 'Planned_Activity',
                    'Travel_Type', 'Accommodation_Type']

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# ------------------------------
# Multi-label Binarization for target
# ------------------------------
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['Essential_Item_Categories'])

# Features
X = df.drop('Essential_Item_Categories', axis=1)

# ------------------------------
# Optional: Scale numeric features
# ------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# Random Forest Classifier
# ------------------------------
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train, y_train)

# ------------------------------
# Predictions
# ------------------------------
y_pred = rf_clf.predict(X_test)

# ------------------------------
# Evaluation
# ------------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("/nClassification Report:/n", classification_report(y_test, y_pred, target_names=mlb.classes_))

# Confusion matrix per label
for i, label in enumerate(mlb.classes_):
    cm = confusion_matrix(y_test[:, i], y_pred[:, i])
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix for {label}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Feature Importance (average across all labels)
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,5))
sns.barplot(x=importances[indices], y=X.columns[indices], palette="viridis")
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()
