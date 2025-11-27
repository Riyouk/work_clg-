import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isna().sum())

# -----------------------------
# 2. Encode Target Column
# -----------------------------
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df["Attrition"] = encoder.fit_transform(df["Attrition"])  # Yes=1, No=0

# -----------------------------
# 3. One-hot Encode ALL other cats
# -----------------------------
df = pd.get_dummies(df, drop_first=True)

print(df.head())

# -----------------------------
# 4. Outlier Plot
# -----------------------------
sns.boxenplot(df)
plt.show()

# -----------------------------
# 5. Train-test split
# -----------------------------
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

X = df.drop(columns=["Attrition"])
y = df["Attrition"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -----------------------------
# 6. Cross-validation
# -----------------------------
stkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# -----------------------------
# 7. RandomForest + GridSearch
# -----------------------------
model = RandomForestClassifier(random_state=42)

param_grid = {
    "max_depth": [3, 4, 5],
    "n_estimators": [10, 15, 20],
    "max_features": ["sqrt", "log2", None],
    "min_samples_split": [2, 3, 4, 5],
    "min_samples_leaf": [2]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=stkf,
    n_jobs=-1,
    scoring="accuracy"
)

grid.fit(x_train, y_train)

# -----------------------------
# 8. Evaluation
# -----------------------------
best_model = grid.best_estimator_
y_pred = best_model.predict(x_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
