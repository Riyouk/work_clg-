# =========================
# Forbes: Multiclass Classifiers + Stacking
# =========================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# 1) Load & quick sanity checks
# -----------------------------
df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/DataSets/DataSets/forbes.csv", index_col=0)

print("\nHead:")
print(df.head(10))
print("\nInfo:")
print(df.info())
print("\nMissing values:")
print(df.isna().sum())

# Peek at categories (these often exist in Forbes-style data)
if 'Sector' in df.columns:
    print("\nUnique Sectors:", df['Sector'].nunique())
    print(df['Sector'].unique()[:20])
if 'Industry' in df.columns:
    print("\nUnique Industries:", df['Industry'].nunique())
    print(df['Industry'].unique()[:20])

# Boxplot for numeric columns (optional EDA)
num_preview = df.select_dtypes(include=[np.number])
if not num_preview.empty:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=num_preview, orient='h')
    plt.title("Numeric Columns (Boxplot)")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------
# 2) Target selection & basic cleaning assumptions
# ------------------------------------------------
# We'll predict 'Sector' as the classification target
TARGET_COL = 'Sector'
assert TARGET_COL in df.columns, f"Expected column '{TARGET_COL}' not found in the CSV."

# Drop rows where target is missing
df = df.dropna(subset=[TARGET_COL]).copy()

# Cast common string categoricals to 'category' dtype (optional but nice)
for col in ['Industry', 'Sector', 'Country', 'Company']:
    if col in df.columns:
        df[col] = df[col].astype('category')

print("\nPost-cleaning info:")
print(df.info())

# -------------------------------------
# 3) Feature/target split & col typing
# -------------------------------------
y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

# Identify column types
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

# -----------------------------------
# 4) Preprocessing (impute/encode/scale)
# -----------------------------------
numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features)
    ],
    remainder='drop'
)

# -----------------------------------
# 5) Train/Test split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# -----------------------------------
# 6) Define models
# -----------------------------------
# Decision Tree (good baseline)
dt_clf = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", DecisionTreeClassifier(
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    ))
])

# Logistic Regression (needs scaling; handled in pipeline)
log_clf = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=2000,
        n_jobs=None,
        random_state=42
    ))
])

# SVM (we’ll use linear kernel for speed; you can try 'rbf' too)
svm_clf = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", SVC(
        kernel="linear",
        probability=True,   # enables predict_proba; helpful for stacking
        C=1.0,
        random_state=42
    ))
])

# -----------------------------------
# 7) Fit individual models
# -----------------------------------
models = {
    "DecisionTree": dt_clf,
    "LogisticRegression": log_clf,
    "SVM": svm_clf
}

print("\n=== Fitting individual models ===")
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[{name}] Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# -----------------------------------
# 8) Stacking (Stacker)
# -----------------------------------
# Base estimators should be strong & diverse. We'll reuse LR & SVM & a shallow tree.
stack_base = [
    ("logreg", Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=2000,
            random_state=42
        ))
    ])),
    ("svm", Pipeline([
        ("prep", preprocess),
        ("clf", SVC(
            kernel="rbf",        # try rbf here for diversity vs linear SVM above
            probability=True,
            C=1.0,
            gamma="scale",
            random_state=42
        ))
    ])),
    ("tree", Pipeline([
        ("prep", preprocess),
        ("clf", DecisionTreeClassifier(
            max_depth=10,        # shallow tree for diversity
            random_state=42
        ))
    ]))
]

# Final estimator (meta-model): typically a simple, well-regularized model
final_est = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=2000,
    random_state=42
)

# IMPORTANT: In sklearn, StackingClassifier expects raw X. If your base estimators
# already include preprocessing in their pipelines (as we do), you pass X directly.
stack_clf = StackingClassifier(
    estimators=stack_base,
    final_estimator=final_est,
    stack_method="predict_proba",   # use class probabilities as meta-features
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    passthrough=False               # set True if you also want raw preprocessed features
)

print("\n=== Fitting Stacking Classifier ===")
stack_clf.fit(X_train, y_train)
y_pred_stack = stack_clf.predict(X_test)
acc_stack = accuracy_score(y_test, y_pred_stack)
print(f"\n[Stacking] Accuracy: {acc_stack:.4f}")
print(classification_report(y_test, y_pred_stack, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_stack))

# -----------------------------------
# 9) (Optional) Cross-validated comparison
# -----------------------------------
print("\n=== Cross-validated Accuracy (5-fold) ===")
for name, pipe in list(models.items()) + [("Stacking", stack_clf)]:
    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"{name}: mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}")

# -----------------------------------
# 10) (Optional) Simple feature importance for a trained tree
# -----------------------------------
# If you want to see which features (after one-hot) matter in the tree:
try:
    # Fit a single pipeline to get transformed feature names
    fitted_preprocess = preprocess.fit(X_train, y_train)
    # Get feature names after OneHotEncoder expansion
    num_names = numeric_features
    cat_names = []
    if categorical_features:
        ohe = fitted_preprocess.named_transformers_['cat'].named_steps['onehot']
        cat_names = ohe.get_feature_names_out(categorical_features).tolist()
    feat_names = num_names + cat_names

    # Train a tree on the transformed space
    from sklearn.tree import DecisionTreeClassifier
    Xtr = fitted_preprocess.transform(X_train)
    tree_for_importance = DecisionTreeClassifier(max_depth=10, random_state=42)
    tree_for_importance.fit(Xtr, y_train)
    importances = pd.Series(tree_for_importance.feature_importances_, index=feat_names)
    print("\nTop 15 features (Decision Tree importance):")
    print(importances.sort_values(ascending=False).head(15))
except Exception as e:
    print("\nFeature importance step skipped:", repr(e))
