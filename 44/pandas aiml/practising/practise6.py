import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import time

# Load dataset
df = pd.read_csv(
    "C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/DataSets/DataSets/winequality-white.csv",
    delimiter=";"
)

# Drop less important/redundant columns
df = df.drop(columns=["density", "free sulfur dioxide"])

# Features and target
X = df.drop(columns=['quality'])
y = df['quality']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.3, stratify=y
)

# Scale features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Models
models = {
    "lr": LogisticRegression(max_iter=3000),
    "dt": DecisionTreeClassifier(),
    "svc": SVC()
}

# Parameter grids
param_grid = {
    "lr": [
        {"penalty": ["l2"], "C": [0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
        {"penalty": ["l1"], "C": [0.1, 1, 10], "solver": ["liblinear", "saga"]}
    ],
    "svc": {
        "C": [0.1, 1, 10],
        "gamma": ["scale", 0.1],
        "kernel": ["rbf", "linear"]
    },
    "dt": {
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "criterion": ["gini", "entropy"]
    }
}

results = {}

# Training loop
for name, model in models.items():
    print(f"\nTraining {name} ...")
    start = time.time()

    grid = GridSearchCV(model, param_grid=param_grid[name], cv=3,
                        scoring="accuracy", n_jobs=-1, error_score="raise")
    grid.fit(x_train, y_train)

    y_pred = grid.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    end = time.time()

    results[name] = {
        "best params": grid.best_params_,
        "train cv accuracy": grid.best_score_,
        "test accuracy": acc,
        "time taken (s)": round(end - start, 2)
    }

    print(f"{name} Results: ")
    print("Best params:", grid.best_params_)
    print("Train CV accuracy:", grid.best_score_)
    print("Test accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Final comparison
print("\nFinal comparison:")
for model, res in results.items():
    print(f"{model}: {res}")
