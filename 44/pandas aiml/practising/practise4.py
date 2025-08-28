import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time

# Load dataset
iris_data = load_iris()

x_data = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
y = iris_data.target
print(x_data.head(10))

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y, random_state=42, stratify=y, test_size=0.3
)

# Models
models = {
    "lr": LogisticRegression(max_iter=500),
    "dt": DecisionTreeClassifier(),
    "svc": SVC()
}

# Parameter grids
param_grid = {
    "lr": {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'solver': ['liblinear', 'saga', 'lbfgs']
    },
    "svc": {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    },
    "dt": {
        'max_depth': [None, 3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],
        'max_leaf_nodes': [None, 5, 10, 20]
    }
}

results = {}

# Training loop
for name, model in models.items():
    print(f"\nTraining {name} ...")
    start = time.time()

    grid = GridSearchCV(model, param_grid=param_grid[name], cv=5,
                        scoring="accuracy", n_jobs=-1)
    grid.fit(x_train, y_train)

    y_pred = grid.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    end = time.time()

    results[name] = {
        "best params": grid.best_params_,
        "train cv accuracy": grid.best_score_,
        "test accuracy": acc,
        "time taken": round(end - start, 2)
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
