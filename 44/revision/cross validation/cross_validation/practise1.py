import pandas as pd
import numpy as np 

df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/revision/cross validation/cross_validation/Iris.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isna().sum())
print(df.duplicated().sum())

#data_preprocess 

# 1 converting to category
df["Species"] = df["Species"].astype("category")
print(df.info())

#tasks 
# 1 
from sklearn.model_selection import train_test_split
X = df.drop(columns=["Species"])
y = df["Species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 2
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l2"],          # use ["l1", "l2"] if you switch solver to "liblinear" or "saga"
    "solver": ["lbfgs"],    # "lbfgs" supports only "l2"; "liblinear"/"saga" support "l1" and "l2"
    "max_iter": [700]
}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)  

# 3
from sklearn.metrics import accuracy_score,classification_report
results = pd.DataFrame(grid.cv_results_)

# Print CV mean accuracy for every parameter combination
print(results[[
    "params",
    "mean_test_score",
    "std_test_score",
]])

print("Best Params:", grid.best_params_)
print(f"Best CV Accuracy: {grid.best_score_:.3f}")
print("Test Accuracy:",np.mean(accuracy_score(y_test, y_pred)))
print("Classification Report:")
print(classification_report(y_test, y_pred))



