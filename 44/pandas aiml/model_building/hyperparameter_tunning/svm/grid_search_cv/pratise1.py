from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Support Vector Machine
svm = SVC()

# Hyperparameter space
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'degree': [2, 3, 4]
}

# Randomized Search
random_svm = RandomizedSearchCV(svm, param_grid_svm, cv=5, n_iter=10,
                                 n_jobs=-1, random_state=42, scoring='accuracy')
random_svm.fit(X_train, y_train)

print("Best SVM Params:", random_svm.best_params_)
print("SVM Accuracy:", accuracy_score(y_test, random_svm.predict(X_test)))
