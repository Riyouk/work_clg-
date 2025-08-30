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
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier



#load_dataset
df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/DataSets/DataSets/titanic.csv")
print(df.head(10))
print(df.isna().sum())

#FEATURE SET 
print(df.columns)
print(df.info())
print(df.describe())

#droping cabin
df.drop(columns=["Cabin","Name","Fare","Ticket"],inplace=True)
print(df.info())

#type conversion 
df["Sex"] = df['Sex'].astype("category")
df["Embarked"] = df['Embarked'].astype("category")
print(df.info())

# df.rename()

#handeling missing values 
# print(df["Embarked"].unique())
df["Age"].fillna(df["Age"].median(),inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)
print(df.info())

#labelencoding 
label = LabelEncoder()
df["Embarked"] = label.fit_transform(df["Embarked"])
df["Sex"] = label.fit_transform(df["Sex"])
print(df.head())

#feature selection
x = [['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
y = df["Survived"]

x = df.drop("Survived",axis=1)
y = df["Survived"]
# print(x.shape)

# split the data 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)
print(x_train.shape)

# Hyperparameter tuning using GridSearchCV
print("\n--- Starting Hyperparameter Tuning ---")
start_time = time.time()

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Create a base model
rf = RandomForestClassifier(random_state=42)

# Instantiate GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring='accuracy'
)

# Fit the grid search to the data
grid_search.fit(x_train, y_train)

# Print the best parameters and best score
print("\nBest Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score: {:.4f}".format(grid_search.best_score_))
print("Time taken for GridSearchCV: {:.2f} seconds".format(time.time() - start_time))

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions with the best model
y_predict = best_model.predict(x_test)

# Evaluation
print("\n--- Model Evaluation with Best Parameters ---")
print("Accuracy:", accuracy_score(y_test, y_predict))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_predict))
print("Classification Report:\n", classification_report(y_test, y_predict))

# Feature importance
feature_importance = pd.DataFrame(
    {'feature': x_train.columns,
     'importance': best_model.feature_importances_}
).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Optional: Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance_rf.png')
plt.close()