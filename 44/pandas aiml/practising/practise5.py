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
df["Age"].fillna(df["Age"].median(),inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)
print(df.info())

#labelencoding 
label = LabelEncoder()
df["Embarked"] = label.fit_transform(df["Embarked"])
df["Sex"] = label.fit_transform(df["Sex"])
print(df.head())

#feature selection
# x = [['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
# y = df["Survived"]

x = df.drop("Survived",axis=1)
y = df["Survived"]
# print(x.shape)

# split the data 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)
models = {"Logestic Regression" : LogisticRegression(max_iter=5000),
          "SVM" : SVC(),
          "Decision Tree" : DecisionTreeClassifier()}

param_grid = {
    "Logestic Regression" : {'C':[0.01,0.1,1,10],'penalty':['l1','l2'],'solver':['saga','liblinear']},
    "SVM" : {'C':[0.01,0.1,1,10],'gamma':[1,0.1,0.01,0.001],'kernel':['rbf','sigmoid','poly'],'max_iter':[1000]},
    "Decision Tree" : {'criterion':['gini','entorpy'],'max_depth':[3,5,8,None]}

}

grid = GridSearchCV(models["Logestic Regression"],param_grid=param_grid["Logestic Regression"],scoring="accuracy",cv=5,n_jobs=-1)
grid.fit(x_train,y_train)

y_pred = grid.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print("Best params",grid.best_params_)
print("Best CV score",grid.best_score_)
print("Best estimator",grid.best_estimator_)


#multiple
results = {}

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

# sns.boxplot(df)
# plt.show()