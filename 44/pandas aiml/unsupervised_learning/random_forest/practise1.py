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


# # #load_dataset
# df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/DataSets/DataSets/titanic.csv")
# print(df.head(10))
# print(df.isna().sum())

# #FEATURE SET 
# print(df.columns)
# print(df.info())
# print(df.describe())

# # feature selection
# x = [['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
# y = df["Survived"]

# x = df.drop("Survived",axis=1)
# y = df["Survived"]
# # print(x.shape)

# # split the data 
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)
# print(x_train.shape)

# #model building 
# model = RandomForestClassifier()
# model.fit(x_train,y_train)

# y_predict = model.predict(x_test)
# print(y_predict)


# # Evaluation
# print("Accuracy:", accuracy_score(y_test, y_predict))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_predict))
# print("Classification Report:\n", classification_report(y_test, y_predict))










































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

#model building 
model = RandomForestClassifier(n_estimators=110,criterion="gini",max_depth=4)
model.fit(x_train,y_train)

y_predict = model.predict(x_test)
print(y_predict)


# Evaluation
print("Accuracy:", accuracy_score(y_test, y_predict))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_predict))
print("Classification Report:\n", classification_report(y_test, y_predict))