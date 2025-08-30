import pandas as pd 
import numpy as np 

import seaborn as sns 
import matplotlib.pyplot as plt 


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import winsorize
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,RocCurveDisplay
# from sklearn.tree import DecisionTreeClassifier,export_text,plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/DataSets/DataSets/breast-cancer.csv")


print(df.head())
# print(df.info())

# print(df.describe())

print(df["diagnosis"].value_counts())

df["diagnosis"] = df["diagnosis"].astype("category")
print(df.info())

print(df.columns)

y = ["diagnosis"]
x = ['radius_mean', 'texture_mean',
       'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

df.drop(columns=["id"],inplace=True)
# df.drop(['perimeter_mean', 'area_mean','perimeter_worst', 'area_worst', 'perimeter_se', 'area_se', 'compactness_mean','concave points_mean', 'compactness_se', 'concave points_se','compactness_worst','concave points_worst'], axis=1, inplace=True)
print(df.info())

# sns.pairplot(df,hue="diagnosis")
# plt.figure(figsize=(20,15))
# plt.show()

# sns.boxplot(df)
# plt.show()

# numreic_col = df.select_dtypes(include=["float64"])
# corr = numreic_col.corr()
corr = df.corr(numeric_only=True)
# plt.figure(figsize=(16,12))
# sns.heatmap(corr,cmap="crest",annot=True) 
# plt.show()

print(corr)

def winsorize_data(data,limits=(0.02,0.02)):
    for col in data.select_dtypes(include=["float64"]):
        data.loc[:,col] = winsorize(data[col], limits=limits)
    return data

df_win = winsorize_data(df)
X = df_win[x]
y = df_win["diagnosis"]

# #model 
# x_train,x_test = train_test_split(x,y,test_size=0.30,random_state=42,stratify=df['diagnosis'])

# scaler = StandardScaler()
# x_train[x] = scaler.fit_transform(x_train[x])
# print(x_train.head(5))
# x_test[x] = scaler.fit_transform(x_test[x])
# print(x_test.head(5))

# model = RandomForestClassifier(random_state=42,max_depth=4)
# model.fit(X=x_train[x],y=x_train[y])

# y_predict = model.predict(x_test[x])
# print(y_predict)



# Define features and target
X = df_win[x]   # features
y = df_win["diagnosis"]   # target (Series, not list)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train[x] = scaler.fit_transform(X_train[x])
X_test[x] = scaler.transform(X_test[x])   # <-- use transform, not fit_transform

# Train model
model = RandomForestClassifier(random_state=42, max_depth=4)
model.fit(X_train, y_train)

# Predictions
y_predict = model.predict(X_test)
print(y_predict)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_predict))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_predict))
print("Classification Report:\n", classification_report(y_test, y_predict))
