import pandas as pd 
import numpy as np 

import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import winsorize
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,RocCurveDisplay
from sklearn.linear_model import LogisticRegression 

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
#model 
x_train,x_test = train_test_split(df_win,test_size=0.30,random_state=42,stratify=df['diagnosis'])

scaler = StandardScaler()
x_train[x] = scaler.fit_transform(x_train[x])
print(x_train.head(5))
x_test[x] = scaler.fit_transform(x_test[x])
print(x_test.head(5))

model = LogisticRegression(class_weight="balanced",max_iter=1000,penalty="l2")
model.fit(x_train[x],x_train[y])

y_pred = model.predict(x_test[x])
# print("the y predict : ",y_pred)

result = pd.DataFrame()
result["y"],result["y_pred"] = x_test[y],y_pred
print(result)
print(model.predict_proba(x_test[x]))

print(confusion_matrix(result['y'],result['y_pred']))
print("--"*5)
print(classification_report(result['y'],result['y_pred']))
print("--"*5)
con_mat = confusion_matrix(result['y'],result['y_pred'])
sns.heatmap(con_mat,annot=True,cmap="crest")
plt.show()
print("--"*5)
print(accuracy_score(result['y'],result['y_pred']))
print("--"*5)
