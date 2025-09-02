import pandas as pd 
import numpy as np 

import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import winsorize
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,RocCurveDisplay
from sklearn.linear_model import LogisticRegression 
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()
df = pd.DataFrame(data=cancer_data.data,columns=cancer_data.feature_names)
df["diagnosis"] = cancer_data.target
print(df.head(5))
print(df.columns)

x = df.drop(columns=["diagnosis"])
# print(x)
y = df["diagnosis"]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components=6)
x_pca = pca.fit_transform(x_scaled)


x_train,x_test,y_train,y_test = train_test_split(x_pca,y,test_size=0.3,random_state=42,stratify=y)


model = LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Accuracy",accuracy_score(y_test,y_pred))

print("explained variance : ",pca.explained_variance_)
print("explained variance in ratio  :",pca.explained_variance_ratio_)