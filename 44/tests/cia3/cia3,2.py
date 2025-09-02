import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

#load dataset 
print("===="*10)
print("loading data set")
print("===="*10)
df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/tests/cia3/tumor_data.csv")
print(df.head(5))

#eda 
print("===="*10)
print("BASIC EDA")
print("===="*10)
print(df.info())
print(df.describe())
print(df.isna().sum())
print(df.columns)
feature_names = df[['Radius', 'Texture', 'Smoothness']]
target_name = df["Tumor_Class"]
print("feature names : ",feature_names)
print("target names : ",target_name)

#spliting the data
print("===="*10)
print("spliting data")
print("===="*10)
x = df[['Radius', 'Texture', 'Smoothness']]
y = df['Tumor_Class']
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.5)
print(x_train.shape)
print(x_test.shape)

#model building 
print("===="*10)
print("model building ")
print("===="*10)
model = LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

#intercept , slope and iterations 

print("C value :",model.intercept_)
print("slope value :",model.coef_)
print("no of iteration : ",model.n_iter_)

#evaluation 
print("===="*10)
print("evaluation")
print("===="*10)
print("\naccuracy_score :",accuracy_score(y_test,y_pred))
print("\nconfusion_matrix :",confusion_matrix(y_test,y_pred))
print("\nclassification_report :",classification_report(y_test,y_pred))

