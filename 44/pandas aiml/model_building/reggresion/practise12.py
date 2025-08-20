import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


df = pd.read_csv("C:/44/pandas aiml/DataSets/DataSets/train_data.csv")

df1 = pd.read_csv("C:/44/pandas aiml/DataSets/DataSets/test_data.csv")

#single 
x_train = df[["carat"]]
x_test = df1[["carat"]]
y_train = df["price"]
y_test = df1['price']

sv = SVR()
sv.fit(x_train,y_train)

y_predict = sv.predict(x_test)

result = pd.DataFrame()
result["x_test"],result["y_test"],result["y_pred"], = x_test,y_test,y_predict
print(result.head(10))




mse = mean_squared_error(y_test,y_predict)
print("MSE",mse)

r_2 = r2_score(y_test,y_predict)
print("R2",r_2)


#multivar
print(df.columns)
x_train = df[['carat','depth','table','price', 'x', 'y', 'z','cut_label','color_label', 'clarity_label']]
x_test = df1[['carat','depth','table','price','x','y','z','cut_label','color_label', 'clarity_label']]
y_train = df["price"]
y_test = df1['price']

print(x_train.columns)
print(x_test.columns)
print("shape of training dataset : ",x_train.shape)
print("shape of training dataset : ",x_test.shape)




#SVR model
sv = SVR()
sv.fit(x_train,y_train)

y_predict = sv.predict(x_test)


mse = mean_squared_error(y_test,y_predict)
print("MSE",mse)

r_2 = r2_score(y_test,y_predict)
print("R2",r_2)

