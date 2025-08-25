import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor


df = pd.read_csv("C:/44/pandas aiml/DataSets/DataSets/train_data.csv")

df1 = pd.read_csv("C:/44/pandas aiml/DataSets/DataSets/test_data.csv")

#single 
x_train = df[["carat"]]
x_test = df1[["carat"]]
y_train = df["price"]
y_test = df1['price']

tree = DecisionTreeRegressor()
tree.fit(x_train,y_train)

y_predict = tree.predict(x_test)

result = pd.DataFrame()
result["x_test"],result["y_test"],result["y_pred"], = x_test,y_test,y_predict
print(result.head(10))




mse = mean_squared_error(y_test,y_predict)
print("MSE",mse)

r_2 = r2_score(y_test,y_predict)
print("R2",r_2)






# #decisiontreeRegreesor 
# tree = DecisionTreeRegressor()
# tree.fit(x_train,y_train)

# y_predict = tree.predict(x_test)


# mse = mean_squared_error(y_test,y_predict)
# print("MSE",mse)

# r_2 = r2_score(y_test,y_predict)
# print("R2",r_2)


