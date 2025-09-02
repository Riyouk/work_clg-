import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler

#loading data set and printing the first five rows
print("===="*10)
print("loading data set")
print("===="*10)
df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/tests/cia3/house_data.csv")
print(df.head(5))

#basic eda 
print("===="*10)
print("BASIC EDA")
print("===="*10)
print(df.shape)
print(df.info())
print(df.describe())
print("missing values",df.isna().sum())
# print(df.duplicated().sum())

#spliting the data 
print("===="*10)
print("spliting data")
print("===="*10)
print("all columns : ",df.columns)
x = df[['Size', 'Rooms', 'Age']]
y = df["Price"]
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.3)
print("x_train shape : ",x_train.shape)
print("x_test shape : ",x_test.shape)

#model building 
print("===="*10)
print("model building ")
print("===="*10)
model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

#intercept , slope
print("C value :",model.intercept_)
print("slope value :",model.coef_)

#evaluation 
print("===="*10)
print("evaluation")
print("===="*10)
print("MASE :",mean_absolute_error(y_test,y_pred))
print("MSE : ",mean_squared_error(y_test,y_pred))
print("R2 SCORE :",r2_score(y_test,y_pred))

