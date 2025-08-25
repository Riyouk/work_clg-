import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder


df_filled = pd.read_csv("C:/44/pandas aiml/DataSets/DataSets/diamonds.csv")


df_filled.dropna(inplace=True)


# df_filled = df_filled.fillna(df_filled.mean(numeric_only=True))
# # print(df_filled.isna().sum())
# print(df_filled.info())




#model building 
x = df_filled[["carat"]]
y = df_filled["price"]

x_train,x_test,y_train,y_test = tts(x,y,test_size=.2,random_state=25)

print(x_train)
print("-"*20)
print(x_test)
print("-"*20)
print(y_train)
print("-"*20)
print(y_test)
print("-"*20)


model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("m value is : ",model.coef_)
print("c value is : ",model.intercept_)


result = pd.DataFrame()
result["x_test"],result["y_test"],result["y_pred"], = x_test,y_test,y_pred
print(result.head(10))


mse = mean_squared_error(y_test,y_pred)
print(mse)


r2 = r2_score(y_test,y_pred)
print(r2)


plt.scatter(x_test,y_test,color = "blue",label = "Actual")
plt.plot(x_test,y_pred,color="Red",label="Predicted")
# sns.regplot(x=x_test,y=y_pred)
plt.legend()
plt.xlabel("Carat")
plt.ylabel("Price")
plt.show()