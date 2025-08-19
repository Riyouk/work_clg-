import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

# from sklearn.datasets import load_boston as lb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error,r2_score

df = pd.read_csv("C:/44/pandas aiml/DataSets/DataSets/Boston.csv",index_col=0)
print(df.info())
print(df.head(10))
print(df.isna().sum())
print(df.describe())
print(df.columns)

y = df["medv"]
x = df[['rm']]

x_train,x_test,y_train,y_test = tts(x,y,random_state=40,test_size=0.3)
print("x_train",x_train.shape)
print("x_test",x_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)



model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
# print(y_pred)


result = pd.DataFrame()

result["x_test"],result["y_test"],result["y_predict"] = x_test,y_test,y_pred
print(result.head(10))



#modelevaluation
mse = mean_squared_error(y_test,y_pred)
print(mse)

r2 = r2_score(y_test,y_pred)
print(r2)


plt.scatter(x_test,y_test,c='blue',label="actual")
plt.plot(x_test,y_pred,c='red',label="predicted")
plt.xlabel("AVG NO OF ROOMS ")
plt.ylabel("HOSE PRICE")
plt.legend()
plt.show()