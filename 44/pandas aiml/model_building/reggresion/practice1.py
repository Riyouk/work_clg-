import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
# sample data - y = f(x) = x^2+2
# data = pd.DataFrame({
#     'x_input' : [1,2,3,4,5,6,7,8,9,10],
#     'target' : [3,6,11,18,27,38,51,66,83,102]
# })

# #feature set 
# x = data[['x_input']]
# y = data['target']

# x_train,x_test,y_train,y_test = tts(x,y,test_size=0.3,random_state=40)
# # print(x.head(10))
# print("xtrain",x_train.head(5))
# print("xtest",x_test.head(5))
# print("ytrain",y_train.head(5))
# print("ytest",y_test.head(5))



df = pd.read_csv("C:/44/pandas aiml/MODEL.csv")
print(df)

x = df[['x_input ']]
y = df['target']
# print(df['x_input '])
x_train,x_test,y_train,y_test = tts(x,y,test_size=0.3,random_state=40)
# print("xtrain",x_train.head(5))
# print("xtest",x_test.head(5))
# print("ytrain",y_train.head(5))
# print("ytest",y_test.head(5))

model = lr()
model.fit(x_train,y_train)

y_predict = model.predict(x_test)
print(y_predict)

# if(y_test==y_predict).empty != True :
#     print("100 % accuracy")

result = pd.DataFrame()

result["x_test"],result["y_test"],result["y_predict"] = x_test,y_test,y_predict
print(result.head(10))




