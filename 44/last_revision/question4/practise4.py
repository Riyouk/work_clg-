import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


df = pd.read_csv("regression_house_prices_anomalous.csv")
print(df.head(10))
print(df.info())
print(df.describe())
print(df.isna().sum())

#replace
df["area_sqft"] = df["area_sqft"].replace("ten thousand",10000)
# print(df["area_sqft"].head(15))

#type
df["area_sqft"] = df["area_sqft"].astype("float64")
# print(df.info())

#handeling nan
df["area_sqft"] = df["area_sqft"].fillna(df["area_sqft"].mean())
df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].mean())
df["price"] = df["price"].fillna(df["price"].mean())
print(df.isna().sum())

# outliers 
sns.boxplot(df)
plt.show()
q1_ar = df["area_sqft"].quantile(0.25)
q3_ar = df["area_sqft"].quantile(0.25)
iqr = q3_ar-q1_ar
lower_sq = q1_ar-1.5*iqr
upper_sq = q3_ar+1.5*iqr
outliers_sq = df[(df["area_sqft"]<lower_sq) | (df["area_sqft"]>upper_sq)]
print("outliers in area_sqft :",outliers_sq)

q1_pr = df["price"].quantile(0.25)
q3_pr = df["price"].quantile(0.25)
IQR = q3_ar-q1_ar
lower_pr = q1_ar-1.5*iqr
upper_pr = q3_ar+1.5*iqr
outliers_pr = df[(df["area_sqft"]<lower_pr) | (df["area_sqft"]>upper_pr)]
print("outliers in price :",outliers_pr)

#handeling 
q1_ar = df["area_sqft"].quantile(0.25)
q3_ar = df["area_sqft"].quantile(0.75)
iqr = q3_ar-q1_ar
df["area_sqft"] = df["area_sqft"].clip(q1_ar-1.5*iqr,q3_ar+1.5*iqr)
lower_sq = q1_ar-1.5*iqr
upper_sq = q3_ar+1.5*iqr
outliers_sq = df[(df["area_sqft"]<lower_sq) | (df["area_sqft"]>upper_sq)]
print("outliers in area_sqft :",outliers_sq)

q1_pr = df["price"].quantile(0.25)
q3_pr = df["price"].quantile(0.75)
IQR = q3_ar-q1_ar
df["price"] = df["price"].clip(q1_pr-1.5*IQR,q3_pr+1.5*IQR)
lower_pr = q1_ar-1.5*iqr
upper_pr = q3_ar+1.5*iqr
outliers_pr = df[(df["area_sqft"]<lower_pr) | (df["area_sqft"]>upper_pr)]
print("outliers in price :",outliers_pr)

# sns.boxplot(df)
# plt.show()

#building model 
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,root_mean_squared_error

#spliting
x = df.drop(columns="price")
y = df["price"]
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.2)

#scaling 
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
x_train_scaled= scalar.fit_transform(x_train)
x_test_scaled = scalar.transform(x_test)

#gridsearch

kfold = KFold(n_splits=5,shuffle=True,random_state=42)
model = RandomForestRegressor()
param_grid = {"max_depth" : [4,5],
              "n_estimators" : [10,20,30],
              "max_features" : ["sqrt","log2",None]}
grid = GridSearchCV(estimator=model,param_grid=param_grid,cv=kfold,n_jobs=-1)
grid.fit(x_train_scaled,y_train)
best_model = grid.best_estimator_
y_pred = best_model.predict(x_test_scaled)

print("MSE :",mean_squared_error(y_test,y_pred))
print("RMSE :",root_mean_squared_error(y_test,y_pred))
