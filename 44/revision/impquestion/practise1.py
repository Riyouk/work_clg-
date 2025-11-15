import pandas as pd 
import numpy as np 

df = pd.read_csv(
    "C:/Users/User/forgit uknow/work_clg-/44/revision/impquestion/regression_house_prices_anomalous.csv",
    na_values=["", " ","\t"], 
    keep_default_na=True
)

# print(df.head(20))
# print(df.info())
# print(df.describe())
# print(df.isna().sum())
# print(df.duplicated().sum())

#fixing the dataset
df["area_sqft"] = df["area_sqft"].replace("ten thousand",10000)

#typeconversion
df["area_sqft"] = df["area_sqft"].astype("float64")
# print(df.info())

#handeling missing values
df[["area_sqft","bedrooms","price"]] = df[["area_sqft","bedrooms","price"]].fillna(df[["area_sqft","bedrooms","price"]].mean())
# print(df.isna().sum())

#detecing outlier
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.stats import zscore
from sklearn.impute import KNNImputer
# using plots 

sns.boxplot(df)
# plt.show()

# using quartile
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3-Q1
# print("Quartile 1 : ",Q1)
# print("Quartile 3 : ",Q1)
# print("INTER Quartile : ",IQR)

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# print("Lower_bound : ",lower_bound)
# print("Upper_bound : ",upper_bound)

# outlier = df[(df["price"] < lower_bound) | (df["price"] > upper_bound)]
# print(outlier)

#using zscore 
# df["zscores"] = zscore(df["price"])
# print("zscores",df["zscores"])
# outliers = df[df["zscores"].abs()>1]
# print(outliers)


#handeling outlier
# using lower bound and upper bound
# df = df[(df["price"] >= lower_bound) & (df["price"] <= upper_bound)]
# print(df)
# sns.boxplot(df)
# plt.show()
# outlier = df[(df["price"] < lower_bound) | (df["price"] > upper_bound)]
# print(outlier)

# using winsorize
# from scipy.stats.mstats import winsorize
# df["price"] = winsorize(df["price"],limits=[0.5,0.5])
# print(df)

#
# df["price_winsorise"] = df["price"].clip(lower=lower_bound,upper=upper_bound)

# using clip 
# import numpy as np

lower = df["price"].quantile(0.01)  # bottom 1%
upper = df["price"].quantile(0.99)  # top 1%

df["price_winsor"] = df["price"].clip(lower=lower, upper=upper)


sns.boxplot(df)
# plt.show()

#transforming 
# print(df.describe())
from sklearn.preprocessing import StandardScaler
# print(df.columns)
scalar = StandardScaler()
df[['area_sqft','bedrooms','age','distance_city_km','price','price_winsor']] = scalar.fit_transform(df[['area_sqft','bedrooms','age','distance_city_km','price','price_winsor']])
# print(df.describe())

#visualiszation 
# corr = df.corr(include=['float64','int64'])
corr = df.corr(numeric_only=True)
# corr = df[['area_sqft','bedrooms','age','distance_city_km','price','price_winsor']].corr()
print(corr)
sns.heatmap(corr,annot=True,cmap="coolwarm")
# plt.show()

df1 = df.drop(columns=["area_sqft","price"])
print(df1.head())

#model building 
#linear regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

x = df1.drop(columns=["price_winsor"])
y = df1["price_winsor"]

# x_train,x_test,y_train,y_test = train_test_split(df1[["bedrooms","age","distance_city_km"]],df1["price_winsor"],test_size=0.2,random_state=42)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print("x_train shape : ",x_train.shape)
print("x_test shape : ",x_test.shape)
print("y_train shape : ",y_train.shape)
print("y_test shape : ",y_test.shape)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("Mean Squared Error : ",mse)
print("R-squared : ",r2)

# with hyperparameter 
from sklearn.model_selection import GridSearchCV

param_grid = {
    'fit_intercept': [True, False],
    'positive': [True, False]
}

grid_search = GridSearchCV(LinearRegression(), param_grid, cv=5, scoring='r2')
grid_search.fit(x_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best R-squared:", grid_search.best_score_)

#using random forest reggresor
from sklearn.ensemble import RandomForestRegressor

Rand_model = RandomForestRegressor()
Rand_model.fit(x_train,y_train)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print("x_train shape : ",x_train.shape)
print("x_test shape : ",x_test.shape)
print("y_train shape : ",y_train.shape)
print("y_test shape : ",y_test.shape)

model1 = RandomForestRegressor()
model1.fit(x_train,y_train)

y_pred1 = model1.predict(x_test)

mse1 = mean_squared_error(y_test,y_pred1)
r2_1 = r2_score(y_test,y_pred1)

print("Mean Squared Error : ",mse1)
print("R-squared : ",r2_1)

# with hyperparameter 
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='r2')
grid_search.fit(x_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best R-squared:", grid_search.best_score_)

#comparing the model performance

# logestic reggresion vs random forest reggresor
loge_mse = mean_squared_error(y_test,y_pred)
loge_r2 = r2_score(y_test,y_pred)

print("Logestic Mean Squared Error : ",loge_mse)
print("Logestic R-squared : ",loge_r2)

rand_mse = mean_squared_error(y_test,y_pred1)
rand_r2 = r2_score(y_test,y_pred1)

print("Random Forest Mean Squared Error : ",rand_mse)
print("Random Forest R-squared : ",rand_r2)

df_compare = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Mean Squared Error": [loge_mse, rand_mse],
    "R-squared": [loge_r2, rand_r2]
})

print(df_compare)



