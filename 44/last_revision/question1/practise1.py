import pandas as pd
import numpy as np

df = pd.read_csv("online_retail_II.csv", nrows=10000)

# Drop unused
df.drop(columns=["Customer ID"], inplace=True)
df.drop(columns=["Invoice"], inplace=True)

# Convert categorical/string columns
# df["Invoice"] = df["Invoice"].astype("string")
df["StockCode"] = df["StockCode"].astype("string")
df["Country"] = df["Country"].astype("category")

# Encode BEFORE MODELING
df = pd.get_dummies(df, drop_first=True)

# Outlier handling
q1 = df["Quantity"].quantile(0.25)
q3 = df["Quantity"].quantile(0.75)
iqr = q3 - q1
df["Quantity"] = df["Quantity"].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)

q1_p = df["Price"].quantile(0.25)
q3_p = df["Price"].quantile(0.75)
iqr_p = q3_p - q1_p
df["Price"] = df["Price"].clip(q1_p - 1.5*iqr_p, q3_p + 1.5*iqr_p)

# Split
x = df.drop(columns=["Price"])
y = df["Price"]

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error,mean_absolute_error,mean_squared_error

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42
)

rdft = RandomForestRegressor(random_state=42)

param_grid = {
    "max_depth": [3,4,5],
    "n_estimators": [10,20,30],
    "max_features": ["sqrt", "log2", None]
}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# FIX #1: wrong scoring → use neg_root_mean_squared_error
# FIX #2: pass kfold object, not class name
grid_search = GridSearchCV(
    estimator=rdft,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=kfold,
    n_jobs=-1
)

grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)

rmse = root_mean_squared_error(y_test, y_pred)
print("RMSE:", rmse)
mae= mean_absolute_error(y_test,y_pred)
print("MAE:",mae)
mse= mean_squared_error(y_test,y_pred)
print("MSE:",mse)
