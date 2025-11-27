# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    r2_score
)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv("C:/Users/hp/Desktop/GHOST/work_clg-/44/revision/model_revesion/linear_model/question1.csv")

# ---------------------------------------------------------
# ENCODING (Manual Label Encoding)
# ---------------------------------------------------------

# Convert to categorical for safety
df[["location", "property_type"]] = df[["location", "property_type"]].astype("category")

# Manual label mapping for location
location_map = {
    'BTM Layout': 0,
    'Indiranagar': 1,
    'Whitefield': 2,
    'Electronic City': 3,
    'HSR Layout': 4,
    'Marathahalli': 5
}
df["location"] = df["location"].map(location_map)

# Manual label mapping for property type
property_map = {'Apartment': 0, 'Studio': 1, 'Villa': 2}
df["property_type"] = df["property_type"].map(property_map)

# ---------------------------------------------------------
# FEATURE SCALING (Standardization)
# ---------------------------------------------------------

scaler = StandardScaler()

num_cols = ['area_sqft', 'bedrooms', 'bathrooms', 'floor',
            'age_years', 'has_parking', 'has_gym']

# Scale only numeric columns
df[num_cols] = scaler.fit_transform(df[num_cols])

# ---------------------------------------------------------
# SIMPLE MODEL → area_sqft  → price
# ---------------------------------------------------------

# X must be 2D → keep double brackets
x = df[["area_sqft"]]
y = df["price"]

# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

# Linear Regression (Simple)
lean_sm = LinearRegression()
lean_sm.fit(x_train, y_train)
y_pred_lr = lean_sm.predict(x_test)

# Random Forest (Simple)
randf_sm = RandomForestRegressor()
randf_sm.fit(x_train, y_train)
y_pred_rf = randf_sm.predict(x_test)

# ---------------------------------------------------------
# MULTIPLE LINEAR MODEL → all features → price
# ---------------------------------------------------------

X = df.drop(columns=["id", "price"])
y = df["price"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Multiple Linear Regression
lean_m = LinearRegression()
lean_m.fit(x_train, y_train)
y_pred_lr_s = lean_m.predict(x_test)

# Random Forest (Multiple)
randf_m = RandomForestRegressor()
randf_m.fit(x_train, y_train)
y_pred_rf_s = randf_m.predict(x_test)

# ---------------------------------------------------------
# MODEL METRICS (MSE, RMSE, R2)
# ---------------------------------------------------------

print("----- SIMPLE LINEAR REGRESSION -----")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("RMSE:", root_mean_squared_error(y_test, y_pred_lr))
print("R2:", r2_score(y_test, y_pred_lr))

print("\n----- SIMPLE RANDOM FOREST -----")
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("RMSE:", root_mean_squared_error(y_test, y_pred_rf))
print("R2:", r2_score(y_test, y_pred_rf))

print("\n----- MULTIPLE LINEAR REGRESSION -----")
print("MSE:", mean_squared_error(y_test, y_pred_lr_s))
print("RMSE:", root_mean_squared_error(y_test, y_pred_lr_s))
print("R2:", r2_score(y_test, y_pred_lr_s))

print("\n----- MULTIPLE RANDOM FOREST -----")
print("MSE:", mean_squared_error(y_test, y_pred_rf_s))
print("RMSE:", root_mean_squared_error(y_test, y_pred_rf_s))
print("R2:", r2_score(y_test, y_pred_rf_s))

# ---------------------------------------------------------
# OUTLIER DETECTION USING BOXPLOT
# ---------------------------------------------------------

plt.figure(figsize=(12,5))
sns.boxplot(data=df)
plt.title("Outlier Detection using Boxplot")
plt.show()

# ---------------------------------------------------------
# K-FOLD CROSS VALIDATION (5 Folds)
# ---------------------------------------------------------

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation for simple LR
scores_lr_s = cross_val_score(lean_sm, x, y, cv=kfold, scoring="r2")

# Cross-validation for simple RF
scores_rf_s = cross_val_score(randf_sm, x, y, cv=kfold, scoring="r2")

# Cross-validation for multiple LR
scores_lr = cross_val_score(lean_m, X, y, cv=kfold, scoring="r2")

# Cross-validation for multiple RF
scores_rf = cross_val_score(randf_m, X, y, cv=kfold, scoring="r2")

print("\n----- K-FOLD R2 SCORES -----")
print("Simple LR CV:", scores_lr_s)
print("Simple RF CV:", scores_rf_s)
print("Multiple LR CV:", scores_lr)
print("Multiple RF CV:", scores_rf)

# ---------------------------------------------------------
# REGULARIZATION MODELS (RIDGE, LASSO, ELASTIC NET)
# ---------------------------------------------------------

# Train on multiple model's training set
ridge = Ridge(alpha=1).fit(x_train, y_train)
lasso = Lasso(alpha=0.1).fit(x_train, y_train)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(x_train, y_train)

# Print R2 scores for each
print("\n----- REGULARIZATION MODEL SCORES (R2) -----")
print("Ridge R2:", ridge.score(x_test, y_test))
print("Lasso R2:", lasso.score(x_test, y_test))
print("ElasticNet R2:", elastic.score(x_test, y_test))
