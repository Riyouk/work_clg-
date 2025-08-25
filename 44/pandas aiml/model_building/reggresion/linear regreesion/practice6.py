import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("C:/44/pandas aiml/DataSets/DataSets/diamonds.csv")
# print(df.info())
# print(df.isna().sum())


# print(df["cut"].unique())
# print(df["color"].unique())
# print(df["clarity"].unique())


# missing_rows = df[df.isna().any(axis=1)]
# print(missing_rows)
# missing_rows_clr = df[df["color"].isna()]
# print(missing_rows_clr)
# missing_rows_cut = df[df['cut'].isna()]
# print(missing_rows_cut)
# missing_rows_clarity = df[df['clarity'].isna()]
# print(missing_rows_clarity)

# checking for outliers 
# sns.boxplot(x=df['carat'])
# plt.show()
# sns.boxplot(x=df['table'])
# plt.show()
# sns.boxplot(x=df['price'])
# plt.show()
# sns.boxplot(x=df['depth'])
# plt.show()



# filling null values 
df_filled = df.fillna(df.median(numeric_only=True))
# print(df_filled.isna().sum())
print(df_filled.info())

# conversitons objects to cat
df_filled["cut"] = df_filled["cut"].astype("category")
df_filled["color"] = df_filled["color"].astype("category")
df_filled["clarity"] = df_filled["clarity"].astype("category")
print(df_filled.info())

# encoding the categorical variables
encode = LabelEncoder()
cut_codes = {"Fair":0,"Good":1,"Very Good":2,"Premium":3,"Ideal":4}
color_codes = {"D":6,"E":5,"F":4,"G":3,"H":2,"I":1,"J":0}
clarity_codes = {"I1":0,"SI2":1,"SI1":2,"VS2":3,"VS1":4,"VVS2":5,"VVS1":6,"IF":7}

df_filled["cut"] = df_filled['cut'].map(cut_codes)
print(df_filled.head())
df_filled["color"] = df_filled['color'].map(color_codes)
print(df_filled.head())
df_filled["clarity"] = df_filled['clarity'].map(clarity_codes)
print(df_filled.head())

#handeling null values 
df_filled["cut"] = df_filled["cut"].fillna(df_filled["cut"].mode()[0])
df_filled["color"] = df_filled["color"].fillna(df_filled["color"].mode()[0])
df_filled["clarity"] = df_filled["clarity"].fillna(df_filled["clarity"].mode()[0])
print(df_filled.isna().sum())
print(df_filled.head(10))

#model building 

x = df_filled[["carat"]]
y = df_filled["price"]

x_train,x_test,y_train,y_test = tts(x,y,test_size=.3,random_state=42)

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