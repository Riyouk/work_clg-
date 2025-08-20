import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error,r2_score
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:/44/pandas aiml/DataSets/DataSets/diamonds.csv")

# accessing the data 
# print(df.info())
# print(df.isna().sum())


# print(df["cut"].unique())
# print(df["color"].unique())
# print(df["clarity"].unique())

# seeing missing index 
# missing_rows = df[df.isna().any(axis=1)]
# print(missing_rows)
# missing_rows_clr = df[df["color"].isna()]
# print(missing_rows_clr)
# missing_rows_cut = df[df['cut'].isna()]
# print(missing_rows_cut)
# missing_rows_clarity = df[df['clarity'].isna()]
# print(missing_rows_clarity)

# handeling missing values 
df_filled = df.fillna(df.median(numeric_only=True))
df_filled.dropna(inplace=True)
# print(df_filled.isna().sum())
print(df_filled.info())

#handeling null values  (category)
df_filled["cut"] = df_filled["cut"].fillna(df_filled["cut"].mode()[0])
df_filled["color"] = df_filled["color"].fillna(df_filled["color"].mode()[0])
df_filled["clarity"] = df_filled["clarity"].fillna(df_filled["clarity"].mode()[0])
print(df_filled.isna().sum())
print(df_filled.head(10))

# checking for outliers 
# sns.boxplot(x=df['carat'])
# plt.show()
# sns.boxplot(x=df['table'])
# plt.show()
# sns.boxplot(x=df['price'])
# plt.show()
# sns.boxplot(x=df['depth'])
# plt.show()

#OUTLIER DETECTION using zscore 
# df_filled[["zcarat","zdepth","ztable","zprice","zx","zy","zz" ]] = zscore(df_filled[['carat','depth','table','price','x','y','z']])
# print("outliers",df_filled[["zcarat","zdepth","ztable","zprice","zx","zy","zz" ]] )

# q1 = df_filled[["zcarat","zdepth","ztable","zprice","zx","zy","zz" ]].quantile(0.25)
# q3 = df_filled[["zcarat","zdepth","ztable","zprice","zx","zy","zz" ]].quantile(0.75)
# iqr = q3 - q1 

# caping with lowe and upper bound 
# lower_b = q1-1.5*iqr
# upper_b = q3+1.5*iqr
# print(q1)
# print(q3)
# print(iqr)

# sns.boxplot(x=df_filled['carat'])
# plt.show()
# sns.boxplot(x=df_filled['table'])
# plt.show()
# sns.boxplot(x=df_filled['price'])
# plt.show()
# sns.boxplot(x=df_filled['depth'])
# plt.show()


# df_filled[["zcarat","zdepth","ztable","zprice","zx","zy","zz" ]] = np.where(df_filled[["zcarat","zdepth","ztable","zprice","zx","zy","zz" ]]>upper_b,upper_b,np.where(df_filled[["zcarat","zdepth","ztable","zprice","zx","zy","zz" ]]<lower_b,lower_b,df_filled[["zcarat","zdepth","ztable","zprice","zx","zy","zz" ]]))


# sns.boxplot(x=df_filled['carat'])
# plt.show()
# sns.boxplot(x=df_filled['table'])
# plt.show()
# sns.boxplot(x=df_filled['price'])
# plt.show()
# sns.boxplot(x=df_filled['depth'])
# plt.show()

# winsorization (handeling outliers ) 

def winsorize_data(data,limits=(0.1,0.1)):
    for col in data.select_dtypes(include=["float64"]):
        data.loc[:,col] = winsorize(df_filled[col], limits=limits)
    return data

win_data = winsorize_data(df_filled)
# sns.boxplot(x=df_filled['carat'])
# plt.show()
# sns.boxplot(x=df_filled['table'])
# plt.show()
# sns.boxplot(x=df_filled['price'])
# plt.show()
# sns.boxplot(x=df_filled['depth'])
# plt.show()

#seeing the relationship after the preprocessing 
df_filled_num = win_data.select_dtypes(include=["float64"])
df_filled_cor = df_filled_num.corr()
print(df_filled_cor)
sns.heatmap(df_filled_cor,cmap="crest",annot=True)
plt.show()


# conversitons objects to cat
df_filled["cut"] = df_filled["cut"].astype("category")
df_filled["color"] = df_filled["color"].astype("category")
df_filled["clarity"] = df_filled["clarity"].astype("category")
# print(df_filled.info())

# encoding the categorical variables
encode = LabelEncoder()
cut_codes = {"Fair":0,"Good":1,"Very Good":2,"Premium":3,"Ideal":4}
color_codes = {"D":6,"E":5,"F":4,"G":3,"H":2,"I":1,"J":0}
clarity_codes = {"I1":0,"SI2":1,"SI1":2,"VS2":3,"VS1":4,"VVS2":5,"VVS1":6,"IF":7}

df_filled["cut"] = df_filled['cut'].map(cut_codes)
# print(df_filled.head())
df_filled["color"] = df_filled['color'].map(color_codes)
# print(df_filled.head())
df_filled["clarity"] = df_filled['clarity'].map(clarity_codes)
# print(df_filled.head())


#model building 

#single var 
# x = df_filled[["carat"]]
# y = df_filled["price"]

# x_train,x_test,y_train,y_test = tts(x,y,test_size=.3,random_state=42)

# print(x_train)
# print("-"*20)
# print(x_test)
# print("-"*20)
# print(y_train)
# print("-"*20)
# print(y_test)
# print("-"*20)


# model = LinearRegression()
# model.fit(x_train,y_train)
# y_pred = model.predict(x_test)


# result = pd.DataFrame()
# result["x_test"],result["y_test"],result["y_pred"], = x_test,y_test,y_pred
# print(result.head(10))


# mse = mean_squared_error(y_test,y_pred)
# print(mse)


# r2 = r2_score(y_test,y_pred)
# print(r2)


# plt.scatter(x_test,y_test,color = "blue",label = "Actual")
# plt.plot(x_test,y_pred,color="Red",label="Predicted")
# # sns.regplot(x=x_test,y=y_pred)
# plt.legend()
# plt.xlabel("Carat")
# plt.ylabel("Price")
# plt.show()

# print(df_filled.info())


#spliting data   
# print(df_filled.columns)
print(win_data.columns)
# x_train = win_data['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y','z']
# x_test = win_data[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y','z']]
# y_train = df["price"]
# y_test = ['price']

x_train,x_test = tts(win_data,test_size=0.2,random_state=25)
print(x_train.head(10))
print(x_test.head(10))

# print(x_train.columns)
# print(x_test.columns)
print("shape of training dataset : ",x_train.shape)
print("shape of training dataset : ",x_test.shape)




# standardization 
# def standardize_data(data,columns=None):
#     if columns is None:
#         columns = df.select_dtypes(include=['number']).columns
#     scaler = StandardScaler()
#     df.loc[:,columns] = scaler.fit_transform(data[columns])
#     return df 


# standardization 
scaler = StandardScaler()

x_train[['carat','depth', 'table', 'price', 'x', 'y','z']] = scaler.fit_transform(x_train[['carat','depth', 'table', 'price', 'x', 'y','z']])
print(x_train.head(10))
x_test[['carat','depth', 'table', 'price', 'x', 'y','z']] = scaler.fit_transform(x_test[['carat','depth', 'table', 'price', 'x', 'y','z']])
print(x_train.head(10))

#asigning var
x = x_train[['carat', 'cut', 'color', 'clarity', 'depth', 'table','x', 'y','z']]
y = x_train[ 'price']
x_t =  x_test[['carat', 'cut', 'color', 'clarity', 'depth', 'table','x', 'y','z']]
y_t = x_test[ 'price']

model = LinearRegression()
model.fit(x,y)

y_predict = model.predict(x_t)


mse = mean_squared_error(y_t,y_predict)
print("mean squared error : ",mse)
r2 = r2_score(y_t,y_predict)
print("R2 score : ",r2)