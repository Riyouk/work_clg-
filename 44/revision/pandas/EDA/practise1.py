import pandas as pd 
import numpy as np 

# 1
df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/revision/pandas/EDA/customer_purchases.csv")
print(df)
print(df.info())
print(df.describe())

# missing values
print("total missing values \n",df.isna().sum())
print("missing values percentage \n",df.isna().sum()/len(df) * 100)

#inconsistent cat
print(df.columns)
print(df['Gender'].value_counts())

# 2
# handeling missing values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Annual_Income (₹)"] = df["Annual_Income (₹)"].fillna(df["Annual_Income (₹)"].median())
df["Membership_Level"] = df["Membership_Level"].fillna(df["Membership_Level"].mode()[0])
df["Spending_Score"] = df["Spending_Score"].fillna(df["Spending_Score"].interpolate())      
# df["Spending_Score"] = df["Spending_Score"].fillna(df["Spending_Score"].mean())      
print(df.isna().sum())

# 3
# label encoding 
var = {"Male":"M","Female":"F","female":"F","MALE":"M"}
df["Gender"] = df["Gender"].map(var)
print(df["Gender"].value_counts())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
print(df["Gender"])

# onehot encoding
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop="first",sparse_output=False)
# df["Membership_Level"] = ohe.fit_transform(df[["Membership_Level"]])
encode = ohe.fit_transform(df[["Membership_Level"]])
# print(df["Membership_Level"])
df1 = pd.DataFrame(encode,columns=[ohe.get_feature_names_out(["Membership_Level"])])
# print(df1)
data = pd.concat([df,df1],axis=1)
print(data)

# 4
# Normalize

#MINMAX SCALAR
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
data[["Age","Annual_Income (₹)","Spending_Score"]] = minmax.fit_transform(data[["Age","Annual_Income (₹)","Spending_Score"]])
print(data[["Age","Annual_Income (₹)","Spending_Score"]])
print(data.describe())

#STANDARDIZE
# from sklearn.preprocessing import StandardScaler
# std = StandardScaler()
# data[["Age","Annual_Income (₹)","Spending_Score"]] = std.fit_transform(data[["Age","Annual_Income (₹)","Spending_Score"]])
# print(data[["Age","Annual_Income (₹)","Spending_Score"]])
# print(data.describe())



#5 
# checking

# 1 
print("NUMBER OF MISSING VALUES \n")
print(data.isna().sum())
# 2
print("ALL CATEGORICAL ENCODED  \n")
print(data[["Gender","Membership_Level"]])
# 3
print("NUMERICAL VALUES ARE SCALED \n")
print(data[["Age","Annual_Income (₹)","Spending_Score"]])


# extra
# changing the types to Catogorical
data[["Gender","Membership_Level","Purchased"]] = data[["Gender","Membership_Level","Purchased"]].astype("category")
print(data.info())


