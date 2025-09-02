import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
from scipy.stats import zscore 
df = pd.read_csv("C:/44/tests/mobile_sales_data.csv")
missing = (df[df.isna().any(axis=1)].index)

#detecting missing values 
print(df.info())
print(df.isna().sum())
print(missing)


#type conversion 
df["Brand"] = df["Brand"].astype('category')
df["Model"] = df["Model"].astype('category')
df["Store"] = df["Store"].astype('category')
df["Payment_Method"] = df["Payment_Method"].astype('category')
df["Sale_Date"] = pd.to_datetime(df["Sale_Date"])

print(df.dtypes)

#handeling missing values 
df["Price"] = df["Price"].fillna(df["Price"].mean())
df["Units_Sold"] = df["Units_Sold"].fillna(df["Units_Sold"].mean())
df["Store"] = df["Store"].fillna(df["Store"].mode()[0])
df["Brand"] = df["Brand"].fillna(df["Brand"].mode()[0])
print(df.isna().sum())

#outliers 
df['zscore'] = zscore(df["Price"])
print("zscore",df["zscore"])
outliers = df[df["zscore"].abs() > 2]
print("zscore outliers",outliers)

sns.boxplot(df["Price"])
plt.xlabel("PRICE")
plt.show()
sns.boxplot(df["Units_Sold"])
plt.xlabel("UNITS_SOLD")
plt.show()
sns.boxplot(df["Total_Sales"])
plt.xlabel("TOTAL_SALES")
plt.show()

# univariate analysis 

sns.histplot(df["Price"]) 
plt.show()

print(df["Brand"].value_counts())

numeric_df = df.select_dtypes(include=["int64","Float64"])
print(numeric_df)

print("-------"*5)
print("mean_values")
print("PRICE mean",numeric_df["Price"].mean())
print("Units_Sold mean",numeric_df["Units_Sold"].mean())
print("Total_Sales mean",numeric_df["Total_Sales"].mean())

print("-------"*5)
print("median_values")
print("PRICE median",numeric_df["Price"].median())
print("Units_Sold median",numeric_df["Units_Sold"].median())
print("Total_Sales median",numeric_df["Total_Sales"].median())

print("-------"*5)
print("mode_values")
print("PRICE mode",numeric_df["Price"].mode())
print("Units_Sold mode",numeric_df["Units_Sold"].mode())
print("Total_Sales mode",numeric_df["Total_Sales"].mode())

print("-------"*5)
print("standard_deviation_values")
print("PRICE std",np.std(numeric_df["Price"]))
print("Units_Sold std",np.std(numeric_df["Units_Sold"]))
print("Total_Sales std",np.std(numeric_df["Total_Sales"]))

print("-------"*5)
print("min_values")
print("PRICE min",numeric_df["Price"].min())
print("Units_Sold min",numeric_df["Units_Sold"].min())
print("Total_Sales min",numeric_df["Total_Sales"].min())

print("-------"*5)
print("max_values")
print("PRICE min",numeric_df["Price"].max())
print("Units_Sold min",numeric_df["Units_Sold"].max())
print("Total_Sales min",numeric_df["Total_Sales"].max())

#bivariate 
sns.scatterplot(df,x="Price",y="Units_Sold")
# plt.scatter(x="Price",y="Units_Sold")
plt.xlabel("price")
plt.ylabel("Units_Sold")
plt.show()

x = df.groupby("Brand")["Price"].mean()
print(x)

sns.barplot(data=df,x="Brand",y="Price")
plt.show()

sns.boxplot(data=df,x=df["Price"],y=df["Brand"])
plt.show()





