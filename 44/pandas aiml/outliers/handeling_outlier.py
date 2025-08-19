import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from scipy.stats import zscore
from sklearn.impute import KNNImputer

# #using boxplot and iqr
# df = pd.DataFrame({"temp" :[-20,-23,-3,20,23,34,43,45,54,33,60,45,32,23,48,90,80,100] })
# # print(df)
# sns.boxplot(data=df,y='temp')
# # plt.xlim(0,100)
# plt.show()

# Q1 = df['temp'].quantile(0.25)
# print("q1 = ",Q1)
# Q3 = df['temp'].quantile(0.75)
# print("q3 = ",Q3)
# IQR = Q3-Q1
# print("iqr = ",IQR)

# lower_bound = Q1 - 1.5 * IQR
# print("lower bound",lower_bound)
# upper_bound = Q3 + 1.5 * IQR
# print("upper_bound",upper_bound)

# outlier = df[(df['temp'] < lower_bound) | (df['temp'] > upper_bound)]
# print("outliers",outlier)

# #zscore 
# df['zscore'] = zscore(df["temp"])
# print("zscore",df["zscore"])
# outliers = df[df["zscore"].abs() > 1]
# print("zscore outliers",outliers)



#titanic dataset 
df = pd.read_csv("C:/44/pandas aiml/titanic.csv")
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df_num = df[numeric_cols]

# Apply KNNImputer to ONLY numeric columns
imputer = KNNImputer(n_neighbors=3)
df_num_imputed = pd.DataFrame(imputer.fit_transform(df_num), columns=numeric_cols, index=df.index)

# If you want to combine back with non-numeric columns:
df_final = df.copy()
df_final[numeric_cols] = df_num_imputed

print(df_final.isna().sum())
print(df.info())
# df.dropna(inplace=True)
sns.boxplot(data=df,y='Age') 
plt.show()
sns.boxplot(data=df,y='Fare') 
plt.show()

Q1_age = df['Age'].quantile(0.25)
print("q1 of age ",Q1_age)
Q3_age = df['Age'].quantile(0.75)
print("q3 of age ",Q3_age)
IQR_age = Q3_age-Q1_age
print("iqr of age ",IQR_age)

lower_bound = Q1_age - 1.5 * IQR_age
print("lower bound of age ",lower_bound)
upper_bound = Q3_age + 1.5 * IQR_age
print("upper bound of age ",upper_bound)
outliers_in_age = df[(df["Age"] < lower_bound) | (df["Age"] > upper_bound)]
print("outliers in age ",outliers_in_age)

Q1_fare = df['Fare'].quantile(0.25)
# print("q1 of fare ",Q1_fare)
Q3_fare = df['Fare'].quantile(0.75)
# print("q3 of fare ",Q3_fare)
IQR_fare = Q3_fare-Q1_fare
# print("iqr of fare ",IQR_fare)
lower_bound_fare = Q1_fare - 1.5 * IQR_fare
# print("lower bound of fare ",lower_bound_fare)
upper_bound_fare = Q3_fare + 1.5 * IQR_fare
# print("upper bound of fare ",upper_bound_fare)
outliers_in_fare = df[(df["Fare"] < lower_bound) | (df["Fare"] > upper_bound)]
# print("outliers in fare ",outliers_in_fare)

#zcore 
df['zscore_age'] = zscore(df["Age"])
# print("column zscore age",df["zscore_age"])
outliers_in_age_zscore = df[df["zscore_age"].abs() > 3]
print("outliers in age zscore",outliers_in_age_zscore)


df['zscore_fare'] = zscore(df["Fare"])
# print("column zscore fare",df["zscore_fare"])
outliers_in_fare_zscore = df[df["zscore_fare"].abs() > 3]
print("outliers in fare zscore",outliers_in_fare_zscore)





