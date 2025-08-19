import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:/44/pandas aiml/DataSets/DataSets/Toyota.csv",na_values=["????","??"],index_col=0)
print(df.info())
print(df.isna().sum())



#label encoding 
df["FuelType"] = df["FuelType"].astype("category")


encoding = LabelEncoder()
df["FuelType"] = encoding.fit_transform(df["FuelType"])
print(df.head())
print(df.dtypes)






#filling the missing values 
df["Age"].fillna(df["Age"].mean(),inplace=True)
df["MetColor"].fillna(df["MetColor"].mean(),inplace=True)
df["HP"].fillna(df["HP"].mean(),inplace=True)
df["KM"].fillna(df["KM"].mean(),inplace=True)
df["FuelType"].fillna(df["FuelType"].mode(),inplace=True)
print(df.isna().sum())


#visualizing 
num_df = df.select_dtypes(include=['int64','float64'])
cor_numdf = num_df.corr()
sns.heatmap(cor_numdf,annot=True,cmap="crest")
plt.tight_layout()
plt.show()

