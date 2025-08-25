import pandas as pd 
import numpy as np 

import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import winsorize
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression 

#loading dataset
df1 = pd.read_csv("C:/44/pandas aiml/DataSets/DataSets/Iris.csv",index_col=0)
df = df1

#preview data
print(df.head(10))
print(df.shape)

#assessing data
print(df.info())
print(df.describe())
print(df.isna().sum())

print(df["Species"].unique())

print(df["Species"].value_counts())

print(df.duplicated().value_counts())

#typeconversion 
df["Species"] = df["Species"].astype("category")
print(df.info())

#eda
# sns.pairplot(df,hue="Species")
# sns.pairplot(df)
# plt.show()
# print(df["Species"].unique())

#for seeing the correlation 
# print(df.columns)
# n_df = df.select_dtypes(include=["float64"])
# corr = n_df.corr()
# sns.heatmap(corr,cmap="crest",annot=True)
# plt.show()

#second_method
# feature_col = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
# corr = df[feature_col].corr()
# sns.heatmap(corr,cmap="crest",annot=True)
# plt.show()

#third method
# x = df.iloc[:,2:5]
# print(x)



#outliers identification and handeling 

Q1 = df["SepalWidthCm"].quantile(0.25)
Q3 = df["SepalWidthCm"].quantile(0.75)
IQR = Q3-Q1
l_l = Q1-1.5*IQR
u_l = Q3+1.5*IQR

outliers = df[(df["SepalWidthCm"]>u_l)|(df["SepalWidthCm"]<l_l)]
print("Out",outliers)

# q1 = df['SepalWidthCm'].quantile(0.25)
# q3 = df['SepalWidthCm'].quantile(0.75)

# IQR = q3-q1

# L_b = q1-1.5*IQR
# U_b = q3+1.5*IQR

# outliers = df[(df['Species']>U_b)|(df['Species']<L_b)]
# print("outliers",outliers)

# sns.boxplot(df)
# plt.show()

df["SepalWidthCm"] = winsorize(df["SepalWidthCm"],limits=[0.1,0.1])

# sns.boxplot(x=df["SepalWidthCm"])
# plt.show()

#label encoding
encoder = LabelEncoder()

ranking = {"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2}
df["Species"] = df["Species"].map(ranking)
print(df["Species"])

df["Species"] = encoder.fit_transform(df["Species"])
print(df["Species"])


# spliting data 

x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df["Species"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

#model 
logestic_model = LogisticRegression()
logestic_model.fit(x_train,y_train)

y_predict = logestic_model.predict(x_test)
print(y_predict)

result = pd.DataFrame()
result["y_test"],result["y_pred"], = y_test,y_predict
print(result.head())

#metrics 
