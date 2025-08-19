import pandas as pd 


#loading data
df = pd.read_csv("C:/44/pandas aiml/DataSets/DataSets/agriculture_dataset.csv")


#preview data
print(df.head(10))

#understand the data structure 
print(df.info())

#understand the data with statistical summury
print(df.describe())

feature_set = df.columns
print(feature_set)
















