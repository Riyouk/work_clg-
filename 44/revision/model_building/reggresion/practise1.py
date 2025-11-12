import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/DataSets/DataSets/Toyota.csv")
print(df.head(10))
print(df.info())
print(df.describe())
print(df.isna().sum())