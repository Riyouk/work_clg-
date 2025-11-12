

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

df = pd.read_csv("C:/44/pandas aiml/DataSets/DataSets/diamonds.csv")
print(df.head(10))
print(df.info())
print(df.describe())
print(df.isna().sum())