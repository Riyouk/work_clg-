import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,RocCurveDisplay
from 

df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/DataSets/DataSets/forbes.csv",index_col=0)
print(df.head())
print(df.info())
print(df.isna().sum())
print(df['Sector'].unique())
print(df['Industry'].unique())

sns.boxplot(df)
plt.show()

df['Industry'] = df['Industry'].astype("category")
df['Sector'] = df['Sector'].astype("category")
df['Country'] = df['Country'].astype("category")
df['Company'] = df['Company'].astype("category")
print(df.info())

df["Sector"]=df['Sector'].fillna(df['Sector'].mode()[0])
df['Industry'] = df['Industry'].fillna(df['Industry'].mode()[0])
print(df.info())



