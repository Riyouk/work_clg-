import pandas as pd 
import numpy as np
import scipy.stats as stats
import seaborn as sns 
import matplotlib.pyplot as plt

sns.set_theme(style="dark")
df = pd.read_csv("C:/44/pandas aiml/Birthweight_reduced_kg_R.csv")
print(df.head(10))
# print(df.isna().count())
print(df.info())
# sns.barplot(data=df,x='Gestation',y='Birthweight')
# sns.pairplot(df)

# sns.regplot(data=df,x=)
print(stats.shapiro(df['Length']))
print(stats.shapiro(df['Birthweight']))
print(stats.chisquare(df['Birthweight']))
plt.show()

