# import pandas as pd

# df = pd.read_csv("C:/44/pandas aiml/titanic.csv")

# print(df.head())
# print(df['Cabin'].unique())

# mode_cabin = df['Cabin'].mode()[0]
# print(mode_cabin)

# df['Cabin'] = df['Cabin'].fillna(mode_cabin,inplace=True)

# df['Cabin'] = df['Cabin'].astype('category').cat.codes
# print(df.head())

# print(df.isna().sum())
# print(df["Embarked"].unique())


# df["Embarked"] = df["Embarked"].astype("category").cat.codes

# print(df.head())
# print(df["Embarked"].unique())


# # Check the mode (most frequent value) of the Embarked column
# mode_embarked = df['Embarked'].mode()
# print(mode_embarked)
# # Fill missing values in Embarked with the mode
# df['Embarked'].fillna(mode_embarked,inplace=True)

# # Verify no missing values remain
# print(df['Embarked'].isna().sum())


import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import seaborn as sns 
df = pd.read_csv("C:/44/pandas aiml/data_loan.csv")
# print(df.info())
# print(df.head())
#minmax
scalar = MinMaxScaler()
df[['Age','Income']] = scalar.fit_transform(df[['Age','Income']])
sns.histplot(df,x='Income',kde=True)
plt.show()
sns.histplot(df,x='Age',kde=True)
plt.show()

# transformed_df = scalar.fit_transform(df[['Age','Income']])
# print(transformed_df)
#standard
standrd = StandardScaler()
df[['Age','Income']] = standrd.fit_transform(df[['Age','Income']])
# print(df[['Age','Income']])

sns.histplot(df,x='Income',kde=True)
plt.show()
sns.histplot(df,x='Age',kde=True)
plt.show()









