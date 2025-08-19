import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
df = pd.read_csv("C:/44/pandas aiml/titanic.csv")
print(df.info())
# print(df.isna().sum().index())
# missing_rows = df[df.isna().any(axis=1)].index
# print(missing_rows)
# missing_in_col = df[df["Embarked"].isna()].index
# missing_in_col2 = df[df["Cabin"].isna()].index
# print(missing_in_col2)
# print(df.isnull().sum())
# print(df.describe())
# df['Age'].fillna(df['Age'].mean,inplace=True)
# df['Age'].fillna(df['Age'].mean,inplace=True)
# imputer = KNNImputer(n_neighbors=3)
# df_imputed = imputer.fit_transform(df['Age'])
# df_imputed = pd.DataFrame(imputer.fit_transform(df['Age']),columns=df.columns)
# print(df_imputed)
# sns.heatmap(df.isna(),cbar=True)
# plt.show()




numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df_num = df[numeric_cols]
print(df_num)

# Apply KNNImputer to ONLY numeric columns
imputer = KNNImputer(n_neighbors=3)
df_num_imputed = pd.DataFrame(imputer.fit_transform(df_num), columns=numeric_cols, index=df.index)

# If you want to combine back with non-numeric columns:
df_final = df.copy()
df_final[numeric_cols] = df_num_imputed

print(df_final.isna().sum())