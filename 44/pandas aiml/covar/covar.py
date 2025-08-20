import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

# df = pd.read_csv("C:/44/pandas aiml/data_loan.csv")
# print(df.info())

# numbers_df = df.select_dtypes(include=['int'])
# # print(numbers_df)
# cor_no_df = numbers_df.corr()
# # print(cor_no_df)

# sns.heatmap(cor_no_df,annot=True)
# plt.show()


# df = pd.read_csv("C:/44/pandas aiml/titanic.csv")
# # print(df.info())
# # print(df.isna().sum())
# df.dropna(inplace=True)
# # print(df.isna().sum())
# num_df = df.select_dtypes(include=['int64','float64'])
# # print(num_df)

# cor_num_df = num_df.corr()
# print(cor_num_df)

# sns.heatmap(cor_num_df,annot=True,cmap='hot')
# plt.tight_layout()
# plt.show()

df = pd.read_csv("C:/44/pandas aiml/bank[1].csv",sep=';')
print(df.info())
print(df.head())
# # print(df.isna().sum())
df.dropna(inplace=True)
# # print(df.isna().sum())
num_df = df.select_dtypes(include=['int64','float64'])
print(num_df)

cor_num_df = num_df.corr()
print(cor_num_df)

sns.heatmap(cor_num_df,annot=True,cmap='crest')
plt.tight_layout()
plt.show()





# df = pd.read_csv("C:/44/pandas aiml/DataSets/DataSets/Boston.csv",index_col=0)
# print(df.info())
# # print(df.head())
# # # print(df.isna().sum())
# # df.dropna(inplace=True)
# # # print(df.isna().sum())
# num_df = df.select_dtypes(include=['int64','float64'])
# # print(num_df)

# cor_num_df = num_df.corr()
# print(cor_num_df)

# sns.heatmap(cor_num_df,annot=True,cmap='crest')
# plt.tight_layout()
# plt.show()
