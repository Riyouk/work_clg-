import statistics
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
sns.set_theme(style='darkgrid')
df = pd.read_csv("C:/44/pandas aiml/forbes[1].csv")
# print(df.head())
df.dropna(inplace=True)
# print(df.isnull().sum())
# print(df.shape)
# print(df.describe())
print(df.info())
# market_value_mean = df["Market Value"].mean()
# market_value_median = df["Market Value"].median()
# diff_btw_mean_median = market_value_mean-market_value_median 
# print("mean = ",market_value_mean)
# print("median =", market_value_median)
# print("differnce between mean and median",diff_btw_mean_median)
# market_value_mode = df['Sector'].mode()
# print("mode = ",market_value_mode)
# print(df['Sector'].value_counts())
# all_companies = (df['Company'].unique())
# val_counts = df['Country'].value_counts()
# val_counts_mode = df['Country'].mode()
# val_counts_mode_industries = df['Industry'].mode()
# print(val_counts_mode_industries)
# print(val_counts_mode)
# print(val_counts)
# plt.bar(df["Company"],df["Market Value"],width=0.6,linewidth=2)
# sns.countplot(data=df,x="Sector",label=df["Sector"].unique(),width=0.6)
# plt.xticks(rotation=90)
# plt.legend()
# # plt.xticks(False)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12,6))
# sns.boxplot(data=df,y=df['Profits'],showmeans=True)
# print(df['Profits'].mean)
# # plt.title()
# plt.xlim(-5,10)
# plt.xticks(range(-5,9,1))
# q_1 = np.quantile(df['Profits'],.25)
# q_3 = np.quantile(df['Profits'],.75)
# iqr = (q_3-q_1)
# print("quartile 1 =",q_1)
# print("quartile 3 =",q_3)
# print("inter quartile range =",iqr)

# max = df['Profits'].max()
# min = df['Profits'].min()
# print("range",max-min)
# q_1 = np.quantile(df['Sales'],.25)
# q_3 = np.quantile(df['Sales'],.75)
# print("iqr = ",q_3-q_1)
# std = np.std(df['Sales'])
# var = np.var(df['Sales'])
# print("std = ",std)
# print("var = ",var)
# plt.show()

# import random 

# def coinfilp(n):
#     result = []
#     for i in range(1,n+1):
#         toss = random.choice(["head_head","head_tail","tail_head","tail_tail"])
#         result.append(toss)
#     return(result)
# x = coinfilp(1000)
# sns.countplot(x=x)
# plt.show()

# def coinfilp(n):
#     result = []
#     for i in range(1,n+1):
#         toss = random.choice(["HHHH", "HHHT", "HHTH", "HTHH", "THHH", "HHTT", "HTHT", "HTTH", 
# "THHT", "THTH", "TTHH", "HTTT", "THTT", "TTHT", "TTTH", "TTTT"])
#         result.append(toss)
#     return(result)
# y = coinfilp(1000)
# sns.countplot(x=y)
# plt.xticks(rotation=90)
# plt.show()
# def dice(n):
#     result = []
#     for i in range(1,n+1):
#         roll = random.choice(['2','3','4','5','6','7','8','9','10','11','12'])
#         result.append(roll)
#     return(result)
# outcome = dice(1000)
# sns.countplot(x=outcome)
# plt.show()

#belcurve 
# normal_dist_data = np.random.normal(loc=0,scale=3,size=10000000)
# normal_dist_data_1 = np.random.normal(loc=2,scale=3,size=10000000)
# print(normal_dist_data[:10])
# plt.hist(normal_dist_data)
# sns.histplot(x=normal_dist_data)
# plt.text(x=4000,y=30000,s='r$')
plt.show()

# df = sns.load_dataset('titanic')
# df = pd.read_csv("C:/44/pandas aiml/titanic.csv")
# df.dropna(inplace=True)
# print(df.head())
# sns.histplot(x=df['Age'],kde=True)
# plt.show()


#kde plot 
# plt.figure(figsize=(6,8))
# sns.kdeplot(x=normal_dist_data)
# sns.kdeplot(x=normal_dist_data_1)
# plt.title("normal distribution")
# plt.show()


