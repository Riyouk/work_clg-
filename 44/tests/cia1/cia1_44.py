import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv('C:/44/tests/dataset_on_product.csv',index_col=0)
# print(df)
# print(df.info())
# print(df.describe())
# print(df['category'].unique())
print("---"*5)
print("first question") 
sns.barplot(data=df,x='region',y='sales')
plt.title("total sales by region")
plt.show()
print("***"*5)

print("second question") 
print("---"*5)
sns.barplot(data=df,x=df['product'],y=df['sales'],hue='months ')
plt.title("monthly sales across all products")
plt.show()
print("***"*5)

print("---"*5)
print("third questuon") 
percentage = df.groupby('category')['sales'].agg(sum)
plt.pie(x=percentage,  autopct='%1.1f%%',labels=df['category'].unique(),shadow=True)
plt.title("percentage of sales by product category")
plt.show()
print("***"*5)

print("---"*5)
print("question 4")
sns.barplot(data=df,x='region',y='sales',hue='category')
plt.title("total sales")
plt.show()
print("***"*5)

print("---"*5)
print("question 5")
sns.regplot(x=df['sales'],y=df['quantity'],data=df)
plt.title("relationship between sales and quantity ")
plt.show()
print("***"*5)

print("---"*5)
print("question 6") 
print(pd.pivot_table(data=df,index='region',values='sales',aggfunc={'sales':sum}))
print("***"*5)


print("---"*5)
print("question 7") 
print(pd.pivot_table(data=df,index='product',values='quantity',aggfunc={'quantity':sum}))
print("***"*5)

print("---"*5)
print("question 8")
print(pd.pivot_table(data=df,index='region',values='sales',aggfunc={'sales':sum},sort=False)[:1])
print()
print("***"*5)

print("---"*5)
print("question 9")
print(pd.pivot_table(data=df,index='months ',values='sales',aggfunc={'sales':sum},sort=False)[0:1])
print("***"*5)

print("---"*5)
print("question 10") 
print(pd.pivot_table(data=df,index='region',columns='category',values='sales',aggfunc={'sales':sum}))
print("***"*5)

print("---"*5)
print("question 11")
print(pd.pivot_table(data=df,index='product',columns='region',values='sales',aggfunc={'sales':max},sort=False)[0:3])
print("***"*5)

print("---"*5)
print("question 12")
print(pd.pivot_table(data=df,columns='category',values='sales',aggfunc={'sales':'mean'}))
print("***"*5)
