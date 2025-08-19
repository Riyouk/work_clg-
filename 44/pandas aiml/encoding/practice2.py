import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler

scalar = MinMaxScaler()
standard = StandardScaler()

df = pd.read_csv("C:/44/pandas aiml/Toyota.csv")

# print(df.info())

# sns.histplot(df,x="Price",kde=True)
# plt.show()
# sns.histplot(df,x="Age",kde=True)
# plt.show()

df[["Age","Price"]] = scalar.fit_transform(df[["Age","Price"]])
# print(df[["Age","Price"]])
# sns.histplot(df,x="Price",kde=True)
# plt.show()
sns.histplot(df,x="Age",kde=True)
plt.show()


df[["Age","Price"]] = standard.fit_transform(df[["Age","Price"]])
print(df[["Age","Price"]])

# sns.histplot(df,x="Price",kde=True)
# plt.show()
# sns.histplot(df,x="Age",kde=True)
# plt.show()