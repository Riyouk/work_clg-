import pandas as pd 

df = pd.read_csv("C:/44/pandas aiml/titanic.csv")
print(df.head())
df_sample = df.sample(frac=0.5,random_state=10)
print(df_sample)
print(df.info())
# group = df_sample.groupby("Sex")['Pclass'].count()
# print(group)
# print(pd.pivot_table(df,index='Embarked',columns=['Sex'],values=['Embarked'],aggfunc={'Embarked':['count']}))
# print(pd.pivot_table(df_sample,index='Sex',columns='Pclass',values="PassengerId",aggfunc='count'))
# resample = df.resample()
# print(df.set_index('date').resample('M').sum())