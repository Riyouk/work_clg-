import pandas as pd 

df1 = pd.read_csv("C:/44/pandas aiml/data integration/transactions_day1.csv")
df2 = pd.read_csv("C:/44/pandas aiml/data integration/transactions_day2.csv")
# print(df1)
# print(df2)
df2.rename(columns={"withdrawa_amount":"withdrawal_amount"},inplace=True)
print(df2)

merge_vr = pd.concat([df1,df2],axis=0,ignore_index=True)
# merge_vr = pd.concat([df1,df2],axis=1,ignore_index=True)

print(merge_vr)
merge_vr.drop_duplicates(inplace=True)
print(merge_vr)

