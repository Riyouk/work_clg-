import pandas as pd 

transaction_file = pd.read_csv("C:/44/pandas aiml/data integration/banks/transactions.csv")
users =pd.read_csv("C:/44/pandas aiml/data integration/banks/users.csv")
logins = pd.read_csv("C:/44/pandas aiml/data integration/banks/logins.csv")

combined = pd.merge(transaction_file,users,on="user_id",how="outer")
total_comb = pd.merge(combined,logins,how="outer",on="user_id")
print(combined)
print(total_comb)