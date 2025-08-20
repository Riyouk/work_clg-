# import pandas as pd 
# import numpy as np 
# import statistics
# import matplotlib.pyplot as plt
# import seaborn as sns 
# df = pd.read_csv("C:/44/pandas aiml/literacy.csv")
# print(df)

# #rual
# print("___"*20)
# print("***"*5,"cental tendancy ","***"*5)
# print("***"*5,"rural data","***"*5)
# r_male_av = df['R_Male'].mean()
# print("rural male avg",r_male_av)
# r_male_md = df['R_Male'].median()
# print("rural male median",r_male_md)
# r_male_mod = df['R_Male'].mode()
# print("rural male mode",r_male_mod)
# r_male_std = np.std(df['R_Male'])
# r_male_var = np.var(df['R_Male'])
# r_female_av = df['R_Female'].mean()
# print("rural female avg",r_female_av)
# r_female_md = df['R_Female'].median()
# print("rural female median",r_female_md)
# r_female_mod = df['R_Female'].mode()
# print("rural female mode",r_female_mod)
# q_1_rm = np.quantile(df["R_Male"],.25)
# q_3_rm = np.quantile(df["R_Male"],.75)
# iqr_rmale = q_3_rm - q_1_rm
# print("the inter quartal range male",iqr_rmale)
# q_1_rfm = np.quantile(df["R_Female"],.25)
# q_3_rfm = np.quantile(df["R_Female"],.75)
# iqr_rfemale = q_3_rfm-q_1_rfm
# print("the inter quartal range rural female ",iqr_rfemale)

# #urban
# print("___"*20)
# print("***"*5,"cental tendancy ","***"*5)
# print("***"*5,"urban data","***"*5)
# u_male_av = df['U_Male'].mean()
# print("urban male avg",u_male_av)
# u_male_md = df['U_Male'].median()
# print("urban male median",u_male_md)
# u_male_mod = df['U_Male'].median()
# print("urban male mode",u_male_mod)
# q_1_um = np.quantile(df["U_Male"],.25)
# q_3_um = np.quantile(df["U_Male"],.75)
# iqr_umale = q_3_um-q_1_um
# print("the inter quartal range male",iqr_rmale)
# u_female_av = df['U_Female'].mean()
# print("urban female avg",u_female_av)
# u_female_md = df['U_Female'].median()
# print("urban female median",u_female_md)
# u_female_mod = df['U_Female'].mode()
# print("urban female mode",u_female_mod)
# q_1_ufm = np.quantile(df["U_Female"],.25)
# q_3_ufm = np.quantile(df["U_Female"].quantile(),.75)
# iqr_ufemale = q_3_ufm-q_1_ufm
# print("the inter quartal range female ",iqr_ufemale)

# fig,ax = plt.subplots(2,2,figsize=(20,20))
# sns.barplot(data=df,x='State',y="R_Male",ax=ax[0,0])
# ax[0,0].set_title("Rural male litracy distribution")
# ax[0,0].tick_params(axis='x',rotation=45)
# sns.barplot(data=df,x='State',y="R_Female",ax=ax[0,1])
# ax[0,1].set_title("Rural male litracy distribution")
# ax[0,1].tick_params(axis='x',rotation=45)
# sns.barplot(data=df,x='State',y="U_Male",ax=ax[1,0])
# ax[1,0].set_title("Rural male litracy distribution")
# ax[1,0].tick_params(axis='x',rotation=45)
# sns.barplot(data=df,x='State',y="U_Female",ax=ax[1,1])
# ax[1,1].set_title("Rural male litracy distribution")
# ax[1,1].tick_params(axis='x',rotation=45)
# plt.show()