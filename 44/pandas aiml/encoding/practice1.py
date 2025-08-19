import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

data = {
    "Name": [
        "Rohan", "Priya", "Amit", "Sana", "Vikram", "Sneha", "Rahul", "Meena", "Alok", "Kavya",
        "Dinesh", "Jaya", "Ritu", "Deepak", "Isha", "Guru", "Pooja", "Anil", "Akash", "Tara"
    ],
    "PreferredMode": [
        "Online", "Offline", "Hybrid", "Online", "Offline", "Online", "Hybrid", "Online", "Offline", "Online",
        "Hybrid", "Offline", "Online", "Hybrid", "Online", "Offline", "Hybrid", "Offline", "Online", "Hybrid"
    ]
}

df = pd.DataFrame(data)
print("CSV created!")

print(df.info())

label = LabelEncoder()

#auto
# df['PreferredMode_label'] = label.fit_transform(df["PreferredMode"])
# print(df[['PreferredMode',"PreferredMode_label"]])

#maunual
# premode = {"Online":0,"Offline":1,"Hybrid":2}
# df["PreferredMode_label_manual"] = df["PreferredMode"].map(premode)
# print(df[["PreferredMode","PreferredMode_label_manual"]])

#onehotencoding drop=first
print(df["PreferredMode"].unique())
one = OneHotEncoder(drop="first",sparse_output=False)
encoded = one.fit_transform(df[["PreferredMode"]])
print(encoded)
feature_name = one.get_feature_names_out(["PreferredMode"])
df1 = pd.DataFrame(encoded,columns=[feature_name])
data = pd.concat([df,df1],axis=1)
print(data)

#onehotencoding drop=if_binary
# print(df["PreferredMode"].unique())
# one = OneHotEncoder(drop='if_binary',sparse_output=False)
# encoded = one.fit_transform(df[["PreferredMode"]])
# print(encoded)
# df1 = pd.DataFrame(encoded,columns=["learning_hybrid",'learning_offline','learning_online'])
# data = pd.concat([df,df1],axis=1)
# print(data)

#dummies in onehotencoder 
# dummies = pd.get_dummies(df,columns=["PreferredMode"],drop_first=True)
# print(dummies)