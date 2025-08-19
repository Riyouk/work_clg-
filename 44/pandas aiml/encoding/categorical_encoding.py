import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

lebel = LabelEncoder()
df = pd.read_csv("C:/44/pandas aiml/data_loan.csv")
print(df.info())

#LABEL ENCODING 
# df["Gender_label"] = lebel.fit_transform(df['Gender'])
# print(df["Gender_label"])
# print(df['Degree'].unique())

#maunual label encoding 
# degree = {'HS      ':0,'College ':1,'Graduate':2}
# df["Degree_label"] = df['Degree'].map(degree)
# print(df[['Degree','Degree_label']])


#onehotencoding 

# df[["college_onelabel","Graduate_label"]] = one.fit_transform(df["Degree"])
# print(df[["college_onelabel","Graduate_label"]])

# one = OneHotEncoder(drop='first')
# df['encoded'] = one.fit_transform(df["Degree"])
# print(df['encoded'])

# dummies = pd.get_dummies(df,columns=["Degree"],drop_first=True)
# print(dummies)

# one = OneHotEncoder(drop='first',sparse_output=True)
# encoded = one.fit_transform(df[["Degree"]])
# print(encoded)
# df1 = pd.DataFrame(encoded,columns=[one.get_feature_names_out(['Degree'])])
# print(df1)
# data = pd.concat([df,df1],axis=1)
# print(data)

one = OneHotEncoder(drop='first', sparse_output=True)
encoded = one.fit_transform(df[["Degree"]])
print(encoded)
df1 = pd.DataFrame(encoded.toarray(), columns=one.get_feature_names_out(['Degree']))
print(df1)
data = pd.concat([df, df1], axis=1)
print(data)
