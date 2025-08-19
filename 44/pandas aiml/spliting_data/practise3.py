from sklearn.model_selection import train_test_split
import pandas as pd 

df = pd.read_csv("C:/44/pandas aiml/data_loan.csv")
print(df.info())
y = df["Loan_length"]
feature_set = ["ID","Default","Loan_type","Gender","Age","Degree","Income","Credit_score","Loan_length","Signers","Citizenship"]
x = df[feature_set]
z = df['Gender']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=40,shuffle=True,stratify=y)
# print('x_train size :',x_train)
# print('x_train size :',x_train["Gender"].value_counts())
# print('x_test size :',x_test["Gender"].value_counts())
print('y_train size :',y_train.value_counts())





# #ai 
# from sklearn.model_selection import train_test_split
# import pandas as pd

# # Example simulated data
# data = {
#     'age': [10, 14, 50, 42, 33, 54, 29, 62, 28, 30],
#     'result': ['Healthy', 'Healthy', 'Disease', 'Healthy', 'Healthy','Disease', 'Healthy', 'Healthy', 'Disease', 'Healthy']
# }
# df = pd.DataFrame(data)
# X = df[['age']]
# y = df['result']

# # Stratified split: this keeps same 'Disease'/'Healthy' ratio in both sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=0, stratify=y
# )
# print("Train distribution:\n", y_train.value_counts(normalize=True))
# print("Test distribution:\n", y_test.value_counts(normalize=True))
