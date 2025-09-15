import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

#load dataset 
df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/tests/cia4/loan_default_dataset.csv")
print(df.head(10))

#eda
print(df.info())
print(df.describe())

#missing values
print(df.isna().sum())

#outliers
sns.boxplot(df)
plt.show()

#coverting Dtypes
df["Education"] = df["Education"].astype("category")
df["Property_Area"] = df["Property_Area"].astype("category")
# print(df.info())


#label encoding
print(df["Education"].unique())
print(df["Property_Area"].unique())

edu_r = {"Graduate":0,"Not Graduate":1}
property_a_r = {"Rural":0,"Urban":1,"Semiurban":2}

df["Education"] = df["Education"].map(edu_r)
df["Property_Area"] = df["Property_Area"].map(property_a_r)

print(df.head(4))

#feature selection 
# print(df.columns)
x = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]


# spliting data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)

#model building 

#desion tree
model_d = DecisionTreeClassifier(criterion="entropy",max_depth=4,random_state=42)
model_d.fit(x_train,y_train)

y_pred = model_d.predict(x_test)

print("accuracy_desition_tree :",accuracy_score(y_test,y_pred))
print("confustion matrix :",confusion_matrix(y_test,y_pred))
print("classification_report :",classification_report(y_test,y_pred))

#random forest
model_r = RandomForestClassifier(random_state=42,)
model_r.fit(x_train,y_train)

y_pred = model_r.predict(x_test)

print("accuracy_desition_tree :",accuracy_score(y_test,y_pred))
print("confustion matrix :",confusion_matrix(y_test,y_pred))
print("classification_report :",classification_report(y_test,y_pred))

#hyper parameter 
param_grid = {"n_estimators" : [100,200,300],
              "max_depth" : [4,5,8],
              "max_features" : ['sqrt', 'log2',None]}

grid_search = GridSearchCV(estimator=model_r,param_grid=param_grid,n_jobs=-1,cv=5,verbose=1)

grid_search.fit(x_train,y_train)

best_est = grid_search.best_estimator_


y_pred = best_est.predict(x_test)

print("accuracy_desition_tree :",accuracy_score(y_test,y_pred))
print("confustion matrix :",confusion_matrix(y_test,y_pred))
print("classification_report :",classification_report(y_test,y_pred))


