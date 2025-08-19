from sklearn.model_selection import train_test_split
import pandas as pd 

df = pd.read_csv("C:/44/pandas aiml/titanic.csv")
print(df.info())
y = df['Survived']
print(y.head(5))
future_set = ["PassengerId","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"] 
x = df[future_set]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=40,shuffle=True)
print('x_train size :',x_train.shape)
print('x_test size :',x_train.shape)
print('y_train size :',x_train.shape)
print('y_test size :',x_train.shape)
print('x_train size :',x_train)
print('x_test size :',x_train)
print('y_train size :',y_train)
print('y_test size :',y_train)