from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd 

iris = load_iris()
x = pd.DataFrame(iris.data,columns=iris.feature_names)
print(x.head(10))
y = iris.target
print(x.info())
print(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,train_size=0.7,random_state=42,shuffle=True)
print('x_train size :',x_train.shape)
print('x_test size :',x_train.shape)
print('y_train size :',x_train.shape)
print('y_test size :',x_train.shape)