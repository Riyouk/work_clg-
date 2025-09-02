import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = {
    'height' : [170,165,180,175,160,172,168,177,162,158],
    'weight' : [65,59,75,68,55,70,62,74,58,54],
    'Age' : [30,25,35,28,22,32,27,33,24,21],
    'gender' : [1,0,1,1,0,1,0,1,0,0]
}

df = pd.DataFrame(data)
print(df)

x = df[['height',"weight","Age"]]
y = df["gender"]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

sns.scatterplot(x=x_scaled[:,0],y=x_scaled[:,1],hue=y)
plt.show()

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
sns.scatterplot(x=x_pca[:,0],y=x_pca[:,1],hue=y)
plt.show()


x_train,x_test,y_train,y_test = train_test_split(x_pca,y,test_size=0.3,random_state=42)

model = LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
