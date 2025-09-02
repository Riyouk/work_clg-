import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import pandas as pd

# Create dataset as dictionary
data = {
    "ID": [1,2,3,4,5,6,7,8,9,10,
           11,12,13,14,15,16,17,18,19,20],
    "Height_cm": [172,158,181,165,174,160,169,155,182,163,
                  178,167,185,161,176,159,180,164,183,157],
    "Weight_kg": [68,54,82,59,75,50,64,48,85,55,
                  72,60,90,52,70,49,78,57,88,51],
    "Age": [25,22,28,24,30,21,27,20,26,23,
            29,25,32,22,28,21,27,24,31,20],
    "Gender": ["M","F","M","F","M","F","M","F","M","F",
               "M","F","M","F","M","F","M","F","M","F"]
}

# Convert to pandas DataFrame
df = pd.DataFrame(data)
print(df)

x = df[['Height_cm',"Weight_kg","Age"]]
y = df["Gender"]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

sns.scatterplot(x=x_scaled[:,0],y=x_scaled[:,1],hue=y)
plt.show()

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)
print(pca.explained_variance_)
sns.scatterplot(x=x_pca[:,0],y=x_pca[:,1],hue=y)
plt.show()




x_train,x_test,y_train,y_test = train_test_split(x_pca,y,test_size=0.3,random_state=42)

model = LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
