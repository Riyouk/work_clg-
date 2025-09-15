import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,silhouette_score
from sklearn.cluster import KMeans

#load data
df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/tests/cia4/customer_segmentation.csv")
print(df.head(10))

#eda 
print(df.info())
print(df.describe())

#missing values
print(df.isna().sum())

#outliers
sns.boxplot(df)
plt.show()

#drop unwanted columns 
df.drop(columns=["CustomerID"],inplace=True)

# feature selection
x = df
print(x.columns)


#standard scalar 
scalar = StandardScaler()
x[['Age','Annual Income (k$)','Spending Score (1-100)']] = scalar.fit_transform(x[['Age','Annual Income (k$)','Spending Score (1-100)']])
print(x.head(5))


c_range = range(2,10)
silhouette = []
inertia = []

for i in c_range:
    k_means = KMeans(n_clusters=i,n_init=10,max_iter=100,random_state=42)
    y_pred = k_means.fit_predict(x)
    inertia.append(k_means.inertia_)
    silhouette.append(silhouette_score(x,y_pred))

    print(f"clusters : {i} ,inertia : {k_means.inertia_}, silloutte : {silhouette_score(x,y_pred)}")


plt.plot(c_range,silhouette)
plt.title("Sillhoutte")
plt.xlabel("range")
plt.ylabel("sillhoutte")
plt.show()

plt.plot(c_range,inertia,marker='o')
plt.xlabel("range")
plt.ylabel("inertia")
plt.title("ELBOW")
plt.show()