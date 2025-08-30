import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import silhouette_score



df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/DataSets/DataSets/crop_recommendation.csv")
print(df.head(10))
print(df.info())
print(df.describe())
print(df.duplicated().sum())
print(df.isna().sum())
print(df.columns)
#visualize 
# sns.scatterplot(data=df,x="temperature",y="humidity",hue="label")
# plt.legend(bbox_to_anchor=(1.05,1),loc="upper left",borderaxespad=0)
# plt.show()


# sns.scatterplot(data=df,x="N",y="P",hue="label")
# plt.legend(bbox_to_anchor=(1.05,1),loc="upper left",borderaxespad=0)
# plt.show()


# sns.scatterplot(data=df,x="N",y="K",hue="label")
# plt.legend(bbox_to_anchor=(1.05,1),loc="upper left",borderaxespad=0)
# plt.show()

# sns.scatterplot(data=df,x="rainfall",y="ph",hue="label")
# plt.legend(bbox_to_anchor=(1.05,1),loc="upper left",borderaxespad=0)
# plt.show()



corr = df.corr(numeric_only=True)
# sns.heatmap(corr,cmap="crest",annot=True)
# sns.heatmap(corr,cmap="crest",annot=True)
# plt.show()

num = df.select_dtypes(include=["int64","float64"])
scaler = StandardScaler()
df1 = pd.DataFrame(scaler.fit_transform(num),columns=num.columns)
# num = scaler.fit_transform(num)
print(df1.head(10))

kmeans = KMeans(n_clusters= 23 , random_state=42 ,n_init=25)
df1['cluster'] = kmeans.fit_predict(df1)
print(df1["cluster"].value_counts())
print(df1["cluster"].head())


# sns.scatterplot(data=df1,x="temperature",y="humidity",hue="cluster",palette="pastel")
# plt.legend(title="cluster",bbox_to_anchor=(1.05,1),loc="upper left",borderaxespad=0)
# plt.show()

centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print(centroids)
print(labels)




sil_score = silhouette_score(df1,labels=labels)
print("silhouette_score : ",sil_score)
print(df["label"].unique())
# plt.scatter()

# k_values = range(2,30)
# scores = []

# for k in k_values:
#     kmeans = KMeans(n_clusters=k,random_state=42,n_init=25)
#     labels = kmeans.fit_predict(df1)

#     #using sample_size for speed 
#     score = silhouette_score(df1,labels=labels,sample_size=500,random_state=42)
#     scores.append(score)
#     print(f"k = {k} , silhouette_score = {score:.3f}")

# #plot silhouette score vs k 
# plt.figure(figsize=(8,5))
# plt.plot(k_values,scores,marker='o')
# plt.xticks(k_values)
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("silhouette score")
# plt.title("silhouette score vs number of clusters ")
# plt.grid(True)
# plt.show()



k_values = range(2,30)
scores = []
inertias = []

for k in k_values:
    kmeans = KMeans(n_clusters=k,random_state=42,n_init=25)
    labels = kmeans.fit_predict(df1)

    #inertia (sum of squared distances to nearest cluster center )
    inertias.append(kmeans.inertia_)


    #using sample_size for speed 
    score = silhouette_score(df1,labels=labels,sample_size=500,random_state=42)
    scores.append(score)
    print(f"k = {k} ,inertia={kmeans.inertia_:2f}, silhouette_score = {score:.3f}")

#plot both curves side by side 
fig,axs = plt.subplots(1,2,figsize=(14,5))

#elbow curve 
axs[0].plot(k_values,inertias,marker="o")
axs[0].set_xlabel("NO OF CLUSTERS (K)")
axs[0].set_ylabel("INERTIA")
axs[0].set_title("Elbow Method (INERTIA VS K)")
axs[0].grid(True)

#silhouette curve
axs[1].plot(k_values,inertias,marker="o")
axs[1].set_xlabel("NO OF CLUSTERS (K)")
axs[1].set_ylabel("silhouette score")
axs[1].set_title("silhouette score VS K)")
axs[1].grid(True)

plt.tight_layout()
plt.show()

