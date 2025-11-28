import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv("Mall_Customers.csv")

# Label encode gender
df["Genre"] = df["Genre"].map({"Female": 0, "Male": 1})

# Add z-score columns (optional for inspection)
df["z_age"] = zscore(df["Age"])
df["z_income"] = zscore(df["Annual Income (k$)"])
df["z_spending"] = zscore(df["Spending Score (1-100)"])

# Drop ID & z-score columns
df.drop(columns=["CustomerID", "z_age", "z_income", "z_spending"], inplace=True)

# ---------------------------------------------------------
# SCALING
# ---------------------------------------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = df.copy()

df_scaled[["Age", "Annual Income (k$)", "Spending Score (1-100)"]] = scaler.fit_transform(
    df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
)

print(df_scaled.head())

# ---------------------------------------------------------
# K-MEANS (Same as before)
# ---------------------------------------------------------
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

n_clusters = range(2, 10)
silhouette_vals = []
inertia_vals = []

for k in n_clusters:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    preds = km.fit_predict(df_scaled)
    inertia_vals.append(km.inertia_)
    silhouette_vals.append(silhouette_score(df_scaled, preds))

# ---------------------------------------------------------
# PCA
# ---------------------------------------------------------
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_coords = pca.fit_transform(df_scaled)

print("\nExplained variance by PCA components:", pca.explained_variance_ratio_)

# # ---------------------------------------------------------
# # VISUALIZE CLUSTERS WITH BEST K USING PCA
# # ---------------------------------------------------------
best_k = 5  # choose based on elbow/silhouette
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(df_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(pca_coords[:, 0], pca_coords[:, 1], c=cluster_labels, cmap="tab10", s=50)
plt.title("Customer Segmentation using PCA + KMeans")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()

# # ---------------------------------------------------------
# # OPTIONAL: 3D PCA (uncomment to use)
# # ---------------------------------------------------------
# from mpl_toolkits.mplot3d import Axes3D
# pca3 = PCA(n_components=3)
# pca3_coords = pca3.fit_transform(df_scaled)
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(pca3_coords[:,0], pca3_coords[:,1], pca3_coords[:,2], c=cluster_labels, cmap='tab10', s=50)
# ax.set_title("3D PCA Visualization")
# ax.set_xlabel("PC1")
# ax.set_ylabel("PC2")
# ax.set_zlabel("PC3")
# plt.show()


# ---------------------------------------------------------
# ELBOW METHOD PLOT
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(n_clusters, inertia_vals, marker='o')
plt.title("Elbow Method - Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.grid(True)
plt.show()

# ---------------------------------------------------------
# SILHOUETTE SCORE PLOT
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(n_clusters, silhouette_vals, marker='o')
plt.title("Silhouette Score vs Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()
