# ====================== IMPORT LIBRARIES ======================
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import time  # For timing the hyperparameter tuning process

# Clustering libraries
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler  # Fixed typo in import
from sklearn.metrics import silhouette_score 

# GridSearch for hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# ====================== DATA LOADING AND EXPLORATION ======================
# Load the dataset
df = pd.read_csv("C:/Users/User/forgit uknow/work_clg-/44/pandas aiml/DataSets/DataSets/crop_recommendation.csv") 

# Display basic information about the dataset
print("===== DATASET OVERVIEW =====")
print(df.head(10))  # Display first 10 rows
print(df.info())    # Display data types and missing values
print(df.describe())  # Display statistical summary
print("Duplicate rows:", df.duplicated().sum())  # Check for duplicates
print("Missing values:\n", df.isna().sum())  # Check for missing values
print("Columns:", df.columns)  # Display column names

# ====================== DATA VISUALIZATION (COMMENTED) ======================
# Uncomment these sections if you want to visualize the data
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

# ====================== DATA PREPROCESSING ======================
# Calculate correlation matrix to understand feature relationships
print("\n===== CORRELATION ANALYSIS =====")
corr = df.corr(numeric_only=True) 
# Uncomment to visualize correlation matrix
# sns.heatmap(corr,cmap="crest",annot=True) 
# plt.show() 

# Select numerical features and standardize them
print("\n===== DATA STANDARDIZATION =====")
num = df.select_dtypes(include=["int64","float64"])  # Select only numeric columns
scaler = StandardScaler()  # Initialize the scaler
df1 = pd.DataFrame(scaler.fit_transform(num), columns=num.columns)  # Standardize the data
print("Standardized data (first 10 rows):")
print(df1.head(10))

# ====================== K-MEANS CLUSTERING WITH FIXED PARAMETERS ======================
print("\n===== K-MEANS CLUSTERING (FIXED PARAMETERS) =====")
# Initialize K-means with fixed parameters
kmeans = KMeans(n_clusters=23, random_state=42, n_init=25)  # 23 clusters, 25 initializations
df1['cluster'] = kmeans.fit_predict(df1)  # Fit model and add cluster labels to dataframe

# Display cluster distribution
print("Cluster distribution:")
print(df1["cluster"].value_counts())
print("First few cluster assignments:")
print(df1["cluster"].head())

# Get cluster centers and labels
centroids = kmeans.cluster_centers_  # Coordinates of cluster centers
labels = kmeans.labels_  # Cluster assignments for each data point
print("Cluster centers:")
print(centroids)
print("Cluster labels:")
print(labels)

# Calculate silhouette score to evaluate clustering quality
sil_score = silhouette_score(df1, labels=labels)
print("Silhouette score:", sil_score)
print("Unique crop labels:", df["label"].unique())

# ====================== HYPERPARAMETER TUNING FOR K-MEANS ======================
print("\n===== HYPERPARAMETER TUNING FOR K-MEANS USING GRIDSEARCHCV =====")
start_time = time.time()

# Define the parameter grid for K-means
print("Defining parameter grid for K-means...")
param_grid = {
    'kmeans__n_clusters': range(20, 26),       # Number of clusters to test
    'kmeans__init': ['k-means++', 'random'],   # Initialization method
    'kmeans__n_init': [10, 25, 30],            # Number of initializations
    'kmeans__max_iter': [100, 300]             # Maximum iterations for each run
}

# Create a pipeline that includes preprocessing and K-means
# This ensures that preprocessing is properly included in cross-validation
pipeline = Pipeline([
    ('scaler', StandardScaler()),              # Step 1: Standardize the data
    ('kmeans', KMeans(random_state=42))        # Step 2: Apply K-means clustering
])

# Create a custom scorer function based on silhouette score
# Note: GridSearchCV maximizes the score, and silhouette score is already higher=better
def silhouette_scorer(estimator, X):
    labels = estimator.predict(X)
    score = silhouette_score(X, labels, sample_size=500, random_state=42)
    return score

# Set up GridSearchCV
print("Setting up GridSearchCV...")
grid_search = GridSearchCV(
    estimator=pipeline,                # The pipeline to tune
    param_grid=param_grid,             # Parameter grid to search
    scoring=silhouette_scorer,         # Custom scoring function
    cv=5,                              # 5-fold cross-validation
    n_jobs=-1,                         # Use all available CPU cores
    verbose=1                          # Show progress
)

# Fit the grid search to the data
print("Fitting GridSearchCV to the data...")
grid_search.fit(num)  # Use original numeric data, as pipeline includes scaling

# Print the results of hyperparameter tuning
print("\n===== HYPERPARAMETER TUNING RESULTS =====")
print("Best Parameters:", grid_search.best_params_)
print("Best Silhouette Score: {:.4f}".format(grid_search.best_score_))
print("Time taken for GridSearchCV: {:.2f} seconds".format(time.time() - start_time))

# Get the best model from GridSearchCV
best_pipeline = grid_search.best_estimator_
best_kmeans = best_pipeline.named_steps['kmeans']

# Apply the best model to the data
print("\n===== APPLYING BEST MODEL =====")
best_labels = best_kmeans.predict(best_pipeline.named_steps['scaler'].transform(num))

# Evaluate the best model
best_sil_score = silhouette_score(num, best_labels, sample_size=500, random_state=42)
print("Silhouette Score with Best Parameters:", best_sil_score)
print("Inertia with Best Parameters:", best_kmeans.inertia_)

# Add the best cluster assignments to the dataframe
df1['best_cluster'] = best_labels
print("First few rows with best cluster assignments:")
print(df1[['best_cluster']].head())

# ====================== ORIGINAL K-MEANS EVALUATION CODE ======================
print("\n===== ORIGINAL K-MEANS EVALUATION =====")
# Test different numbers of clusters and evaluate with silhouette score
k_values = range(20, 26)
scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=25)
    labels = kmeans.fit_predict(df1)

    # Using sample_size for speed 
    score = silhouette_score(df1, labels=labels, sample_size=500, random_state=42)
    scores.append(score)
    print(f"k = {k}, silhouette_score = {score:.3f}")

# Plot silhouette score vs k 
plt.figure(figsize=(8, 5))
plt.plot(k_values, scores, marker='o')
plt.xticks(k_values)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette score vs number of clusters")
plt.grid(True)
plt.show()

# Test different numbers of clusters and evaluate with both silhouette score and inertia
k_values = range(20, 26)
scores = []
inertias = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=25)
    labels = kmeans.fit_predict(df1)

    # Inertia (sum of squared distances to nearest cluster center)
    inertias.append(kmeans.inertia_)

    # Using sample_size for speed 
    score = silhouette_score(df1, labels=labels, sample_size=500, random_state=42)
    scores.append(score)
    print(f"k = {k}, inertia={kmeans.inertia_:.2f}, silhouette_score = {score:.3f}")

# Plot both curves side by side 
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Elbow curve 
axs[0].plot(k_values, inertias, marker="o")
axs[0].set_xlabel("NUMBER OF CLUSTERS (K)")
axs[0].set_ylabel("INERTIA")
axs[0].set_title("Elbow Method (INERTIA VS K)")
axs[0].grid(True)

# Silhouette curve
axs[1].plot(k_values, scores, marker="o")  # Fixed: plot scores instead of inertias
axs[1].set_xlabel("NUMBER OF CLUSTERS (K)")
axs[1].set_ylabel("SILHOUETTE SCORE")
axs[1].set_title("SILHOUETTE SCORE VS K")
axs[1].grid(True)

plt.tight_layout()
plt.show()

# ====================== COMPARE ORIGINAL AND OPTIMIZED MODELS ======================
print("\n===== COMPARISON OF ORIGINAL AND OPTIMIZED MODELS =====")
print(f"Original model (k=23): Silhouette score = {sil_score:.4f}")
print(f"Optimized model (k={best_kmeans.n_clusters}): Silhouette score = {best_sil_score:.4f}")
print(f"Improvement: {(best_sil_score - sil_score) / sil_score * 100:.2f}%")

