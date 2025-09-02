# ====================== IMPORT LIBRARIES ======================
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import time

# Machine learning libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from scipy.stats.mstats import winsorize 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, RocCurveDisplay, roc_auc_score
from sklearn.linear_model import LogisticRegression 
from sklearn.decomposition import PCA 
from sklearn.datasets import load_breast_cancer 

# ====================== LOAD AND EXPLORE DATA ======================
print("===== LOADING BREAST CANCER DATASET =====")
# Load the breast cancer dataset
cancer_data = load_breast_cancer() 
df = pd.DataFrame(data=cancer_data.data, columns=cancer_data.feature_names) 
df["diagnosis"] = cancer_data.target 
print("First 5 rows:")
print(df.head(5)) 
print("Dataset shape:", df.shape)
print("Feature names:", df.columns.tolist())
print("Target distribution:", df["diagnosis"].value_counts())

# ====================== DATA PREPARATION ======================
print("\n===== PREPARING DATA =====")
# Split features and target
X = df.drop(columns=["diagnosis"]) 
y = df["diagnosis"] 

# Split data into training and testing sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Training set: {X_train_raw.shape[0]} samples")
print(f"Testing set: {X_test_raw.shape[0]} samples")

# ====================== FIND OPTIMAL NUMBER OF PCA COMPONENTS ======================
print("\n===== FINDING OPTIMAL NUMBER OF PCA COMPONENTS =====")
# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Create PCA with all components
pca_full = PCA()
pca_full.fit(X_train_scaled)

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Plot explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
plt.grid(True)
plt.legend()

# Find number of components for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
plt.annotate(f'95% variance: {n_components_95} components', 
             xy=(n_components_95, 0.95), 
             xytext=(n_components_95+1, 0.9),
             arrowprops=dict(arrowstyle='->'))
# plt.savefig('pca_variance_explained.png')
plt.close()

print(f"Number of components for 95% variance: {n_components_95}")
print("Top 10 components explained variance ratio:")
for i, ratio in enumerate(pca_full.explained_variance_ratio_[:10]):
    print(f"Component {i+1}: {ratio:.4f} ({ratio*100:.2f}%)")

# ====================== HYPERPARAMETER TUNING ======================
print("\n===== HYPERPARAMETER TUNING WITH GRIDSEARCHCV =====")
start_time = time.time()

# Create a pipeline with PCA and Logistic Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Define parameter grid
param_grid = {
    'pca__n_components': [n_components_95, min(n_components_95+2, X.shape[1]), min(n_components_95+4, X.shape[1])],
    'classifier__C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'classifier__solver': ['liblinear', 'lbfgs'],
    'classifier__penalty': ['l2']
}

# Create GridSearchCV
grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit GridSearchCV
grid_search.fit(X_train_raw, y_train)

# Print results
print("\n===== GRID SEARCH RESULTS =====")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
print(f"Time taken: {time.time() - start_time:.2f} seconds")

# ====================== MODEL EVALUATION ======================
print("\n===== EVALUATING BEST MODEL =====")
# Get best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test_raw)
y_pred_proba = best_model.predict_proba(X_test_raw)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {auc:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=cancer_data.target_names,
            yticklabels=cancer_data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('pca_confusion_matrix.png')
plt.close()

# Plot ROC curve
RocCurveDisplay.from_predictions(
    y_test,
    y_pred_proba,
    name=f"ROC Curve (AUC = {auc:.4f})",
    plot_chance_level=True
)
plt.title('ROC Curve')
plt.grid(True)
plt.savefig('pca_roc_curve.png')
plt.close()

# ====================== FEATURE IMPORTANCE ANALYSIS ======================
print("\n===== FEATURE IMPORTANCE ANALYSIS =====")
# Get the PCA components from the best model
best_pca = best_model.named_steps['pca']
n_components = best_pca.n_components_

# Get the feature names
feature_names = X.columns

# Create a DataFrame to store component loadings
component_df = pd.DataFrame()

# For each component, get the loadings and store in the DataFrame
for i in range(n_components):
    component_df[f'PC{i+1}'] = best_pca.components_[i]

# Set the index to be the feature names
component_df.index = feature_names

# Get the top 5 features for each component
print("Top 5 features for each principal component:")
for i in range(min(3, n_components)):  # Show only first 3 components
    print(f"\nPrincipal Component {i+1}:")
    sorted_features = component_df[f'PC{i+1}'].abs().sort_values(ascending=False)
    print(sorted_features.head(5))

# Plot feature importance for the first component
plt.figure(figsize=(12, 8))
component_df['PC1'].abs().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title('Top 10 Features in First Principal Component')
plt.ylabel('Absolute Loading Value')
plt.tight_layout()
plt.savefig('pca_feature_importance.png')
plt.close()

# ====================== COMPARISON WITH ORIGINAL MODEL ======================
print("\n===== COMPARING WITH ORIGINAL MODEL =====")
# Train the original model (6 components)
original_pca = PCA(n_components=6)
X_train_pca_original = original_pca.fit_transform(X_train_scaled)
X_test_pca_original = original_pca.transform(X_test_scaled)

original_model = LogisticRegression()
original_model.fit(X_train_pca_original, y_train)

original_pred = original_model.predict(X_test_pca_original)
original_accuracy = accuracy_score(y_test, original_pred)

print(f"Original model (6 components) accuracy: {original_accuracy:.4f}")
print(f"Optimized model ({best_pca.n_components_} components) accuracy: {accuracy:.4f}")
print(f"Improvement: {(accuracy - original_accuracy) * 100:.2f}%")

print("\n===== SUMMARY =====")
print(f"1. Optimal number of PCA components for 95% variance: {n_components_95}")
print(f"2. Best model parameters: {grid_search.best_params_}")
print(f"3. Best model accuracy: {accuracy:.4f}")
print("4. Visualizations saved: pca_variance_explained.png, pca_confusion_matrix.png, pca_roc_curve.png, pca_feature_importance.png")