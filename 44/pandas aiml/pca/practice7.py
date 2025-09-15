import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
cancer_data = load_breast_cancer()
X, y = cancer_data.data, cancer_data.target

# Split train/test first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try different n_components
components_range = range(1, X_train_scaled.shape[1] + 1)
accuracies = []
explained_variances = []

for n in components_range:
    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)

    accuracies.append(accuracy_score(y_test, y_pred))
    explained_variances.append(np.sum(pca.explained_variance_ratio_))

# Plot accuracy vs n_components
plt.figure(figsize=(12,5))

# plt.subplot(1,2,1)
# plt.plot(components_range, accuracies, marker='o')
# plt.xlabel("Number of PCA Components")
# plt.ylabel("Test Accuracy")
# plt.title("Accuracy vs Number of Components")

plt.subplot(1,2,2)
plt.plot(components_range, explained_variances, marker='o')
plt.xlabel("Number of PCA Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance vs Components")

plt.tight_layout()
plt.show()